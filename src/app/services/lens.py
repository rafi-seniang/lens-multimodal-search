"""Lens: semantic and hybrid photograph search backed by Typesense (or numpy in tests) plus image ingestion."""

import asyncio
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.errors import InvalidAction, NotFound
from app.engines.embedder import Embedder
from app.engines.keyword import KeywordIndex
from app.repositories.collection import Collection, remove_uploaded_file, save_uploads, write_uploaded_files
from app.repositories.typesense_store import TypesensePhotographStore
from app.schemas.search import Hit, IngestResult, Mode, OriginFilter, SearchResult


def top_positions(scores: np.ndarray, limit: int) -> np.ndarray:
    """Positions of the highest scores, best first. Only the top `limit` are sorted."""
    if len(scores) == 0:
        return np.empty(0, dtype=np.intp)
    limit = min(limit, len(scores))
    top = np.argpartition(-scores, limit - 1)[:limit]
    return top[np.argsort(-scores[top])]


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank of every photograph by score, 1 for the best."""
    order = np.argsort(-scores)  # fusion needs every rank across the active slice
    positions = np.empty_like(order)
    positions[order] = np.arange(1, len(scores) + 1)
    return positions


class Lens:
    def __init__(
        self,
        collection: Collection,
        embedder: Embedder,
        keyword: KeywordIndex,
        fusion_k: int,
        picture_weight: float = 0.8,
        store: TypesensePhotographStore | None = None,
        collection_dir: Path | None = None,
    ) -> None:
        self.collection = collection
        self._embedder = embedder
        self._keyword = keyword
        self._fusion_k = fusion_k
        self._picture_weight = picture_weight
        self._store = store
        self._collection_dir = collection_dir
        self._write_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self.collection.model

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vector_store_name(self) -> str:
        return "Typesense 27.1" if self._store is not None else "in-memory"

    def _allowed_positions(self, origin: OriginFilter) -> np.ndarray:
        if origin == "all":
            return np.arange(len(self.collection.photographs))
        return np.array(
            [index for index, item in enumerate(self.collection.photographs) if item.origin == origin],
            dtype=np.intp,
        )

    def _memory_result(
        self,
        query: str,
        mode: str,
        scores: np.ndarray,
        caption_scores: np.ndarray,
        limit: int,
        origin: OriginFilter,
        started: float,
    ) -> SearchResult:
        allowed = self._allowed_positions(origin)
        if len(allowed) == 0:
            return SearchResult(
                query=query, mode=mode, took_ms=round((time.perf_counter() - started) * 1000, 1), hits=[]
            )
        sliced_scores = scores[allowed]
        hits = [
            Hit(
                photograph=self.collection.photographs[int(allowed[local_pos])],
                score=round(float(sliced_scores[local_pos]), 4),
                in_caption=bool(caption_scores[int(allowed[local_pos])] > 0),
            )
            for local_pos in top_positions(sliced_scores, limit)
            if sliced_scores[local_pos] > 0 or mode != "caption"
        ]
        return SearchResult(query=query, mode=mode, took_ms=round((time.perf_counter() - started) * 1000, 1), hits=hits)

    async def search_text(self, query: str, mode: Mode, limit: int, origin: OriginFilter = "all") -> SearchResult:
        started = time.perf_counter()
        caption_scores = self._keyword.scores(query)
        if self._store is not None:
            caption_ids = {
                self.collection.photographs[index].id for index, score in enumerate(caption_scores) if score > 0
            }
            vector = None if mode == "caption" else await self._embedder.text(query)
            hits = await self._store.search_text(
                query=query,
                vector=vector,
                mode=mode,
                alpha=self._picture_weight,
                limit=limit,
                origin=origin,
                caption_ids=caption_ids,
            )
            return SearchResult(
                query=query, mode=mode, took_ms=round((time.perf_counter() - started) * 1000, 1), hits=hits
            )

        if mode == "caption":
            return self._memory_result(query, mode, caption_scores, caption_scores, limit, origin, started)
        picture_scores = self.collection.vectors @ await self._embedder.text(query)
        if mode == "picture":
            return self._memory_result(query, mode, picture_scores, caption_scores, limit, origin, started)
        # Weighted reciprocal rank fusion: a strong caption match lifts a top picture, weak function words do not
        fused = self._picture_weight / (self._fusion_k + ranks(picture_scores))
        fused += np.where(caption_scores > 0, (1 - self._picture_weight) / (self._fusion_k + ranks(caption_scores)), 0)
        return self._memory_result(query, mode, fused, caption_scores, limit, origin, started)

    async def search_image(self, image: Image.Image, limit: int, origin: OriginFilter = "all") -> SearchResult:
        started = time.perf_counter()
        vector = await self._embedder.image(image)
        if self._store is not None:
            hits = await self._store.search_vector(vector, limit, origin)
            return SearchResult(
                query="an uploaded picture",
                mode="picture",
                took_ms=round((time.perf_counter() - started) * 1000, 1),
                hits=hits,
            )
        scores = self.collection.vectors @ vector
        return self._memory_result(
            "an uploaded picture", "picture", scores, np.zeros_like(scores), limit, origin, started
        )

    async def similar(self, photograph_id: str, limit: int) -> SearchResult:
        started = time.perf_counter()
        if self._store is not None:
            title, hits = await self._store.similar(photograph_id, limit)
            return SearchResult(
                query=f"like {title}",
                mode="picture",
                took_ms=round((time.perf_counter() - started) * 1000, 1),
                hits=hits,
            )
        position = self.collection.position_by_id.get(photograph_id)
        if position is None:
            raise NotFound(f"No photograph {photograph_id} in the collection.")
        scores = self.collection.vectors @ self.collection.vectors[position]
        scores[position] = -np.inf  # a photograph is always most like itself
        title = self.collection.photographs[position].title
        return self._memory_result(f"like {title}", "picture", scores, np.zeros_like(scores), limit, "all", started)

    async def list_photographs(self, origin: OriginFilter, limit: int, tag: str | None = None) -> SearchResult:
        started = time.perf_counter()
        cleaned_tag = tag.strip() if tag and tag.strip() else None
        label = cleaned_tag or ("uploaded pictures" if origin == "upload" else "collection")
        if self._store is not None:
            hits = await self._store.list_recent(origin, limit, tag=cleaned_tag)
            return SearchResult(
                query=label, mode="browse", took_ms=round((time.perf_counter() - started) * 1000, 1), hits=hits
            )
        allowed = [
            int(pos)
            for pos in self._allowed_positions(origin)
            if cleaned_tag is None or cleaned_tag in self.collection.photographs[int(pos)].tags
        ]
        # Browse shows newest uploads first when created_at is set, keeping archive order otherwise
        ordered = sorted(
            allowed,
            key=lambda pos: (self.collection.photographs[pos].created_at, -pos),
            reverse=True,
        )[:limit]
        hits = [Hit(photograph=self.collection.photographs[pos], score=1.0, in_caption=False) for pos in ordered]
        return SearchResult(
            query=label, mode="browse", took_ms=round((time.perf_counter() - started) * 1000, 1), hits=hits
        )

    async def add_photographs(
        self,
        uploads: list[tuple[str, Image.Image]],
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> IngestResult:
        if self._collection_dir is None:
            raise InvalidAction("Collection directory is not configured for uploads.")
        started = time.perf_counter()
        cleaned_title = title.strip() if title and title.strip() else None
        cleaned_tags = [tag.strip() for tag in (tags or []) if tag.strip()]
        vectors = await self._embedder.images([image for _, image in uploads])
        added = await asyncio.to_thread(
            write_uploaded_files, self._collection_dir, uploads, cleaned_title, cleaned_tags
        )
        async with self._write_lock:
            if self._store is not None:
                await self._store.upsert(added, vectors)
            self.collection.append(added, vectors)
            self._keyword = KeywordIndex(self.collection.photographs)
            await asyncio.to_thread(
                save_uploads, self._collection_dir, self.collection.photographs, self.collection.vectors
            )
        return IngestResult(
            added=added,
            total_photographs=len(self.collection.photographs),
            took_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def delete_photograph(self, photograph_id: str) -> int:
        if self._collection_dir is None:
            raise InvalidAction("Collection directory is not configured for uploads.")
        async with self._write_lock:
            position = self.collection.position_by_id.get(photograph_id)
            if position is None:
                raise NotFound(f"No photograph {photograph_id} in the collection.")
            target = self.collection.photographs[position]
            if target.origin != "upload":
                raise InvalidAction(
                    "Archive photographs from The Met are read-only. Only uploaded pictures can be deleted."
                )
            if self._store is not None:
                await self._store.delete(photograph_id)
            self.collection.remove(photograph_id)
            self._keyword = KeywordIndex(self.collection.photographs)
            await asyncio.to_thread(
                save_uploads, self._collection_dir, self.collection.photographs, self.collection.vectors
            )
            await asyncio.to_thread(remove_uploaded_file, self._collection_dir, target.image)
            return len(self.collection.photographs)
