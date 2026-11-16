"""Lens: photograph search by meaning alone, backed by Typesense (or numpy in tests), plus image ingestion.

A query, written or a picture, becomes a Gemini Embedding 2 vector and is compared with each picture's vector. No
caption, title or tag takes part, so a hit is found from what the picture shows.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.engines.embedder import Embedder
from app.features.search.collection import Collection, remove_uploaded_file, save_uploads, write_uploaded_files
from app.features.search.schemas import Hit, IngestResult, OriginFilter, SearchResult
from app.features.search.typesense_store import TypesensePhotographStore
from app.platform.errors import InvalidAction, NotFound


def top_positions(scores: np.ndarray, limit: int) -> np.ndarray:
    """Positions of the highest scores, best first. Only the top `limit` are sorted."""
    if len(scores) == 0:
        return np.empty(0, dtype=np.intp)
    limit = min(limit, len(scores))
    top = np.argpartition(-scores, limit - 1)[:limit]
    return top[np.argsort(-scores[top])]


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


class Lens:
    def __init__(
        self,
        collection: Collection,
        embedder: Embedder,
        min_picture_score: float = 0.0,
        store: TypesensePhotographStore | None = None,
        collection_dir: Path | None = None,
    ) -> None:
        self.collection = collection
        self._embedder = embedder
        self._min_picture_score = min_picture_score
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

    def _memory_hits(self, scores: np.ndarray, limit: int, origin: OriginFilter, cutoff: float) -> list[Hit]:
        """The best photographs by cosine similarity, at or above the cutoff."""
        allowed = self._allowed_positions(origin)
        if len(allowed) == 0:
            return []
        sliced = scores[allowed]
        return [
            Hit(photograph=self.collection.photographs[int(allowed[position])], score=round(float(sliced[position]), 4))
            for position in top_positions(sliced, limit)
            if sliced[position] >= cutoff
        ]

    async def _search_vector(self, query: str, vector: np.ndarray, limit: int, origin: OriginFilter) -> SearchResult:
        started = time.perf_counter()
        if self._store is not None:
            hits = await self._store.search_vector(vector, limit, origin, min_picture_score=self._min_picture_score)
        else:
            hits = self._memory_hits(self.collection.vectors @ vector, limit, origin, self._min_picture_score)
        return SearchResult(query=query, mode="picture", took_ms=elapsed_ms(started), hits=hits)

    async def search_text(self, query: str, limit: int, origin: OriginFilter = "all") -> SearchResult:
        return await self._search_vector(query, await self._embedder.text(query), limit, origin)

    async def search_image(self, image: Image.Image, limit: int, origin: OriginFilter = "all") -> SearchResult:
        return await self._search_vector("an uploaded picture", await self._embedder.image(image), limit, origin)

    async def similar(self, photograph_id: str, limit: int) -> SearchResult:
        started = time.perf_counter()
        if self._store is not None:
            title, hits = await self._store.similar(photograph_id, limit)
            return SearchResult(query=f"like {title}", mode="picture", took_ms=elapsed_ms(started), hits=hits)
        position = self.collection.position_by_id.get(photograph_id)
        if position is None:
            raise NotFound(f"No photograph {photograph_id} in the collection.")
        scores = self.collection.vectors @ self.collection.vectors[position]
        scores[position] = -np.inf  # a photograph is always most like itself
        title = self.collection.photographs[position].title
        hits = self._memory_hits(scores, limit, "all", cutoff=-np.inf)  # the nearest, however near
        return SearchResult(query=f"like {title}", mode="picture", took_ms=elapsed_ms(started), hits=hits)

    async def list_photographs(self, origin: OriginFilter, limit: int) -> SearchResult:
        started = time.perf_counter()
        label = "uploaded pictures" if origin == "upload" else "collection"
        if self._store is not None:
            hits = await self._store.list_recent(origin, limit)
        else:
            # Newest uploads first when created_at is set, archive order otherwise
            ordered = sorted(
                (int(position) for position in self._allowed_positions(origin)),
                key=lambda position: (self.collection.photographs[position].created_at, -position),
                reverse=True,
            )[:limit]
            hits = [Hit(photograph=self.collection.photographs[position], score=1.0) for position in ordered]
        return SearchResult(query=label, mode="browse", took_ms=elapsed_ms(started), hits=hits)

    async def add_photographs(self, uploads: list[tuple[str, Image.Image]]) -> IngestResult:
        """Embeds and stores pictures as they are: no title or tag is asked for, since search never reads one."""
        if self._collection_dir is None:
            raise InvalidAction("Collection directory is not configured for uploads.")
        started = time.perf_counter()
        vectors = await self._embedder.images([image for _, image in uploads])
        added = await asyncio.to_thread(write_uploaded_files, self._collection_dir, uploads)
        async with self._write_lock:
            if self._store is not None:
                await self._store.upsert(added, vectors)
            self.collection.append(added, vectors)
            await asyncio.to_thread(
                save_uploads, self._collection_dir, self.collection.photographs, self.collection.vectors
            )
        return IngestResult(
            added=added,
            total_photographs=len(self.collection.photographs),
            took_ms=elapsed_ms(started),
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
                raise InvalidAction("Archive photographs are read-only. Only uploaded pictures can be deleted.")
            if self._store is not None:
                await self._store.delete(photograph_id)
            self.collection.remove(photograph_id)
            await asyncio.to_thread(
                save_uploads, self._collection_dir, self.collection.photographs, self.collection.vectors
            )
            await asyncio.to_thread(remove_uploaded_file, self._collection_dir, target.image)
            return len(self.collection.photographs)
