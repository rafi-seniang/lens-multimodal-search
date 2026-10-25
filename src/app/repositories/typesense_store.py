"""Typesense vector store over its REST API. Embeddings are computed on the GPU and stored in Typesense."""

import json
from typing import Any

import httpx
import numpy as np

from app.core.errors import NotFound, StoreUnavailable
from app.schemas.photographs import Photograph
from app.schemas.search import Hit, Mode, OriginFilter

TEXT_FIELDS = "title,artist,date,tags"
IMPORT_BATCH_SIZE = 100


def collection_schema(name: str, dimensions: int) -> dict[str, Any]:
    return {
        "name": name,
        "fields": [
            {"name": "title", "type": "string"},
            {"name": "artist", "type": "string", "optional": True},
            {"name": "date", "type": "string", "optional": True},
            {"name": "medium", "type": "string", "optional": True},
            {"name": "tags", "type": "string[]", "facet": True, "optional": True},
            {"name": "image", "type": "string"},
            {"name": "width", "type": "int32"},
            {"name": "height", "type": "int32"},
            {"name": "source_url", "type": "string", "optional": True},
            {"name": "licence", "type": "string", "optional": True},
            {"name": "origin", "type": "string", "facet": True},
            {"name": "created_at", "type": "int64", "sort": True},
            {"name": "embedding", "type": "float[]", "num_dim": dimensions},
        ],
        "default_sorting_field": "created_at",
    }


def origin_clause(origin: OriginFilter, tag: str | None = None) -> str:
    clauses: list[str] = []
    if origin != "all":
        clauses.append(f"origin:=`{origin}`")
    if tag:
        clauses.append(f"tags:=`{tag}`")
    return " && ".join(clauses)


def format_vector(vector: np.ndarray) -> str:
    return ",".join(f"{float(value):.6f}" for value in vector)


def vector_query(vector_text: str, k: int, ef: int, alpha: float | None = None) -> str:
    """Typesense's vector_query. `vector_text` is the formatted vector, or empty with `id:` to search by a document."""
    hybrid_weight = "" if alpha is None else f", alpha:{alpha}"
    return f"embedding:({vector_text}, k:{k}{hybrid_weight}, ef:{ef})"


def document_from_photograph(photograph: Photograph, vector: np.ndarray) -> dict[str, Any]:
    payload = photograph.model_dump()
    payload["embedding"] = [float(value) for value in vector]
    return payload


def hit_from_typesense(raw: dict[str, Any], mode: Mode, caption_ids: set[str]) -> Hit:
    doc = raw["document"]
    photograph = Photograph.model_validate(doc)
    distance = raw.get("vector_distance")
    if mode == "hybrid":
        score = float(raw.get("hybrid_search_info", {}).get("rank_fusion_score", 0.0))
    elif mode == "picture":
        score = 1.0 - float(distance if distance is not None else 1.0)
    else:
        score = float(raw.get("text_match", 0))
    in_caption = photograph.id in caption_ids if mode != "caption" else True
    return Hit(photograph=photograph, score=round(score, 4), in_caption=in_caption)


class TypesensePhotographStore:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        search_ef: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._collection = collection_name
        self._search_ef = search_ef
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=30.0,
            headers={"X-TYPESENSE-API-KEY": api_key},
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise StoreUnavailable(f"Typesense is unreachable: {error}") from error
        if response.status_code >= 400:
            raise StoreUnavailable(f"Typesense {method} {path} returned {response.status_code}: {response.text[:240]}")
        return response

    async def bootstrap(self, photographs: list[Photograph], vectors: np.ndarray) -> None:
        dimensions = int(vectors.shape[1])
        try:
            response = await self._http.get(f"/collections/{self._collection}")
        except httpx.HTTPError as error:
            raise StoreUnavailable(
                f"Typesense is unreachable at {self._http.base_url}. Run: docker compose up -d typesense"
            ) from error

        needs_create = response.status_code == 404
        existing_docs = 0
        if not needs_create:
            if response.status_code >= 400:
                raise StoreUnavailable(f"Typesense returned {response.status_code}: {response.text[:240]}")
            info = response.json()
            existing_dim = next(
                (field.get("num_dim", 0) for field in info.get("fields", []) if field["name"] == "embedding"), 0
            )
            existing_docs = int(info.get("num_documents", 0))
            if existing_dim != dimensions:
                await self._call("DELETE", f"/collections/{self._collection}")
                needs_create = True

        if needs_create:
            await self._call("POST", "/collections", json=collection_schema(self._collection, dimensions))

        if needs_create or existing_docs != len(photographs):
            await self.upsert(photographs, vectors)
            return

        upload_indices = [index for index, item in enumerate(photographs) if item.origin == "upload"]
        if upload_indices:
            await self.upsert([photographs[index] for index in upload_indices], vectors[upload_indices])

    async def upsert(self, photographs: list[Photograph], vectors: np.ndarray) -> None:
        # Typesense JSONL bulk import handles 100 vectorized documents per call cleanly
        for start in range(0, len(photographs), IMPORT_BATCH_SIZE):
            batch_photos = photographs[start : start + IMPORT_BATCH_SIZE]
            batch_vectors = vectors[start : start + IMPORT_BATCH_SIZE]
            lines = "\n".join(
                json.dumps(document_from_photograph(photo, vector))
                for photo, vector in zip(batch_photos, batch_vectors, strict=True)
            )
            response = await self._call(
                "POST",
                f"/collections/{self._collection}/documents/import",
                params={"action": "upsert"},
                content=lines,
                headers={"Content-Type": "text/plain"},
            )
            results = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            failed = [row for row in results if not row.get("success")]
            if failed:
                raise StoreUnavailable(f"Typesense rejected {len(failed)} pictures: {failed[0].get('error')}")

    async def delete(self, photograph_id: str) -> None:
        response = await self._http.delete(f"/collections/{self._collection}/documents/{photograph_id}")
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise StoreUnavailable(f"Typesense delete failed ({response.status_code}): {response.text[:240]}")

    async def search_text(
        self,
        query: str,
        vector: np.ndarray | None,
        mode: Mode,
        alpha: float,
        limit: int,
        origin: OriginFilter,
        caption_ids: set[str],
    ) -> list[Hit]:
        params: dict[str, Any] = {
            "collection": self._collection,
            "per_page": limit,
            "exclude_fields": "embedding",
        }
        if mode == "caption":
            params |= {"q": query, "query_by": TEXT_FIELDS}
        elif mode == "picture":
            assert vector is not None
            params |= {"q": "*", "vector_query": vector_query(f"[{format_vector(vector)}]", limit, self._search_ef)}
        else:
            assert vector is not None
            params |= {
                "q": query,
                "query_by": TEXT_FIELDS,
                "vector_query": vector_query(f"[{format_vector(vector)}]", max(limit, 50), self._search_ef, alpha),
            }
        if clause := origin_clause(origin):
            params["filter_by"] = clause

        result = (await self._call("POST", "/multi_search", json={"searches": [params]})).json()["results"][0]
        if "error" in result:
            raise StoreUnavailable(f"Typesense search error: {result['error']}")
        return [hit_from_typesense(raw, mode, caption_ids) for raw in result.get("hits", [])]

    async def search_vector(self, vector: np.ndarray, limit: int, origin: OriginFilter) -> list[Hit]:
        params: dict[str, Any] = {
            "collection": self._collection,
            "q": "*",
            "vector_query": vector_query(f"[{format_vector(vector)}]", limit, self._search_ef),
            "per_page": limit,
            "exclude_fields": "embedding",
        }
        if clause := origin_clause(origin):
            params["filter_by"] = clause

        result = (await self._call("POST", "/multi_search", json={"searches": [params]})).json()["results"][0]
        if "error" in result:
            raise StoreUnavailable(f"Typesense vector search error: {result['error']}")
        return [hit_from_typesense(raw, "picture", set()) for raw in result.get("hits", [])]

    async def similar(self, photograph_id: str, limit: int) -> tuple[str, list[Hit]]:
        try:
            doc_response = await self._http.get(
                f"/collections/{self._collection}/documents/{photograph_id}",
                params={"exclude_fields": "embedding"},
            )
        except httpx.HTTPError as error:
            raise StoreUnavailable(f"Typesense is unreachable: {error}") from error
        if doc_response.status_code == 404:
            raise NotFound(f"No photograph {photograph_id} in the collection.")
        if doc_response.status_code >= 400:
            raise StoreUnavailable(f"Typesense read failed ({doc_response.status_code}): {doc_response.text[:240]}")

        title = str(doc_response.json().get("title", photograph_id))
        params: dict[str, Any] = {
            "collection": self._collection,
            "q": "*",
            "vector_query": vector_query(f"[], id: {photograph_id}", limit, self._search_ef),
            "per_page": limit,
            "exclude_fields": "embedding",
        }
        result = (await self._call("POST", "/multi_search", json={"searches": [params]})).json()["results"][0]
        if "error" in result:
            raise StoreUnavailable(f"Typesense neighbour search error: {result['error']}")
        hits = [
            hit_from_typesense(raw, "picture", set())
            for raw in result.get("hits", [])
            if raw["document"]["id"] != photograph_id
        ]
        return title, hits[:limit]

    async def list_recent(self, origin: OriginFilter, limit: int, tag: str | None = None) -> list[Hit]:
        params: dict[str, Any] = {
            "q": "*",
            "sort_by": "created_at:desc",
            "per_page": limit,
            "exclude_fields": "embedding",
        }
        if clause := origin_clause(origin, tag):
            params["filter_by"] = clause
        result = (await self._call("GET", f"/collections/{self._collection}/documents/search", params=params)).json()
        return [
            Hit(photograph=Photograph.model_validate(raw["document"]), score=1.0, in_caption=False)
            for raw in result.get("hits", [])
        ]
