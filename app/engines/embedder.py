"""Gemini Embedding 2: text and pictures in one normalized vector space, so words and pictures match."""

import io
import os
from collections import OrderedDict
from typing import Any, Protocol

import httpx
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image

from app.platform.errors import EmbedderUnavailable

# gemini-embedding-2 ignores task_type. Google's "task: search result | query:" prefix ranked the same here and narrowed
# the gap between real and no-match queries (README "Query prefix"), so queries are sent bare like the pictures
# Repeated queries (the page's default and its examples) cost no API call. About 3 KB each at 768 dimensions
QUERY_CACHE_SIZE = 1024


class Embedder(Protocol):
    name: str

    async def text(self, query: str) -> np.ndarray: ...

    async def image(self, image: Image.Image) -> np.ndarray: ...

    async def images(self, batch: list[Image.Image]) -> np.ndarray: ...


def _image_part(image: Image.Image) -> types.Part:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)


def _load_api_keys() -> list[str]:
    """Read primary and backup Gemini API keys from env, supporting comma-separated lists."""
    raw_values = [
        os.environ.get(name, "")
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEYS")
    ]
    keys = [part.strip() for raw in raw_values for part in raw.split(",") if part.strip()]
    unique = list(dict.fromkeys(keys))
    if not unique:
        raise SystemExit("GEMINI_API_KEY is empty in the secrets file or environment. Lens needs it.")
    return unique


class GeminiEmbedder:
    """Async Gemini Embedding 2 client. Vectors return unit-length so dot product equals cosine similarity."""

    def __init__(
        self,
        model_id: str = "gemini-embedding-2",
        dimensions: int = 768,
        timeout_seconds: float = 30.0,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_keys = _load_api_keys()
        self.name = model_id
        self._model = model_id
        self._dimensions = dimensions
        retry = types.HttpRetryOptions(
            attempts=1 if http_async_client is not None else 2,
            initial_delay=0.5,
            max_delay=2.0,
        )
        http_options = types.HttpOptions(
            timeout=int(timeout_seconds * 1000),
            retry_options=retry,
            httpx_async_client=http_async_client,
        )
        self._clients = [genai.Client(api_key=key, http_options=http_options) for key in api_keys]
        self._active_index = 0
        self._query_vectors: OrderedDict[str, np.ndarray] = OrderedDict()

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aio.aclose()

    def _to_vectors(self, response: types.EmbedContentResponse) -> np.ndarray:
        embeddings = response.embeddings or []
        if not embeddings:
            raise EmbedderUnavailable("Gemini returned no embeddings.")
        matrix = np.asarray([item.values for item in embeddings], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimensions:
            raise EmbedderUnavailable(
                f"Gemini returned embeddings with shape {matrix.shape}, expected (*, {self._dimensions})."
            )
        return _normalize(matrix)

    async def _embed(self, contents: Any) -> np.ndarray:
        count = len(self._clients)
        for offset in range(count):
            index = (self._active_index + offset) % count
            client = self._clients[index]
            try:
                response = await client.aio.models.embed_content(
                    model=self._model,
                    contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=self._dimensions),
                )
            except APIError as error:
                detail = (error.message or "")[:300]
                if error.code == 429 and offset + 1 < count:
                    continue
                raise EmbedderUnavailable(f"Gemini ({self._model}) answered {error.code}: {detail}") from error
            except httpx.HTTPError as error:
                raise EmbedderUnavailable(f"Gemini did not answer: {type(error).__name__}") from error
            self._active_index = index
            return self._to_vectors(response)
        raise EmbedderUnavailable(f"Gemini ({self._model}) exhausted all API keys.")

    async def text(self, query: str) -> np.ndarray:
        cached = self._query_vectors.get(query)
        if cached is not None:
            self._query_vectors.move_to_end(query)
            return cached
        content = types.Content(parts=[types.Part.from_text(text=query)])
        vector = np.asarray((await self._embed([content]))[0], dtype=np.float32)
        self._query_vectors[query] = vector
        if len(self._query_vectors) > QUERY_CACHE_SIZE:
            self._query_vectors.popitem(last=False)
        return vector

    async def image(self, image: Image.Image) -> np.ndarray:
        return np.asarray((await self.images([image]))[0], dtype=np.float32)

    async def images(self, batch: list[Image.Image]) -> np.ndarray:
        if not batch:
            return np.empty((0, self._dimensions), dtype=np.float32)
        contents = [types.Content(parts=[_image_part(image)]) for image in batch]
        vectors = await self._embed(contents)
        if len(vectors) != len(batch):
            raise EmbedderUnavailable(f"Gemini returned {len(vectors)} embeddings for {len(batch)} images.")
        return vectors
