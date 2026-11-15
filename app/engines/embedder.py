"""Gemini Embedding 2: text and pictures in one normalized vector space, so words and pictures match."""

import io
import os
from typing import Any, Protocol

import httpx
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image

from app.platform.errors import EmbedderUnavailable


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


class GeminiEmbedder:
    """Async Gemini Embedding 2 client. Vectors return unit-length so dot product equals cosine similarity."""

    def __init__(
        self,
        model_id: str = "gemini-embedding-2",
        dimensions: int = 768,
        timeout_seconds: float = 30.0,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is empty in the secrets file or environment. Lens needs it.")
        self.name = model_id
        self._model = model_id
        self._dimensions = dimensions
        retry = types.HttpRetryOptions(
            attempts=1 if http_async_client is not None else 2,
            initial_delay=0.5,
            max_delay=2.0,
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                retry_options=retry,
                httpx_async_client=http_async_client,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _embed(self, contents: Any, task_type: str) -> np.ndarray:
        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=self._dimensions, task_type=task_type),
            )
        except APIError as error:
            detail = (error.message or "")[:300]
            raise EmbedderUnavailable(f"Gemini ({self._model}) answered {error.code}: {detail}") from error
        except httpx.HTTPError as error:
            raise EmbedderUnavailable(f"Gemini did not answer: {type(error).__name__}") from error
        embeddings = response.embeddings or []
        if not embeddings:
            raise EmbedderUnavailable("Gemini returned no embeddings.")
        matrix = np.asarray([item.values for item in embeddings], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimensions:
            raise EmbedderUnavailable(
                f"Gemini returned embeddings with shape {matrix.shape}, expected (*, {self._dimensions})."
            )
        return _normalize(matrix)

    async def text(self, query: str) -> np.ndarray:
        content = types.Content(parts=[types.Part.from_text(text=query)])
        return np.asarray((await self._embed([content], "RETRIEVAL_QUERY"))[0], dtype=np.float32)

    async def image(self, image: Image.Image) -> np.ndarray:
        return np.asarray((await self.images([image]))[0], dtype=np.float32)

    async def images(self, batch: list[Image.Image]) -> np.ndarray:
        if not batch:
            return np.empty((0, self._dimensions), dtype=np.float32)
        contents = [types.Content(parts=[_image_part(image)]) for image in batch]
        vectors = await self._embed(contents, "RETRIEVAL_DOCUMENT")
        if len(vectors) != len(batch):
            raise EmbedderUnavailable(f"Gemini returned {len(vectors)} embeddings for {len(batch)} images.")
        return vectors
