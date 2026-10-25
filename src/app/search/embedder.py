"""EmbeddingGemma 2: text and pictures in one vector space, so words find pictures and pictures find pictures."""

import asyncio
from typing import Protocol

import numpy as np
from PIL import Image

# Text queries carry the model card's retrieval prefix. Pictures get no prefix.
QUERY_PROMPT = "task: search result | query: "


class Embedder(Protocol):
    name: str

    async def text(self, query: str) -> np.ndarray: ...

    async def image(self, image: Image.Image) -> np.ndarray: ...

    async def images(self, batch: list[Image.Image]) -> np.ndarray: ...


class GemmaEmbedder:
    """Loaded once at startup. Vectors come back unit length, so a dot product is the cosine similarity."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch  # heavy imports, only where the model is used
        from sentence_transformers import SentenceTransformer

        # bfloat16 on the GPU, float32 on the CPU. Never float16: EmbeddingGemma overflows it without an error.
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.name = model_id
        self._model = SentenceTransformer(model_id, device=device, model_kwargs={"dtype": dtype})
        self._model.default_prompt_name = None

    def encode_images(self, batch: list[Image.Image]) -> np.ndarray:
        vectors = self._model.encode([{"image": image} for image in batch], normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    async def text(self, query: str) -> np.ndarray:
        vectors = await asyncio.to_thread(self._model.encode, [query], prompt=QUERY_PROMPT, normalize_embeddings=True)
        return np.asarray(vectors[0], dtype=np.float32)

    async def image(self, image: Image.Image) -> np.ndarray:
        vectors = await asyncio.to_thread(self.encode_images, [image])
        return np.asarray(vectors[0], dtype=np.float32)

    async def images(self, batch: list[Image.Image]) -> np.ndarray:
        return await asyncio.to_thread(self.encode_images, batch)
