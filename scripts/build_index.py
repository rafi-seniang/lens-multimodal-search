"""Embed every photograph in the collection (plus existing uploads) and write the index loaded at startup.

Run from the repository root after scripts/fetch_collection.py: python scripts/build_index.py
"""

import asyncio
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.engines.embedder import GeminiEmbedder
from app.features.search.collection import (
    UPLOAD_VECTORS_FILE,
    UPLOADS_FILE,
    read_photographs,
    write_index,
)
from app.features.search.schemas import Photograph
from app.platform.errors import EmbedderUnavailable
from app.platform.resources import load_secrets
from config.settings import get_settings


def _image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_cache(
    images_dir: Path, photographs: list[Photograph], cache_dir: Path
) -> tuple[list[np.ndarray | None], list[tuple[int, Path, Path]], int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_by_index: list[np.ndarray | None] = [None] * len(photographs)
    pending: list[tuple[int, Path, Path]] = []
    cached_hits = 0
    for idx, photograph in enumerate(photographs):
        image_path = images_dir / photograph.image
        cache_file = cache_dir / f"{_image_digest(image_path)}.npy"
        if cache_file.is_file():
            vectors_by_index[idx] = np.load(cache_file).astype(np.float32)
            cached_hits += 1
        else:
            pending.append((idx, cache_file, image_path))
    return vectors_by_index, pending, cached_hits


def _open_rgb_batch(chunk: list[tuple[int, Path, Path]]) -> list[Image.Image]:
    opened: list[Image.Image] = []
    for _, _, image_path in chunk:
        with Image.open(image_path) as raw:
            opened.append(raw.convert("RGB"))
    return opened


async def _embed_all(
    embedder: GeminiEmbedder,
    images_dir: Path,
    photographs: list[Photograph],
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    vectors_by_index, pending, cached_hits = _prepare_cache(images_dir, photographs, cache_dir)
    if cached_hits:
        print(f"Loaded {cached_hits}/{len(photographs)} cached vectors from {cache_dir}")

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        opened = _open_rgb_batch(chunk)
        for attempt in range(6):
            try:
                batch_vectors = await embedder.images(opened)
                break
            except EmbedderUnavailable as error:
                if "429" not in error.detail or attempt == 5:
                    raise
                wait_seconds = 15 * (attempt + 1)
                print(f"Rate-limited at {start}/{len(pending)}, waiting {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
        for (idx, cache_file, _), vector in zip(chunk, batch_vectors, strict=True):
            vec = vector.astype(np.float32)
            np.save(cache_file, vec)
            vectors_by_index[idx] = vec
        print(f"Embedded {cached_hits + min(start + batch_size, len(pending))}/{len(photographs)}")

    return np.stack([v for v in vectors_by_index if v is not None]).astype(np.float32)


async def main() -> None:
    settings = get_settings()
    load_secrets(settings)
    photographs = read_photographs(settings.collection_dir)
    images_dir = settings.collection_dir / "images"
    cache_dir = (
        settings.collection_dir / ".cache" / "vectors" / f"{settings.embedding_model}_{settings.embedding_dimensions}"
    )
    embedder = GeminiEmbedder(
        model_id=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    started = time.perf_counter()
    try:
        vectors = await _embed_all(embedder, images_dir, photographs, settings.index_batch_size, cache_dir)
        write_index(settings.collection_dir, settings.embedding_model, photographs, vectors)

        uploads_path = settings.collection_dir / UPLOADS_FILE
        if uploads_path.is_file():
            uploads = [
                Photograph.model_validate_json(line)
                for line in uploads_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if uploads:
                print(f"Re-embedding {len(uploads)} existing uploads...")
                upload_vectors = await _embed_all(embedder, images_dir, uploads, settings.index_batch_size, cache_dir)
                np.save(settings.collection_dir / UPLOAD_VECTORS_FILE, upload_vectors.astype(np.float32))
    finally:
        await embedder.aclose()

    seconds = time.perf_counter() - started
    each = seconds / len(photographs) * 1000
    print(f"{len(photographs)} archive photographs embedded in {seconds:.1f} s ({each:.0f} ms each)")


if __name__ == "__main__":
    asyncio.run(main())
