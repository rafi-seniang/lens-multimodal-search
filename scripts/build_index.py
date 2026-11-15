"""Embed every photograph in the collection (plus existing uploads) and write the index loaded at startup.

Run from the repository root after scripts/fetch_collection.py: python scripts/build_index.py
"""

import asyncio
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


async def _embed_all(
    embedder: GeminiEmbedder, images_dir: Path, photographs: list[Photograph], batch_size: int
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(photographs), batch_size):
        batch = photographs[start : start + batch_size]
        opened = [Image.open(images_dir / photograph.image).convert("RGB") for photograph in batch]
        for attempt in range(6):
            try:
                batches.append(await embedder.images(opened))
                break
            except EmbedderUnavailable as error:
                if "429" not in error.detail or attempt == 5:
                    raise
                wait_seconds = 15 * (attempt + 1)
                print(f"Rate-limited at {start}/{len(photographs)}, waiting {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
        print(f"Embedded {min(start + batch_size, len(photographs))}/{len(photographs)}")
    return np.concatenate(batches)


async def main() -> None:
    settings = get_settings()
    load_secrets(settings)
    photographs = read_photographs(settings.collection_dir)
    images_dir = settings.collection_dir / "images"
    embedder = GeminiEmbedder(
        model_id=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    started = time.perf_counter()
    try:
        vectors = await _embed_all(embedder, images_dir, photographs, settings.index_batch_size)
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
                upload_vectors = await _embed_all(embedder, images_dir, uploads, settings.index_batch_size)
                np.save(settings.collection_dir / UPLOAD_VECTORS_FILE, upload_vectors.astype(np.float32))
    finally:
        await embedder.aclose()

    seconds = time.perf_counter() - started
    each = seconds / len(photographs) * 1000
    print(f"{len(photographs)} archive photographs embedded in {seconds:.1f} s ({each:.0f} ms each)")


if __name__ == "__main__":
    asyncio.run(main())
