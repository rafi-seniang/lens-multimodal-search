"""Embed every photograph in the collection and write the index the service loads at startup.

Run from the repository root after scripts/fetch_collection.py: python scripts/build_index.py
"""

import time

import numpy as np
import torch
from PIL import Image

from app.engines.embedder import GemmaEmbedder
from app.features.search.collection import read_photographs, write_index
from app.platform.resources import load_secrets
from config.settings import get_settings


def main() -> None:
    settings = get_settings()
    load_secrets(settings)
    photographs = read_photographs(settings.collection_dir)
    images_dir = settings.collection_dir / "images"
    embedder = GemmaEmbedder(settings.embedding_model, settings.device)
    started = time.perf_counter()
    batches = []
    for start in range(0, len(photographs), settings.index_batch_size):  # batches keep GPU memory flat
        batch = photographs[start : start + settings.index_batch_size]
        images = [Image.open(images_dir / photograph.image).convert("RGB") for photograph in batch]
        batches.append(embedder.encode_images(images))
    vectors = np.concatenate(batches)
    write_index(settings.collection_dir, settings.embedding_model, photographs, vectors)
    seconds = time.perf_counter() - started
    each = seconds / len(photographs) * 1000
    print(f"{len(photographs)} photographs embedded in {seconds:.1f} s ({each:.0f} ms each)")
    if settings.device == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
