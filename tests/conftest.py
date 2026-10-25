from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from app.api.deps import get_lens
from app.core.config import Settings
from app.main import create_app
from app.search.collection import Photograph, load_collection, write_index
from app.search.keyword import KeywordIndex
from app.search.service import Lens

FAKE_MODEL = "fake-embedder"
# Three directions stand in for what a picture shows: a dog, a boat, a bridge
DOG, BOAT, BRIDGE = np.eye(3, dtype=np.float32)
WORDS = {"dog": DOG, "anjing": DOG, "boat": BOAT, "perahu": BOAT, "bridge": BRIDGE}


class FakeEmbedder:
    """Puts a text on the axis of the first word it knows, and a picture on the axis of its colour."""

    name = FAKE_MODEL

    async def text(self, query: str) -> np.ndarray:
        known = [WORDS[word] for word in query.lower().split() if word in WORDS]
        return known[0] if known else np.full(3, 3**-0.5, dtype=np.float32)

    async def image(self, image: Image.Image) -> np.ndarray:
        red, green, blue = image.getpixel((0, 0))
        return DOG if red > max(green, blue) else BOAT if blue > green else BRIDGE

    async def images(self, batch: list[Image.Image]) -> np.ndarray:
        rows = [await self.image(item) for item in batch]
        return np.stack(rows).astype(np.float32)


def photograph(photograph_id: str, title: str, tags: list[str]) -> Photograph:
    return Photograph(
        id=photograph_id,
        title=title,
        artist="Unknown",
        date="1890",
        medium="Albumen print",
        tags=tags,
        image=f"{photograph_id}.jpg",
        width=640,
        height=480,
        source_url=f"https://example.org/{photograph_id}",
        licence="CC0 1.0",
    )


PHOTOGRAPHS = [
    photograph("1", "[Dog]", ["Dogs"]),
    photograph("2", "Untitled", ["Dogs", "Men"]),  # a dog the caption does not mention
    photograph("3", "Harbour at dawn", ["Boats"]),
    photograph("4", "Brooklyn Bridge", ["Bridges"]),
]
VECTORS = np.stack([DOG, 0.9 * DOG + 0.1 * BOAT, BOAT, BRIDGE])


@pytest.fixture
def collection_dir(tmp_path: Path) -> Path:
    (tmp_path / "collection.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in PHOTOGRAPHS), encoding="utf-8"
    )
    write_index(tmp_path, FAKE_MODEL, PHOTOGRAPHS, VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True))
    return tmp_path


@pytest.fixture
def lens(collection_dir: Path) -> Lens:
    collection = load_collection(collection_dir, FAKE_MODEL)
    return Lens(
        collection,
        FakeEmbedder(),
        KeywordIndex(collection.photographs),
        fusion_k=60,
        collection_dir=collection_dir,
    )


def make_app(settings: Settings, lens: Lens) -> Any:
    app = create_app(settings)
    app.dependency_overrides[get_lens] = lambda: lens
    return app


@pytest.fixture
async def client(lens: Lens, collection_dir: Path) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        _env_file=None,
        environment="test",
        store_backend="memory",
        api_keys="test-key",
        collection_dir=collection_dir,
        max_results=3,
        max_upload_bytes=50_000,
    )
    transport = httpx.ASGITransport(app=make_app(settings, lens))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as http:
        yield http
