"""The collection: Met CC0 photographs plus user-uploaded pictures and their precomputed image vectors."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

COLLECTION_FILE = "collection.jsonl"
VECTORS_FILE = "vectors.npy"
INDEX_FILE = "index.json"
UPLOADS_FILE = "uploads.jsonl"
UPLOAD_VECTORS_FILE = "upload_vectors.npy"

Origin = Literal["met", "upload"]


class Photograph(BaseModel):
    id: str
    title: str
    artist: str
    date: str
    medium: str
    tags: list[str]
    image: str
    width: int
    height: int
    source_url: str
    licence: str
    origin: Origin = "met"
    created_at: int = 0


@dataclass
class Collection:
    photographs: list[Photograph]
    vectors: np.ndarray  # one unit-length row per photograph, in the same order
    model: str
    position_by_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.rebuild_positions()

    def rebuild_positions(self) -> None:
        self.position_by_id = {photograph.id: position for position, photograph in enumerate(self.photographs)}

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def append(self, added_photographs: list[Photograph], added_vectors: np.ndarray) -> None:
        self.photographs.extend(added_photographs)
        self.vectors = np.vstack([self.vectors, added_vectors.astype(np.float32)])
        self.rebuild_positions()

    def remove(self, photograph_id: str) -> Photograph | None:
        position = self.position_by_id.get(photograph_id)
        if position is None:
            return None
        removed = self.photographs.pop(position)
        self.vectors = np.delete(self.vectors, position, axis=0)
        self.rebuild_positions()
        return removed


def read_photographs(directory: Path) -> list[Photograph]:
    with (directory / COLLECTION_FILE).open(encoding="utf-8") as file:
        return [Photograph.model_validate_json(line) for line in file]


def load_uploads(directory: Path, dimensions: int) -> tuple[list[Photograph], np.ndarray]:
    uploads_path = directory / UPLOADS_FILE
    vectors_path = directory / UPLOAD_VECTORS_FILE
    if not uploads_path.is_file() or not vectors_path.is_file():
        return [], np.empty((0, dimensions), dtype=np.float32)
    with uploads_path.open(encoding="utf-8") as file:
        uploads = [Photograph.model_validate_json(line) for line in file if line.strip()]
    for item in uploads:
        if not item.tags or item.tags == ["Uploaded"]:
            item.tags = ["Untagged"]
    if not uploads:
        return [], np.empty((0, dimensions), dtype=np.float32)
    vectors = np.load(vectors_path).astype(np.float32)
    if vectors.shape != (len(uploads), dimensions):
        raise RuntimeError(f"{UPLOADS_FILE} and {UPLOAD_VECTORS_FILE} are out of step.")
    return uploads, vectors


def save_uploads(directory: Path, photographs: list[Photograph], vectors: np.ndarray) -> None:
    uploads_path = directory / UPLOADS_FILE
    vectors_path = directory / UPLOAD_VECTORS_FILE
    upload_indices = [index for index, item in enumerate(photographs) if item.origin == "upload"]
    if not upload_indices:
        uploads_path.unlink(missing_ok=True)
        vectors_path.unlink(missing_ok=True)
        return
    uploads_text = "".join(photographs[index].model_dump_json() + "\n" for index in upload_indices)
    uploads_path.write_text(uploads_text, encoding="utf-8")
    np.save(vectors_path, vectors[upload_indices].astype(np.float32))


def load_collection(directory: Path, model: str) -> Collection:
    """The photographs and their vectors. Fails when the index is missing or was built by another model."""
    photographs = read_photographs(directory)
    index_path = directory / INDEX_FILE
    if not index_path.is_file():
        raise RuntimeError(f"No index in {directory}. Run: python scripts/build_index.py")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["model"] != model:
        raise RuntimeError(f"The index was built with {index['model']}, the service uses {model}. Rebuild it.")
    if index["ids"] != [photograph.id for photograph in photographs]:
        raise RuntimeError(f"The index and {COLLECTION_FILE} list different photographs. Rebuild the index.")
    vectors = np.load(directory / VECTORS_FILE).astype(np.float32)
    uploads, upload_vectors = load_uploads(directory, int(vectors.shape[1]))
    if uploads:
        photographs = [*photographs, *uploads]
        vectors = np.vstack([vectors, upload_vectors])
    return Collection(photographs=photographs, vectors=vectors, model=model)


def write_index(directory: Path, model: str, photographs: list[Photograph], vectors: np.ndarray) -> None:
    np.save(directory / VECTORS_FILE, vectors.astype(np.float32))
    index = {"model": model, "dimensions": int(vectors.shape[1]), "ids": [photograph.id for photograph in photographs]}
    (directory / INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")
