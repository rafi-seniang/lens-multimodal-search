"""Photographs, search inputs and responses. Inputs are query parameters and uploaded files, validated in the router."""

from typing import Literal

from pydantic import BaseModel

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


Mode = Literal["hybrid", "picture", "caption"]
OriginFilter = Literal["all", "upload", "met"]


class Hit(BaseModel):
    photograph: Photograph
    # Cosine similarity for a picture search, text match for caption, rank fusion score for hybrid
    score: float
    # Whether a query word is in the photograph's title, photographer or date
    in_caption: bool


class SearchResult(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[Hit]


class IngestResult(BaseModel):
    added: list[Photograph]
    total_photographs: int
    took_ms: float


class DeleteResult(BaseModel):
    deleted_id: str
    total_photographs: int


class SubjectCount(BaseModel):
    subject: str
    photographs: int


class CollectionInfo(BaseModel):
    photographs: int
    uploaded: int
    archive: int
    vector_store: str
    subjects: list[SubjectCount]
    model: str
    dimensions: int
    source: str
    licence: str
