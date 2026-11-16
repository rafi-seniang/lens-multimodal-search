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


OriginFilter = Literal["all", "upload", "met"]


class Hit(BaseModel):
    photograph: Photograph
    # Cosine similarity between the query's vector and the picture's, 1.0 when browsing
    score: float


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


class CollectionInfo(BaseModel):
    photographs: int
    uploaded: int
    archive: int
    vector_store: str
    model: str
    dimensions: int
    source: str
    licence: str
