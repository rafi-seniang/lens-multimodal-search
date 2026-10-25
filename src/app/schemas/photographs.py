"""A photograph in the collection: a Met CC0 print or a local upload."""

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
