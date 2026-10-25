"""Liveness touches nothing (the lobby polls it). /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_lens
from app.schemas import ServiceMeta
from app.search.service import Lens

router = APIRouter(tags=["health"])
VERSION = "0.2.0"


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(lens: Annotated[Lens, Depends(get_lens)]) -> ServiceMeta:
    return ServiceMeta(
        id="lens",
        name="Lens",
        summary="Semantic image search with EmbeddingGemma 2 and Typesense, plus custom picture indexing.",
        version=VERSION,
        engine=lens.model,
        own_ui_path="/docs",
        docs_path="/docs",
    )
