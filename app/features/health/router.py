"""Liveness touches nothing (the lobby polls it). /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import VERSION
from app.features.health.schemas import ServiceMeta
from app.platform.dependencies import get_resources
from app.platform.resources import Resources

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(resources: Annotated[Resources, Depends(get_resources)]) -> ServiceMeta:
    return ServiceMeta(
        id="lens",
        name="Lens",
        summary="Semantic image search with EmbeddingGemma 2 and Typesense, plus custom picture indexing.",
        version=VERSION,
        engine=resources.lens.model,
        own_ui_path="/docs",
        docs_path="/docs",
    )
