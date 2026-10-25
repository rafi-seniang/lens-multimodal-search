"""Search by words, by an uploaded picture, or add pictures directly into the Typesense collection."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, UploadFile

from app.api.deps import get_search_controller
from app.controllers.search import SearchController
from app.schemas.search import CollectionInfo, DeleteResult, IngestResult, Mode, OriginFilter, SearchResult

router = APIRouter(tags=["search"])
Controller = Annotated[SearchController, Depends(get_search_controller)]
Limit = Annotated[int | None, Query(ge=1, description="Results to return. The service's default when left out.")]


@router.get("/search")
async def search(
    controller: Controller,
    q: Annotated[str, Query(min_length=1, max_length=500, description="What to look for, in any language")],
    mode: Mode = "hybrid",
    origin: OriginFilter = "all",
    limit: Limit = None,
) -> SearchResult:
    """`picture` runs pure semantic vector search in Typesense, `caption` matches words, `hybrid` fuses both."""
    return await controller.search_text(q, mode, origin, limit)


@router.post("/search/image")
async def search_by_image(
    controller: Controller, file: UploadFile, origin: OriginFilter = "all", limit: Limit = None
) -> SearchResult:
    """Photographs that look like the uploaded picture."""
    return await controller.search_image(file, origin, limit)


@router.get("/photographs")
async def list_photographs(
    controller: Controller,
    origin: OriginFilter = "upload",
    tag: Annotated[str | None, Query(max_length=120, description="Filter by subject or Untagged")] = None,
    limit: Limit = None,
) -> SearchResult:
    """Recent photographs in the collection, filtered by origin (upload, met or all) and optional tag."""
    return await controller.list_photographs(origin, tag, limit)


@router.post("/photographs", status_code=201)
async def add_photographs(
    controller: Controller,
    files: list[UploadFile],
    title: Annotated[str | None, Form()] = None,
    tags: Annotated[str, Form()] = "",
) -> IngestResult:
    """Embed one or more pictures with EmbeddingGemma 2 and index them into Typesense."""
    return await controller.add_photographs(files, title, tags)


@router.delete("/photographs/{photograph_id}")
async def delete_photograph(controller: Controller, photograph_id: str) -> DeleteResult:
    """Delete a user-uploaded picture from Typesense and local storage."""
    return await controller.delete_photograph(photograph_id)


@router.get("/photographs/{photograph_id}/similar")
async def similar(controller: Controller, photograph_id: str, limit: Limit = None) -> SearchResult:
    """Photographs that look like one in the collection, using its stored vector in Typesense."""
    return await controller.similar(photograph_id, limit)


@router.get("/collection")
async def collection(controller: Controller) -> CollectionInfo:
    return controller.collection_info()
