"""Search by words or by a picture, by meaning alone, and add pictures to the Typesense collection."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, UploadFile

from app.features.search.controller import SearchController
from app.features.search.dependencies import get_search_controller
from app.features.search.schemas import CollectionInfo, DeleteResult, IngestResult, OriginFilter, SearchResult
from app.platform.dependencies import enforce_upload_limit

router = APIRouter(tags=["search"])
Controller = Annotated[SearchController, Depends(get_search_controller)]
Limit = Annotated[int | None, Query(ge=1, description="Results to return. The service's default when left out.")]


@router.get("/search")
async def search(
    controller: Controller,
    q: Annotated[str, Query(min_length=1, max_length=500, description="What to look for, in any language")],
    origin: OriginFilter = "all",
    limit: Limit = None,
) -> SearchResult:
    """Photographs whose picture means what the words say. Captions, titles and tags are never read."""
    return await controller.search_text(q, origin, limit)


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
    limit: Limit = None,
) -> SearchResult:
    """Recent photographs in the collection, filtered by origin (upload, met or all)."""
    return await controller.list_photographs(origin, limit)


@router.post("/photographs", status_code=201, dependencies=[Depends(enforce_upload_limit)])
async def add_photographs(
    controller: Controller,
    files: list[UploadFile],
) -> IngestResult:
    """Embed one or more pictures with Gemini Embedding 2 and index them into Typesense."""
    return await controller.add_photographs(files)


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
