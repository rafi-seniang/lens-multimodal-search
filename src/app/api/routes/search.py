"""Search by words, by an uploaded picture, or add pictures directly into the Typesense collection."""

import asyncio
import io
from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.api.deps import get_lens
from app.core.config import Settings
from app.core.errors import ImageTooLarge, InvalidAction, UnreadableImage
from app.core.security import app_settings
from app.schemas import CollectionInfo, DeleteResult, IngestResult, SearchResult, SubjectCount
from app.search.service import Lens, Mode
from app.search.typesense_store import OriginFilter

router = APIRouter(tags=["search"])
LensDep = Annotated[Lens, Depends(get_lens)]
SettingsDep = Annotated[Settings, Depends(app_settings)]
Limit = Annotated[int | None, Query(ge=1, description="Results to return. The service's default when left out.")]


def result_count(limit: int | None, settings: Settings) -> int:
    return min(limit or settings.default_results, settings.max_results)


async def decode_upload(file: UploadFile, max_upload_bytes: int) -> Image.Image:
    content = await file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise ImageTooLarge(f"The picture is over {max_upload_bytes // 1_000_000} MB.")
    try:
        opened = Image.open(io.BytesIO(content))
        return ImageOps.exif_transpose(opened).convert("RGB")
    except UnidentifiedImageError as error:
        raise UnreadableImage("The file is not a JPEG, PNG, WebP or another picture format Pillow reads.") from error


@router.get("/search")
async def search(
    lens: LensDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=500, description="What to look for, in any language")],
    mode: Mode = "hybrid",
    origin: OriginFilter = "all",
    limit: Limit = None,
) -> SearchResult:
    """`picture` runs pure semantic vector search in Typesense, `caption` matches words, `hybrid` fuses both."""
    return await lens.search_text(q, mode, result_count(limit, settings), origin=origin)


@router.post("/search/image")
async def search_by_image(
    lens: LensDep,
    settings: SettingsDep,
    file: UploadFile,
    origin: OriginFilter = "all",
    limit: Limit = None,
) -> SearchResult:
    """Photographs that look like the uploaded picture."""
    image = await decode_upload(file, settings.max_upload_bytes)
    return await lens.search_image(image, result_count(limit, settings), origin=origin)


@router.get("/photographs")
async def list_photographs(
    lens: LensDep,
    settings: SettingsDep,
    origin: OriginFilter = "upload",
    tag: Annotated[str | None, Query(max_length=120, description="Filter by subject or Untagged")] = None,
    limit: Limit = None,
) -> SearchResult:
    """Recent photographs in the collection, filtered by origin (upload, met or all) and optional tag."""
    return await lens.list_photographs(origin, result_count(limit, settings), tag=tag)


@router.post("/photographs", status_code=201)
async def add_photographs(
    lens: LensDep,
    settings: SettingsDep,
    files: list[UploadFile],
    title: Annotated[str | None, Form()] = None,
    tags: Annotated[str, Form()] = "",
) -> IngestResult:
    """Embed one or more pictures with EmbeddingGemma 2 and index them into Typesense."""
    if not files:
        raise InvalidAction("Select at least one picture file.")
    if len(files) > settings.max_batch_uploads:
        raise InvalidAction(f"Upload at most {settings.max_batch_uploads} pictures at once.")
    images = await asyncio.gather(*(decode_upload(file, settings.max_upload_bytes) for file in files))
    decoded = [(file.filename or "upload", image) for file, image in zip(files, images, strict=True)]
    parsed_tags = [item.strip() for item in tags.split(",") if item.strip()]
    return await lens.add_photographs(decoded, title=title, tags=parsed_tags)


@router.delete("/photographs/{photograph_id}")
async def delete_photograph(lens: LensDep, photograph_id: str) -> DeleteResult:
    """Delete a user-uploaded picture from Typesense and local storage."""
    remaining = await lens.delete_photograph(photograph_id)
    return DeleteResult(deleted_id=photograph_id, total_photographs=remaining)


@router.get("/photographs/{photograph_id}/similar")
async def similar(lens: LensDep, settings: SettingsDep, photograph_id: str, limit: Limit = None) -> SearchResult:
    """Photographs that look like one in the collection, using its stored vector in Typesense."""
    return await lens.similar(photograph_id, result_count(limit, settings))


@router.get("/collection")
async def collection(lens: LensDep) -> CollectionInfo:
    photographs = lens.collection.photographs
    uploaded = sum(1 for item in photographs if item.origin == "upload")
    subjects = Counter(tag for photograph in photographs for tag in (photograph.tags or ["Untagged"]))
    subject_list = [SubjectCount(subject=tag, photographs=count) for tag, count in subjects.most_common()]
    if "Untagged" not in subjects:
        subject_list.append(SubjectCount(subject="Untagged", photographs=0))
    return CollectionInfo(
        photographs=len(photographs),
        uploaded=uploaded,
        archive=len(photographs) - uploaded,
        vector_store=lens.vector_store_name,
        subjects=subject_list,
        model=lens.model,
        dimensions=lens.collection.dimensions,
        source="The Metropolitan Museum of Art (CC0) and local uploads",
        licence="CC0 1.0 and user uploads",
    )
