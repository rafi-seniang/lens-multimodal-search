"""Turns search and upload requests into calls on Lens: decodes pictures, applies result limits, shapes responses."""

import asyncio
import io
from collections import Counter

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.features.search.schemas import (
    CollectionInfo,
    DeleteResult,
    IngestResult,
    Mode,
    OriginFilter,
    SearchResult,
    SubjectCount,
)
from app.features.search.service import Lens
from app.platform.errors import ImageTooLarge, InvalidAction, UnreadableImage
from config.settings import Settings


async def decode_upload(file: UploadFile, max_upload_bytes: int) -> Image.Image:
    content = await file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise ImageTooLarge(f"The picture is over {max_upload_bytes // 1_000_000} MB.")
    try:
        opened = Image.open(io.BytesIO(content))
        return ImageOps.exif_transpose(opened).convert("RGB")
    except UnidentifiedImageError as error:
        raise UnreadableImage("The file is not a JPEG, PNG, WebP or another picture format Pillow reads.") from error


class SearchController:
    def __init__(self, lens: Lens, settings: Settings) -> None:
        self._lens = lens
        self._settings = settings

    def _result_count(self, limit: int | None) -> int:
        return min(limit or self._settings.default_results, self._settings.max_results)

    async def search_text(self, query: str, mode: Mode, origin: OriginFilter, limit: int | None) -> SearchResult:
        return await self._lens.search_text(query, mode, self._result_count(limit), origin=origin)

    async def search_image(self, file: UploadFile, origin: OriginFilter, limit: int | None) -> SearchResult:
        image = await decode_upload(file, self._settings.max_upload_bytes)
        return await self._lens.search_image(image, self._result_count(limit), origin=origin)

    async def list_photographs(self, origin: OriginFilter, tag: str | None, limit: int | None) -> SearchResult:
        return await self._lens.list_photographs(origin, self._result_count(limit), tag=tag)

    async def add_photographs(self, files: list[UploadFile], title: str | None, tags: str) -> IngestResult:
        if not files:
            raise InvalidAction("Select at least one picture file.")
        if len(files) > self._settings.max_batch_uploads:
            raise InvalidAction(f"Upload at most {self._settings.max_batch_uploads} pictures at once.")
        images = await asyncio.gather(*(decode_upload(file, self._settings.max_upload_bytes) for file in files))
        decoded = [(file.filename or "upload", image) for file, image in zip(files, images, strict=True)]
        parsed_tags = [item.strip() for item in tags.split(",") if item.strip()]
        return await self._lens.add_photographs(decoded, title=title, tags=parsed_tags)

    async def delete_photograph(self, photograph_id: str) -> DeleteResult:
        remaining = await self._lens.delete_photograph(photograph_id)
        return DeleteResult(deleted_id=photograph_id, total_photographs=remaining)

    async def similar(self, photograph_id: str, limit: int | None) -> SearchResult:
        return await self._lens.similar(photograph_id, self._result_count(limit))

    def collection_info(self) -> CollectionInfo:
        photographs = self._lens.collection.photographs
        uploaded = sum(1 for item in photographs if item.origin == "upload")
        subjects = Counter(tag for photograph in photographs for tag in (photograph.tags or ["Untagged"]))
        subject_list = [SubjectCount(subject=tag, photographs=count) for tag, count in subjects.most_common()]
        if "Untagged" not in subjects:
            subject_list.append(SubjectCount(subject="Untagged", photographs=0))
        return CollectionInfo(
            photographs=len(photographs),
            uploaded=uploaded,
            archive=len(photographs) - uploaded,
            vector_store=self._lens.vector_store_name,
            subjects=subject_list,
            model=self._lens.model,
            dimensions=self._lens.collection.dimensions,
            source="The Metropolitan Museum of Art (CC0) and local uploads",
            licence="CC0 1.0 and user uploads",
        )
