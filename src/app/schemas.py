"""Response models. Request inputs are query parameters and uploaded files, validated in the routes."""

from pydantic import BaseModel

from app.search.collection import Photograph


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


class ServiceMeta(BaseModel):
    """What the lab frontend's lobby shows about this backend (the Lab backend contract)."""

    id: str
    name: str
    summary: str
    version: str
    engine: str
    own_ui_path: str
    docs_path: str
