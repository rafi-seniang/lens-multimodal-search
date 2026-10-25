"""The model, the collection and the Typesense store, built once in the lifespan and held on app.state."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI

from app.core.config import Settings
from app.engines.embedder import GemmaEmbedder
from app.engines.keyword import KeywordIndex
from app.repositories.collection import load_collection
from app.repositories.typesense_store import TypesensePhotographStore
from app.services.lens import Lens


@dataclass
class Resources:
    lens: Lens
    store: TypesensePhotographStore | None = None


def load_secrets(settings: Settings) -> None:
    if settings.secrets_env_path:
        if not settings.secrets_env_path.is_file():
            raise RuntimeError(f"SECRETS_ENV_PATH points to a missing file: {settings.secrets_env_path}")
        load_dotenv(settings.secrets_env_path, override=False)  # HF_TOKEN for a first download


async def build_resources(settings: Settings) -> Resources:
    load_secrets(settings)
    collection = load_collection(settings.collection_dir, settings.embedding_model)
    store: TypesensePhotographStore | None = None
    if settings.store_backend == "typesense":
        store = TypesensePhotographStore(
            url=settings.typesense_url,
            api_key=settings.typesense_api_key.get_secret_value(),
            collection_name=settings.typesense_collection,
            search_ef=settings.vector_search_ef,
        )
        await store.bootstrap(collection.photographs, collection.vectors)
    lens = Lens(
        collection=collection,
        embedder=GemmaEmbedder(settings.embedding_model, settings.device),
        keyword=KeywordIndex(collection.photographs),
        fusion_k=settings.fusion_k,
        picture_weight=settings.picture_weight,
        store=store,
        collection_dir=settings.collection_dir,
    )
    return Resources(lens=lens, store=store)


def lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        resources = await build_resources(settings)  # model and Typesense load here, before the first request
        app.state.resources = resources
        try:
            yield
        finally:
            if resources.store is not None:
                await resources.store.close()

    return run
