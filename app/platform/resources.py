"""The model, the collection and the Typesense store, built once in the lifespan and held on app.state."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI
from redis.asyncio import Redis

from app.engines.embedder import GeminiEmbedder
from app.engines.keyword import KeywordIndex
from app.features.search.collection import load_collection
from app.features.search.service import Lens
from app.features.search.typesense_store import TypesensePhotographStore
from app.platform.rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.platform.redis import connect_redis
from config.settings import Settings


@dataclass
class Resources:
    lens: Lens
    rate_limiter: RateLimiter
    store: TypesensePhotographStore | None = None
    redis: Redis | None = None


def load_secrets(settings: Settings) -> None:
    if settings.secrets_env_path:
        if not settings.secrets_env_path.is_file():
            raise RuntimeError(f"SECRETS_ENV_PATH points to a missing file: {settings.secrets_env_path}")
        load_dotenv(settings.secrets_env_path, override=False)  # GEMINI_API_KEY when not passed in env


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
        await store.bootstrap(collection.photographs, collection.vectors, model=settings.embedding_model)
    lens = Lens(
        collection=collection,
        embedder=GeminiEmbedder(
            model_id=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.gemini_timeout_seconds,
        ),
        keyword=KeywordIndex(collection.photographs),
        fusion_k=settings.fusion_k,
        picture_weight=settings.picture_weight,
        min_picture_score=settings.min_picture_score,
        store=store,
        collection_dir=settings.collection_dir,
    )
    redis = await connect_redis(settings)
    rate_limiter: RateLimiter = RedisRateLimiter(redis) if redis is not None else InMemoryRateLimiter()
    return Resources(lens=lens, rate_limiter=rate_limiter, store=store, redis=redis)


def lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        resources = await build_resources(settings)  # collection, Gemini client, and Typesense load here
        app.state.settings = settings
        app.state.resources = resources
        try:
            yield
        finally:
            close_embedder = getattr(resources.lens.embedder, "aclose", None)
            if close_embedder is not None:
                await close_embedder()
            if resources.store is not None:
                await resources.store.close()
            if resources.redis is not None:
                await resources.redis.aclose()

    return run
