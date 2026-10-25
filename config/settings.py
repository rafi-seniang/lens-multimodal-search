"""Typed settings. Every environment value the service reads is declared here, nowhere else."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "lens"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    # Comma-separated keys accepted in the X-API-Key header. Empty disables auth (local only).
    api_keys: SecretStr = SecretStr("")
    # Browser origins allowed to call the API (the lab frontend in development and preview)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    # The personal secrets file with HF_TOKEN, if the model has to be downloaded. Empty when it is cached.
    secrets_env_path: Path | None = None

    # Vector store backend: typesense (Docker on 8108) or memory (unit tests and offline eval)
    store_backend: Literal["typesense", "memory"] = "typesense"
    typesense_url: str = "http://localhost:8108"
    typesense_api_key: SecretStr = SecretStr("lens-local-typesense")
    typesense_collection: str = "lens_photographs"
    # HNSW search width. Typesense defaults to 10, which skipped the best match for "plane" and "bumblebee".
    # 200 matches exact search on this collection. Keep it at least the largest k (50 in hybrid mode).
    vector_search_ef: int = Field(default=200, gt=0)

    # The collection: images/, collection.jsonl, uploads.jsonl, and the vector files
    collection_dir: Path = Path("../../assets/lens")
    # One model for text and images (Apache 2.0). bfloat16 on the GPU, float32 on the CPU.
    embedding_model: str = "google/embeddinggemma-2"
    device: str = "cuda"
    # Pictures embedded per forward pass when building the index
    index_batch_size: int = Field(default=8, gt=0)

    # Results per search when the caller does not say
    default_results: int = Field(default=24, gt=0)
    max_results: int = Field(default=60, gt=0)
    # Reciprocal rank fusion constant and picture share (0.8 keeps function words in descriptions from outvoting images)
    fusion_k: int = Field(default=60, gt=0)
    picture_weight: float = Field(default=0.8, gt=0, le=1)
    # Largest picture accepted per file and max files per ingest request
    max_upload_bytes: int = Field(default=8_000_000, gt=0)
    max_batch_uploads: int = Field(default=12, gt=0)

    # Redis for the shared rate limit. Empty counts per process: local runs only
    redis_url: SecretStr = SecretStr("")
    redis_timeout_seconds: float = Field(default=2, gt=0)
    # Requests per caller (API key, or client address without one) per minute on /v1
    rate_limit_per_minute: int = Field(default=120, gt=0)

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
