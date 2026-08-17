"""Environment-backed settings for the RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    data_dir: str
    storage_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Read settings from the environment (with sane defaults) each time it's called."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        data_dir=os.getenv("RAG_DATA_DIR", "data"),
        storage_dir=os.getenv("RAG_STORAGE_DIR", "storage"),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 150),
        top_k=_env_int("RAG_TOP_K", 4),
    )
