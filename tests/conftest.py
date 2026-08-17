"""Shared pytest fixtures.

The application always uses real OpenAI (see `rag/providers.py`), but the
test suite injects deterministic test doubles (`tests/fakes.py`) in place of
the real network calls so it stays hermetic, fast, and free to run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag.config import Settings
from tests.fakes import FakeChatModel, FakeEmbeddings


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = FakeEmbeddings()
    llm = FakeChatModel()
    monkeypatch.setattr("rag.ingest.get_embeddings", lambda settings: embeddings)
    monkeypatch.setattr("rag.chain.get_llm", lambda settings: llm)


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "policy.md").write_text(
        "# Refund Policy\n\n"
        "Customers may request a full refund within 30 days of purchase.\n"
        "After 30 days, refunds are handled case by case.\n"
    )
    (data_dir / "faq.txt").write_text(
        "Our product supports Windows, macOS, and Linux.\n"
        "Subscriptions can be cancelled anytime from account settings.\n"
    )
    return data_dir


@pytest.fixture
def test_settings(tmp_path: Path, sample_data_dir: Path) -> Settings:
    storage_dir = tmp_path / "storage"
    return Settings(
        openai_api_key="sk-test-placeholder-not-real",
        openai_chat_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        data_dir=str(sample_data_dir),
        storage_dir=str(storage_dir),
        chunk_size=200,
        chunk_overlap=20,
        top_k=2,
    )


@pytest.fixture(autouse=True)
def cleanup_default_storage():
    yield
    default_storage = Path("storage")
    if default_storage.exists() and default_storage.is_dir():
        shutil.rmtree(default_storage, ignore_errors=True)
