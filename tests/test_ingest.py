from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingest import build_index, load_documents, load_index, split_documents


def test_load_documents_reads_txt_and_md(sample_data_dir: Path) -> None:
    docs = load_documents(str(sample_data_dir))
    sources = {Path(d.metadata["source"]).name for d in docs}
    assert sources == {"policy.md", "faq.txt"}


def test_load_documents_raises_when_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_documents(str(empty_dir))


def test_split_documents_respects_chunk_size(sample_data_dir: Path) -> None:
    docs = load_documents(str(sample_data_dir))
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= len(docs)
    for chunk in chunks:
        # Some slack allowed since the splitter avoids breaking mid-word.
        assert len(chunk.page_content) <= 80


def test_build_and_load_index_roundtrip(test_settings: Settings) -> None:
    build_index(test_settings)
    assert Path(test_settings.storage_dir).exists()

    vector_store = load_index(test_settings)

    # The offline fake embedding is deterministic-but-not-semantic (it hashes
    # text to a pseudo-random vector), so querying with the *exact* text of a
    # known chunk is the only reliable way to assert retrieval correctness
    # without a real embedding model.
    docs = load_documents(str(test_settings.data_dir))
    chunks = split_documents(docs, test_settings.chunk_size, test_settings.chunk_overlap)
    refund_chunk = next(c for c in chunks if "refund" in c.page_content.lower())

    results = vector_store.similarity_search(refund_chunk.page_content, k=1)
    assert len(results) == 1
    assert "refund" in results[0].page_content.lower()


def test_load_index_missing_raises(test_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(test_settings)
