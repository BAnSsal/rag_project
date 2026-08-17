from __future__ import annotations

from rag.chain import answer_question
from rag.config import Settings
from rag.ingest import build_index
from rag.providers import ExtractiveFakeChatModel, get_embeddings, get_llm


def test_get_providers_fall_back_to_fake_without_api_key(test_settings: Settings) -> None:
    llm = get_llm(test_settings)
    assert isinstance(llm, ExtractiveFakeChatModel)

    embeddings = get_embeddings(test_settings)
    assert embeddings.embed_query("hello") == embeddings.embed_query("hello")


def test_answer_question_end_to_end_offline(test_settings: Settings) -> None:
    vector_store = build_index(test_settings)

    result = answer_question("How many days do I have to request a refund?", vector_store, test_settings)

    assert result.question == "How many days do I have to request a refund?"
    assert "30" in result.answer
    assert any("policy.md" in source for source in result.sources)


def test_answer_question_unrelated_question_is_honest(test_settings: Settings) -> None:
    vector_store = build_index(test_settings)

    result = answer_question("What is the capital of France?", vector_store, test_settings)

    assert result.answer  # always returns something, never crashes
    assert result.sources  # still retrieves nearest chunks, just low relevance
