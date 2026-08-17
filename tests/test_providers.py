from __future__ import annotations

import dataclasses

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import Settings
from rag.providers import get_embeddings, get_llm


def test_get_llm_and_get_embeddings_return_real_openai_classes(test_settings: Settings) -> None:
    # Constructing these clients does not make a network call, so this is
    # safe to run without hitting the OpenAI API.
    assert isinstance(get_llm(test_settings), ChatOpenAI)
    assert isinstance(get_embeddings(test_settings), OpenAIEmbeddings)


def test_get_llm_raises_without_api_key(test_settings: Settings) -> None:
    settings_without_key = dataclasses.replace(test_settings, openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm(settings_without_key)


def test_get_embeddings_raises_without_api_key(test_settings: Settings) -> None:
    settings_without_key = dataclasses.replace(test_settings, openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embeddings(settings_without_key)
