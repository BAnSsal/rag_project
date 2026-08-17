"""Factory functions for embeddings and chat models.

Both factories require ``OPENAI_API_KEY`` to be set and always talk to the
real OpenAI API -- there is no offline/mock fallback in production code.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import Settings


def _require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file (see "
            ".env.example) or export it in your shell before running this "
            "project."
        )
    return settings.openai_api_key


def get_embeddings(settings: Settings) -> Embeddings:
    api_key = _require_api_key(settings)
    return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=api_key)


def get_llm(settings: Settings) -> BaseChatModel:
    api_key = _require_api_key(settings)
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=api_key,
        temperature=0,
    )
