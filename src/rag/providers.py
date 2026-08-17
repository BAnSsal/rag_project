"""Factory functions for embeddings and chat models.

When ``OPENAI_API_KEY`` is configured, real OpenAI embeddings and chat
completions are used. Otherwise the pipeline transparently falls back to a
deterministic, offline "fake" provider so ingestion, retrieval, and the CLI
all still work end-to-end without network access or an API key -- handy for
demos, tests, and CI.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag.config import Settings

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "and", "or", "for", "what", "when", "where", "who", "why", "how",
    "does", "do", "did", "can", "could", "would", "should", "with", "that",
    "this", "it", "its", "be", "as", "by", "from",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class ExtractiveFakeChatModel(BaseChatModel):
    """A dependency-free, deterministic "chat model" for offline use.

    It does not call any external service. Instead it looks at the context
    block injected by the RAG prompt and returns the sentence(s) that share
    the most keywords with the question, so the offline pipeline still
    produces a grounded, non-random answer.
    """

    @property
    def _llm_type(self) -> str:
        return "extractive-fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_text = "\n".join(m.content for m in messages if isinstance(m.content, str))

        context_match = re.search(r"Context:\s*(.*?)\n\s*Question:", full_text, re.DOTALL)
        question_match = re.search(r"Question:\s*(.*)", full_text, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else full_text

        answer = self._extractive_answer(context, question)
        message = AIMessage(content=answer)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _extractive_answer(context: str, question: str) -> str:
        if not context.strip():
            return "I don't have enough context to answer that."

        # Strip the "[source]" tags injected by the prompt formatter and drop
        # markdown heading lines so headings aren't mistaken for answers.
        cleaned_lines = [
            line
            for line in context.splitlines()
            if not re.match(r"^\s*\[[^\]]+\]\s*$", line) and not re.match(r"^\s*#+\s", line)
        ]
        cleaned_context = " ".join(cleaned_lines)

        q_keywords = _keywords(question)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_context)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return "I don't have enough context to answer that."

        scored = []
        for sentence in sentences:
            overlap = len(_keywords(sentence) & q_keywords)
            scored.append((overlap, sentence))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_sentence = scored[0]

        if best_score == 0:
            return "The provided documents don't seem to answer that question."

        return best_sentence


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.has_openai_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_community.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=256)


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.has_openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    return ExtractiveFakeChatModel()
