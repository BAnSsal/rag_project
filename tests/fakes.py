"""Deterministic test doubles for embeddings/chat models.

These exist ONLY to keep the test suite hermetic (no network calls, no API
key, no cost) -- they are not shipped as part of the `rag` package, and the
application itself always talks to real OpenAI (see `rag/providers.py`).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "and", "or", "for", "what", "when", "where", "who", "why", "how",
    "does", "do", "did", "can", "could", "would", "should", "with", "that",
    "this", "it", "its", "be", "as", "by", "from",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class FakeEmbeddings(Embeddings):
    """Hashes text into a pseudo-random but reproducible vector.

    Not semantically meaningful -- only useful for exercising the index
    build/save/load/search plumbing in tests.
    """

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def _vector(self, text: str) -> List[float]:
        import hashlib

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = __import__("random").Random(seed)
        return [rng.uniform(-1, 1) for _ in range(self.size)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class FakeChatModel(BaseChatModel):
    """Extractive "answering" for tests: picks the context sentence with the
    most keyword overlap with the question, so test assertions can check for
    expected facts without calling a real LLM."""

    @property
    def _llm_type(self) -> str:
        return "fake-extractive-chat-model"

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
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    @staticmethod
    def _extractive_answer(context: str, question: str) -> str:
        if not context.strip():
            return "I don't have enough context to answer that."

        cleaned_lines = [
            line
            for line in context.splitlines()
            if not re.match(r"^\s*\[[^\]]+\]\s*$", line) and not re.match(r"^\s*#+\s", line)
        ]
        cleaned_context = " ".join(cleaned_lines)

        q_keywords = _keywords(question)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_context) if s.strip()]
        if not sentences:
            return "I don't have enough context to answer that."

        scored = sorted(
            ((len(_keywords(s) & q_keywords), s) for s in sentences),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score, best_sentence = scored[0]
        if best_score == 0:
            return "The provided documents don't seem to answer that question."
        return best_sentence
