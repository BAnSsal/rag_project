"""The retrieval-augmented generation chain: retriever -> prompt -> LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from rag.config import Settings, get_settings
from rag.providers import get_llm

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "you don't know instead of guessing. Be concise."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
    )


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: List[str]


def build_rag_chain(vector_store: FAISS, settings: Settings | None = None):
    """Build an LCEL chain that returns a RagAnswer for a given question string."""
    settings = settings or get_settings()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})
    llm = get_llm(settings)

    setup = RunnableParallel(
        question=lambda x: x,
        docs=retriever,
    )

    def _respond(payload: dict) -> RagAnswer:
        docs: List[Document] = payload["docs"]
        question: str = payload["question"]
        context = _format_docs(docs)
        messages = PROMPT.invoke({"context": context, "question": question})
        answer = (llm | StrOutputParser()).invoke(messages)
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        return RagAnswer(question=question, answer=answer, sources=sources)

    return setup | RunnableLambda(_respond)


def answer_question(
    question: str, vector_store: FAISS, settings: Settings | None = None
) -> RagAnswer:
    chain = build_rag_chain(vector_store, settings)
    return chain.invoke(question)
