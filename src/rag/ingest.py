"""Load documents, split them into chunks, embed them, and persist a FAISS index."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings, get_settings
from rag.providers import get_embeddings


def load_documents(data_dir: str) -> List[Document]:
    """Load every .txt/.md file under ``data_dir`` as a LangChain Document."""
    path = Path(data_dir)
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"No documents found in '{data_dir}'. Add some .txt/.md files first."
        )

    documents: List[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            data_dir,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=False,
        )
        documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(
            f"No .txt or .md documents found in '{data_dir}'. Add some documents first."
        )
    return documents


def split_documents(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(settings: Settings | None = None) -> FAISS:
    """Run the full ingestion pipeline and persist the resulting FAISS index."""
    settings = settings or get_settings()

    documents = load_documents(settings.data_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    embeddings = get_embeddings(settings)
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(settings.storage_dir, exist_ok=True)
    vector_store.save_local(settings.storage_dir)

    return vector_store


def load_index(settings: Settings | None = None) -> FAISS:
    """Load a previously built FAISS index from disk."""
    settings = settings or get_settings()

    if not Path(settings.storage_dir).exists():
        raise FileNotFoundError(
            f"No index found in '{settings.storage_dir}'. Run ingestion first: "
            "`python -m rag.cli ingest`."
        )

    embeddings = get_embeddings(settings)
    return FAISS.load_local(
        settings.storage_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    build_index()
    print("Index built successfully.")
