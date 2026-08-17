"""Command-line interface for the RAG project.

Usage:
    python -m rag.cli ingest
    python -m rag.cli ask "What is the refund policy?"
"""

from __future__ import annotations

import argparse
import sys

from rag.chain import answer_question
from rag.config import get_settings
from rag.ingest import build_index, load_index


def cmd_ingest(_: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.has_openai_key:
        print(
            "Error: OPENAI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before running ingestion.",
            file=sys.stderr,
        )
        return 1
    print(f"Loading documents from '{settings.data_dir}'...")
    build_index(settings)
    print(f"Index built and saved to '{settings.storage_dir}'.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.has_openai_key:
        print(
            "Error: OPENAI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before asking questions.",
            file=sys.stderr,
        )
        return 1
    try:
        vector_store = load_index(settings)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result = answer_question(args.question, vector_store, settings)
    print(f"\nQuestion: {result.question}")
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources: {', '.join(result.sources) if result.sources else 'none'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Simple RAG project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the vector index from data/")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
