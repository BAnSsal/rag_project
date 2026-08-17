# rag_project

A simple **Retrieval-Augmented Generation (RAG)** project built with
[LangChain](https://python.langchain.com/). It answers questions about your
own documents by retrieving the most relevant chunks and asking a chat model
to answer using only that context.

## How it works

1. **Ingest** – documents in `data/` are loaded, split into overlapping
   chunks, embedded, and stored in a local [FAISS](https://github.com/facebookresearch/faiss)
   vector index under `storage/`.
2. **Ask** – your question is embedded the same way, the most similar chunks
   are retrieved from the index, and a chat model answers using only those
   chunks as context. The answer is printed along with the source files it
   came from.

```text
data/*.md, *.txt --(load + split)--> chunks --(embed)--> FAISS index (storage/)
                                                              |
question --(embed)--> similarity search --------------------+
                                                              v
                                            context + question --(LLM)--> answer + sources
```

## Project layout

```text
rag_project/
├── data/                  # sample documents to index (swap in your own!)
├── src/rag/
│   ├── config.py          # environment-backed settings
│   ├── providers.py       # embeddings/LLM factory (OpenAI, or offline fake)
│   ├── ingest.py          # load → split → embed → save FAISS index
│   ├── chain.py           # LCEL retrieval-augmented generation chain
│   └── cli.py             # `ingest` and `ask` commands
└── tests/                 # pytest suite (runs fully offline)
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and, optionally, add your OpenAI API key:

```bash
cp .env.example .env
```

- **With `OPENAI_API_KEY` set:** real OpenAI embeddings (`text-embedding-3-small`
  by default) and chat completions (`gpt-4o-mini` by default) are used.
- **Without a key:** the project automatically falls back to a deterministic,
  offline provider — a hashed "fake" embedding plus a keyword-based extractive
  answerer — so ingestion, retrieval, and the CLI all still work end-to-end
  with no network access. This keeps the project runnable out of the box and
  makes the test suite fast and reliable. Answers in this mode are simple
  sentence extracts rather than fluent generated text, and retrieval quality
  is lower since the fake embedding isn't semantically meaningful — add a
  real API key for good answers.

## Usage

Build the index from `data/`:

```bash
python -m rag.cli ingest
```

Ask a question:

```bash
python -m rag.cli ask "How many days do I have to request a refund?"
```

Example output:

```text
Question: How many days do I have to request a refund?

Answer: Customers may request a full refund within 30 days of purchase, no questions asked.

Sources: data/company_handbook.md
```

To ask questions about your **own** documents, replace the files in `data/`
with your own `.md`/`.txt` files and re-run `ingest`.

## Configuration

All settings are read from the environment (see `.env.example`):

| Variable                  | Default                   | Description                              |
| -------------------------- | -------------------------- | ----------------------------------------- |
| `OPENAI_API_KEY`           | *(unset)*                  | Enables real OpenAI embeddings/chat       |
| `OPENAI_CHAT_MODEL`        | `gpt-4o-mini`              | Chat model used for answering             |
| `OPENAI_EMBEDDING_MODEL`   | `text-embedding-3-small`   | Embedding model used for indexing/queries |
| `RAG_DATA_DIR`             | `data`                     | Folder of documents to index              |
| `RAG_STORAGE_DIR`          | `storage`                  | Where the FAISS index is persisted        |
| `RAG_CHUNK_SIZE`           | `1000`                     | Max characters per chunk                  |
| `RAG_CHUNK_OVERLAP`        | `150`                      | Overlap between consecutive chunks        |
| `RAG_TOP_K`                | `4`                        | Number of chunks retrieved per question   |

## Tests

The test suite forces the offline fake provider, so it runs fully
deterministically with no network calls or API key required:

```bash
pytest
```
