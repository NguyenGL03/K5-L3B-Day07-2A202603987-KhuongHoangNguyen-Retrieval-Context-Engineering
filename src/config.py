"""Shared paths and connection settings for the Day 7 agentic RAG lab.

Everything that identifies the Day 6 index (table, strategy, embedding model,
revision, prompts) is pinned here so query embeddings stay compatible with the
stored document vectors.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CHUNKS_PATH = DATA / "day6" / "chunks.json"
QUESTIONS_PATH = DATA / "benchmark" / "questions.json"   # answerable, Day 7 schema
CASES_PATH = DATA / "benchmark" / "cases.json"           # answerable + unanswerable

# Day 6 workspace (source of the index and the frozen benchmark).
DAY6_ROOT = Path(os.environ.get(
    "DAY6_ROOT",
    ROOT.parents[1] / "day06" / "K5-L3B-Day06-KhuongHoangNguyen-2A202603987-DataIngestionEngineering"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; plain env vars also work
        return
    load_dotenv(ROOT / ".env")


_load_env()
# The pinned embedding revision is read from the local Hugging Face cache; set
# HF_HUB_OFFLINE=0 only for the first download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Day 6 vector store: PostgreSQL + pgvector, table rag_chunks, index B (paragraph <=800).
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
PG_TABLE = "rag_chunks"
PG_STRATEGY = "B"

# Day 6 embedding model (must match the index; see Day 6 results/run_config.json).
EMBEDDING_MODEL = "google/embeddinggemma-2"
EMBEDDING_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EMBEDDING_DIM = 768

# Agent model served by Ollama (local, cost 0).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "6144"))
LLM_SEED = 7
