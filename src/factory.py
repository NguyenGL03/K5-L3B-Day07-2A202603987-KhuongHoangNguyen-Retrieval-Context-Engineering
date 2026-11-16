"""Build retrievers by name so retrieval-only and agent runs share one definition."""
from __future__ import annotations

import json
from functools import lru_cache

from . import config
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

METHODS = {
    "dense": "Dense: embeddinggemma-2 query -> pgvector index B (Day 6), cosine, exact search",
    "bm25": "Lexical: Okapi BM25 (k1=1.5, b=0.75), Vietnamese syllable tokens",
    "bm25_bigram": "Lexical: BM25 with syllable unigrams + bigrams",
    "hybrid": "Hybrid: RRF(k=60) of dense top-20 and bm25 top-20",
    "hybrid_bigram": "Hybrid: RRF(k=60) of dense top-20 and bm25_bigram top-20",
}


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    return json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def embedder():
    from .embedding import QueryEmbedder
    return QueryEmbedder()


@lru_cache(maxsize=None)
def build(name: str):
    if name == "dense":
        return DenseRetriever(embedder())
    if name == "bm25":
        return BM25Retriever(load_chunks())
    if name == "bm25_bigram":
        return BM25Retriever(load_chunks(), bigrams=True)
    if name == "hybrid":
        return HybridRetriever([build("dense"), build("bm25")])
    if name == "hybrid_bigram":
        return HybridRetriever([build("dense"), build("bm25_bigram")])
    raise ValueError(f"unknown retriever {name!r}; choose from {sorted(METHODS)}")
