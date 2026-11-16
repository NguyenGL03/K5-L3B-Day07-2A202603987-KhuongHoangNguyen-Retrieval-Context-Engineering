"""Three retrieval methods with one output contract.

Every result: id, text, source_id, source, page, metadata, score, rank, strategy
(+ method-specific diagnostics). Filters are user-provided metadata (here only
`section`), applied identically by every method; gold never enters this module.

- DenseRetriever : Day 6 pgvector index B, cosine distance, exact search, SQL filter.
- BM25Retriever  : Okapi BM25 over the same chunks; tokenizer = Vietnamese syllables,
                   optionally + syllable bigrams (approximates multi-syllable words).
- HybridRetriever: Reciprocal Rank Fusion of dense and BM25 candidate lists.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Protocol

from . import config

FILTERABLE = {"section"}


class Retriever(Protocol):
    name: str

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        ...


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - FILTERABLE
    if bad:
        raise ValueError(f"filter keys not allowed: {sorted(bad)}")
    return filters


def _matches(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in filters.items())


# ---------------------------------------------------------------- dense
class DenseRetriever:
    name = "dense"

    def __init__(self, embedder, database_url: str = config.DATABASE_URL,
                 strategy: str = config.PG_STRATEGY) -> None:
        import psycopg

        self.embedder, self.strategy = embedder, strategy
        self.conn = psycopg.connect(database_url, connect_timeout=5, autocommit=True)
        row = self.conn.execute(
            f"SELECT count(*), min(embedding_model), max(embedding_model) FROM {config.PG_TABLE} "
            "WHERE strategy = %s", (strategy,)).fetchone()
        if not row[0] or row[1] != config.EMBEDDING_MODEL or row[2] != config.EMBEDDING_MODEL:
            raise RuntimeError(f"index {strategy} missing or embedded with another model: {row}")
        self.size = row[0]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        vec = "[" + ",".join(repr(x) for x in self.embedder.embed_query(query)) + "]"
        where, params = ["strategy = %s"], [self.strategy]
        for key, value in filters.items():
            where.append(f"{key} = %s")
            params.append(value)
        rows = self.conn.execute(
            f"SELECT chunk_id, document_id, source, page, section, section_no, text, metadata, "
            f"1 - (embedding <=> %s::vector) FROM {config.PG_TABLE} WHERE {' AND '.join(where)} "
            f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s",
            [vec, *params, vec, top_k]).fetchall()
        out = []
        for rank, (cid, doc, src, page, section, section_no, text, meta, score) in enumerate(rows, 1):
            meta = {**(meta or {}), "section": section, "section_no": section_no}
            out.append({"id": cid, "text": text, "source_id": doc, "source": src, "page": page,
                        "metadata": meta, "score": round(float(score), 6), "rank": rank,
                        "strategy": self.name})
        return out


# ---------------------------------------------------------------- lexical
def tokenize(text: str, bigrams: bool = False) -> list[str]:
    syl = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    if not bigrams:
        return syl
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]], *, bigrams: bool = False,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.name = "bm25_bigram" if bigrams else "bm25"
        self.chunks, self.bigrams, self.k1, self.b = chunks, bigrams, k1, b
        self.tf = [Counter(tokenize(c["text"], bigrams)) for c in chunks]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for t in query_tokens:
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        q = list(dict.fromkeys(tokenize(query, self.bigrams)))  # unique query terms
        scored = [(self.score(q, i), c) for i, c in enumerate(self.chunks) if _matches(c, filters)]
        scored = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]["id"]))[:top_k]
        return [{**c, "score": round(s, 6), "rank": r, "strategy": self.name}
                for r, (s, c) in enumerate(scored, 1)]


# ---------------------------------------------------------------- hybrid
class HybridRetriever:
    """RRF: score(d) = sum over retrievers of 1 / (k + rank_r(d)).

    Rank-based fusion avoids mixing cosine similarity with unbounded BM25 scores.
    """

    def __init__(self, retrievers: list, *, k: int = 60, pool: int = 20) -> None:
        self.retrievers, self.k, self.pool = retrievers, k, pool
        self.name = "hybrid_rrf(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for r in self.retrievers:
            for item in r.search(query, filters=filters, top_k=self.pool):
                entry = fused.setdefault(item["id"], {**item, "score": 0.0, "components": {}})
                entry["score"] += 1 / (self.k + item["rank"])
                entry["components"][r.name] = {"rank": item["rank"], "score": item["score"]}
        ranked = sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]
        return [{**c, "score": round(c["score"], 6), "rank": i, "strategy": self.name}
                for i, c in enumerate(ranked, 1)]
