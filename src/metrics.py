"""Evidence matching and retrieval/context metrics (same hit definition as Day 6).

A chunk *matches* a gold evidence when source_id and page are equal and the
evidence quote is contained in the chunk text (whitespace-insensitive, casefold).

gold_mode "any": gold items are alternatives; a list of chunks covers the
question when one chunk matches one gold item.
gold_mode "all": every gold item is required; a list covers the question only
when each item is matched by some chunk. Hit@1 is then impossible by design and
is reported separately (`multi_evidence`).
"""
from __future__ import annotations

import re
from typing import Any


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def matches(chunk: dict[str, Any], gold: dict[str, Any]) -> bool:
    return (chunk["source_id"] == gold["source_id"] and chunk["page"] == gold["page"]
            and collapse(gold["evidence"]) in collapse(chunk["text"]))


def is_hit(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    """Chunk contains at least one gold evidence item."""
    return any(matches(chunk, g) for g in question["gold"])


def covered(chunks: list[dict[str, Any]], question: dict[str, Any]) -> bool:
    found = [any(matches(c, g) for c in chunks) for g in question["gold"]]
    return all(found) if question.get("gold_mode") == "all" else any(found)


def first_cover_rank(results: list[dict[str, Any]], question: dict[str, Any]) -> int | None:
    for k in range(1, len(results) + 1):
        if covered(results[:k], question):
            return k
    return None


def retrieval_row(question: dict[str, Any], results: list[dict[str, Any]],
                  context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = first_cover_rank(results, question)
    return {"hit1": first == 1, "hit3": first is not None and first <= 3,
            "first_hit_rank": first, "rr": (1 / first) if first else 0.0,
            "context_evidence": covered(context_chunks, question)}


def rate(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "N": d, "rate": round(n / d, 3) if d else None}
