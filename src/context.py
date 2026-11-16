"""Context selection: dedupe, ordering policy and a character budget.

Budget unit = characters of the final context string, headers included (the
same unit as the starter). Token counts actually sent to the model are taken
from Ollama's prompt_eval_count and logged by the agent.

Policies over the candidates gathered from all searches of one question:
- "append"    : keep chunks in arrival order (first search first) — starter behaviour.
- "best_rank" : order by the best rank a chunk reached in any search, then arrival
                order; a strong hit from a later, better query is not starved by
                weaker chunks from the first query.
"""
from __future__ import annotations

from typing import Any


def block(c: dict[str, Any]) -> str:
    section = c.get("metadata", {}).get("section_title") or c.get("metadata", {}).get("section") or ""
    return f"[{c['id']} | {c['source']} | trang {c['page']} | {section}]\n{c['text']}"


def order(candidates: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    first_seen: dict[str, int] = {}
    best: dict[str, tuple[int, int]] = {}
    unique: dict[str, dict[str, Any]] = {}
    for pos, c in enumerate(candidates):
        first_seen.setdefault(c["id"], pos)
        unique.setdefault(c["id"], c)
        key = (c.get("rank", 10**6), first_seen[c["id"]])
        best[c["id"]] = min(best.get(c["id"], key), key)
    if policy == "append":
        return sorted(unique.values(), key=lambda c: first_seen[c["id"]])
    if policy == "best_rank":
        return sorted(unique.values(), key=lambda c: best[c["id"]])
    raise ValueError(f"unknown context policy {policy!r}")


def pack(candidates: list[dict[str, Any]], budget: int, policy: str = "append",
         next_chunk: dict[str, dict[str, Any]] | None = None) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Return (context string, kept ids, kept chunks); chunks that do not fit are skipped.

    With `next_chunk` (id -> following chunk in document order), after all
    retrieved chunks are placed, the chunk that directly follows each kept one is
    added if it still fits (lists and tables often continue in the next chunk).
    Neighbours are marked `neighbor_of` and never displace a retrieved chunk.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    blocks, kept = [], []

    def add(c: dict[str, Any]) -> None:
        if c["id"] in {k["id"] for k in kept}:
            return
        b = block(c)
        if len("\n\n".join(blocks + [b])) <= budget:
            blocks.append(b)
            kept.append(c)

    for c in order(candidates, policy):
        add(c)
    if next_chunk:
        for c in list(kept):
            nxt = next_chunk.get(c["id"])
            if nxt is not None:
                add({**nxt, "neighbor_of": c["id"]})
        # keep document reading order inside the final string for neighbours
        pos = {c["id"]: i for i, c in enumerate(kept)}
        kept.sort(key=lambda c: pos[c.get("neighbor_of", c["id"])] + (0.5 if "neighbor_of" in c else 0))
        blocks = [block(c) for c in kept]
    return "\n\n".join(blocks), [c["id"] for c in kept], kept
