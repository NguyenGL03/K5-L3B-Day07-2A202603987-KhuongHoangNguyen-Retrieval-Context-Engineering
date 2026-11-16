"""Retrieval-only benchmark on the ORIGINAL question (no rewriting, no agent).

For each method: top-3 ranking (Hit@1, Evidence Hit@3, MRR@3), plus the context
that top-k produces under the character budget (Context Evidence Rate).
Only answerable questions enter the denominators. The user-provided `filter`
(section) is applied to every method unless --no-filter.

Run:  python -m src.retrieval_eval --split dev --methods dense bm25 bm25_bigram hybrid
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from . import config, context, factory, metrics


def load_questions(split: str) -> list[dict]:
    if split == "test" and not (config.RESULTS / "selection.json").exists():
        raise SystemExit("Chốt cấu hình (results/selection.json) trước khi chạy test.")
    qs = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [q for q in qs if q["split"] == split]


def run(method: str, questions: list[dict], *, top_k: int, budget: int, use_filter: bool,
        policy: str = "append") -> tuple[list[dict], dict]:
    retriever = factory.build(method)
    if method != "bm25" and not method.startswith("bm25_"):
        factory.embedder()._cache.clear()  # time includes query embedding for every method
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=max(3, top_k))
        elapsed = (time.perf_counter() - start) * 1000
        top3, topk = results[:3], results[:top_k]
        ctx, ids, kept = context.pack(topk, budget, policy)
        row = metrics.retrieval_row(q, top3, kept)
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q["type"], "query": q["question"],
            "query_kind": "original", "filter": filters, "method": method, "top_k": top_k,
            "budget_chars": budget, "context_policy": policy,
            "results": [{"rank": r["rank"], "id": r["id"], "page": r["page"],
                         "section": r["metadata"].get("section"), "score": r["score"],
                         "components": r.get("components"), "hit": metrics.is_hit(r, q)}
                        for r in results],
            "context_ids": ids, "context_chars": len(ctx), "elapsed_ms": round(elapsed, 2), **row})
    n = len(traces)
    summary = {
        "method": method, "description": factory.METHODS[method], "split": questions[0]["split"],
        "top_k": top_k, "budget_chars": budget, "context_policy": policy, "filter": use_filter,
        "N": n,
        "hit_at_1": metrics.rate(sum(t["hit1"] for t in traces), n),
        "evidence_hit_at_3": metrics.rate(sum(t["hit3"] for t in traces), n),
        "mrr_at_3": round(sum(t["rr"] for t in traces) / n, 3),
        "context_evidence_rate": metrics.rate(sum(t["context_evidence"] for t in traces), n),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "cost_usd": 0.0,
        "misses_at_3": [t["query_id"] for t in traces if not t["hit3"]],
        "not_top1": [t["query_id"] for t in traces if not t["hit1"]],
    }
    return traces, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--methods", nargs="+", default=["dense", "bm25", "bm25_bigram", "hybrid", "hybrid_bigram"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=2600)
    p.add_argument("--policy", default="append")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--tag", default="", help="suffix for the output folder name")
    args = p.parse_args()

    questions = load_questions(args.split)
    out_dir = config.RESULTS / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    all_summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for method in args.methods:
        traces, summary = run(method, questions, top_k=args.top_k, budget=args.budget,
                              use_filter=not args.no_filter, policy=args.policy)
        key = method + ("_nofilter" if args.no_filter else "") + (f"_{args.tag}" if args.tag else "")
        (out_dir / key).mkdir(exist_ok=True)
        (out_dir / key / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_summaries[key] = summary
        print(f"{key:28s} Hit@1 {summary['hit_at_1']['n']}/{summary['N']}  "
              f"Hit@3 {summary['evidence_hit_at_3']['n']}/{summary['N']}  MRR {summary['mrr_at_3']}  "
              f"CtxEv {summary['context_evidence_rate']['n']}/{summary['N']}  "
              f"{summary['median_ms']} ms  miss@3 {summary['misses_at_3']}  not@1 {summary['not_top1']}")
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
