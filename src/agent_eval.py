"""Run the real agent on one split for one named configuration.

Outputs (results/agent/<split>/<config>/):
  traces.json     one record per case (query_id, status, events, citations, context, tokens, time)
  auto.json       metrics computed from traces + gold (tool use, searches, context evidence,
                  first-search Hit@1/@3, status counts, latency, tokens) — no answer grading
  review_todo.json skeleton for the manual labels (answer_grade, citation_support, abstention_correct)

Run:  python -m src.agent_eval --split dev --config A0_dense
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections import Counter

from . import config, factory, metrics
from .agent import AgentConfig, run_agent
from .llm import OllamaChat

CONFIGS: dict[str, dict] = {
    # baseline: dense retrieval on the Day 6 index, starter-like context policy
    "A0_dense": {"retriever": "dense", "top_k": 3, "budget": 2600, "context_policy": "append", "max_searches": 3},
    # method 2 / 3 inside the agent (same budget, same loop)
    "A1_bm25": {"retriever": "bm25_bigram", "top_k": 3, "budget": 2600, "context_policy": "append", "max_searches": 3},
    "A2_hybrid": {"retriever": "hybrid_bigram", "top_k": 3, "budget": 2600, "context_policy": "append", "max_searches": 3},
    # improvement 1 (dev errors q02/q15/q19: agent-rewritten query loses evidence the original question finds)
    "A3_hybrid_orig": {"retriever": "hybrid_bigram", "top_k": 3, "budget": 2600, "context_policy": "append",
                       "max_searches": 3, "original_first": True},
    # improvement 2 (q09: list continues in the next chunk; agent never re-searches because the budget is full)
    "A4_hybrid_orig_ctx": {"retriever": "hybrid_bigram", "top_k": 3, "budget": 4500, "context_policy": "best_rank",
                           "max_searches": 3, "original_first": True, "neighbors": True, "research_prompt": True},
}


def load_cases(split: str) -> tuple[list[dict], dict[str, dict]]:
    if split == "test" and not (config.RESULTS / "selection.json").exists():
        raise SystemExit("Chốt cấu hình (results/selection.json) trước khi chạy test.")
    cases = [c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8")) if c["split"] == split]
    gold = {q["id"]: q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))}
    return cases, gold


def scope_note(filters: dict | None, chunks: list[dict]) -> str:
    if not filters:
        return ""
    titles = {c["metadata"].get("section"): c["metadata"].get("section_title") for c in chunks}
    return ", ".join(f"chương '{titles.get(v, v)}'" for k, v in filters.items())


def auto_metrics(traces: list[dict], gold: dict[str, dict], chunk_map: dict[str, dict]) -> dict:
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    unans = [t for t in ok if not t["answerable"]]
    first_hit1 = first_hit3 = ctx_ev = 0
    for t in ans:
        q = gold[t["query_id"]]
        first = next((e for e in t["events"] if e["event"] == "search"), None)
        if first:
            ranked = [chunk_map[r["id"]] for r in first["results"]]
            rank = metrics.first_cover_rank(ranked[:3], q)
            first_hit1 += rank == 1
            first_hit3 += rank is not None
        ctx_ev += metrics.covered(t["final_context_chunks"], q)
    searches = Counter(t["n_searches"] for t in ok)
    return {
        "cases": len(traces), "errors": len(traces) - len(ok),
        "tool_use": metrics.rate(sum(t["n_searches"] > 0 for t in ok), len(ok)),
        "searches_distribution": {str(k): v for k, v in sorted(searches.items())},
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "first_search_hit_at_1": metrics.rate(first_hit1, len(ans)),
        "first_search_evidence_hit_at_3": metrics.rate(first_hit3, len(ans)),
        "context_evidence_rate": metrics.rate(ctx_ev, len(ans)),
        "status_answerable": dict(Counter(t["status"] for t in ans)),
        "status_unanswerable": dict(Counter(t["status"] for t in unans)),
        "abstain_on_unanswerable": metrics.rate(sum(t["status"] == "abstain" for t in unans), len(unans)),
        "abstain_on_answerable": metrics.rate(sum(t["status"] == "abstain" for t in ans), len(ans)),
        "answers_with_citation": metrics.rate(sum(bool(t["citations"]) for t in ok if t["status"] == "answered"),
                                              sum(t["status"] == "answered" for t in ok)),
        "end_events": dict(Counter(t["events"][-1]["event"] for t in ok if t["events"])),
        "median_latency_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
        "total_latency_s": round(sum(t["elapsed_s"] for t in ok), 1),
        "mean_llm_calls": round(statistics.mean(len(t["llm_calls"]) for t in ok), 2) if ok else None,
        "mean_prompt_tokens": round(statistics.mean(t["prompt_tokens"] for t in ok), 1) if ok else None,
        "mean_completion_tokens": round(statistics.mean(t["completion_tokens"] for t in ok), 1) if ok else None,
        "mean_final_context_chars": round(statistics.mean(len(t["final_context"]) for t in ok), 1) if ok else None,
        "cost_usd": 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--only", nargs="*", help="run only these case IDs (debug; not for reporting)")
    args = p.parse_args()

    cfg = AgentConfig(**CONFIGS[args.config])
    cases, gold = load_cases(args.split)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    chunks = factory.load_chunks()
    chunk_map = {c["id"]: c for c in chunks}
    # document order = page, then chunk id (zero-padded paragraph index)
    ordered = sorted(chunks, key=lambda c: (c["source_id"], c["page"], c["id"]))
    next_chunk = {a["id"]: b for a, b in zip(ordered, ordered[1:]) if a["source_id"] == b["source_id"]}
    retriever = factory.build(cfg.retriever)
    llm = OllamaChat()
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    traces = []
    for case in cases:
        filters = case.get("filter")
        try:
            result = run_agent(case["question"], llm, retriever, cfg, filters=filters,
                               scope_note=scope_note(filters, chunks), next_chunk=next_chunk)
        except Exception as exc:  # report provider/store errors; never substitute fake output
            result = {"status": "error", "answer": "", "citations": [], "events": [{"event": "error", "error": repr(exc),
                      "traceback": traceback.format_exc(limit=3)}], "final_context": "", "final_context_ids": [],
                      "final_context_chunks": [], "queries": [], "n_searches": 0, "llm_calls": [],
                      "prompt_tokens": 0, "completion_tokens": 0, "elapsed_s": 0.0}
        trace = {"query_id": case["id"], "split": case["split"], "type": case["type"],
                 "answerable": case["answerable"], "question": case["question"], "filter": filters, **result}
        traces.append(trace)
        print(f"{case['id']:4s} {trace['status']:8s} searches={trace['n_searches']} "
              f"{trace['elapsed_s']:6.1f}s  {trace['answer'][:110]!r}", flush=True)

    summary = auto_metrics(traces, gold, chunk_map)
    out = config.RESULTS / "agent" / args.split / (args.config + ("_partial" if args.only else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config_name": args.config, "agent_config": CONFIGS[args.config],
            "retriever": factory.METHODS[cfg.retriever], "llm": llm.describe(),
            "embedding": factory.embedder().describe() if "dense" in cfg.retriever or "hybrid" in cfg.retriever else None,
            "run_started": run_started, "split": args.split}
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "auto.json").write_text(json.dumps({**meta, "metrics": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    todo = [{"query_id": t["query_id"], "answerable": t["answerable"], "status": t["status"],
             "citations": t["citations"], "answer_grade": None, "citation_support": [None] * len(t["citations"]),
             **({} if t["answerable"] else {"abstention_correct": None}), "reason": ""}
            for t in traces if t["status"] != "error"]
    if not (out / "reviews.json").exists():
        (out / "review_todo.json").write_text(json.dumps(todo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
