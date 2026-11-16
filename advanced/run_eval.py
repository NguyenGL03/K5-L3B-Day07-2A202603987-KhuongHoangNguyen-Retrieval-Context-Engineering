"""Retrieval and agent evaluation for the advanced benchmark.

  python -m advanced.run_eval retrieval --split dev --methods dense bm25 hybrid --top-k 4
  python -m advanced.run_eval agent --split dev --config H_hybrid
  python -m advanced.run_eval summarize --split dev --config H_hybrid   (after reviews.json exists)

Retrieval metric = page level on the ORIGINAL question: a result hits when its page is one of the
question's accepted evidence pages; multi-hop questions also report whether every required page
group is covered. The test split is refused until advanced/results/selection.json exists.
Gold is only read here, after retrieval/agent outputs are produced, never passed to them.
"""
from __future__ import annotations

import argparse
import re
import json
import statistics
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

from .agent import AgentConfig, run_agent

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GROUPS = ["table", "visual", "chart", "multihop", "text", "unanswerable"]

CONFIGS: dict[str, dict] = {
    "A_dense": {"retriever": "dense", "top_k": 4, "budget": 6000, "max_searches": 4, "original_first": True},
    "A_bm25": {"retriever": "bm25", "top_k": 4, "budget": 6000, "max_searches": 4, "original_first": True},
    "A_hybrid": {"retriever": "hybrid", "top_k": 4, "budget": 6000, "max_searches": 4, "original_first": True},
    "B_hybrid_plan": {"retriever": "hybrid", "top_k": 4, "budget": 7000, "max_searches": 4, "original_first": True,
                      "plan": True},
    "C_hybrid_plan_guard": {"retriever": "hybrid", "top_k": 4, "budget": 7000, "max_searches": 4, "original_first": True,
                            "plan": True, "num_guard": True},
    # C2 = C re-run after fixing two bugs found on dev (guard number regex, citation repair step)
    "C2_hybrid_plan_guard": {"retriever": "hybrid", "top_k": 4, "budget": 7000, "max_searches": 4, "original_first": True,
                             "plan": True, "num_guard": True},
    "D_hybrid600_plan_guard": {"retriever": "hybrid_600", "top_k": 6, "budget": 7000, "max_searches": 4,
                               "original_first": True, "plan": True, "num_guard": True},
    "E_hybrid600_plan_guard_cite": {"retriever": "hybrid_600", "top_k": 6, "budget": 7000, "max_searches": 4,
                                    "original_first": True, "plan": True, "num_guard": True, "cite_guard": True},
    "F_hybrid600_v2": {"retriever": "hybrid_600", "top_k": 6, "budget": 8000, "max_searches": 4, "original_first": True,
                       "plan": True, "num_guard": True, "cite_guard": True, "v2": True},
    "G_hybrid600_v2_pagehead": {"retriever": "hybrid_600", "top_k": 6, "budget": 10000, "max_searches": 4,
                                "original_first": True, "plan": True, "num_guard": True, "cite_guard": True, "v2": True,
                                "page_head": True, "guard_rounds": 2},
    "H_hybrid600_v2_pagehead_operands": {"retriever": "hybrid_600", "top_k": 6, "budget": 10000, "max_searches": 4,
                                         "original_first": True, "plan": True, "num_guard": True, "cite_guard": True,
                                         "v2": True, "page_head": True, "guard_rounds": 2, "cite_operands": True},
    "I_hybrid600_full": {"retriever": "hybrid_600", "top_k": 6, "budget": 10000, "max_searches": 4,
                         "original_first": True, "plan": True, "num_guard": True, "cite_guard": True, "v2": True,
                         "page_head": True, "guard_rounds": 2, "cite_operands": True, "unit_v3": True},
}
# Round 2: tuned on the `tune` split (dev + the round-1 test, which was seen once its result was read);
# the new test (`test2`) is written and frozen after the round-2 selection and run once.
_H = CONFIGS["H_hybrid600_v2_pagehead_operands"]
CONFIGS["J_H_display_unit"] = {**_H, "display": True, "unit_fix": True, "abstain_start": True}
CONFIGS["K_J_plan3_retry"] = {**CONFIGS["J_H_display_unit"], "plan_v3": True, "rules_v4": True,
                              "abstain_retry": True, "max_searches": 6}

# after labelling J/K on `tune`: unit heading carried into continuation chunks (display only),
# named-source rule, recheck of abstentions that follow a real calculation
CONFIGS["M_J_source_recheck"] = {**CONFIGS["J_H_display_unit"], "source_rule": True, "calc_recheck": True}
CONFIGS["L_K_source_recheck"] = {**CONFIGS["K_J_plan3_retry"], "source_rule": True, "calc_recheck": True}
# per-number unit fix (v1 skipped an answer that mentioned RM'000 anywhere)
CONFIGS["N_M_unit2"] = {**CONFIGS["M_J_source_recheck"], "unit_fix_v2": True}
CONFIGS["P_L_unit2"] = {**CONFIGS["L_K_source_recheck"], "unit_fix_v2": True}
# same pipeline with the thinking variant of Qwen3-4B (reasoning tokens are not shown to the pipeline)
CONFIGS["Q_N_think"] = {**CONFIGS["N_M_unit2"], "_llm": "qwen3:4b"}

# Hosted model chosen by the user for round 2 (Gemini API). Same pipeline as P.
GEMINI_PRICES = {"gemini-3.5-flash-lite": {"price_in": 0.30, "price_out": 2.50}}
CONFIGS["R_P_gemini35lite"] = {**CONFIGS["P_L_unit2"], "_llm": "gemini-3.5-flash-lite"}
# after a 3-question smoke test of R: the guard rejected correct unit conversions (RM1,182,608,000) and the model
# then looped on calculate -> accept scaled context numbers, do not re-run identical calculations
CONFIGS["S_R_scaled_dedupe"] = {**CONFIGS["R_P_gemini35lite"], "scaled_ok": True, "dedupe_calc": True}

SPLITS = ("dev", "test", "tune", "test2")


def questions(split: str) -> list[dict]:
    if split == "test" and not (RESULTS / "selection.json").exists():
        raise SystemExit("Freeze the configuration on dev (advanced/results/selection.json) before running test.")
    if split == "test2":
        if not (RESULTS / "selection_r2.json").exists() or not (ROOT / "benchmark" / "FROZEN_test2.json").exists():
            raise SystemExit("test2 needs results/selection_r2.json (chosen on `tune`) and a frozen benchmark/test2.json.")
        return json.loads((ROOT / "benchmark" / "test2.json").read_text(encoding="utf-8"))["questions"]
    qs = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))["questions"]
    if split == "tune":
        return [q for q in qs if q["split"] in ("dev", "test")]
    return [q for q in qs if q["split"] == split]


@lru_cache(maxsize=None)
def chunks(size: str = "") -> list[dict]:
    return json.loads((ROOT / "data" / (f"chunks_{size}.json" if size else "chunks.json")).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def embedder():
    from src.embedding import QueryEmbedder
    return QueryEmbedder()


@lru_cache(maxsize=None)
def retriever(name: str):
    """dense | bm25 | hybrid, optionally suffixed with an index size, e.g. hybrid_600 (600-char chunks)."""
    base, size = re.fullmatch(r"(dense|bm25|hybrid)(?:_(\d+))?", name).groups()
    size = size or ""
    sfx = f"_{size}" if size else ""
    if base == "dense":
        r = DenseRetriever(embedder(), strategy="PBB" + size)
        r.name = name
        return r
    if base == "bm25":
        r = BM25Retriever(chunks(size))
        r.name = name
        return r
    if base == "hybrid":
        return HybridRetriever([retriever("dense" + sfx), retriever("bm25" + sfx)])
    raise ValueError(name)


def page_of(chunk_id: str) -> int:
    return int(re.search(r"-p(\d{3})-", chunk_id).group(1))


def page_hits(q: dict, pages: list[int]) -> dict:
    acc = set(q.get("accept_pages", []))
    first = next((i + 1 for i, p in enumerate(pages) if p in acc), None)
    req = q.get("required_pages")
    all_req = all(any(p in pages for p in grp) for grp in req) if req else first is not None
    return {"first_hit_rank": first, "all_required": all_req}


def cmd_retrieval(args) -> None:
    qs = [q for q in questions(args.split) if q["group"] != "unanswerable"]
    out_dir = RESULTS / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = out_dir / "summary.json"
    summ = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    for m in args.methods:
        r = retriever(m)
        if not m.startswith("bm25"):
            embedder()._cache.clear()
        traces = []
        for q in qs:
            t0 = time.perf_counter()
            res = r.search(q["question"], filters=None, top_k=args.top_k)
            ms = (time.perf_counter() - t0) * 1000
            pages = [x["page"] for x in res]
            traces.append({"id": q["id"], "group": q["group"], "query": q["question"], "method": m,
                           "results": [{"rank": x["rank"], "id": x["id"], "page": x["page"], "kind": x["metadata"].get("kind"),
                                        "score": x["score"]} for x in res],
                           "ms": round(ms, 2), **page_hits(q, pages)})
        by = defaultdict(list)
        for t in traces:
            by[t["group"]].append(t)
        key = f"{m}_k{args.top_k}"
        summ[key] = {"method": m, "top_k": args.top_k, "N": len(traces),
                     "hit_at_1": sum(t["first_hit_rank"] == 1 for t in traces),
                     f"hit_at_{args.top_k}": sum(t["first_hit_rank"] is not None for t in traces),
                     "all_required_pages": sum(t["all_required"] for t in traces),
                     "by_group": {g: f"{sum(t['first_hit_rank'] is not None for t in v)}/{len(v)}" for g, v in by.items()},
                     "misses": [t["id"] for t in traces if t["first_hit_rank"] is None],
                     "median_ms": round(statistics.median(t["ms"] for t in traces), 1)}
        (out_dir / f"{key}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(key, json.dumps(summ[key], ensure_ascii=False))
    summ_path.write_text(json.dumps(summ, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_agent(args) -> None:
    from src.llm import LLMError, OllamaChat
    spec = dict(CONFIGS[args.config])
    model = spec.pop("_llm", None)              # optional LLM override (default: src.config.LLM_MODEL)
    cfg = AgentConfig(**spec)
    if model and model.startswith("gemini"):
        from .llm_gemini import GeminiChat
        llm = GeminiChat(model, **GEMINI_PRICES.get(model, {}))
    else:
        llm = OllamaChat(model=model) if model else OllamaChat()
    r = retriever(cfg.retriever)
    size = re.fullmatch(r"(?:dense|bm25|hybrid)(?:_(\d+))?", cfg.retriever).group(1) or ""
    heads: dict[int, dict] = {}
    for c in chunks(size):                       # first table/text chunk of every page, document order
        if c["metadata"]["kind"].startswith("table"):
            heads.setdefault(c["page"], c)
    display = {c["id"]: c.get("display") or c["text"] for c in chunks(size)} if cfg.display else None
    qs = questions(args.split)
    if args.only:
        qs = [q for q in qs if q["id"] in args.only]
    out = RESULTS / "agent" / args.split / args.config
    out.mkdir(parents=True, exist_ok=True)
    traces = []
    for q in qs:
        try:
            tr = run_agent(q["question"], llm, r, cfg, page_heads=heads, display=display)
        except (LLMError, Exception) as exc:  # recorded, counted as failure, never replaced
            tr = {"status": "error", "error": repr(exc), "answer": "", "citations": [], "elapsed_s": None}
        cpages = sorted({page_of(c) for c in tr.get("citations", [])})
        tr.update({"id": q["id"], "group": q["group"], "question": q["question"], "cited_pages": cpages})
        if q["group"] != "unanswerable":
            tr["context_page_hit"] = page_hits(q, tr.get("final_context_pages", []))
        traces.append(tr)
        print(f"{q['id']:5s} {tr['status']:8s} {tr.get('elapsed_s')}s searches={tr.get('n_searches')} "
              f"calc={len(tr.get('calculations', []))} cited={cpages} | {tr['answer'][:150]!r}")
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ok = [t for t in traces if t["status"] != "error"]
    auto = {"config": CONFIGS[args.config], "llm": llm.describe(), "N": len(traces),
            "errors": len(traces) - len(ok),
            "abstain_on_unanswerable": f"{sum(t['status'] == 'abstain' for t in traces if t['group'] == 'unanswerable')}/{sum(t['group'] == 'unanswerable' for t in traces)}",
            "abstain_on_answerable": sum(t["status"] == "abstain" for t in traces if t["group"] != "unanswerable"),
            "context_has_required_pages": f"{sum(t.get('context_page_hit', {}).get('all_required', False) for t in traces)}/{sum(t['group'] != 'unanswerable' for t in traces)}",
            "median_latency_s": statistics.median(t["elapsed_s"] for t in ok) if ok else None,
            "total_latency_s": round(sum(t["elapsed_s"] for t in ok), 1),
            "mean_prompt_tokens": round(statistics.mean(t["prompt_tokens"] for t in ok), 1) if ok else None,
            "mean_llm_calls": round(statistics.mean(len(t["llm_calls"]) for t in ok), 2) if ok else None,
            "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
            "cost_usd": round(sum(c.get("cost_usd", 0.0) for t in ok for c in t["llm_calls"]), 4),
            "cost_note": ("paid-tier list price (USD/1M tokens) " + json.dumps(GEMINI_PRICES.get(model or "", {}))
                          + ", checked 2026-11-16 on aistudio.google.com/docs/pricing; 0 if the key is on the free tier")
            if model and model.startswith("gemini") else "local model, no API cost"}
    (out / "auto.json").write_text(json.dumps(auto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in auto.items() if k != "llm"}, ensure_ascii=False))


def cmd_summarize(args) -> None:
    out = RESULTS / "agent" / args.split / args.config
    traces = {t["id"]: t for t in json.loads((out / "traces.json").read_text(encoding="utf-8"))}
    reviews = json.loads((out / "reviews.json").read_text(encoding="utf-8"))
    assert set(r["id"] for r in reviews) == set(traces), "every question needs one label"
    by = defaultdict(lambda: [0, 0])
    for r in reviews:
        g = traces[r["id"]]["group"]
        by[g][1] += 1
        by[g][0] += bool(r["pass"])
    n = sum(v[1] for v in by.values())
    p = sum(v[0] for v in by.values())
    summary = {"split": args.split, "config": args.config, "pass": p, "N": n, "rate": round(p / n, 4),
               "by_group": {g: f"{by[g][0]}/{by[g][1]}" for g in GROUPS if g in by},
               "errors": sum(t["status"] == "error" for t in traces.values()),
               "failed_ids": [r["id"] for r in reviews if not r["pass"]]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("retrieval")
    a.add_argument("--split", required=True, choices=SPLITS)
    a.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    a.add_argument("--top-k", type=int, default=4)
    b = sub.add_parser("agent")
    b.add_argument("--split", required=True, choices=SPLITS)
    b.add_argument("--config", required=True)
    b.add_argument("--only", nargs="*")
    c = sub.add_parser("summarize")
    c.add_argument("--split", required=True, choices=SPLITS)
    c.add_argument("--config", required=True)
    args = p.parse_args()
    {"retrieval": cmd_retrieval, "agent": cmd_agent, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
