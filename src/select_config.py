"""Freeze the agent configuration on DEV only, then write the dev comparison table.

Criterion (applied in this order; written before any Day 7 test run):
  1. higher mean grounded-answer grade (/2, manual labels) on dev answerable questions
  2. fewer wrong abstentions + more correct abstentions on dev
  3. higher valid-citation rate (manual citation_support)
  4. higher Context Evidence Rate (gold evidence in the final context)
  5. lower median latency per question, then fewer prompt tokens (cost proxy; $0 local)
Baseline for the test comparison = A0_dense (agent + dense Day 6 retrieval).

Writes results/selection.json (unlocks --split test) and results/dev_comparison.md.
Run: python -m src.select_config
"""
from __future__ import annotations

import json
import time

from . import config
from .agent_eval import CONFIGS

BASELINE = "A0_dense"
CRITERION = __doc__.split("Criterion")[1].split("Baseline")[0].strip()


def load(name: str) -> dict:
    run = config.RESULTS / "agent" / "dev" / name
    auto = json.loads((run / "auto.json").read_text(encoding="utf-8"))
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return {"name": name, "cfg": auto["agent_config"], "auto": auto["metrics"], "labels": summ}


def key(r: dict) -> tuple:
    a, s = r["auto"], r["labels"]
    cit = s["citation_validity"]
    return (-(s["grounded_answer"]["mean"] or 0),
            s["false_abstention"]["n"] - s["abstention"]["n"],
            -(cit["n"] / cit["N"] if cit["N"] else 0),
            -a["context_evidence_rate"]["n"],
            a["median_latency_s"], a["mean_prompt_tokens"])


def fmt(x: dict) -> str:
    return f"{x['n']}/{x['N']} ({x['n'] / x['N']:.0%})" if x["N"] else "0/0"


def main() -> None:
    rows = [load(n) for n in CONFIGS if (config.RESULTS / "agent" / "dev" / n / "summary.json").exists()]
    ranked = sorted(rows, key=key)
    chosen = ranked[0]
    ties = [r["name"] for r in ranked[1:] if key(r)[:4] == key(chosen)[:4]]

    retr = json.loads((config.RESULTS / "retrieval" / "dev" / "summary.json").read_text(encoding="utf-8"))
    lines = ["# So sánh trên dev (Day 7)", "",
             "## Retrieval trên câu hỏi gốc (không qua agent), top-3, budget 2600 ký tự", "",
             "| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | MRR@3 | Context Evidence | Median ms/query |",
             "|---|---|---|---|---|---|---|"]
    for k, s in retr.items():
        lines.append(f"| {k} | {'có' if s['filter'] else 'không'} | {fmt(s['hit_at_1'])} | {fmt(s['evidence_hit_at_3'])} | "
                     f"{s['mrr_at_3']} | {fmt(s['context_evidence_rate'])} | {s['median_ms']} |")
    lines += ["", "## Agent (qwen3:4b-instruct, Ollama) trên dev: 17 câu có đáp án + 2 câu thiếu evidence", "",
              "| Cấu hình | Retriever, top-k, budget, policy, thêm | Hit@1 / Hit@3 lượt tìm 1 | Context Evidence | Grounded 0/1/2 (TB /2) | Citation hợp lệ | Từ chối đúng | Từ chối nhầm | Lượt tìm TB | LLM calls TB | Median s/câu | Prompt tok TB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, s, c = r["auto"], r["labels"], r["cfg"]
        extra = [k for k in ("original_first", "neighbors", "research_prompt") if c.get(k)]
        g = s["grounded_answer"]
        lines.append(
            f"| {r['name']} | {c['retriever']}, k={c['top_k']}, {c['budget']} ký tự, {c['context_policy']}"
            f"{', ' + ', '.join(extra) if extra else ''} | {fmt(a['first_search_hit_at_1'])} / {fmt(a['first_search_evidence_hit_at_3'])} | "
            f"{fmt(a['context_evidence_rate'])} | {g['0']}/{g['1']}/{g['2']} ({g['mean']:.2f}) | {fmt(s['citation_validity'])} | "
            f"{fmt(s['abstention'])} | {fmt(s['false_abstention'])} | {a['mean_searches']} | {a['mean_llm_calls']} | "
            f"{a['median_latency_s']} | {a['mean_prompt_tokens']} |")
    lines += ["", f"**Tiêu chí chọn:**\n\n```\n{CRITERION}\n```", "",
              f"**Thứ hạng:** {' > '.join(r['name'] for r in ranked)}", "",
              f"**Chọn:** `{chosen['name']}`" + (f" (hoà 4 tiêu chí chất lượng với {', '.join(ties)}, thắng ở thời gian/token)" if ties else "")
              + f"; baseline test: `{BASELINE}`.", "",
              "Chi phí dịch vụ = $0 (embedding và LLM chạy local); thời gian đo trên laptop RTX 3050 Ti 4 GB, model ~77% GPU."]
    (config.RESULTS / "dev_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    selection = {"selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "split_used": "dev",
                 "criterion": CRITERION, "ranking": [r["name"] for r in ranked],
                 "selected": chosen["name"], "selected_config": CONFIGS[chosen["name"]],
                 "quality_ties": ties, "baseline": BASELINE, "baseline_config": CONFIGS[BASELINE],
                 "llm": config.LLM_MODEL, "embedding": f"{config.EMBEDDING_MODEL}@{config.EMBEDDING_REVISION[:7]}",
                 "note": "Chốt trên dev trước khi chạy bất kỳ lần test nào của Day 7. Test chạy đúng 2 cấu hình: baseline và selected."}
    (config.RESULTS / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
