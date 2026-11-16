"""Print one agent run side by side with gold, for MANUAL grading only.

Shows question, expected answer, agent answer, each citation's chunk text and
whether gold evidence reached the final context. The grader writes labels to
results/agent/<split>/<config>/reviews.json (schema of starter/agent_metrics.py
plus a free-text `reason`). This script never assigns labels.

Run:  python -m src.review_view --split dev --config A0_dense [--ids q06 q12]
"""
from __future__ import annotations

import argparse
import json

from . import config, factory, metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--chars", type=int, default=700)
    args = p.parse_args()
    run = config.RESULTS / "agent" / args.split / args.config
    traces = json.loads((run / "traces.json").read_text(encoding="utf-8"))
    gold = {q["id"]: q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))}
    cases = {c["id"]: c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8"))}
    chunks = {c["id"]: c for c in factory.load_chunks()}
    for t in traces:
        if args.ids and t["query_id"] not in args.ids:
            continue
        q = gold.get(t["query_id"])
        print("=" * 100)
        print(f"{t['query_id']} [{t['type']}] status={t['status']} searches={t['queries']}")
        print("Q:", t["question"])
        if q:
            print("EXPECTED:", q["expected_answer"])
            print("CTX has gold:", metrics.covered(t["final_context_chunks"], q),
                  "| gold chunks:", [c for c in chunks if metrics.is_hit(chunks[c], q)])
        else:
            print("UNANSWERABLE:", cases[t["query_id"]]["absence_reason"])
        print("CTX ids:", t["final_context_ids"])
        print("ANSWER:", t["answer"])
        for cid in t["citations"]:
            print(f"  -- {cid}: {chunks[cid]['text'][:args.chars]!r}")


if __name__ == "__main__":
    main()
