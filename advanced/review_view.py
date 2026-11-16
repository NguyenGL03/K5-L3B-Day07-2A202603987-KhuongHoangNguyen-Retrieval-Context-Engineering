"""Print agent answers next to the gold 'must' facts, accepted pages and cited pages (manual grading aid).

Run: python -m advanced.review_view --split dev --config A_hybrid [--ids q1 q2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ids", nargs="*")
    a = p.parse_args()
    qs = {q["id"]: q for f in ("questions.json", "test2.json") if (ROOT / "benchmark" / f).exists()
          for q in json.loads((ROOT / "benchmark" / f).read_text(encoding="utf-8"))["questions"]}
    trs = json.loads((ROOT / "results" / "agent" / a.split / a.config / "traces.json").read_text(encoding="utf-8"))
    for t in trs:
        if a.ids and t["id"] not in a.ids:
            continue
        q = qs[t["id"]]
        print("=" * 100)
        print(f"{t['id']} [{q['group']}] status={t['status']} cited_pages={t.get('cited_pages')} accept={q.get('accept_pages')} "
              f"required={q.get('required_pages')} ctx_pages={t.get('final_context_pages')}")
        print("Q:", q["question"])
        print("EXPECTED:", q.get("answer") or q.get("expected_behavior"), "| MUST:", q.get("must"))
        print("SEARCHES:", t.get("queries"), "| CALC:", [c["result"] for c in t.get("calculations", [])])
        print("ANSWER:", t["answer"])


if __name__ == "__main__":
    main()
