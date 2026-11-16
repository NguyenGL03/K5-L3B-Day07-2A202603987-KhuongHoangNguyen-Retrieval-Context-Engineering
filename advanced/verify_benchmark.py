"""Check the advanced benchmark against the ORIGINAL PDF (not against the index), then freeze it.

* group minima from ADVANCED.md (test >=30: 6 table, 5 visual, 5 chart, 4 multihop, 4 text,
  6 unanswerable; dev >=15 answerable + 3 unanswerable)
* every non-visual evidence string must appear on its gold page of the raw PDF text
  (whitespace/'|' ignored; '...' separates fragments that must all appear)
* every absence_check regex must have 0 matches in the whole PDF text (all 328 pages)
* visual evidence was checked by reading the rendered page (recorded below)
* writes advanced/benchmark/verification.json and FROZEN.json (sha256 + time)

Run: python -m advanced.verify_benchmark [--freeze]
     python -m advanced.verify_benchmark --test2 [--freeze]   (round-2 test in benchmark/test2.json;
     also checks that no test2 question repeats the answer facts of a round-1 question)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "benchmark" / "questions.json"
PDF = ROOT.parent / "docs" / "publicbank_fr-2025_hr.pdf"
MIN_TEST = {"table": 6, "visual": 5, "chart": 5, "multihop": 4, "text": 4, "unanswerable": 6}
VISUAL_CHECK = ("Visual evidence (donut/bar/diagram/flow) read on pages rendered with PyMuPDF "
                "(p5, p8, p9, p11, p278, p283, p318, p322, p323, p325); p11 donut values also equal the p10 table; "
                "p8/p9 bar values equal the p6/p7 tables.")


def compact(s: str) -> str:
    return re.sub(r"[\s|]+", "", s).casefold().replace("’", "'").replace("“", '"').replace("”", '"')


def main() -> None:
    t2 = "--test2" in sys.argv
    bench = ROOT / "benchmark" / ("test2.json" if t2 else "questions.json")
    sfx = "_test2" if t2 else ""
    test_split = "test2" if t2 else "test"
    data = json.loads(bench.read_text(encoding="utf-8"))
    qs = data["questions"]
    doc = pymupdf.open(PDF)
    pages = [p.get_text() for p in doc]
    full = "\n".join(pages)
    report, problems = [], []
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    for q in qs:
        row = {"id": q["id"], "split": q["split"], "group": q["group"]}
        if q["group"] == "unanswerable":
            hits = {pat: len(re.findall(pat, full, re.I)) for pat in q["absence_check"]}
            row["absence_hits"] = hits
            if any(hits.values()):
                problems.append(f"{q['id']}: absence regex matched {hits}")
        else:
            checks = []
            for g in q["gold"]:
                ev = g["evidence"]
                if ev.startswith("visual"):
                    checks.append({"page": g["page"], "kind": "visual", "ok": True})
                    continue
                frags = [f for f in (x.strip() for x in ev.split("...")) if f]
                ok = all(compact(f) in compact(pages[g["page"] - 1]) for f in frags)
                checks.append({"page": g["page"], "kind": "text", "ok": ok})
                if not ok:
                    problems.append(f"{q['id']}: evidence not on page {g['page']}: {ev[:80]}")
            row["evidence_checks"] = checks
        report.append(row)
    counts = Counter((q["split"], q["group"]) for q in qs)
    for g, n in MIN_TEST.items():
        if counts[(test_split, g)] < n:
            problems.append(f"{test_split} {g}: {counts[(test_split, g)]} < {n}")
    n_test = sum(v for (s, _), v in counts.items() if s == test_split)
    if t2:
        if n_test < 30 or n_test != len(qs):
            problems.append("test2 size below minimum or mixed splits")
        old = json.loads(BENCH.read_text(encoding="utf-8"))["questions"]
        old_q = {compact(q["question"]) for q in old}
        old_must = {frozenset(compact(m) for m in q.get("must", [])) for q in old if q.get("must")}
        old_abs = {compact(q["question"]) for q in old if q["group"] == "unanswerable"}
        for q in qs:
            if compact(q["question"]) in old_q | old_abs:
                problems.append(f"{q['id']}: same question as round 1")
            if q.get("must") and frozenset(compact(m) for m in q["must"]) in old_must:
                problems.append(f"{q['id']}: same answer facts as a round-1 question")
    else:
        dev_ans = sum(v for (s, g), v in counts.items() if s == "dev" and g != "unanswerable")
        if n_test < 30 or dev_ans < 15 or counts[("dev", "unanswerable")] < 3:
            problems.append("split size below minimum")
    out = {"counts": {f"{s}/{g}": v for (s, g), v in sorted(counts.items())}, "test_total": n_test,
           "visual_check": VISUAL_CHECK, "problems": problems, "rows": report}
    (ROOT / "benchmark" / f"verification{sfx}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("counts", "test_total", "problems")}, ensure_ascii=False, indent=1))
    if problems:
        sys.exit(1)
    if "--freeze" in sys.argv:
        frozen = ROOT / "benchmark" / f"FROZEN{sfx}.json"
        if frozen.exists():
            sys.exit(f"already frozen: {frozen.read_text(encoding='utf-8')}")
        note = ("Round-2 test, written after the round-2 configuration was selected on `tune` "
                "(results/selection_r2.json) and frozen before any run on it." if t2 else
                "Frozen before any retrieval/agent experiment on this document.")
        frozen.write_text(json.dumps({"sha256": hashlib.sha256(bench.read_bytes()).hexdigest(),
                                      "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                      "note": note},
                                     indent=1) + "\n", encoding="utf-8")
        print("frozen")


if __name__ == "__main__":
    main()
