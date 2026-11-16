"""Seeded draw of the evidence targets for the round-2 test (test2), run AFTER results/selection_r2.json was written.

The question writer does not choose where the questions come from: targets are drawn at random (fixed seed) from
pools that exclude every page/visual already used by a round-1 dev/test question. Questions are then written by
hand for the drawn targets and checked against the original PDF (advanced.verify_benchmark --test2).

Pools
  table  : table chunks (>= 3 numeric rows) of pages not used in round 1, one per note/section
  chart  : the 22 extracted charts minus those asked about in round 1
  visual : the 7 diagrams (all were used in round 1, so a not-yet-asked element is drawn for each)
  multihop: anchor rows (not asked in round 1) from the summary pages p4-p12; the 2nd hop is a related figure
           on a different page, chosen by the writer
  text   : prose pages (few numbers) not used in round 1

Run: python -m advanced.benchmark.sample_test2   -> advanced/benchmark/test2_targets.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261116
NUM = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+")


def main() -> None:
    rnd = random.Random(SEED)
    qs = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))["questions"]
    used_pages = {p for q in qs for p in (q.get("accept_pages") or [])} | {g["page"] for q in qs for g in q.get("gold", [])}
    chunks = json.loads((ROOT / "data" / "chunks_600.json").read_text(encoding="utf-8"))
    used_vis = {"p005-v1", "p005-v3", "p005-v4", "p008-v6", "p009-v1", "p009-v5", "p009-v6", "p011-v1", "p011-v4", "p011-v5"}

    # tables: one chunk per note/section, numeric rows >= 3
    by_note: dict[str, list[dict]] = {}
    for c in chunks:
        if not c["metadata"]["kind"].startswith("table") or c["page"] in used_pages:
            continue
        rows = [l for l in c["display"].split("\n") if len(NUM.findall(l)) >= 2]
        if len(rows) >= 3:
            by_note.setdefault(c["metadata"].get("note") or c["metadata"]["section"], []).append(c)
    notes = sorted(by_note)
    table = []
    for n in rnd.sample(notes, 6):
        c = rnd.choice(sorted(by_note[n], key=lambda x: x["id"]))
        rows = [l for l in c["display"].split("\n") if len(NUM.findall(l)) >= 2]
        table.append({"chunk": c["id"], "page": c["page"], "note": n, "row": rnd.choice(rows)[:400]})

    charts = sorted(c["id"] for c in chunks if c["metadata"]["kind"].startswith("chart")
                    and c["id"].split("fs-")[1] not in used_vis)
    chart = rnd.sample(charts, 5)

    diagrams = sorted(c["id"] for c in chunks if c["metadata"]["kind"] == "diagram")
    visual = rnd.sample(diagrams, 5)

    anchors = []
    for c in chunks:
        if 4 <= c["page"] <= 12 and c["metadata"]["kind"].startswith("table"):
            anchors += [(c["id"], l[:300]) for l in c["display"].split("\n") if len(NUM.findall(l)) >= 2]
    multihop = [{"chunk": a, "row": r} for a, r in rnd.sample(sorted(anchors), 4)]

    prose_pages = sorted({c["page"] for c in chunks if c["metadata"]["kind"] == "table_text" and c["page"] not in used_pages
                          and len(NUM.findall(c["text"])) <= 4 and len(c["text"]) > 450})
    text = rnd.sample(prose_pages, 4)

    # element of each drawn diagram: drawn (separate RNG, SEED + 1) from the elements no round-1 question asked about.
    # Listed from the rendered pages; positions are as seen on the page, not as extracted.
    elements = {
        "pbb-ar2025-fs-p278-v2": ["first step and what it considers", "last step and in light of what",
                                  "cascade step: to which levels via the budgeting process"],
        "pbb-ar2025-fs-p322-v1": ["Identification activities", "Assessment & Measurement activities", "Monitoring activities",
                                  "oversight box on the left", "oversight box on the right", "oversight box at the bottom",
                                  "stage at the bottom-left", "stage at the top-left"],
        "pbb-ar2025-fs-p318-v1": ["Identification activities", "Assessment & Measurement activities", "Monitoring activities",
                                  "activity prohibited without Board approval", "oversight box on the left",
                                  "oversight box on the right", "oversight box at the bottom", "stage at the top-left"],
        "pbb-ar2025-fs-p323-v1": ["Identification activities", "Control & Mitigation activities", "Monitoring activities",
                                  "oversight box on the left", "oversight box on the right", "oversight box at the bottom",
                                  "stage at the bottom-left", "stage at the top-left"],
        "pbb-ar2025-fs-p325-v1": ["Identification activities", "Assessment & Measurement activities",
                                  "Control & Mitigation activities", "Monitoring activities", "oversight box on the left",
                                  "oversight box on the right", "oversight box at the bottom", "stage at the bottom-left"],
    }
    rnd2 = random.Random(SEED + 1)
    visual = [{"diagram": d, "element": rnd2.choice(elements[d])} if d in elements else {"diagram": d} for d in visual]

    out = {"seed": SEED, "excluded_pages_round1": sorted(used_pages), "excluded_visuals_round1": sorted(used_vis),
           "table": table, "chart": chart, "visual": visual, "multihop_anchors": multihop, "text_pages": text}
    (ROOT / "benchmark" / "test2_targets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                            encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
