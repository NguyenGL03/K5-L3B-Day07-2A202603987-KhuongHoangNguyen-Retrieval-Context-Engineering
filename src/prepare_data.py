"""Export the Day 6 index and benchmark into the Day 7 JSON contract.

- Chunks are read from the live Day 6 vector store (table rag_chunks, strategy B),
  so chunk IDs/text are exactly what dense retrieval returns.
- Answerable questions come from the frozen Day 6 benchmark (same dev/test split).
- Unanswerable questions are added here (near-topic, absence checked below).
Gold is written to data files only; it never reaches the agent or the retrievers.

Run: python -m src.prepare_data
"""
from __future__ import annotations

import json
import re
from collections import Counter

import psycopg

from . import config

# Near-topic questions whose answer is NOT in the corpus. `absence_check` lists the
# terms searched in every chunk (text + OCR) of index A and B; see verify_absence().
UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "Công ty Nước khoáng Quảng Ninh có tổng cộng bao nhiêu lao động (nhân viên) trong năm 2022?",
     "absence_reason": "Tài liệu chỉ nói định tính về nguồn lao động/nhân sự (tr.8, 13, 25); không có số lượng lao động.",
     "absence_check": [r"\d[\d.,]*\s*(người|lao động|nhân viên|công nhân)", r"số lượng (lao động|nhân viên)"]},
    {"id": "u02", "split": "dev",
     "question": "Giá bán lẻ một chai nước khoáng Quảng Ninh loại 500ml là bao nhiêu đồng?",
     "absence_reason": "Tài liệu có giá trị tồn kho và ngưỡng giao hàng miễn phí 500.000 VNĐ, không có giá bán lẻ sản phẩm.",
     "absence_check": [r"giá bán", r"đồng\s*/\s*chai", r"\d[\d.,]*\s*(đ|đồng|vnđ|vnd)\s*/\s*(chai|bình)"]},
    {"id": "u03", "split": "test",
     "question": "Ai đang giữ chức Tổng giám đốc của Công ty Cổ phần Nước khoáng Quảng Ninh?",
     "absence_reason": "Tài liệu nói chung về 'ban lãnh đạo', không nêu tên hay chức danh giám đốc.",
     "absence_check": [r"giám đốc", r"chủ tịch"]},
    {"id": "u04", "split": "test",
     "question": "Vốn điều lệ của Công ty Cổ phần Nước khoáng Quảng Ninh là bao nhiêu?",
     "absence_reason": "Tài liệu có doanh thu, tồn kho, chỉ số P/E, P/B nhưng không có vốn điều lệ.",
     "absence_check": [r"vốn điều lệ", r"điều lệ"]},
]


# Day 6 gold quotes that cross a chunk boundary of index B cannot satisfy the
# "evidence inside one chunk" contract. They become multi-evidence questions:
# gold_mode "all" = every listed evidence is required (see src/metrics.py).
# q20: the 7 principles are listed over two consecutive chunks of page 22
# (customer focus/leadership/engagement, then process/system/improvement/evidence).
MULTI_EVIDENCE = {
    "q20": [
        {"page": 22, "evidence": "– Định hướng bởi khách hàng"},
        {"page": 22, "evidence": "– Tiếp cận theo quá trình"},
    ],
    # q21: the 4 advantages end on page 14 chunk 001; the 4th one is chunk 002 alone.
    "q21": [
        {"page": 14, "evidence": "– Tiếp cận nhanh các công nghệ sản xuất tiên tiến."},
        {"page": 14, "evidence": "– Tổng chi phí thấp hơn do tính kinh tế về quy mô"},
    ],
}


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def export_chunks() -> list[dict]:
    with psycopg.connect(config.DATABASE_URL, connect_timeout=5) as conn:
        rows = conn.execute(
            f"SELECT chunk_id, document_id, source, page, section, section_no, text, metadata, "
            f"embedding_model FROM {config.PG_TABLE} WHERE strategy = %s ORDER BY page, chunk_id",
            (config.PG_STRATEGY,)).fetchall()
    if not rows:
        raise RuntimeError("Day 6 vector store has no rows for strategy B; run Day 6 ingest first")
    models = {r[8] for r in rows}
    if models != {config.EMBEDDING_MODEL}:
        raise RuntimeError(f"index embedded with {models}, expected {config.EMBEDDING_MODEL}")
    chunks = []
    for cid, doc, source, page, section, section_no, text, meta, _ in rows:
        meta = dict(meta or {})
        meta.update({"section": section, "section_no": section_no, "index": f"day6-{config.PG_STRATEGY}"})
        chunks.append({"id": cid, "text": text, "source_id": doc, "source": source,
                       "page": page, "metadata": meta})
    return chunks


def convert_questions(day6: dict) -> list[dict]:
    out = []
    for q in day6["questions"]:
        gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
        gold += [{"source_id": a.get("gold_source_id", q["gold_source_id"]), "page": a["gold_page"],
                  "evidence": a["evidence_quote"]} for a in q.get("alternatives", [])]
        item = {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": gold, "gold_mode": "any"}
        if q["id"] in MULTI_EVIDENCE:
            item["day6_evidence_quote"] = q["evidence_quote"]
            item["gold"] = [{"source_id": q["gold_source_id"], **g} for g in MULTI_EVIDENCE[q["id"]]]
            item["gold_mode"] = "all"
        if q.get("filter"):
            item["filter"] = q["filter"]
            item["filter_context"] = q.get("filter_context", "")
        out.append(item)
    return out


def verify_gold(chunks: list[dict], questions: list[dict]) -> dict:
    """Each gold evidence must sit inside one chunk of the right source/page."""
    report = {}
    for q in questions:
        per_gold = []
        for g in q["gold"]:
            same = [c for c in chunks if c["source_id"] == g["source_id"] and c["page"] == g["page"]]
            per_gold.append((any(g["evidence"] in c["text"] for c in same),
                             any(collapse(g["evidence"]) in collapse(c["text"]) for c in same)))
        agg = all if q["gold_mode"] == "all" else any
        exact, norm = agg(e for e, _ in per_gold), agg(n for _, n in per_gold)
        if not norm:
            raise ValueError(f"{q['id']}: gold evidence is not contained in any single chunk")
        report[q["id"]] = {"gold_mode": q["gold_mode"], "exact_substring": exact,
                           "whitespace_insensitive": norm}
    return report


def verify_absence(all_texts: list[str]) -> dict:
    report = {}
    for u in UNANSWERABLE:
        found = {p: [m.group(0) for t in all_texts for m in re.finditer(p, t, re.I)]
                 for p in u["absence_check"]}
        report[u["id"]] = found
        if any(found.values()):
            raise ValueError(f"{u['id']}: absence check matched {found}; question may be answerable")
    return report


def day6_all_texts() -> list[str]:
    """Text of both Day 6 indexes (A fixed-size + B paragraph), including OCR blocks."""
    texts = []
    for name in ("chunks_A.jsonl", "chunks_B.jsonl"):
        path = config.DAY6_ROOT / "starter" / "results" / name
        texts += [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts


def main() -> None:
    chunks = export_chunks()
    day6 = json.loads((config.DAY6_ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = convert_questions(day6)
    gold_report = verify_gold(chunks, questions)
    absence_report = verify_absence(day6_all_texts())

    cases = [{"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
              "answerable": True, "expected_behavior": "search_and_answer",
              **({"filter": q["filter"]} if q.get("filter") else {})} for q in questions]
    cases += [{"id": u["id"], "split": u["split"], "type": "thieu_bang_chung", "question": u["question"],
               "answerable": False, "expected_behavior": "search_and_abstain",
               "absence_reason": u["absence_reason"], "absence_check": u["absence_check"]}
              for u in UNANSWERABLE]

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, obj in ((config.CHUNKS_PATH, chunks), (config.QUESTIONS_PATH, questions),
                      (config.CASES_PATH, cases),
                      (config.DATA / "benchmark" / "verification.json",
                       {"gold": gold_report, "absence_matches": absence_report,
                        "day6_benchmark_frozen_at": day6.get("frozen_at"),
                        "day6_split_rule": day6.get("split_rule")})):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split = Counter((c["split"], c["answerable"]) for c in cases)
    print(json.dumps({"chunks": len(chunks), "pages": len({c["page"] for c in chunks}),
                      "cases": {f"{s}/{'answerable' if a else 'unanswerable'}": n for (s, a), n in sorted(split.items())},
                      "gold_exact_substring": sum(v["exact_substring"] for v in gold_report.values()),
                      "gold_whitespace_insensitive": sum(v["whitespace_insensitive"] for v in gold_report.values())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
