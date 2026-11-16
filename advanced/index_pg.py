"""Embed advanced/data/chunks.json into the Day 6 pgvector table under strategy 'PBB'.

Same model/revision/dtype/normalisation as Day 6 (google/embeddinggemma-2, 768d), document
prompt for chunks (query prompt is used at search time by src.embedding.QueryEmbedder).
Idempotent: rows of strategy PBB are replaced in one transaction.

Run: python -m advanced.index_pg
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from src import config

import sys

SIZE = sys.argv[1] if len(sys.argv) > 1 else ""          # "" = default 1200-char index, "600" = small-chunk index
STRATEGY = "PBB" + SIZE
CHUNKS = Path(__file__).resolve().parent / "data" / (f"chunks_{SIZE}.json" if SIZE else "chunks.json")


def main() -> None:
    import psycopg
    import torch
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    model = SentenceTransformer(config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION, device="cpu",
                                model_kwargs={"dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    t0 = time.time()
    vecs = model.encode([c["text"] for c in chunks], prompt_name="document", batch_size=16,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    if vecs.shape[1] != config.EMBEDDING_DIM or any(math.isnan(float(x)) for x in vecs.ravel()[:1000]):
        raise RuntimeError(f"bad embeddings {vecs.shape}")
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.0f}s")
    with psycopg.connect(config.DATABASE_URL, autocommit=False) as conn:
        conn.execute(f"DELETE FROM {config.PG_TABLE} WHERE strategy = %s", (STRATEGY,))
        with conn.cursor() as cur:
            for c, v in zip(chunks, vecs):
                cur.execute(
                    f"INSERT INTO {config.PG_TABLE} (strategy, chunk_id, document_id, source, page, section, section_no, "
                    "text, metadata, content_hash, embedding_model, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    (STRATEGY, c["id"], c["source_id"], c["source"], c["page"], c["metadata"].get("section"), None,
                     c["text"], json.dumps(c["metadata"], ensure_ascii=False),
                     hashlib.sha256(c["text"].encode("utf-8")).hexdigest(), config.EMBEDDING_MODEL,
                     "[" + ",".join(repr(float(x)) for x in v) + "]"))
        conn.commit()
        n = conn.execute(f"SELECT count(*) FROM {config.PG_TABLE} WHERE strategy = %s", (STRATEGY,)).fetchone()[0]
    print(f"stored {n} rows (strategy={STRATEGY})")


if __name__ == "__main__":
    main()
