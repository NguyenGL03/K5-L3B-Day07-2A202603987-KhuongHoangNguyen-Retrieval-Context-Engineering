"""Query encoder compatible with the Day 6 index (google/embeddinggemma-2, 768d).

Same model, revision, dtype, normalization and *query* prompt as Day 6
(`task: search result | query: `). No random/mock fallback: load errors raise.
"""
from __future__ import annotations

import math

from . import config


class QueryEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 revision: str = config.EMBEDDING_REVISION, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name, self.revision = model_name, revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            model_name, revision=revision, device=self.device,
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None})
        prompts = getattr(self.model, "prompts", {}) or {}
        if "query" not in prompts:
            raise RuntimeError(f"{model_name} has no 'query' prompt")
        self.query_prompt = prompts["query"]
        self._cache: dict[str, list[float]] = {}
        self.dimensions = len(self.embed_query("kiểm tra"))
        if self.dimensions != config.EMBEDDING_DIM:
            raise RuntimeError(f"query dim {self.dimensions} != index dim {config.EMBEDDING_DIM}")

    def embed_query(self, query: str) -> list[float]:
        if query not in self._cache:
            vec = self.model.encode([query], prompt_name="query", normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)[0]
            row = [float(x) for x in vec]
            if any(math.isnan(x) or math.isinf(x) for x in row):
                raise RuntimeError("query embedding contains NaN/Inf")
            self._cache[query] = row
        return self._cache[query]

    def describe(self) -> dict:
        return {"model": self.model_name, "revision": self.revision, "dimensions": self.dimensions,
                "device": self.device, "dtype": "float32", "normalized": True,
                "query_prompt": self.query_prompt, "cost_usd": 0.0}
