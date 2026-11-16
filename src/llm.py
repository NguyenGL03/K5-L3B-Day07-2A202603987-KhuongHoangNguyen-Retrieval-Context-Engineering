"""Minimal Ollama /api/chat client with tool calling (stdlib only, no fallback)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from . import config


class LLMError(RuntimeError):
    pass


class OllamaChat:
    def __init__(self, model: str = config.LLM_MODEL, url: str = config.OLLAMA_URL,
                 num_ctx: int = config.LLM_NUM_CTX, seed: int = config.LLM_SEED,
                 temperature: float = 0.0, timeout: int = 300) -> None:
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": self.options, "keep_alive": "30m"}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc
        if "message" not in data:
            raise LLMError(f"Ollama returned no message: {str(data)[:300]}")
        msg = data["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
                           for tc in msg.get("tool_calls") or []],
            "usage": {"prompt_tokens": data.get("prompt_eval_count", 0),
                      "completion_tokens": data.get("eval_count", 0),
                      "wall_s": round(time.perf_counter() - start, 3)},
        }

    def describe(self) -> dict[str, Any]:
        info = {"provider": "Ollama (local, Docker)", "model": self.model, "options": self.options}
        try:
            req = urllib.request.Request(f"{self.url}/api/show", data=json.dumps({"model": self.model}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
            info.update({"details": d.get("details"), "capabilities": d.get("capabilities"),
                         "digest_modified_at": d.get("modified_at")})
        except Exception as exc:  # describe is informational only
            info["describe_error"] = str(exc)
        return info
