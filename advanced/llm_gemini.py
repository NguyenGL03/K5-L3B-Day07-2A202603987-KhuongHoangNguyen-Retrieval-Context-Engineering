"""Gemini API client with the same contract as src.llm.OllamaChat (chat(messages, tools) -> content, tool_calls,
usage), so advanced.agent runs unchanged. Stdlib only (urllib), native generateContent REST with function calling.

* The API key is read from GEMINI_API_KEY (.env) and sent only in the x-goog-api-key header; it is never logged.
* The agent rebuilds the conversation itself (synthetic search/calculate turns), so replayed functionCall parts carry
  the documented placeholder thought signature for injected calls.
* 429 / 5xx are retried with backoff; anything else raises LLMError (recorded as a system error, never replaced).
* usage: prompt, completion and thinking tokens; cost is computed from per-1M-token prices given at construction.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from src.llm import LLMError

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SKIP_SIG = "skip_thought_signature_validator"


def _key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            from dotenv import dotenv_values
            from pathlib import Path
            key = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("GEMINI_API_KEY")
        except ImportError:
            key = None
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    return key


def to_contents(messages: list[dict[str, Any]], flatten: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """flatten=True (no tools offered in this turn): earlier tool calls/results become plain text, otherwise the
    model keeps emitting function calls it has seen in the history."""
    system, contents, last_call = [], [], "search_documents"
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append(m["content"])
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif role == "assistant":
            parts = [{"text": m["content"]}] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                last_call = fn["name"]
                if flatten:
                    parts.append({"text": f"(I called {fn['name']} with {json.dumps(fn.get('arguments') or {})})"})
                else:
                    parts.append({"functionCall": {"name": fn["name"], "args": fn.get("arguments") or {}},
                                  "thoughtSignature": SKIP_SIG})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            if flatten:
                contents.append({"role": "user", "parts": [{"text": f"Result of {last_call}:\n{m['content']}"}]})
            else:
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": last_call, "response": {"content": m["content"]}}}]})
    return "\n\n".join(system), contents


class GeminiChat:
    def __init__(self, model: str, price_in: float | None = None, price_out: float | None = None,
                 temperature: float | None = None, seed: int = 7, timeout: int = 180, max_retries: int = 6) -> None:
        self.model, self.timeout, self.max_retries = model, timeout, max_retries
        self.temperature, self.seed = temperature, seed
        self.price_in, self.price_out = price_in, price_out      # USD per 1M tokens (output includes thinking)
        self._key = _key()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict[str, Any]:
        system, contents = to_contents(messages, flatten=not tools)
        gen: dict[str, Any] = {"seed": self.seed}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        body: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [t["function"] for t in tools]}]
        data = json.dumps(body).encode("utf-8")
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(API.format(model=self.model), data=data,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self._key})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise LLMError(f"Gemini HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise LLMError(f"Gemini call failed: {exc}") from exc
        cands = out.get("candidates") or []
        if not cands:
            raise LLMError(f"Gemini returned no candidate: {str(out)[:300]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        calls = [{"name": p["functionCall"]["name"], "arguments": p["functionCall"].get("args") or {}}
                 for p in parts if "functionCall" in p]
        u = out.get("usageMetadata") or {}
        pin, pout, pth = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), u.get("thoughtsTokenCount", 0)
        usage = {"prompt_tokens": pin, "completion_tokens": pout + pth, "thinking_tokens": pth,
                 "wall_s": round(time.perf_counter() - start, 3), "finish_reason": cands[0].get("finishReason")}
        if self.price_in is not None and self.price_out is not None:
            usage["cost_usd"] = round(pin / 1e6 * self.price_in + (pout + pth) / 1e6 * self.price_out, 6)
        return {"content": text, "tool_calls": calls, "usage": usage}

    def describe(self) -> dict[str, Any]:
        return {"provider": "Google Gemini API (generateContent)", "model": self.model,
                "options": {"temperature": self.temperature if self.temperature is not None else "model default",
                            "seed": self.seed},
                "price_usd_per_1M": {"input": self.price_in, "output_incl_thinking": self.price_out}}
