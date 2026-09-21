"""Thin streaming client for OpenAI-compatible /chat/completions.

No SDK: plain httpx + manual SSE line parsing so any compatible
endpoint (Ollama, LM Studio, OpenRouter, vLLM, ...) works.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dep in practice

    def load_dotenv(*args, **kwargs):
        return False


class LLMConfigError(RuntimeError): ...


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMConfig:
        if env is None:
            load_dotenv()  # loads .env if present; existing env wins (os_default)
            env = dict(os.environ)
        env = dict(env)
        base_url = env.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        api_key = env.get("LLM_API_KEY", "unused")
        model = env.get("LLM_MODEL")
        if not model:
            raise LLMConfigError("LLM_MODEL is not set. Copy .env.example to .env and set it.")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def with_updates(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> LLMConfig:
        """Return a copy with the given fields replaced (None = keep current).

        The resulting ``model`` must be non-empty: a config without a model is
        invalid and would fail only at the (remote) LLM endpoint.
        """
        next_model = self.model if model is None else model
        if not next_model:
            raise LLMConfigError("refusing to apply an LLM config with an empty model")
        return self.__class__(
            base_url=self.base_url if base_url is None else base_url.rstrip("/"),
            api_key=self.api_key if api_key is None else api_key,
            model=next_model,
            timeout=self.timeout if timeout is None else timeout,
        )


@dataclass
class LLMProbe:
    """Outcome of one minimal "does this config work" chat probe."""

    ok: bool
    error: str | None = None
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    """Aggregated result of one (possibly streaming) chat completion."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None


def parse_sse_line(line: str | bytes) -> dict[str, Any] | None:
    """Parse one SSE line; None for non-data lines, empty payloads, or [DONE].

    Accepts str or bytes: httpx's iter_lines() yields str, but callers may
    hand over raw bytes.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def fold_stream(events: list[dict[str, Any]]) -> Completion:
    """Fold chat-completion stream chunks into one Completion.

    Handles incremental ``delta`` content and multi-index tool_calls whose
    id/name/arguments arrive across several chunks.
    """
    completion = Completion()
    tool_buffers: dict[int, dict[str, Any]] = {}

    for event in events:
        if event.get("usage"):
            completion.usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}

            content = delta.get("content")
            if content:
                completion.content += content

            for tc in delta.get("tool_calls") or []:
                slot = tool_buffers.setdefault(
                    tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    for idx in sorted(tool_buffers):
        slot = tool_buffers[idx]
        try:
            args = json.loads(slot["arguments"]) if slot["arguments"] else {}
        except json.JSONDecodeError:
            args = {"_raw": slot["arguments"]}
        completion.tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))

    return completion


class OpenAICompatClient:
    """Minimal streaming chat-completions client."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        config: LLMConfig | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Run one chat completion, streaming tokens; returns the full Completion.

        ``config`` overrides this client's default for this call only (the core
        passes the run's current model from ``RunContext.current_model``);
        ``max_tokens`` caps the response (set by ``Agent.test_model`` probes).
        """
        import httpx

        cfg = config or self.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        events: list[dict[str, Any]] = []
        with (
            httpx.Client(timeout=cfg.timeout) as client,
            client.stream(
                "POST",
                f"{cfg.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload).encode(),
            ) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                event = parse_sse_line(line)
                if event is None:
                    continue
                events.append(event)
                if on_token is not None:
                    for choice in event.get("choices") or []:
                        token = (choice.get("delta") or {}).get("content")
                        if token:
                            on_token(token)

        return fold_stream(events)
