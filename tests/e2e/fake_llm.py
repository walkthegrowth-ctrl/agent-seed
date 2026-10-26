"""A tiny OpenAI-compatible LLM endpoint for the e2e battery.

Runs a ``ThreadingHTTPServer`` on a free loopback port and answers
``POST .../chat/completions`` by streaming back a *scripted* completion as
``text/event-stream``. The real backend subprocess is pointed at this over
its ``LLM_BASE_URL``, so the full HTTP/SSE + parsing path is exercised while
the suite stays network-free and deterministic.

The scripted responses are pulled from a FIFO queue: each test seeds exactly
the completions it expects, in order. A request for a model name marked
"failed" is answered with HTTP 500 (to drive the backend's error path).
"""

from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_TOKEN_RE = re.compile(r"\S+\s*")


@dataclass(frozen=True)
class TextScript:
    text: str


@dataclass(frozen=True)
class ToolScript:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailScript:
    """When popped, the fake answers the request with HTTP 500 (drives the
    backend's LLM-error path: an ``error`` SSE event, no traceback)."""

    message: str = "upstream LLM error (scripted)"


Script = TextScript | ToolScript | FailScript


def text(text: str) -> TextScript:
    return TextScript(text=text)


def tool(name: str, **arguments: Any) -> ToolScript:
    return ToolScript(name=name, arguments=arguments)


def fail(message: str = "upstream LLM error (scripted)") -> FailScript:
    """A script that, when popped, makes the fake answer with HTTP 500."""
    return FailScript(message)


def _split_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text) or [""]


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-e2e",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _usage_chunk() -> dict[str, Any]:
    return {
        "id": "chatcmpl-e2e",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _chunks(script: Script) -> list[dict[str, Any]]:
    if isinstance(script, TextScript):
        return [_chunk({"content": tok}) for tok in _split_tokens(script.text)] + [_usage_chunk()]
    tc = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": script.name, "arguments": json.dumps(script.arguments)},
    }
    return [_chunk({"tool_calls": [tc]}, finish="tool_calls"), _usage_chunk()]


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    out = b"".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode() for c in chunks)
    return out + b"data: [DONE]\n\n"


def _json_error(message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode()


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeLLM/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence request logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        self.server.record(self.path, body)  # wired in FakeLLMServer.__init__
        model = str(body.get("model") or "")
        if model in self.server.fail_models:  # shared set, wired in __init__
            self._send(500, _json_error(f"no such model: {model}"), "application/json")
            return
        script = self.server.pop_script()  # wired in FakeLLMServer.__init__
        if script is None:
            self._send(
                500, _json_error("no scripted response in the e2e fake LLM"), "application/json"
            )
            return
        if isinstance(script, FailScript):
            self._send(500, _json_error("upstream LLM error (scripted)"), "application/json")
            return
        self._send(200, _sse(_chunks(script)), "text/event-stream")


class FakeLLMServer:
    def __init__(self) -> None:
        self._q: queue.Queue[Script] = queue.Queue()
        self.fail_models: set[str] = set()
        self.recorded: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.fail_models = self.fail_models  # type: ignore[attr-defined]
        self.httpd.record = self.record  # type: ignore[attr-defined]
        self.httpd.pop_script = self.pop_script  # type: ignore[attr-defined]

    # -- queue / scripting ----------------------------------------------------
    def script(self, *scripts: Script) -> FakeLLMServer:
        for s in scripts:
            self._q.put(s)
        return self

    def reset(self, *scripts: Script) -> None:
        with self._lock:
            self._q = queue.Queue()
        for s in scripts:
            self._q.put(s)
        self.fail_models.clear()
        self.recorded.clear()

    def fail(self, model: str) -> FakeLLMServer:
        self.fail_models.add(model)
        return self

    # -- introspection --------------------------------------------------------
    def record(self, path: str, body: dict[str, Any]) -> None:  # type: ignore[name-defined]
        with self._lock:
            self.recorded.append({"path": path, "body": body})

    def pop_script(self) -> Script | None:  # type: ignore[name-defined]
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    @property
    def last_request(self) -> dict[str, Any] | None:  # type: ignore[name-defined]
        with self._lock:
            return self.recorded[-1] if self.recorded else None

    # -- lifecycle ------------------------------------------------------------
    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> FakeLLMServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
