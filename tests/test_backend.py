"""End-to-end: real FastAPI app + SSE stream, with a mocked LLM client."""

import json

import pytest
from fastapi.testclient import TestClient

from agent_seed.backends.http import server
from agent_seed.tools import ToolRegistry, calculator
from agent_seed.tools import registry as default_registry


def _events(raw_text):
    """Parse an SSE body into an ordered list of (event, data_dict)."""
    events = []
    event = "message"
    data = []
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
        elif line == "":
            if data:
                events.append((event, json.loads("\n".join(data))))
            event, data = "message", []
    if data:
        events.append((event, json.loads("\n".join(data))))
    return events


def _make_app(script):
    """Build the real app with a mocked LLM client that pops scripted completions."""
    from agent_seed.llm import Completion  # noqa: F401 - ensure import path works

    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None):
                completion = script.pop(0)
                if on_token is not None and completion.content:
                    on_token(completion.content)
                return completion

        return _Fake()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry)
    return app, registry, monkeypatch


@pytest.fixture
def tool_client(monkeypatch):
    from tests.conftest import text_completion, tool_completion

    script = [
        tool_completion("calculator", {"expression": "6 * 7"}),
        text_completion("42.0"),
    ]
    app, registry, mp = _make_app(script)
    with TestClient(app) as test_client:
        yield test_client, registry
    mp.undo()


def test_health(monkeypatch):
    app, _, mp = _make_app([])
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    mp.undo()
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tools_endpoint(tool_client):
    test_client, _ = tool_client
    response = test_client.get("/tools")
    assert response.status_code == 200
    body = response.json()
    assert body["names"] == ["calculator"]
    assert body["tools"][0]["function"]["name"] == "calculator"


def test_chat_streams_tool_then_token_then_done(tool_client):
    test_client, _ = tool_client
    with test_client.stream("POST", "/chat", json={"message": "what is 6 * 7?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = _events(body)
    kinds = [k for k, _ in events]
    assert kinds == ["tool", "token", "done"]

    (_, tool_data), (_, token_data), (_, done_data) = events
    assert tool_data["name"] == "calculator"
    assert tool_data["arguments"] == {"expression": "6 * 7"}
    assert tool_data["result"] == "42.0"
    assert token_data == {"text": "42.0"}
    assert done_data["session_id"]


def test_chat_missing_model_returns_500(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    # ensure no .env in cwd can re-provide LLM_MODEL at request time
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent_seed.llm.load_dotenv", lambda *a, **k: False)

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry)
    with TestClient(app) as test_client:
        response = test_client.post("/chat", json={"message": "hi"})
    assert response.status_code == 500
    assert "LLM_MODEL" in response.json()["error"]


# -- agents ------------------------------------------------------------------


def test_agents_endpoint_lists_subdirs(monkeypatch, tmp_path):
    (tmp_path / "hook_modules" / "alpha").mkdir(parents=True)
    (tmp_path / "hook_modules" / "gamma").mkdir(parents=True)
    monkeypatch.setattr(
        "agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path / "hook_modules")
    )
    from agent_seed.discovery import reset_cache

    reset_cache()

    registry = ToolRegistry()
    app = server.create_app(registry)
    with TestClient(app) as test_client:
        response = test_client.get("/agents")
    assert response.status_code == 200
    assert response.json() == {"agents": ["alpha", "gamma"]}


def test_chat_agent_name_selects_subdir_hooks(monkeypatch, tmp_path):
    # Use an after_tool_call hook: it rewrites ctx.tool_result, which the
    # backend emits in the SSE `tool` event — so the marker appears on the
    # wire and we can assert which folder's hook ran.
    hooks_root = tmp_path / "hook_modules"
    (hooks_root / "alpha").mkdir(parents=True)
    (hooks_root / "alpha" / "a.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 100\n"
        "def make() -> Hooks:\n    return Hooks(after_tool_call=after_tool_call)\n"
    )
    (hooks_root / "root.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 1000\n"
        "def make() -> Hooks:\n    return Hooks(after_tool_call=after_tool_call)\n"
    )

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))

    from agent_seed.discovery import reset_cache

    reset_cache()

    from tests.conftest import text_completion, tool_completion

    script = [
        tool_completion("calculator", {"expression": "1+1"}),
        text_completion("done"),
        tool_completion("calculator", {"expression": "1+1"}),
        text_completion("done"),
    ]

    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None):
                completion = script.pop(0)
                if on_token is not None and completion.content:
                    on_token(completion.content)
                return completion

        return _Fake()

    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)
    app = server.create_app(default_registry)

    with TestClient(app) as test_client:
        # agent_name = "alpha": alpha's after_tool_call (x100) runs
        with test_client.stream(
            "POST", "/chat", json={"message": "hi", "agent_name": "alpha"}
        ) as response:
            body = "".join(response.iter_text())
        assert "200.0" in body and "2000.0" not in body

        # no agent_name: root's after_tool_call (x1000) runs
        with test_client.stream("POST", "/chat", json={"message": "hi"}) as response:
            body = "".join(response.iter_text())
        assert "2000.0" in body and "200.0" not in body
