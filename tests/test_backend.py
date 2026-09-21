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


def _make_app(script, hooks=None):
    """Build the real app with a mocked LLM client that pops scripted completions.

    Under strict tool-whitelist policy, ``hooks`` (if provided) must include an
    ``enter_agent`` that sets ``ctx.allowed_tools``; otherwise the agent offers
    no tools. ``app.state.fakes`` collects every created client (each carries
    ``config`` and ``last_chat_config``) so tests can assert which model the
    core actually called with.
    """
    from agent_seed.llm import Completion  # noqa: F401 - ensure import path works

    class _Fake:
        def __init__(self, config):
            self.config = config

        def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
            self.last_chat_config = config
            completion = script.pop(0)
            if on_token is not None and completion.content:
                on_token(completion.content)
            return completion

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry, hooks=hooks)
    app.state.fakes = []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def fake_client_factory(cfg):
        fake = _Fake(cfg)
        app.state.fakes.append(fake)
        return fake

    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)
    return app, registry, monkeypatch


@pytest.fixture
def tool_client(monkeypatch):
    from agent_seed.hooks import Hooks, RunContext
    from tests.conftest import text_completion, tool_completion

    # Strict tool-whitelist policy: declare `calculator` for the agent.
    def _enter(ctx: RunContext) -> None:
        ctx.allowed_tools = ["calculator"]

    hooks = Hooks(enter_agent=_enter)
    script = [
        tool_completion("calculator", {"expression": "6 * 7"}),
        text_completion("42.0"),
    ]
    app, registry, mp = _make_app(script, hooks=hooks)
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


def test_tools_endpoint_agent_name_reflects_whitelist(monkeypatch, tmp_path):
    from agent_seed.discovery import reset_cache

    hooks_root = tmp_path / "hook_modules"
    (hooks_root / "alpha").mkdir(parents=True)
    (hooks_root / "alpha" / "a.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def make() -> Hooks:\n    return Hooks(enter_agent=enter_agent)\n"
    )
    (hooks_root / "bare").mkdir()
    (hooks_root / "bare" / "b.py").write_text(
        "from agent_seed.hooks import Hooks\ndef make() -> Hooks:\n    return Hooks()\n"
    )
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))
    reset_cache()

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry)
    with TestClient(app) as test_client:
        # no agent_name: the whole catalog
        full = test_client.get("/tools").json()
        assert full["names"] == ["calculator"]
        # alpha whitelists calculator
        alpha = test_client.get("/tools", params={"agent_name": "alpha"}).json()
        assert alpha["names"] == ["calculator"]
        # bare declares none -> strict policy: no tools
        bare = test_client.get("/tools", params={"agent_name": "bare"}).json()
        assert bare == {"tools": [], "names": []}
        # unknown agent -> falls back to the root hooks (none) -> no tools
        unknown = test_client.get("/tools", params={"agent_name": "nope"}).json()
        assert unknown == {"tools": [], "names": []}


def test_tools_endpoint_uses_apps_registry_not_global(monkeypatch):
    # Regression (1.1): the /tools catalog must reflect the registry the app
    # was built with, not agent_seed.tools.registry. We register a tool under a
    # UNIQUE name that is absent from the global default registry, so the old
    # bug (reporting the global catalog) would return ['calculator'] and fail.
    from agent_seed.hooks import Hooks

    reg = ToolRegistry()

    @reg.register
    def weather(city: str) -> str:
        """Get current weather for a city."""
        return "sunny"

    hooks = Hooks(enter_agent=lambda ctx: setattr(ctx, "allowed_tools", ["weather"]))
    app = server.create_app(reg, hooks=hooks)

    with TestClient(app) as test_client:
        full = test_client.get("/tools").json()
        assert full["names"] == ["weather"]
        assert full["tools"][0]["function"]["name"] == "weather"
        # and an agent that whitelists the same name resolves against this app
        per_agent = test_client.get("/tools", params={"agent_name": "nope"}).json()
        # "nope" falls back to the (hermetic, empty) root hooks -> no tools
        assert per_agent == {"tools": [], "names": []}


def test_chat_streams_tool_then_token_then_done(tool_client):
    test_client, _ = tool_client
    with test_client.stream("POST", "/chat", json={"message": "what is 6 * 7?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        header_sid = response.headers["x-session-id"]
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
    # the X-Session-Id header must carry the same session id as the done event
    assert header_sid == done_data["session_id"]


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
    # alpha whitelists calculator AND rewrites the result (x100)
    (hooks_root / "alpha" / "a.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 100\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent, after_tool_call=after_tool_call)\n"
    )
    # root whitelists calculator AND rewrites the result (x1000)
    (hooks_root / "root.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 1000\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent, after_tool_call=after_tool_call)\n"
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
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
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


def test_chat_mid_stream_error_emits_error_event(monkeypatch):
    # A mid-stream failure must surface as `event: error` and terminate the
    # turn — no `done`. (Covers server.py's error branch + protocol.md's
    # `error` event.)
    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                raise RuntimeError("boom")

        return _Fake()

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)

    registry = ToolRegistry()
    app = server.create_app(registry)
    with TestClient(app) as test_client:
        response = test_client.post("/chat", json={"message": "hi"})
        assert response.status_code == 200
        body = response.text

    events = _events(body)
    events_map = {k: d for k, d in events}
    assert "error" in events_map
    assert "done" not in events_map
    assert "boom" in events_map["error"]["message"]


def test_chat_agent_switch_mid_session_refires_establishment(tmp_path, monkeypatch):
    # Posting the *same* session_id under two different agent_name values must
    # re-resolve hooks on the live agent (server.py:72-75 branch) and apply the
    # new agent's whitelist + hooks. Previously uncovered.
    from agent_seed.discovery import reset_cache
    from tests.conftest import text_completion, tool_completion

    hooks_root = tmp_path / "hook_modules"
    (hooks_root / "alpha").mkdir(parents=True)
    (hooks_root / "alpha" / "a.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 100\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent, after_tool_call=after_tool_call)\n"
    )
    (hooks_root / "beta").mkdir(parents=True)
    (hooks_root / "beta" / "b.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 1000\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent, after_tool_call=after_tool_call)\n"
    )

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))
    reset_cache()

    registry = ToolRegistry()
    registry.register(calculator)

    script = [
        tool_completion("calculator", {"expression": "1 + 1"}),
        text_completion("first"),
        tool_completion("calculator", {"expression": "1 + 1"}),
        text_completion("second"),
    ]

    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                completion = script.pop(0)
                if on_token is not None and completion.content:
                    on_token(completion.content)
                return completion

        return _Fake()

    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)
    app = server.create_app(registry)

    with TestClient(app) as test_client:
        # First turn: mint a session, use agent alpha (x100).
        with test_client.stream(
            "POST",
            "/chat",
            json={"message": "hi", "agent_name": "alpha"},
        ) as response:
            sid = response.headers["x-session-id"]
            body1 = "".join(response.iter_text())
        assert "200.0" in body1 and "2000.0" not in body1

        # Second turn: SAME session_id, agent beta (x1000). The live agent
        # must re-resolve to beta's hooks — the mid-session switch branch.
        with test_client.stream(
            "POST",
            "/chat",
            json={"message": "hi", "session_id": sid, "agent_name": "beta"},
        ) as response:
            assert response.headers["x-session-id"] == sid
            body2 = "".join(response.iter_text())
        assert "2000.0" in body2 and "200.0" not in body2


# -- runtime model selection (/config) ----------------------------------------


def test_config_get_returns_base_and_current(monkeypatch):
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi")])
    with TestClient(app) as test_client:
        with test_client.stream("POST", "/chat", json={"message": "hi"}) as response:
            sid = response.headers["x-session-id"]

        body = test_client.post(
            "/config", json={"session_id": sid, "model": "picked-model"}
        ).json()
        assert body["ok"] and body["llm"]["model"] == "picked-model"

        got = test_client.get("/config", params={"session_id": sid})
        assert got.status_code == 200
        data = got.json()
        assert data["llm"]["model"] == "picked-model"
        # base is whatever the session started from (env test-model)
        assert data["base"]["model"] == "test-model"
    mp.undo()


def test_config_set_changes_the_model_the_session_calls(monkeypatch):
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi"), text_completion("hi")])
    with TestClient(app) as test_client:
        with test_client.stream("POST", "/chat", json={"message": "a"}) as response:
            sid = response.headers["x-session-id"]

        response = test_client.post("/config", json={"session_id": sid, "model": "model-b"})
        assert response.status_code == 200

        with test_client.stream("POST", "/chat", json={"session_id": sid, "message": "b"}) as _r:
            fake = app.state.fakes[0]
        assert fake.last_chat_config is not None
        assert fake.last_chat_config.model == "model-b"
    mp.undo()


def test_config_set_global_applies_to_live_and_new_sessions(monkeypatch):
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi"), text_completion("hi"), text_completion("hi")])
    with TestClient(app) as test_client:
        # one live session exists before we set the global default
        with test_client.stream("POST", "/chat", json={"message": "a"}) as response:
            sid = response.headers["x-session-id"]
        live_fake = app.state.fakes[0]

        # global set: becomes the app default AND applies to the live session
        response = test_client.post("/config", json={"model": "global-model"})
        assert response.status_code == 200
        assert response.json()["llm"]["model"] == "global-model"

        with test_client.stream("POST", "/chat", json={"session_id": sid, "message": "b"}) as _r:
            pass
        assert live_fake.last_chat_config.model == "global-model"

        # a NEW session now starts from the global app default
        with test_client.stream("POST", "/chat", json={"message": "c"}) as _r2:
            pass
        assert app.state.fakes[-1].config.model == "global-model"
    mp.undo()


def test_config_set_hook_veto_returns_409_and_keeps_state(monkeypatch):
    from agent_seed.hooks import Hooks
    from tests.conftest import text_completion

    def veto(ctx):
        raise RuntimeError("policy: no model changes")

    app, _, mp = _make_app([text_completion("hi")], hooks=Hooks(on_model_change=veto))
    with TestClient(app) as test_client:
        with test_client.stream("POST", "/chat", json={"message": "a"}) as response:
            sid = response.headers["x-session-id"]

        response = test_client.post("/config", json={"session_id": sid, "model": "model-x"})
        assert response.status_code == 409
        assert "policy" in response.json()["error"]

        got = test_client.get("/config", params={"session_id": sid})
        assert got.json()["llm"]["model"] == "test-model"  # unchanged
    mp.undo()


def test_config_validation_and_unknown_session(monkeypatch):
    app, _, mp = _make_app([])
    with TestClient(app) as test_client:
        assert test_client.post("/config", json={}).status_code == 400
        # An unknown session_id is treated as "not yet created": POST records a
        # pending config (200); GET on it reports the pending config (200); GET
        # on a session with nothing pending is still a 404.
        assert (
            test_client.post("/config", json={"model": "m", "session_id": "nope"}).status_code
            == 200
        )
        got = test_client.get("/config", params={"session_id": "nope"})
        assert got.status_code == 200 and got.json()["llm"]["model"] == "m"
        assert (
            test_client.get("/config", params={"session_id": "never-touched"}).status_code == 404
        )
    mp.undo()


def test_config_set_before_first_message_applies_at_creation(monkeypatch):
    """Regression: frontends mint a session_id up front, so `/model <name>` can
    arrive before any /chat. It must be honored, not 404'd."""
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi"), text_completion("hi")])
    fresh_sid = "pre-minted-session"
    with TestClient(app) as test_client:
        # no session exists yet — this is exactly the failing case
        response = test_client.post(
            "/config", json={"session_id": fresh_sid, "model": "qwen3.8:27b"}
        )
        assert response.status_code == 200
        assert response.json()["llm"]["model"] == "qwen3.8:27b"
        # the GET reflects the pending choice too (200, not 404)
        got = test_client.get("/config", params={"session_id": fresh_sid})
        assert got.status_code == 200
        assert got.json()["llm"]["model"] == "qwen3.8:27b"

        # the first /chat creates the session and ADOPTS the pending model
        with test_client.stream(
            "POST", "/chat", json={"message": "hi", "session_id": fresh_sid}
        ) as _r:
            pass
        assert app.state.fakes[0].config.model == "qwen3.8:27b"
        # and the turn ran against it
        assert app.state.fakes[0].last_chat_config.model == "qwen3.8:27b"
    mp.undo()


def _probe_client_factory(behavior):
    """A fake OpenAICompatClient whose chat() delegates to behavior."""

    def factory(config):
        class _Probe:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                return behavior(self)

        return _Probe()

    return factory


def test_config_test_endpoint_success(monkeypatch):
    from tests.conftest import text_completion

    app, _, mp = _make_app([])
    monkeypatch.setattr(
        "agent_seed.core.OpenAICompatClient",
        _probe_client_factory(lambda self: text_completion("ok")),
    )
    with TestClient(app) as test_client:
        response = test_client.post("/config/test", json={"model": "probe-model"})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["model"] == "probe-model"
        assert body["latency_ms"] is not None
    mp.undo()


def test_config_test_endpoint_failure(monkeypatch):
    app, _, mp = _make_app([])

    def boom(self):
        raise RuntimeError("boom probe")

    monkeypatch.setattr("agent_seed.core.OpenAICompatClient", _probe_client_factory(boom))
    with TestClient(app) as test_client:
        response = test_client.post("/config/test", json={"model": "bad-model"})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert "boom probe" in body["error"]
    mp.undo()
