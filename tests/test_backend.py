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


def test_tools_raises_hook_returns_clean_500_json(monkeypatch, tmp_path):
    """Regression: an agent whose enter_agent hook **raises** must return a
    clean 500 JSON body (not a FastAPI traceback, not a 200 with the default
    body). Before the fix the hook raised straight out of the request handler
    into FastAPI's error middleware — the test client would have received a 500
    with ``text/plain`` body, or a 200 when the hook wrote a marker before
    raising.
    """
    from agent_seed.discovery import reset_cache

    hooks_root = tmp_path / "hk"
    (hooks_root / "boom").mkdir(parents=True)
    (hooks_root / "boom" / "b.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx):\n"
        "    # a broken agent hook: it raises before declaring tools.\n"
        "    raise RuntimeError('enter_agent exploded')\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent)\n"
    )
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))
    reset_cache()

    reg = ToolRegistry()
    reg.register(calculator)
    app = server.create_app(reg)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert "boom" in client.get("/agents").json().get("agents", [])
        response = client.get("/tools", params={"agent_name": "boom"})
        assert response.status_code == 500
        assert (response.headers.get("content-type") or "").startswith("application/json"), (
            f"expected a JSON error body, got {response.headers.get('content-type')!r}"
        )
        body = response.json()
        assert "error" in body and "exploded" in body["error"]
        assert "Traceback" not in response.text


def test_chat_with_broken_agent_switch_returns_clean_500(monkeypatch, tmp_path):
    """Regression: ``POST /chat`` with ``agent_name=<broken>`` must return a
    clean JSON 500 (the agent hook folder failed to load / entered). Before
    the fix, only ``LLMConfigError`` was caught here and a ``DiscoveryError``
    (raised by an enter_agent that raises, or by a broken hook file) escaped
    into FastAPI's middleware with a traceback body.
    """
    from agent_seed.discovery import reset_cache

    hooks_root = tmp_path / "hk"
    (hooks_root / "main").mkdir(parents=True)
    (hooks_root / "main" / "m.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent)\n"
    )
    (hooks_root / "bad_agent").mkdir()
    (hooks_root / "bad_agent" / "b.py").write_text(
        "from agent_seed.discovery import DiscoveryError\n"
        "from agent_seed.hooks import Hooks\n"
        "def make() -> Hooks:\n"
        "    raise DiscoveryError('broken agent folder')\n"
    )
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))
    reset_cache()

    reg = ToolRegistry()
    reg.register(calculator)
    app = server.create_app(reg)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert "main" in client.get("/agents").json().get("agents", [])
        assert "bad_agent" in client.get("/agents").json().get("agents", [])
        response = client.post("/chat", json={"message": "hi", "agent_name": "bad_agent"})
        assert response.status_code == 500
        assert (response.headers.get("content-type") or "").startswith("application/json"), (
            f"expected a JSON error body, got {response.headers.get('content-type')!r}"
        )
        body = response.json()
        assert "broken agent folder" in body.get("error", "")
        assert "Traceback" not in response.text


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


def _usage_hooks():
    from agent_seed.hooks import Hooks, RunContext

    def _enter(ctx: RunContext) -> None:
        ctx.allowed_tools = ["calculator"]

    return Hooks(enter_agent=_enter)


class _UsageFake:
    """A fake LLM that pops scripted completions (each may carry usage)."""

    def __init__(self, config, script):
        self.config = config
        self._script = script

    def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
        self.last_chat_config = config
        completion = self._script.pop(0)
        if on_token is not None and completion.content:
            on_token(completion.content)
        return completion


def _usage_app(script):
    """Build the app from a usage-bearing script. Returns (app, monkeypatch)."""

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry, hooks=_usage_hooks())
    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_MODEL", "test-model")

    def fake_client_factory(cfg):
        return _UsageFake(cfg, script)

    mp.setattr(server, "OpenAICompatClient", fake_client_factory)
    return app, mp


def test_done_event_carries_per_turn_usage(monkeypatch):
    from fastapi.testclient import TestClient

    from tests.conftest import text_completion, tool_completion

    # One turn (tool round + final answer); each scripted LLM call reports
    # its own usage. Per-turn usage = sum of both calls in this turn.
    script = [
        tool_completion(
            "calculator",
            {"expression": "6 * 7"},
            usage={"prompt_tokens": 2, "completion_tokens": 1},
        ),
        text_completion("42.0", usage={"prompt_tokens": 3, "completion_tokens": 2}),
    ]
    app, mp = _usage_app(script)
    with TestClient(app) as client:
        with client.stream("POST", "/chat", json={"message": "6 * 7?"}) as response:
            body = "".join(response.iter_text())
        events = dict(_events(body))
        done = events["done"]
        # per-turn = sum of both LLM calls in this turn (2+3, 1+2)
        assert done["usage"]["prompt_tokens"] == 5
        assert done["usage"]["completion_tokens"] == 3
        # session total for a fresh session equals the first (only) turn
        assert done["usage_session"]["prompt_tokens"] == 5
        assert done["usage_session"]["completion_tokens"] == 3
    mp.undo()


def test_session_usage_accumulates_and_reset_clears(monkeypatch):
    from fastapi.testclient import TestClient

    from tests.conftest import text_completion

    # One text completion per turn (no tool round). 3 script items for 3 turns.
    script = [
        text_completion("one", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        text_completion("two", usage={"prompt_tokens": 2, "completion_tokens": 3}),
        text_completion("three", usage={"prompt_tokens": 4, "completion_tokens": 5}),
    ]
    app, mp = _usage_app(script)
    with TestClient(app) as client:
        sid = client.post("/chat", json={"message": "one"}).headers.get("x-session-id")
        with client.stream("POST", "/chat", json={"message": "two", "session_id": sid}) as r:
            events = dict(_events("".join(r.iter_text())))
        done = events["done"]
        # second turn: 2/3 for that turn; 3/4 for the session (1+2, 1+3)
        assert done["usage"] == {"prompt_tokens": 2, "completion_tokens": 3}
        assert done["usage_session"] == {"prompt_tokens": 3, "completion_tokens": 4}

        # reset -> new session starts at zero; one turn => exactly one turn's usage
        client.post("/reset", json={"session_id": sid})
        with client.stream("POST", "/chat", json={"message": "three"}) as r:
            events = dict(_events("".join(r.iter_text())))
        done = events["done"]
        assert done["usage"] == {"prompt_tokens": 4, "completion_tokens": 5}
        assert done["usage_session"] == {"prompt_tokens": 4, "completion_tokens": 5}
    mp.undo()


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


# -- /prompt ---------------------------------------------------------------------


def test_prompt_endpoint_scratch_default(monkeypatch, tmp_path):
    """Scratch (default) resolves the agent's declared prompts from its folder."""
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    from agent_seed.core import DEFAULT_SYSTEM_PROMPT
    from agent_seed.discovery import reset_cache

    # Empty folder: the scratch view falls back to the default identity.
    reset_cache()
    app = server.create_app(ToolRegistry())
    with TestClient(app) as test_client:
        body = test_client.get("/prompt").json()
        assert body["live"] is False
        assert body["identity"] == DEFAULT_SYSTEM_PROMPT
        assert body["context"] is None
        assert body["examples"] == []
        assert body["addendum"] is None
        assert body["system"] == DEFAULT_SYSTEM_PROMPT

    # Populated folder: scratch reflects the declared files.
    (tmp_path / "prompt.md").write_text("You are alpha.\n")
    (tmp_path / "context.md").write_text("state=on\n")
    (tmp_path / "examples.jsonl").write_text(
        '{"role": "user", "content": "e1"}\n{"role": "assistant", "content": "a1"}\n'
    )
    reset_cache()
    app = server.create_app(ToolRegistry())
    with TestClient(app) as test_client:
        body = test_client.get("/prompt").json()
        assert body["live"] is False
        assert body["identity"] == "You are alpha."
        assert body["context"] == "state=on"
        assert body["examples"] == [
            {"role": "user", "content": "e1"},
            {"role": "assistant", "content": "a1"},
        ]
        # system is composed: identity + context block (blank-line separated).
        assert body["system"] == "You are alpha.\n\nstate=on"
        assert body["addendum"] is None


def test_prompt_endpoint_agent_name_selects_subdir(monkeypatch, tmp_path):
    (tmp_path / "alpha").mkdir(parents=True)
    (tmp_path / "alpha" / "prompt.md").write_text("ALPHA IDENTITY\n")
    (tmp_path / "beta").mkdir(parents=True)
    (tmp_path / "beta" / "prompt.md").write_text("BETA IDENTITY\n")
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    from agent_seed.discovery import reset_cache

    reset_cache()
    app = server.create_app(ToolRegistry())
    with TestClient(app) as test_client:
        alpha = test_client.get("/prompt", params={"agent_name": "alpha"}).json()
        beta = test_client.get("/prompt", params={"agent_name": "beta"}).json()
        assert alpha["identity"] == "ALPHA IDENTITY"
        assert beta["identity"] == "BETA IDENTITY"


def test_prompt_endpoint_unknown_agent_falls_back_to_root(monkeypatch, tmp_path):
    """Unknown agent name falls back to the (empty) root: default identity."""
    from agent_seed.core import DEFAULT_SYSTEM_PROMPT
    from agent_seed.discovery import reset_cache

    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()
    app = server.create_app(ToolRegistry())
    with TestClient(app) as test_client:
        body = test_client.get("/prompt", params={"agent_name": "nope"}).json()
        assert body["live"] is False
        assert body["identity"] == DEFAULT_SYSTEM_PROMPT


def test_prompt_endpoint_live_requires_session_and_reports_unknown(monkeypatch):
    """live=true without a session_id is 400; with an unknown id is 404."""
    app = server.create_app(ToolRegistry())
    with TestClient(app) as test_client:
        assert test_client.get("/prompt", params={"live": "true"}).status_code == 400
        assert (
            test_client.get("/prompt", params={"session_id": "nope", "live": "true"}).status_code
            == 404
        )


def test_prompt_endpoint_live_reflects_last_composed_value(monkeypatch, tmp_path):
    """live=1 with a session_id returns that session's last composed prompt
    (the sticky identity + that round's context/addendum), not the scratch view."""
    from agent_seed.core import DEFAULT_SYSTEM_PROMPT
    from agent_seed.discovery import reset_cache
    from agent_seed.hooks import Hooks, RunContext
    from agent_seed.llm import Completion

    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    monkeypatch.setenv("LLM_MODEL", "test-model")
    reset_cache()

    def set_addendum(ctx: RunContext) -> None:
        ctx.addendum = "round-note"

    hooks = Hooks(context="ctx-baseline", before_llm=set_addendum)

    script = [Completion(content="ok")]

    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                return script.pop(0)

        return _Fake()

    monkeypatch.setattr(server, "OpenAICompatClient", fake_client_factory)
    app = server.create_app(ToolRegistry(), hooks=hooks)
    with TestClient(app) as test_client:
        # Create a session and run one turn so the agent's per-round fields are set.
        with test_client.stream("POST", "/chat", json={"message": "hi"}) as response:
            session_id = response.headers["x-session-id"]
            assert response.status_code == 200

        # live=1 reflects the session's last composed value (addendum "round-note",
        # context baseline "ctx-baseline") — distinct from the scratch view, which
        # would report addendum=None (scratch has no live round to read).
        live = test_client.get("/prompt", params={"session_id": session_id, "live": "true"}).json()
        assert live["live"] is True
        assert live["session_id"] == session_id
        assert live["identity"] == DEFAULT_SYSTEM_PROMPT
        assert live["context"] == "ctx-baseline"
        assert live["addendum"] == "round-note"
        assert live["system"] == (DEFAULT_SYSTEM_PROMPT + "\n\nctx-baseline\n\nround-note")

        # Contrast: the scratch view for the same agent reports addendum=None.
        scratch = test_client.get("/prompt").json()
        assert scratch["live"] is False
        assert scratch["addendum"] is None


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


def test_chat_agent_omitted_name_selects_main_agent(tmp_path, monkeypatch):
    # Regression (REPL `/agent tutor` then `/agent default` then a greeting
    # still answers as the tutor): an omitted ``agent_name`` on the *same*
    # session must select the MAIN agent (the hooks root), not keep whatever is
    # currently loaded. Previously ``SessionStore.get`` guarded the switch with
    # ``agent_name is not None`` and skipped ``set_agent_name(None)``, so a
    # session that had used a sub-agent silently stayed on it. See protocol.md
    # ("An omitted agent_name ... selects the default (main) agent").
    from agent_seed.discovery import reset_cache
    from tests.conftest import text_completion, tool_completion

    hooks_root = tmp_path / "hook_modules"
    (hooks_root / "alpha").mkdir(parents=True)
    # alpha: whitelist calculator, rewrite the result x100 -> "200.0"
    (hooks_root / "alpha" / "a.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def after_tool_call(ctx): ctx.tool_result = (ctx.tool_result or 0) * 100\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent, after_tool_call=after_tool_call)\n"
    )
    # root (main agent): whitelist calculator, rewrite the result x1000 -> "2000.0"
    (hooks_root / "root.py").write_text(
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
        # Turn 1: use agent alpha (x100 -> "200.0"). Mints a session.
        with test_client.stream(
            "POST", "/chat", json={"message": "hi", "agent_name": "alpha"}
        ) as response:
            sid = response.headers["x-session-id"]
            body1 = "".join(response.iter_text())
        assert "200.0" in body1 and "2000.0" not in body1

        # Turn 2: SAME session_id, NO agent_name. Omission must re-resolve to
        # the main (root) agent (x1000 -> "2000.0") — not stay on alpha.
        with test_client.stream(
            "POST", "/chat", json={"message": "hi", "session_id": sid}
        ) as response:
            assert response.headers["x-session-id"] == sid
            body2 = "".join(response.iter_text())
        assert "2000.0" in body2 and "200.0" not in body2


def test_agent_selection_changes_composed_system_prompt(tmp_path, monkeypatch):
    # E2E (the REPL flow the user actually drove): selecting an agent — and
    # going back to the main (default) agent — must change the *exact* system
    # prompt the core sends to the LLM. The model's own "who are you" reply is
    # temperature-dependent and can self-report its base identity even when the
    # system prompt is correct, so we do NOT assert on it. Instead the fake LLM
    # records the precise ``messages`` it is called with and we assert on
    # ``messages[0]`` — the composed system prompt the model actually receives.
    #
    # This mirrors the user's real hook layout: a main (root) agent defined by
    # a ``prompt.md`` identity plus a calculator whitelist, and a named
    # sub-agent ``tutor`` that is a ``prompt.md`` identity file ("You are a
    # patient math tutor.") — exactly the shape of hook_modules/tutor/.
    from agent_seed.discovery import reset_cache
    from tests.conftest import text_completion

    hooks_root = tmp_path / "hook_modules"
    hooks_root.mkdir(parents=True)
    # Main (root) agent: a file-driven identity (hook_modules/prompt.md).
    (hooks_root / "prompt.md").write_text("You are a concise, helpful assistant.\n")
    # ...plus a python tool-whitelist declaration (hook_modules/00_tools.py).
    (hooks_root / "00_tools.py").write_text(
        "from agent_seed.hooks import Hooks, RunContext\n"
        "def enter_agent(ctx):\n"
        "    ctx.allowed_tools = ['calculator']\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent)\n"
    )
    # Named sub-agent 'tutor': a prompt.md identity, like the user's folder.
    (hooks_root / "tutor").mkdir(parents=True)
    (hooks_root / "tutor" / "prompt.md").write_text("You are a patient math tutor.\n")
    (hooks_root / "tutor" / "tools.py").write_text(
        "from agent_seed.hooks import Hooks, RunContext\n"
        "def enter_agent(ctx):\n"
        "    ctx.allowed_tools = ['calculator']\n"
        "def make() -> Hooks:\n"
        "    return Hooks(enter_agent=enter_agent)\n"
    )

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(hooks_root))
    reset_cache()

    registry = ToolRegistry()
    registry.register(calculator)

    # Records the EXACT messages each LLM call receives (messages[0] is the
    # composed system prompt the model sees). No tool is ever called, so each
    # script entry is a plain reply.
    seen: list = []

    def fake_chat(messages, tools=None, on_token=None, config=None, max_tokens=None):
        seen.append([dict(m) for m in messages])
        completion = text_completion("hi")
        if on_token is not None and completion.content:
            on_token(completion.content)
        return completion

    class _Fake:
        def __init__(self, config):
            self.config = config

        def chat(self, *a, **k):
            return fake_chat(*a, **k)

    monkeypatch.setattr(server, "OpenAICompatClient", lambda cfg: _Fake(cfg))
    app = server.create_app(registry)  # hooks=None -> discovery, like the real backend

    ROOT = "concise, helpful assistant"
    TUTOR = "patient math tutor"

    def system_of(i: int) -> str:
        return seen[i][0]["content"]

    with TestClient(app) as test_client:
        # Sanity: tutor is discoverable (mirrors `/agents` showing it).
        assert "tutor" in test_client.get("/agents").json()["agents"]

        # 1) main agent (no agent_name) -> root identity, not tutor.
        with test_client.stream("POST", "/chat", json={"message": "Hi, who are you?"}) as response:
            sid = response.headers["x-session-id"]
            "".join(response.iter_text())
        assert ROOT in system_of(0), system_of(0)
        assert TUTOR not in system_of(0), system_of(0)
        # The core must record that the active agent is the main one (None).
        live = test_client.get("/prompt", params={"session_id": sid, "live": "true"}).json()
        assert live["agent_name"] is None

        # 2) SAME session, select tutor -> tutor identity, not root.
        with test_client.stream(
            "POST",
            "/chat",
            json={"message": "Hi, who are you?", "session_id": sid, "agent_name": "tutor"},
        ) as response:
            "".join(response.iter_text())
        assert TUTOR in system_of(1), system_of(1)
        assert ROOT not in system_of(1), system_of(1)
        live = test_client.get("/prompt", params={"session_id": sid, "live": "true"}).json()
        assert live["agent_name"] == "tutor"

        # 3) SAME session, back to main (omitted agent_name) -> root identity again.
        with test_client.stream(
            "POST", "/chat", json={"message": "Hi, who are you?", "session_id": sid}
        ) as response:
            "".join(response.iter_text())
        assert ROOT in system_of(2), system_of(2)
        assert TUTOR not in system_of(2), system_of(2)
        live = test_client.get("/prompt", params={"session_id": sid, "live": "true"}).json()
        assert live["agent_name"] is None

    reset_cache()


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
            # Drain the turn to completion so no producer thread is still
            # writing fake state when we assert on it below.
            "".join(response.iter_text())

        response = test_client.post("/config", json={"session_id": sid, "model": "model-b"})
        assert response.status_code == 200

        with test_client.stream("POST", "/chat", json={"session_id": sid, "message": "b"}) as _r:
            # Read the stream to the `done` frame: the backend emits `done`
            # only after the producer thread finished the LLM call, so the fake
            # state observed below is guaranteed to reflect THIS turn.
            body = "".join(_r.iter_text())
        assert "done" in [k for k, _ in _events(body)]
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
            # Read the stream to the `done` frame: the backend emits `done`
            # only after the producer thread finished the LLM call, so
            # live_fake.last_chat_config below is guaranteed to be THIS turn's.
            body = "".join(_r.iter_text())
        assert "done" in [k for k, _ in _events(body)]
        assert live_fake.last_chat_config.model == "global-model"

        # a NEW session now starts from the global app default
        with test_client.stream("POST", "/chat", json={"message": "c"}) as _r2:
            body2 = "".join(_r2.iter_text())
        assert "done" in [k for k, _ in _events(body2)]
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


def test_config_test_no_base_returns_400_not_fake_default(monkeypatch, tmp_path):
    """With no base LLM to merge over (LLM_MODEL unset, no pending/app config,
    and the body supplying neither base_url nor api_key) the probe must return
    a clear 400 — it must NOT fabricate a default and report a bogus model or a
    false success."""
    from tests.conftest import text_completion  # noqa: F401 - parity with _make_app

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry)
    app.state.fakes = []

    mp2 = pytest.MonkeyPatch()
    mp2.delenv("LLM_MODEL", raising=False)
    mp2.chdir(tmp_path)  # no .env in CWD
    mp2.setattr("agent_seed.llm.load_dotenv", lambda *a, **k: False)

    with TestClient(app) as test_client:
        response = test_client.post("/config/test", json={"model": "candidate"})
        assert response.status_code == 400
        body = response.json()
        assert body["ok"] is False
        # No fabricated probe ran: there is NO `model` field carrying a made-up
        # default (the old bug reported model="probe").
        assert body.get("model") is None
        assert "latency_ms" not in body or body.get("latency_ms") is None
    mp2.undo()


def test_config_set_global_hook_veto_returns_409_and_keeps_state(monkeypatch):
    """POST /config (no session_id) with a veto hook on any live session must
    return 409, leave every live session unchanged, AND leave the app default
    unchanged (atomic rollback)."""
    from agent_seed.hooks import Hooks
    from tests.conftest import text_completion

    def veto(ctx):
        raise RuntimeError("policy: no model changes")

    app, _, mp = _make_app([text_completion("hi")], hooks=Hooks(on_model_change=veto))
    with TestClient(app) as test_client:
        # Two live sessions so the partial-commit case is actually exercised.
        with test_client.stream("POST", "/chat", json={"message": "a"}) as response:
            sid1 = response.headers["x-session-id"]
        with test_client.stream("POST", "/chat", json={"message": "b"}) as response:
            sid2 = response.headers["x-session-id"]

        # Global set (no session_id) must be vetoed.
        response = test_client.post("/config", json={"model": "global-x"})
        assert response.status_code == 409
        assert "policy" in response.json()["error"]

        # Both live sessions are unchanged.
        for sid in (sid1, sid2):
            got = test_client.get("/config", params={"session_id": sid})
            assert got.json()["llm"]["model"] == "test-model"

        # The app default is unchanged (not committed).
        got_global = test_client.get("/config")
        assert got_global.json()["llm"]["model"] == "test-model"
    mp.undo()


def test_config_test_with_session_id_uses_session_as_merge_base(monkeypatch):
    """POST /config/test with a session_id uses the session's current model as
    the merge base, so the candidate (body has no model) reflects the session's
    model, not the app default."""
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi")])
    monkeypatch.setattr(
        "agent_seed.core.OpenAICompatClient",
        _probe_client_factory(lambda self: text_completion("ok")),
    )
    with TestClient(app) as test_client:
        # Create a session and switch its model.
        with test_client.stream("POST", "/chat", json={"message": "a"}) as response:
            sid = response.headers["x-session-id"]
        resp = test_client.post("/config", json={"session_id": sid, "model": "session-model"})
        assert resp.status_code == 200

        # Probe with session_id (no model in body): the merge base is the
        # session's "session-model", so the candidate model is "session-model".
        # (If the base were the app default "test-model", it would be "test-model".)
        resp = test_client.post("/config/test", json={"session_id": sid})
        assert resp.status_code == 200
        assert resp.json()["model"] == "session-model"
    mp.undo()


def test_config_get_pending_session_with_no_base_returns_base_null(monkeypatch, tmp_path):
    """GET /config for a pending session when the app default cannot be
    determined (LLM_MODEL unset, no explicit app default) must return 200
    with base=null, not 500."""
    from tests.conftest import text_completion

    app, _, mp = _make_app([text_completion("hi")])
    with TestClient(app) as test_client:
        # Record a pending config (LLM_MODEL is set by _make_app so default_llm works).
        r = test_client.post("/config", json={"session_id": "X", "model": "x-model"})
        assert r.status_code == 200

        # Now make the app default un-determinable.
        mp.delenv("LLM_MODEL", raising=False)
        mp.chdir(tmp_path)
        mp.setattr("agent_seed.llm.load_dotenv", lambda *a, **k: False)

        # The pending GET must still be 200 (the pending config is valid) and
        # base must be null (no base to report), not a 500.
        got = test_client.get("/config", params={"session_id": "X"})
        assert got.status_code == 200
        data = got.json()
        assert data["llm"]["model"] == "x-model"
        assert data["base"] is None
    mp.undo()


def test_pending_config_survives_unrelated_chat(monkeypatch, tmp_path):
    """A config set for session ``X`` before X's first /chat is 'pending'.
    Chats for a *different* session (``Y``) must not consume or clobber it;
    X must still adopt its pending config when it is finally created."""
    from tests.conftest import text_completion

    registry = ToolRegistry()
    registry.register(calculator)
    app = server.create_app(registry)
    app.state.fakes = []
    mp2 = pytest.MonkeyPatch()
    mp2.setenv("LLM_MODEL", "env-default")

    def fake_client_factory(cfg):
        class _Fake:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                self.last_chat_config = config
                return text_completion("ok")

        fake = _Fake()
        fake.config = cfg
        app.state.fakes.append(fake)
        return fake

    mp2.setattr(server, "OpenAICompatClient", fake_client_factory)

    with TestClient(app) as test_client:
        # 1. Mint X and record a pending config for it (before any /chat).
        r = test_client.post("/config", json={"session_id": "X", "model": "x-model"})
        assert r.status_code == 200

        # 2. An unrelated session Y runs a full turn.
        payload = {"message": "hi", "session_id": "Y"}
        with test_client.stream("POST", "/chat", json=payload) as resp:
            assert resp.status_code == 200
            "".join(resp.iter_text())

        # 3. X's pending config must still be intact (not consumed / 404).
        got = test_client.get("/config", params={"session_id": "X"})
        assert got.status_code == 200
        assert got.json()["llm"]["model"] == "x-model"

        # 4. When X is finally created it adopts its pending model; Y used env.
        fakes = app.state.fakes
        assert fakes[0].config.model == "env-default"  # Y started from env
        payload_x = {"message": "hi", "session_id": "X"}
        with test_client.stream("POST", "/chat", json=payload_x) as resp:
            "".join(resp.iter_text())
        assert app.state.fakes[-1].config.model == "x-model"

    mp2.undo()


async def test_chat_streams_in_isolation_across_concurrent_handles(monkeypatch):
    """Exercise the `async def` SSE handler on a real event loop (the sync
    TestClient can't, and this is the project's only genuine async surface).
    Two concurrent chat requests must each receive their own scripted
    completions — no cross-session script corruption."""
    import asyncio

    import httpx2

    from agent_seed.hooks import Hooks
    from tests.conftest import text_completion

    monkeypatch.setenv("LLM_MODEL", "test-model")

    class _Fake:
        def __init__(self, config):
            self.config = config
            self._answer = None

        def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
            # Echo the request's message so each concurrent session's
            # completion is distinct and we can assert isolation.
            last_user = next(m["content"] for m in reversed(messages) if m.get("role") == "user")
            self._answer = f"echo:{last_user}"
            if on_token is not None:
                on_token(self._answer)
            return text_completion(self._answer)

    monkeypatch.setattr(server, "OpenAICompatClient", _Fake)
    registry = ToolRegistry()
    registry.register(calculator)
    hooks = Hooks(enter_agent=lambda ctx: setattr(ctx, "allowed_tools", ["calculator"]))
    app = server.create_app(registry, hooks=hooks)

    transport = httpx2.ASGITransport(app=app)

    async def _chat(session_id: str, message: str) -> str:
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"session_id": session_id, "message": message}
            )
            return response.text

    body1, body2 = await asyncio.gather(_chat("s1", "h1"), _chat("s2", "h2"))

    # Each session's stream must contain exactly its own echoed message.
    assert "echo:h1" in body1 and "echo:h2" not in body1
    assert "echo:h2" in body2 and "echo:h1" not in body2
    # Both reached the terminal `done` event.
    assert "event: done" in body1 and "event: done" in body2
