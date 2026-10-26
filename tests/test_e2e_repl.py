"""End-to-end battery: a real ``agent-seed`` server + a real ``agent-repl``.

Topology (see ``docs/protocol.md``):

    [pytest]                [subprocess: real backend]                [subprocess: real REPL]
    FakeLLMServer  ◀── LLM (SSE) ──  uv run agent-seed  ◀── HTTP/SSE ──  uv run agent-repl (pty)

Both binaries are exactly the user's commands (``uv run ...``); the only fake is
the LLM endpoint, which is scripted per test so the suite is deterministic and
needs no network. Each test gets a *fresh* REPL (the user-facing surface); the
backend is shared per module.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.fake_llm import FakeLLMServer, fail, text, tool
from tests.e2e.repl_driver import ReplDriver, spawn_server, wait_until_healthy

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model names the fake recognises (the REPL's ``config.model`` echoes these).
FAKE_MODEL = "fake-model"
GOOD_MODEL = "good-model"
FORCED_MODEL = "forced-model"
BAD_MODEL = "unreachable-model"

MAIN_AGENT_HOOKS = (
    "from agent_seed.hooks import Hooks, RunContext\n"
    "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
    "def make(): return Hooks(enter_agent=enter_agent)\n"
)
ALPHA_AGENT_HOOKS = (
    "from agent_seed.hooks import Hooks, RunContext\n"
    "def enter_agent(ctx):\n"
    "    ctx.allowed_tools = []\n"
    "    ctx.system_prompt = 'You are an alpha agent.'\n"
    "def make(): return Hooks(enter_agent=enter_agent)\n"
)


def _write_hooks_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "00_main.py").write_text(MAIN_AGENT_HOOKS)
    (root / "alpha").mkdir()
    (root / "alpha" / "00_alpha.py").write_text(ALPHA_AGENT_HOOKS)


# -- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_server() -> Iterator[FakeLLMServer]:
    server = FakeLLMServer().start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def backend(
    fake_server: FakeLLMServer, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, Any]]:
    hooks_root = tmp_path_factory.mktemp("hooks_root")
    _write_hooks_root(hooks_root)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = {
        "LLM_MODEL": FAKE_MODEL,
        "LLM_API_KEY": "unused",
        "LLM_BASE_URL": fake_server.url,
        "AGENT_SERVER_HOST": "127.0.0.1",
        "AGENT_SERVER_PORT": str(port),
        "HOOKS_ROOT": str(hooks_root),
    }
    proc = spawn_server(env, str(PROJECT_ROOT))
    url = f"http://127.0.0.1:{port}"
    try:
        wait_until_healthy(url, timeout=25)
    except TimeoutError as exc:
        proc.terminate()
        out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        raise TimeoutError(f"backend failed to start:\n{out}\n{err}") from exc
    yield {"proc": proc, "url": url, "hooks_root": hooks_root}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


@pytest.fixture
def repl(backend: dict[str, Any]) -> Iterator[ReplDriver]:
    """A fresh, ready REPL for every test."""
    driver = ReplDriver(backend["url"], cwd=str(PROJECT_ROOT), project_root=str(PROJECT_ROOT))
    try:
        driver.wait_ready(timeout=30)
        yield driver
    finally:
        try:
            driver.close(timeout=15)
        except Exception:
            driver._kill()


def _last_chat_request(fake: FakeLLMServer) -> dict[str, Any]:
    for record in reversed(fake.recorded):
        if record["path"].endswith("/chat/completions"):
            return record["body"]
    raise AssertionError("no /chat/completions request was recorded on the fake LLM")


# -- scenarios ------------------------------------------------------------------


def test_help_lists_the_commands(repl: ReplDriver) -> None:
    out = repl.send("/help")
    for command in ("/help", "/reset", "/agent", "/agents", "/tools", "/model", "/quit"):
        assert command in out, f"help output missing {command!r}:\n{out[:1200]}"


def test_tools_lists_calculator(repl: ReplDriver) -> None:
    out = repl.send("/tools")
    assert "calculator" in out, out[:1200]


def test_agents_lists_alpha(repl: ReplDriver) -> None:
    out = repl.send("/agents")
    assert "agents:" in out and "alpha" in out, out[:1200]


def test_agent_switch_acknowledged(repl: ReplDriver) -> None:
    out = repl.send("/agent alpha")
    assert "agent: alpha" in out, out[:1200]
    out = repl.send("/agent")  # no argument: show the current agent
    assert "alpha" in out, out[:1200]


def test_agent_switch_reflects_in_tools(repl: ReplDriver) -> None:
    # After switching, /tools reflects the *active* agent's whitelist.
    repl.send("/agent alpha")
    out = repl.send("/tools")
    assert "calculator" not in out, (
        "alpha declares no tools; /tools must reflect that:\n" + out[:1200]
    )


def test_alpha_agent_receives_no_tool_schemas(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    # alpha's whitelist is empty: the core must not offer the calculator tool.
    fake_server.reset(text("alpha says hello"))
    repl.send("/agent alpha")
    out = repl.send("hi from alpha?")
    assert "alpha says hello" in out, out[:1200]
    request = _last_chat_request(fake_server)
    assert request.get("model") == FAKE_MODEL
    assert "tools" not in request, "strict whitelist: alpha declares no tools"


def test_main_agent_tool_round_trip(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    # Main agent: LLM calls the whitelisted calculator, core runs it, LLM answers.
    fake_server.reset(
        tool("calculator", expression="12 * 13"),
        text("12 times 13 is 156."),
    )
    out = repl.send("what is 12 times 13?")
    assert "calculator" in out, "the tool event must be rendered:\n" + out[:1200]
    assert "156.0" in out, "the computed tool result must be rendered:\n" + out[:1200]
    assert "12 times 13 is 156." in out, "the final answer must be rendered:\n" + out[:1200]
    request = _last_chat_request(fake_server)
    assert request.get("model") == FAKE_MODEL
    tool_names = [t["function"]["name"] for t in request.get("tools", [])]
    assert "calculator" in tool_names, "main agent must be offered the calculator tool"


def test_model_shows_session_config(repl: ReplDriver) -> None:
    out = repl.send("/model")
    assert FAKE_MODEL in out, f"the current model should be shown:\n{out[:1200]}"


def test_model_test_probes_success(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    fake_server.reset(text("pong"))  # the probe's minimal completion
    out = repl.send(f"/model test {GOOD_MODEL}")
    assert "ok:" in out, f"probe should report ok:\n{out[:1200]}"
    assert GOOD_MODEL in out, out[:1200]
    request = _last_chat_request(fake_server)
    assert request.get("model") == GOOD_MODEL, "the probe must target the candidate model"


def test_model_test_reports_failure_cleanly(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    fake_server.reset(fail())
    out = repl.send(f"/model test {BAD_MODEL}")
    assert "failed" in out.lower() or "error" in out.lower(), out[:1200]
    assert BAD_MODEL in out, out[:1200]
    assert "Traceback" not in out, "failure must be reported, not raised:\n" + out[:1200]


def test_model_force_switches_and_is_used(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    # `/model force <m>` commits without a probe; the *next* LLM call uses it.
    fake_server.reset(text("forced reply"))
    out = repl.send(f"/model force {FORCED_MODEL}")
    assert FORCED_MODEL in out, out[:1200]
    out = repl.send("hello after force")
    assert "forced reply" in out, out[:1200]
    request = _last_chat_request(fake_server)
    assert request.get("model") == FORCED_MODEL, (
        f"the core must call the forced model, got {request.get('model')}"
    )


def test_reset_starts_fresh_history(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    fake_server.reset(text("first answer"), text("second answer"))
    first = repl.send("remember this fact")
    assert "first answer" in first, first[:1200]
    out = repl.send("/reset")
    assert "session reset" in out, out[:1200]
    second = repl.send("what did I say?")
    assert "second answer" in second, second[:1200]
    # The post-reset turn must not carry the pre-reset user message.
    request = _last_chat_request(fake_server)
    user_messages = [m for m in request["messages"] if m.get("role") == "user"]
    assert len(user_messages) == 1, f"reset should clear history: {user_messages}"
    assert "remember this fact" not in user_messages[0]["content"]


def test_block_mode_joins_multiline_message(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    # A multi-line block: the REPL joins the lines into one message.
    fake_server.reset(text("three lines, answered"))
    out = repl.send_block(["first line", "second  line", "third line"])
    assert "three lines, answered" in out, out[:1200]
    request = _last_chat_request(fake_server)
    user_messages = [m for m in request["messages"] if m.get("role") == "user"]
    assert user_messages[-1]["content"] == "first line\nsecond  line\nthird line", user_messages[
        -1
    ]


def test_llm_error_surfaced_without_traceback(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    # The LLM 500s mid-turn; the backend must emit an ``error`` event and the
    # REPL must show it — not a Python traceback.
    fake_server.reset(fail())
    out = repl.send("this will fail")
    assert "error" in out.lower(), f"failure should be surfaced:\n{out[:1200]}"
    assert "Traceback" not in out and "Exception" not in out, out[:1200]


def test_unknown_command_keeps_repl_alive(repl: ReplDriver) -> None:
    out = repl.send("/nope")
    assert "unknown" in out.lower(), out[:1200]
    # The REPL is still usable afterwards: a real command works.
    out = repl.send("/agents")
    assert "alpha" in out, out[:1200]


def test_quit_exits_cleanly(repl: ReplDriver, backend: dict[str, Any]) -> None:
    out = repl.send("/quit")
    assert "bye" in out.lower(), out[:1200]
    # The backend is still healthy after the REPL has exited.
    with httpx.Client(timeout=5) as client:
        assert client.get(f"{backend['url']}/health").status_code == 200
