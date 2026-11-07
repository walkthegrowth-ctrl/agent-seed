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

import httpx2
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
    for command in ("/help", "/reset", "/agent", "/agents", "/tools", "/model", "/usage", "/quit"):
        assert command in out, f"help output missing {command!r}:\n{out[:1200]}"


def test_tools_lists_calculator(repl: ReplDriver) -> None:
    out = repl.send("/tools")
    assert "calculator" in out, out[:1200]


def test_per_turn_usage_renders_and_usage_command_accumulates(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    # The fake LLM reports usage (prompt=1, completion=1) on each completion;
    # the core turns that into a per-turn usage, the backend emits it on ``done``,
    # and the REPL renders a ``in=/out=`` line and tracks the session total.
    fake_server.reset(text("hello world"))
    first = repl.send("say hello")
    assert "in=1" in first and "out=1" in first, (
        "the per-turn usage line must be rendered after the answer:\n" + first[:1200]
    )
    out = repl.send("/usage")
    assert "session:" in out and "in=1" in out, (
        "/usage must show the session cumulative total:\n" + out[:1200]
    )


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


# ---------------------------------------------------------------------------
# Per-command battery: every standard REPL command, driven in a *real*
# interactive session (no monkeypatching of the command logic). The key
# property these assert on top of "the right text": a command must *never*
# take the REPL down. Because the pty driver settles on a *dead* process too,
# a mid-turn exception would otherwise be invisible — so each test also guards
# that no Python traceback leaked and the REPL is still alive afterwards.
# ---------------------------------------------------------------------------


def _assert_repl_alive(repl: ReplDriver, out: str) -> None:
    """A command must not crash the REPL with a Python traceback / markup error."""
    assert "Traceback" not in out, f"traceback leaked into the REPL:\n{out[:1200]}"
    assert "MarkupError" not in out, "Rich markup error leaked into the REPL:\n" + out[:1200]
    assert repl.proc.poll() is None, (
        "the REPL process exited; a command crashed it:\n" + out[:1200]
    )


def test_usage_renders_the_session_total_without_crashing(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    """Regression: the /usage output used to contain a ``[/reset]`` substring
    that Rich parsed as a closing tag, raising MarkupError and killing the REPL
    *after* printing the correct session line. We now assert on the line Rich
    chokes on, not just the one it prints before it."""
    fake_server.reset(text("hello"))
    repl.send("hi")
    out = repl.send("/usage")
    assert "session:" in out and "in=1" in out, (
        "/usage must render the session cumulative total:\n" + out[:1200]
    )
    # ...and it must not die the instant it prints that line.
    _assert_repl_alive(repl, out)


def test_usage_before_any_turn_shows_empty_hint(repl: ReplDriver) -> None:
    out = repl.send("/usage")
    assert "no usage recorded" in out, out[:1200]
    _assert_repl_alive(repl, out)


def test_model_probe_then_set_uses_new_model(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    """``/model <n>`` (the plain form) probes first and *commits* on success; the
    next chat must run against the new model. (The committed-on-probe path was
    uncovered — only ``force`` was.)"""
    fake_server.reset(text("probe pong"), text("running with the new model"))
    first = repl.send(f"/model {GOOD_MODEL}")
    assert f"model: {GOOD_MODEL}" in first, (
        "a successful probe should commit and report the effective model:\n" + first[:1200]
    )
    second = repl.send("hello on the new model")
    assert "running with the new model" in second, second[:1200]
    request = _last_chat_request(fake_server)
    assert request.get("model") == GOOD_MODEL, (
        f"the next chat must use the committed model, got {request.get('model')}"
    )


def test_model_plain_probe_failure_aborts_keeps_current(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    """A failed probe on the plain ``/model <n>`` form must abort WITHOUT
    committing the bad model — the current model is left in place."""
    fake_server.reset(fail(), text("still on the old model"))
    out = repl.send(f"/model {BAD_MODEL}")
    assert "not set:" in out, "a failed probe must abort the commit:\n" + out[:1200]
    assert BAD_MODEL in out, out[:1200]
    _assert_repl_alive(repl, out)
    # The old model still answers — the failed candidate was never committed.
    again = repl.send("still on the old model")
    assert "still on the old model" in again, again[:1200]


def test_model_test_missing_name_shows_usage_hint(repl: ReplDriver) -> None:
    out = repl.send("/model test")
    assert "usage:" in out, "/model test with no name must show its usage hint:\n" + out[:1200]
    assert "/model test" in out, out[:1200]
    _assert_repl_alive(repl, out)


def test_model_force_missing_name_shows_usage_hint(repl: ReplDriver) -> None:
    out = repl.send("/model force")
    assert "usage:" in out, "/model force with no name must show its usage hint:\n" + out[:1200]
    assert "/model force" in out, out[:1200]
    _assert_repl_alive(repl, out)


def test_agent_no_arg_reports_the_main_agent(repl: ReplDriver) -> None:
    out = repl.send("/agent")
    assert "agent:" in out and "(default)" in out, (
        "/agent with no arg must show the current (main) agent:\n" + out[:1200]
    )
    _assert_repl_alive(repl, out)


def test_exit_is_an_alias_for_quit(repl: ReplDriver, backend: dict[str, Any]) -> None:
    out = repl.send("/exit")
    assert "bye" in out.lower(), out[:1200]
    with httpx2.Client(timeout=5) as client:
        assert client.get(f"{backend['url']}/health").status_code == 200


def test_blank_input_is_a_noop_and_repl_stays_alive(repl: ReplDriver) -> None:
    out = repl.send("")
    assert "agent>" not in out, "an empty line must not chat:\n" + out[:1200]
    assert "you>" in out, "an empty line must re-prompt:\n" + out[:1200]
    _assert_repl_alive(repl, out)
    # And still fully functional.
    out = repl.send("/agents")
    assert "agents:" in out, out[:1200]


def test_block_mode_keeps_blank_lines(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    """A multi-line block keeps interior blank lines (help: "blank lines are kept")."""
    fake_server.reset(text("joined fine"))
    out = repl.send_block(["line one", "", "line three"])
    assert "joined fine" in out, out[:1200]
    request = _last_chat_request(fake_server)
    user_messages = [m for m in request["messages"] if m.get("role") == "user"]
    assert user_messages[-1]["content"] == "line one\n\nline three", user_messages[-1]


def test_block_mode_slash_lines_are_plain_text(
    fake_server: FakeLLMServer, repl: ReplDriver
) -> None:
    """Inside a block, a line that *looks* like a command (``/reset``) is literal
    text sent to the LLM — it must NOT be executed."""
    fake_server.reset(text("echoed the slash as text"))
    out = repl.send_block(["/reset", "hello there"])
    assert "echoed the slash as text" in out, (
        "the block must be sent as a chat, not executed as commands:\n" + out[:1200]
    )
    assert "session reset" not in out.lower(), (
        "/reset inside a block must not have been executed:\n" + out[:1200]
    )


def test_block_mode_ctrl_c_cancels_the_draft(repl: ReplDriver) -> None:
    """Ctrl-C in block mode discards the draft and returns to normal mode."""
    # Open the block; the prompt is now "…>" (not "you>"), so settle on that marker.
    repl._send_marked("/", "...>", settle=1.2, timeout=30)
    repl._send_marked("some draft", "...>", settle=1.2, timeout=30)
    # A raw Ctrl-C (\x03 -> SIGINT) cancels the draft and the REPL returns to
    # normal mode, re-prompting "you> " — which is exactly where send_raw settles.
    out = repl.send_raw(b"\x03", settle=1.2, timeout=30, what="Ctrl-C in block mode")
    assert "draft discarded" in out.lower(), (
        "Ctrl-C in block mode must report the discarded draft:\n" + out[:1200]
    )
    # Back in normal mode: a real command still works and nothing was sent.
    out = repl.send("/agents")
    assert "agents:" in out, out[:1200]


def test_quit_exits_cleanly(repl: ReplDriver, backend: dict[str, Any]) -> None:
    out = repl.send("/quit")
    assert "bye" in out.lower(), out[:1200]
    # The backend is still healthy after the REPL has exited.
    with httpx2.Client(timeout=5) as client:
        assert client.get(f"{backend['url']}/health").status_code == 200


def test_agent_unknown_is_rejected_cleanly(repl: ReplDriver) -> None:
    """`/agent <n>` where <n> is not a discoverable agent must be rejected by
    the REPL, not silently fall back to the default via the backend."""
    out = repl.send("/agent nope_agent")
    assert "unknown agent" in out.lower(), out[:1200]
    # The main agent is still in effect (no switch was committed).
    out2 = repl.send("/agent")
    assert "(default)" in out2, out2[:1200]
    # REPL is still alive.
    _assert_repl_alive(repl, out)


def test_agent_default_returns_to_main_agent(repl: ReplDriver) -> None:
    out = repl.send("/agent alpha")
    assert "agent: alpha" in out, out[:1200]
    # Now switch back to main using the reserved name.
    out2 = repl.send("/agent default")
    assert "agent:" in out2 and "(default)" in out2, out2[:1200]
    # Confirm the *current* agent is now the main one.
    out3 = repl.send("/agent")
    assert "(default)" in out3, out3[:1200]
    _assert_repl_alive(repl, out2)


def test_usage_toggle_off_suppresses_line(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    """`/usage off` disables the per-turn `in=/out=` line, but `/usage` still
    shows the session total (the toggle is display-only, not state)."""
    fake_server.reset(text("hi there"))
    out = repl.send("/usage off")
    assert "per-turn" in out.lower() and "off" in out.lower(), out[:1200]
    # Turn: the REPL must NOT print an in=/out= line even though the backend
    # recorded usage.
    out2 = repl.send("hello again")
    assert "in=" not in out2 or "out=" not in out2, (
        f"per-turn report must be off after /usage off:\n{out2[:1200]}"
    )
    # But /usage still shows the cumulative session total.
    out3 = repl.send("/usage")
    assert "session:" in out3, out3[:1200]
    # Per-turn report state is still "off" after this.
    assert "per-turn" in out3.lower() and "off" in out3.lower(), out3[:1200]


def test_usage_toggle_on_restores_line(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    fake_server.reset(text("hi"), text("hello"))
    repl.send("/usage off")
    # A single turn is now silent about per-turn tokens.
    repl.send("a message")
    # Re-enable; the next turn shows the line again.
    out = repl.send("/usage on")
    assert "per-turn" in out.lower() and "on" in out.lower(), out[:1200]
    out2 = repl.send("another message")
    assert "in=" in out2 and "out=" in out2, (
        f"per-turn report must be back on after /usage on:\n{out2[:1200]}"
    )


def test_per_turn_usage_line_is_on_own_line(fake_server: FakeLLMServer, repl: ReplDriver) -> None:
    """The per-turn usage line must not glue onto the final streamed token."""
    fake_server.reset(text("one two"))
    out = repl.send("say hi")
    # The last token of the answer is "two"; the usage line must be on its own
    # line (not on the same line as "two").
    # We can't parse ANSI cleanly here, but the plain "in=" substring must
    # appear somewhere on a line that is *not* the final "two".
    assert "in=" in out and "out=" in out, (
        f"per-turn line must be rendered (default is on):\n{out[:1200]}"
    )
    # The `in=...out=...` string must appear on a single line that does NOT
    # also contain "two" (the last token of "one two").
    # (The previous bug was this: the final print landed after the last token
    # without a newline, so the line contained e.g. "two in=4 out=6 ...".)
    found_on_same_line = False
    for line in out.splitlines():
        if "in=" in line and "two" in line:
            # This line contains the final token AND the usage — the bug.
            found_on_same_line = True
            break
    assert not found_on_same_line, (
        f"the per-turn line must be on its own line, not glued to the last token:\n{out[:1200]}"
    )


def test_help_output_is_column_aligned(repl: ReplDriver) -> None:
    """Every command row's description must start at the same visual column.

    The old hand-padded string left `/model force <n>` and `/model test <n>`
    two spaces short of `/model <n>`'s descriptions. The new layout pads *every*
    command to the widest one, so all rows' descriptions begin in the same
    column. We assert this against the *REPL's own* `HELP` string: it's the
    exact text passed to `rich.panel.Panel`, so any misalignment there is a
    misalignment on screen (the panel only wraps, it never re-indents).
    """
    from agent_seed.frontends.repl.app import HELP

    # Reference: the short `/help` row.
    help_row = next(ln for ln in HELP.splitlines() if ln.lstrip().startswith("/help"))
    ref = help_row.index("show this help")
    # Each previously-misaligned command (and `/model <n>` as a second reference)
    # must have its description at the same column.
    for cmd in ("/model force <n>", "/model test <n>", "/model <n>"):
        row = next(ln for ln in HELP.splitlines() if cmd in ln)
        idx = row.index(cmd) + len(cmd)
        col = idx
        while col < len(row) and row[col] == " ":
            col += 1
        assert col > idx, f"{cmd} row has no padding gap:\n{row}"
        assert col == ref, (
            f"{cmd} description at column {col} (row: {row!r}), "
            f"expected {ref} (reference: {help_row!r})"
        )
    # The panel itself must still render a /help block that contains the
    # reference row's description text (sanity: the REPL did not regress the
    # help body while adding the new rows).
    out = repl.send("/help")
    assert "show this help" in out, out[:1200]
    assert "/agent default" in out, f"the /agent default row must be in the help:\n{out[:1200]}"
    assert "/usage" in out, out[:1200]
