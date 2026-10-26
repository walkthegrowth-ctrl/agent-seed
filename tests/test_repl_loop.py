from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from prompt_toolkit.history import FileHistory

from agent_seed.frontends.repl import app as app_module


class _FakeSession:
    """Records the message strings Repl._turn would POST to the backend.

    Also implements the model-related surface used by ``/model`` commands:
    ``model`` (returns the current config), ``test_model`` (probe), and
    ``set_model`` (commit). ``_model_probe`` / ``_model_set`` call these.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.model_calls: list[str] = []
        self.test_model_calls: list[tuple[str, str | None]] = []
        self.set_model_calls: list[tuple[str, str | None]] = []
        # usage accumulates as turns complete (populated by Repl._turn on done)
        self.usage_session: dict[str, Any] = {}
        # configurable probe outcome
        self.probe_ok: bool = True
        self.probe_error: str | None = None

    def chat_stream(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self.sent.append(message)
        yield "token", {"text": "[response]"}
        yield (
            "done",
            {
                "session_id": "s",
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                "usage_session": {"prompt_tokens": 8, "completion_tokens": 11, "total_tokens": 19},
            },
        )

    def model(self) -> dict[str, Any] | None:
        self.model_calls.append("current")
        return {"llm": {"model": "initial-model"}}

    def test_model(self, model: str, base_url: str | None = None) -> dict[str, Any]:
        self.test_model_calls.append((model, base_url))
        if self.probe_ok:
            return {"ok": True, "model": model, "latency_ms": 1.0, "error": None}
        return {"ok": False, "model": model, "error": self.probe_error or "boom"}

    def set_model(self, model: str, base_url: str | None = None) -> dict[str, Any]:
        self.set_model_calls.append((model, base_url))
        return {"ok": True, "llm": {"model": model}}

    def close(self) -> None:
        pass


def _run_with_inputs(
    inputs: Sequence[Any],
    history: Any = None,
    session: _FakeSession | None = None,
) -> tuple[_FakeSession, list[str]]:
    """Drive Repl._loop with a scripted sequence of prompt() results.

    ``inputs`` items are either strings (a user line) or an Exception class
    (prompt raises an instance). The loop must always exit, so the
    harness guarantees an EOFError once the script is exhausted.
    """
    if session is None:
        session = _FakeSession()
    it = iter(list(inputs))
    prints: list[str] = []

    def fake_print(*args: Any, **kw: Any) -> None:
        if args:
            prints.append(str(args[0]))

    def fake_prompt(message: Any, history: Any, **kw: Any) -> str:
        try:
            item = next(it)
        except StopIteration:
            raise EOFError from None
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        return item

    repl = app_module.Repl("http://example.invalid", use_history=False)
    with pytest.MonkeyPatch().context() as m:
        m.setattr(app_module, "prompt", fake_prompt)
        m.setattr(app_module.CONSOLE, "print", fake_print)
        return_code = repl._loop(session, history)

    assert return_code == 0
    return session, prints


def test_typed_slash_alone_opens_block_then_sends_joined_body() -> None:
    session, _ = _run_with_inputs(
        ["/", "line one", "  indented line  ", "", "line four", "/", "next", "/quit"]
    )
    assert session.sent == [
        "line one\n  indented line  \n\nline four",
        "next",
    ]


def test_empty_block_is_silently_dropped() -> None:
    session, prints = _run_with_inputs(["/", "/", "turn-after-empty-block", "/quit"])
    assert session.sent == ["turn-after-empty-block"]
    assert not any("empty" in t.lower() for t in prints)


def test_slash_commands_inside_block_are_content() -> None:
    session, _ = _run_with_inputs(["/", "/help", "/", "single line", "/quit"])
    assert session.sent == ["/help", "single line"]


def test_trailing_whitespace_around_close_slash_is_forgiving() -> None:
    session, _ = _run_with_inputs(["/", "hello", "   /   ", "then-this", "/quit"])
    assert session.sent == ["hello", "then-this"]


def test_block_with_only_whitespace_lines_drops_silently() -> None:
    session, _ = _run_with_inputs(["/", "   \t ", "", "   ", "/", "next", "/quit"])
    assert session.sent == ["next"]


def test_ctrl_c_in_block_discards_draft_and_stays_in_normal() -> None:
    session, prints = _run_with_inputs(
        ["/", "draft line", KeyboardInterrupt, "recovered-message", "/quit"]
    )
    assert session.sent == ["recovered-message"]
    assert any("draft discarded" in t for t in prints)


def test_ctrl_c_in_normal_mode_exits() -> None:
    session, prints = _run_with_inputs(["hi", KeyboardInterrupt])
    assert session.sent == ["hi"]
    assert any("bye" in t for t in prints)


def test_eof_in_block_exits_without_sending() -> None:
    session, prints = _run_with_inputs(["/", "draft", EOFError])
    assert session.sent == []
    assert any("bye" in t for t in prints)


def test_blank_line_in_normal_mode_is_ignored() -> None:
    session, _ = _run_with_inputs(["", "   ", "real", "/quit"])
    assert session.sent == ["real"]


def test_unknown_slash_command_prints_error_and_stays_in_normal() -> None:
    session, prints = _run_with_inputs(["/nope", "still-here", "/quit"])
    assert session.sent == ["still-here"]
    assert any("unknown command" in t for t in prints)


def test_joined_block_is_written_to_file_history_once() -> None:
    import os

    tmp = "/tmp/opencode/repl_hist_test.txt"
    if os.path.exists(tmp):
        os.remove(tmp)
    hist = FileHistory(tmp)
    try:
        session, _ = _run_with_inputs(["/", "a", "b", "c", "/", "single", "/quit"], history=hist)
        assert session.sent == ["a\nb\nc", "single"]
        stored = list(hist.load_history_strings())
        assert stored == ["a\nb\nc"]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# -- /model command routes ----------------------------------------------------


def _mk_session(**kw: Any) -> _FakeSession:
    session = _FakeSession()
    for key, value in kw.items():
        setattr(session, key, value)
    session.agent_name = None
    return session


def test_model_no_arg_shows_current_config() -> None:
    session = _mk_session()
    _, _ = _run_with_inputs(["/model", "/quit"], session=session)
    assert session.model_calls == ["current"]
    assert session.set_model_calls == []


def test_model_test_only_probes_no_set() -> None:
    session = _mk_session()
    _run_with_inputs(["/model test qwen3.8:27b", "/quit"], session=session)
    assert session.test_model_calls == [("qwen3.8:27b", None)]
    assert session.set_model_calls == []


def test_model_plain_probes_then_sets_when_probe_ok() -> None:
    session = _mk_session()
    _run_with_inputs(["/model qwen3.8:27b", "/quit"], session=session)
    assert session.test_model_calls == [("qwen3.8:27b", None)]
    assert session.set_model_calls == [("qwen3.8:27b", None)]


def test_model_plain_aborts_when_probe_fails_keeps_model() -> None:
    session = _mk_session(probe_ok=False, probe_error="connection refused")
    _, prints = _run_with_inputs(["/model qwen3.8:27b", "/quit"], session=session)
    assert session.test_model_calls == [("qwen3.8:27b", None)]
    assert session.set_model_calls == []
    assert any("not set" in t for t in prints)
    assert any("qwen3.8:27b" in t for t in prints)  # the abort message mentions it


def test_model_force_sets_without_probe() -> None:
    session = _mk_session()
    _run_with_inputs(["/model force qwen3.8:27b http://x/v1", "/quit"], session=session)
    # force must NOT probe at all
    assert session.test_model_calls == []
    assert session.set_model_calls == [("qwen3.8:27b", "http://x/v1")]


def test_model_usage_errors_remain_in_normal_mode() -> None:
    # /model test with no name -> prints usage, stays alive
    session = _mk_session()
    session2, prints = _run_with_inputs(["/model test", "still-here", "/quit"], session=session)
    assert session2.set_model_calls == []
    assert session2.test_model_calls == []
    assert any("usage" in t.lower() for t in prints)


# -- /usage command -----------------------------------------------------------


def test_usage_shows_session_total_after_turns() -> None:
    session = _mk_session()
    session, prints = _run_with_inputs(["hi there", "/usage", "/quit"], session=session)
    # the turn's done event populated the session mirror
    assert session.usage_session["prompt_tokens"] == 8
    # the per-turn line was rendered
    assert any("in=4" in t and "out=6" in t for t in prints)
    # /usage prints the session total
    assert any("session:" in t and "in=8" in t for t in prints)


def test_usage_with_no_turns_reports_empty() -> None:
    session = _mk_session()
    session, prints = _run_with_inputs(["/usage", "/quit"], session=session)
    assert not session.usage_session
    assert any("no usage recorded" in t for t in prints)


def test_reset_clears_session_usage_mirror() -> None:
    session = _mk_session()

    # seed a real reset on the fake
    def fake_reset() -> None:
        session.usage_session = {}

    session.reset = fake_reset  # type: ignore[method-assign]
    session.usage_session = {"prompt_tokens": 8, "completion_tokens": 11}
    session, prints = _run_with_inputs(["a", "/reset", "/usage", "/quit"], session=session)
    assert session.usage_session == {}
    assert any("no usage recorded" in t for t in prints)
