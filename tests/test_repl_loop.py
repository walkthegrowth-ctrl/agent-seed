from collections.abc import Iterator, Sequence
from pathlib import Path
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
        # agents(): the backend-reported discoverable agents.
        self._agents: list[str] = ["alpha", "beta"]
        self.agent_set_calls: list[str | None] = []
        self.agent_name: str | None = None

    def agents(self) -> list[str]:
        return list(self._agents)

    def set_agent(self, name: str | None) -> None:
        self.agent_set_calls.append(name)
        self.agent_name = name

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
    session.repl = repl  # back-reference so tests can inspect Repl state
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


def test_joined_block_is_written_to_file_history_once(tmp_path: Path) -> None:
    # tmp_path is a guaranteed-existing, unique dir (pytest provides it), so
    # FileHistory's open("ab") has a real home on any runner — a hardcoded
    # /tmp/... path would fail on a clean CI box that lacks that directory.
    hist = FileHistory(str(tmp_path / "repl_hist_test.txt"))
    session, _ = _run_with_inputs(["/", "a", "b", "c", "/", "single", "/quit"], history=hist)
    assert session.sent == ["a\nb\nc", "single"]
    # only the joined block is written to the file history, once
    stored = list(hist.load_history_strings())
    assert stored == ["a\nb\nc"]


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


# -- /agent command -----------------------------------------------------------


def test_agent_no_arg_shows_current() -> None:
    session = _mk_session()
    session.agent_name = "alpha"
    _run_with_inputs(["/agent", "/quit"], session=session)
    # No switch committed just by displaying the current agent.
    assert session.agent_set_calls == []
    assert session.agent_name == "alpha"


def test_agent_selects_a_known_agent() -> None:
    session = _mk_session()
    _run_with_inputs(["/agent alpha", "/quit"], session=session)
    assert session.agent_set_calls == ["alpha"]
    assert session.agent_name == "alpha"


def test_agent_rejects_unknown_name() -> None:
    # `gamma` is not in the fake's discoverable agents -> rejected, no switch.
    session = _mk_session()
    session, prints = _run_with_inputs(["/agent gamma", "/quit"], session=session)
    assert session.agent_set_calls == []
    assert session.agent_name is None
    assert any("unknown agent" in t for t in prints)


def test_agent_default_returns_to_main() -> None:
    session = _mk_session()
    session.agent_name = "alpha"
    session, prints = _run_with_inputs(["/agent default", "/quit"], session=session)
    # "default" is the reserved main agent -> set to None, even though there is
    # no discoverable folder named "default".
    assert session.agent_set_calls == [None]
    assert session.agent_name is None
    assert any("(default)" in t for t in prints)


def test_agent_default_allowed_even_when_unknown_to_backend() -> None:
    # Even with no backend agents at all, "default" must still select the main
    # agent (it is special, not a discoverable name).
    session = _mk_session()
    session._agents = []
    _run_with_inputs(["/agent default", "/quit"], session=session)
    assert session.agent_set_calls == [None]


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


# -- /usage on|off|toggle (the per-turn report) --------------------------------


def test_usage_off_suppresses_the_per_turn_line() -> None:
    # The per-turn line is off after `/usage off`, but the session's total can
    # still be inspected with a bare `/usage`.
    session, prints = _run_with_inputs(
        ["/usage off", "hi there", "/usage", "/quit"], session=_mk_session()
    )
    # off was applied
    assert session.repl.show_usage is False
    # the turn's done event still updated the session mirror (backend state
    # is independent of the display toggle)
    assert session.usage_session.get("prompt_tokens") == 8
    # but the per-turn in=/out= line did NOT appear
    assert not any("in=4" in t and "out=6" in t for t in prints)
    # the command reported its new state
    assert any("per-turn" in t.lower() and "off" in t.lower() for t in prints)


def test_usage_toggle_flips_the_state() -> None:
    # Within one REPL (one Repl instance), on -> off -> on.
    session, prints = _run_with_inputs(
        ["/usage toggle", "/usage toggle", "/quit"], session=_mk_session()
    )
    # default is on; two toggles bring it back to on.
    assert session.repl.show_usage is True
    # the second (off→on) toggle reported `on`.
    assert any("per-turn" in t.lower() and "on" in t.lower() for t in prints)


def test_usage_toggle_from_off_back_to_on_allows_line_again() -> None:
    session, prints = _run_with_inputs(
        ["/usage off", "first", "/usage on", "second", "/quit"],
        session=_mk_session(),
    )
    assert session.repl.show_usage is True
    # `second` should have the per-turn line (report is back on); `first`
    # should not. Both appear in the printed buffer in order — distinguish by
    # counting the per-turn lines rendered (exactly one, for `second`).
    per_turn = [t for t in prints if "in=4" in t and "out=6" in t]
    assert len(per_turn) == 1, prints


def test_usage_unknown_arg_is_clean_error() -> None:
    session, prints = _run_with_inputs(["/usage maybe", "/quit"], session=_mk_session())
    assert session.repl.show_usage is True  # unchanged
    assert any("usage:" in t for t in prints)
