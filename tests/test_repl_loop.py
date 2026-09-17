from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from prompt_toolkit.history import FileHistory

from agent_seed.frontends.repl import app as app_module


class _FakeSession:
    """Records the message strings Repl._turn would POST to the backend."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def chat_stream(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self.sent.append(message)
        # One token event so Repl._turn's "agent> " branch runs harmlessly.
        yield "token", {"text": "[response]"}
        yield "done", {}

    def close(self) -> None:
        pass


def _run_with_inputs(
    inputs: Sequence[Any],
    history: Any = None,
) -> tuple[_FakeSession, list[str]]:
    """Drive Repl._loop with a scripted sequence of prompt() results.

    ``inputs`` items are either strings (a user line) or an Exception class
    (prompt raises an instance of it). The loop must always exit, so the
    harness guarantees an EOFError once the script is exhausted.
    """
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
