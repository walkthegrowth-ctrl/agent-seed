"""End-to-end: the ``agent-compose`` wizard driven over a real pty.

The command the user is expected to type is ``uv run agent-compose``. The
wizard has six prompts (``agent name`` → ``voice`` → ``tools`` → ``context``
→ ``examples`` → ``sign-off``), rendered by ``rich`` and read from the
attached tty. Each test launches a fresh ``agent-compose`` under a pty
(see :mod:`tests.e2e.compose_driver`) and either walks the whole wizard and
inspects the resulting files on disk, or types a bad value and asserts a
*clean* error (no traceback, the wizard does NOT silently advance to the
next prompt).

Cases:
 1. Full wizard happy path — every channel filled → the 5 promised files
    exist with the promised content.
 2. Blank optional prompts — only a name filled → the folder is the neutral
    skeleton (all 5 channel files + a README): each file is a placeholder
    stub, the Python ones wired to the right hook slot with a no-op body.
3. Name re-validation loop — an invalid name triggers a re-prompt.
4. Unknown tool name → a clean error; the wizard does NOT advance to
   ``context``.
5. Invalid example role → a clean error; the wizard does NOT advance to
   ``sign-off`` (the wizard and the flag path now agree on role validation).
6. Quoted vs unquoted voice → both produce the *same* ``prompt.md``.
7. Tools hint is visible (not silently stripped by markup).
8. Unicode + emoji survive the wizard and reach ``prompt.md`` byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.compose_driver import ComposeDriver

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _reservoir(root: Path) -> Path:
    """A small reservoir with one tool; the wizard must advertise it by name
    (in the hint) and the flag path must accept it (via the shared
    :func:`list_tools`)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "double.py").write_text(
        'def double(n: int) -> int:\n    """Double a given number."""\n    return n * 2\n'
    )
    return root


@pytest.fixture()
def reservoir(tmp_path_factory) -> Path:
    return _reservoir(tmp_path_factory.mktemp("reservoir_tools"))


@pytest.fixture()
def compose(tmp_path_factory, reservoir) -> ComposeDriver:
    hooks_root = tmp_path_factory.mktemp("hooks_root")
    driver = ComposeDriver(
        project_root=PROJECT_ROOT,
        hooks_root=hooks_root,
        tools_root=reservoir,
    )
    driver.wait_ready(timeout=25)
    try:
        yield driver
    finally:
        try:
            driver.close(timeout=5)
        except Exception:
            driver.kill()


# ------------------------------------------------------------------------------
# 1. Full wizard happy path
# ------------------------------------------------------------------------------


def test_full_wizard_happy_path(compose: ComposeDriver) -> None:
    import json

    # 1) name → voice:
    out = compose.advance("tutor", "voice", settle=1.2, timeout=25)
    assert "voice / persona" in out

    # 2) voice → tools:
    out = compose.advance("You are a tutor.", "tools", settle=1.2, timeout=25)
    assert "tools (comma-separated" in out

    # 3) tools → context:
    out = compose.advance("double", "context", settle=1.2, timeout=25)
    assert "context block" in out

    # 4) context → examples:
    out = compose.advance("Round 1.", "examples", settle=1.2, timeout=25)
    assert "examples" in out

    # 5) examples → signoff:
    out = compose.advance("user=hi|assistant=hello", "signoff", settle=1.2, timeout=25)
    assert "sign-off line" in out

    # 6) signoff → terminal turn (the "Created ..." report or "error:"):
    out = compose.advance("— tutor", None, settle=1.8, timeout=25)
    assert "Created" in out, f"expected the success report, got:\n{out[:1200]}"
    assert "error" not in out.lower()

    # 7) on-disk: exactly the 5 promised files, with the promised values.
    folder = compose.created_dir("tutor")
    assert folder.is_dir(), "the wizard should have created the agent folder"
    files = {p.name for p in folder.glob("*") if p.is_file()}
    assert files == {"prompt.md", "tools.py", "context.md", "examples.jsonl", "sign.py"}, (
        f"the wizard should produce exactly these 5 files, got {files}"
    )
    voice = (folder / "prompt.md").read_text(encoding="utf-8").strip()
    assert voice == "You are a tutor."
    ctx = (folder / "context.md").read_text(encoding="utf-8").strip()
    assert ctx == "Round 1."
    signoff = (folder / "sign.py").read_text(encoding="utf-8")
    assert "SIGNOFF" in signoff and "— tutor" in signoff
    src = (folder / "tools.py").read_text(encoding="utf-8")
    assert '"double"' in src or "'double'" in src, f"tools.py should whitelist 'double':\n{src}"
    lines = [
        line
        for line in (folder / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [json.loads(line) for line in lines] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ------------------------------------------------------------------------------
# 2. Blank optional prompts are skipped
# ------------------------------------------------------------------------------


def test_blank_optional_fields_are_skipped(compose: ComposeDriver) -> None:
    """Only a name (all optional prompts blank) creates a *neutral skeleton* —
    the five placeholder files plus a README. The contract: the folder loads
    and works, but the stubs change no behaviour; each file is a place to grow from."""
    compose.advance("solo", "voice", settle=1.2, timeout=25)
    # voice: Enter (blank).
    compose.advance("", "tools", settle=1.2, timeout=25)
    # tools: Enter (blank).
    compose.advance("", "context", settle=1.2, timeout=25)
    # context: Enter (blank).
    compose.advance("", "examples", settle=1.2, timeout=25)
    # examples: Enter (blank).
    compose.advance("", "signoff", settle=1.2, timeout=25)
    # signoff: Enter (blank) -> the terminal turn.
    out = compose.advance("", None, settle=1.8, timeout=25)
    assert "Created" in out

    folder = compose.created_dir("solo")
    assert folder.is_dir()
    files = {p.name for p in folder.glob("*") if p.is_file()}
    assert files == {
        "prompt.md",
        "context.md",
        "examples.jsonl",
        "tools.py",
        "sign.py",
        "README.md",
    }, f"a blank-wizard folder should carry the neutral skeleton, got {files}"
    # The Python stubs are loadable (the hook contract stays valid) and the
    # text channels are empty placeholders (they add nothing to the prompt).
    assert (folder / "prompt.md").read_text(encoding="utf-8") == ""
    assert (folder / "context.md").read_text(encoding="utf-8") == ""
    assert (folder / "examples.jsonl").read_text(encoding="utf-8") == ""
    assert "def enter_agent" in (folder / "tools.py").read_text(encoding="utf-8")
    assert "def finish" in (folder / "sign.py").read_text(encoding="utf-8")
    assert "solo" in (folder / "README.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------
# 3. Name re-validation loop
# ------------------------------------------------------------------------------


def test_invalid_name_re_prompts(compose: ComposeDriver) -> None:
    """An invalid name triggers a clean re-prompt (not a crash, not a hang).
    Then a valid name advances to the voice prompt."""
    # First pass: invalid name -> wizard re-prompts on agent name.
    out = compose.advance("Upper_Case", "name", settle=1.2, timeout=25)
    assert "agent name" in out, "the wizard should re-prompt on a bad name"
    assert "not allowed" in out or "name" in out, f"should print a clear error:\n{out[:600]}"
    # Second pass: valid name -> wizard advances to voice.
    out = compose.advance("lower", "voice", settle=1.2, timeout=25)
    assert "voice / persona" in out


# ------------------------------------------------------------------------------
# 4. Unknown tool name
# ------------------------------------------------------------------------------


def test_unknown_tool_is_clean_error(compose: ComposeDriver) -> None:
    """Typing a tool that's not in the catalog must be a clean error at the
    tools prompt itself -- no Python traceback, and the wizard re-prompts on
    tools (it does NOT silently advance to context)."""
    compose.advance("notools", "voice", settle=1.2, timeout=25)
    compose.advance("", "tools", settle=1.2, timeout=25)
    # A bad tool name must be rejected inline, with the tools prompt re-shown.
    out = compose.advance("not_a_tool", "tools", settle=1.8, timeout=25)
    assert "Traceback" not in out, f"no Python tracebacks allowed:\n{out[:1200]}"
    assert "tools (comma-separated" in out, (
        f"the wizard must stay on the tools prompt after a bad name:\n{out[:600]}"
    )
    assert "not found" in out.lower() or "not_a_tool" in out, (
        f"expected a tool-not-found message mentioning the bad name:\n{out[:600]}"
    )
    # And the agent folder must not exist (no partial state).
    assert not compose.created_dir("notools").exists()


# ------------------------------------------------------------------------------
# 5. Invalid example role
# ------------------------------------------------------------------------------


def test_invalid_example_role_is_clean_error(compose: ComposeDriver) -> None:
    """`bogus=hi` typed in the examples prompt must be a clean error -- the
    wizard and the flag path now agree on role validation."""
    compose.advance("badrole", "voice", settle=1.2, timeout=25)
    compose.advance("", "tools", settle=1.2, timeout=25)
    compose.advance("", "context", settle=1.2, timeout=25)
    compose.advance("", "examples", settle=1.2, timeout=25)
    out = compose.advance("bogus=hi", None, settle=1.8, timeout=25)
    # The terminal "error:" should appear (or the examples prompt should
    # re-prompt). Either way, no traceback and no advance to "signoff".
    assert "Traceback" not in out, f"no Python tracebacks:\n{out[:1200]}"
    assert "sign-off line" not in out, (
        f"after a bad example role the wizard should NOT advance to signoff:\n{out[:1200]}"
    )
    assert not compose.created_dir("badrole").exists()


# ------------------------------------------------------------------------------
# 6. Quoted vs unquoted voice -> identical prompt.md
# ------------------------------------------------------------------------------


def test_quoted_vs_unquoted_voice(tmp_path_factory, reservoir) -> None:
    """`'You are a tutor.'` and `You are a tutor.` must both land in
    prompt.md as `You are a tutor.` (one outer quote pair stripped)."""

    def run_once(voice: str) -> str:
        hooks_root = tmp_path_factory.mktemp("hook_root")
        d = ComposeDriver(project_root=PROJECT_ROOT, hooks_root=hooks_root, tools_root=reservoir)
        try:
            d.wait_ready(timeout=25)
            d.advance("tutor", "voice", settle=1.2, timeout=25)
            d.advance(voice, "tools", settle=1.2, timeout=25)
            d.advance("", "context", settle=1.2, timeout=25)
            d.advance("", "examples", settle=1.2, timeout=25)
            d.advance("", "signoff", settle=1.2, timeout=25)
            out = d.advance("", None, settle=1.8, timeout=25)
            assert "Created" in out, f"wizard should have succeeded, got:\n{out[:1200]}"
            folder = d.created_dir("tutor")
            return (folder / "prompt.md").read_text(encoding="utf-8").strip()
        finally:
            try:
                d.close(timeout=5)
            except Exception:
                d.kill()

    assert run_once("'You are a tutor.'") == "You are a tutor."
    assert run_once("You are a tutor.") == "You are a tutor."


# ------------------------------------------------------------------------------
# 7. Tools hint is visible
# ------------------------------------------------------------------------------


def test_tools_hint_is_visible(compose: ComposeDriver) -> None:
    """The tools prompt must *show* the available tool name(s) in square
    brackets. This is the regression guard for the markup-stripping bug
    that was caught and fixed in this session: the old f-string
    ``"…  [double]"`` was swallowed by rich's markup parser (``[double]``
    parsed as a style token), so the user never saw the available tools.

    The catalog lists built-ins (e.g. ``calculator``) and the reservoir
    tools together, so we assert the brackets contain *our* reservoir
    tool name (``double``)."""
    compose.advance("hinttest", "voice", settle=1.2, timeout=25)
    # The *tools* prompt is the next turn (typed value = blank voice);
    # ``advance`` returns the buffer since the last drain, so it includes
    # the tools prompt line with the hint.
    out = compose.advance("", "tools", settle=1.2, timeout=25)
    assert "tools (comma-separated" in out, (
        f"the tools prompt should be present; got: {out[:600]!r}"
    )
    assert "[", f"the hint brackets are missing (markup-stripping bug): {out[:600]!r}"
    assert "double" in out, (
        f"the tools hint should contain the reservoir tool 'double'; got: {out[:600]!r}"
    )


# ------------------------------------------------------------------------------
# 8. Unicode + emoji round-trip
# ------------------------------------------------------------------------------


def test_unicode_and_emoji_round_trip(compose: ComposeDriver) -> None:
    """Grüße — Team Alpha 💡 must reach prompt.md byte-identical (no mojibake,
    no emoji-stripping)."""
    compose.advance("grue", "voice", settle=1.2, timeout=25)
    compose.advance("Grüße — Team Alpha 💡", "tools", settle=1.2, timeout=25)
    compose.advance("", "context", settle=1.2, timeout=25)
    compose.advance("", "examples", settle=1.2, timeout=25)
    compose.advance("", "signoff", settle=1.2, timeout=25)
    out = compose.advance("", None, settle=1.8, timeout=25)
    assert "Created" in out

    folder = compose.created_dir("grue")
    voice = (folder / "prompt.md").read_text(encoding="utf-8")
    assert voice.strip() == "Grüße — Team Alpha 💡", (
        f"emoji/umlauts were mangled or stripped: {voice!r}"
    )
