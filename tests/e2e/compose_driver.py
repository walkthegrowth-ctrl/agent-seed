"""Drive a real ``agent-compose`` under a pseudo-terminal (pty).

The command the user is expected to type is ``uv run agent-compose [<NAME>]
[--root PATH]``. ``agent-compose`` offers a 6-prompt wizard when arguments are
not passed; the wizard's prompts are rendered by ``rich`` (rich's
``Console.input`` reads from ``stdin`` when the tty is attached to a real
terminal, so we need a pty, not a pipe).

The pty / pump / settle / ansi-strip plumbing lives in
:mod:`tests.e2e.pty_common`; this module holds the composer-specific parts:

- launch: ``uv run --no-sync agent-compose --root <tmp>``;
- the wizard's known prompt order (``agent name`` → ``voice / persona`` →
  ``tools`` → ``context`` → ``examples`` → ``sign-off line``);
- the terminal markers: ``Created`` (success) or ``error:`` (failure);
- :meth:`advance` — type one line and settle on the *next* prompt (or the
  terminal marker).

A single background thread pumps the pty master into a shared buffer; tests
only ever read from that buffer (never the fd directly).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .pty_common import (
    PtySession,
    PtyTimeout,
    base_env,
    find_uv,
)


class ComposeDriver(PtySession):
    """A live ``agent-compose`` attached to a pty, driven by the test.

    ``wait_ready`` consumes the startup output up to the first prompt.
    ``advance(line, next_key)`` types one line and settles on the next prompt
    (or on the terminal ``Created``/``error:`` marker when ``next_key`` is
    ``None``).
    """

    # The wizard's known prompt order. The keys are the wizard *slot* name;
    # the values are the exact prefix the prompt starts with (ANSI-stripped);
    # the settle matches "starts with <prefix>", so the trailing default/hint
    # and the style codes do not need to be matched verbatim.
    PROMPTS: dict[str, str] = {
        "name": "agent name ",
        "voice": "voice / persona ",
        "tools": "tools (comma-separated, blank for none)",
        "context": "context block (blank for none)",
        "examples": "examples (role=content pairs, blank for none)",
        "signoff": "sign-off line (blank for none)",
    }

    # The terminal turn ends on one of these substrings (the wizard always ends
    # with either a "Created N file(s) ..." report or an "error: " line).
    MARKER = ("Created", "error:")

    def __init__(
        self,
        project_root: str | Path,
        hooks_root: Path,
        tools_root: Path | None = None,
        rows: int = 50,
        cols: int = 160,
    ) -> None:
        self._hooks_root = Path(hooks_root)
        cmd = [find_uv(), "run", "--no-sync", "agent-compose", "--root", str(hooks_root)]
        env = base_env()
        if tools_root is not None:
            env["TOOLS_ROOT"] = str(tools_root)
        super().__init__(
            cmd=cmd,
            cwd=str(project_root),
            env=env,
            rows=rows,
            cols=cols,
        )

    # -- the wizard ----------------------------------------------------------

    def wait_ready(self, timeout: float = 25.0) -> str:
        """Consume the startup output up to the first prompt (``agent name``);
        returns the text.

        The wizard's first prompt appears before we type anything, so we drain
        to it to establish a clean baseline for :meth:`advance`.
        """
        out = self.read_until(self.PROMPTS["name"], settle=1.2, timeout=timeout, what="startup")
        if self.PROMPTS["name"] not in out:
            raise PtyTimeout("agent name prompt", timeout, out)
        return out

    def prompt_visible(self, which: str, timeout: float = 30.0) -> str:
        """Assert a specific wizard prompt is visible; returns the captured
        text so the caller can also assert *what* it says (e.g. the
        tool-name hint on the ``tools`` prompt)."""
        out = self.read_until(
            self.PROMPTS[which], settle=1.2, timeout=timeout, what=f"prompt {which!r}"
        )
        if self.PROMPTS[which] not in out:
            raise PtyTimeout(f"prompt {which!r}", timeout, out)
        return out

    def advance(
        self, line: str, next_key: str | None, settle: float = 1.2, timeout: float = 30.0
    ) -> str:
        """Type one line (Enter) and settle on the *next* wizard prompt.

        ``next_key`` is one of ``PROMPTS`` keys, or ``None`` to settle on the
        terminal turn (``Created`` or ``error:``).
        """
        marker = self.MARKER if next_key is None else self.PROMPTS[next_key]
        return self.type_to(line, marker, settle=settle, timeout=timeout)

    # -- inspection ----------------------------------------------------------

    def created_dir(self, name: str) -> Path:
        """``<hooks_root>/<name>`` — the agent folder the wizard would have
        written (the driver's ``--root`` argument is ``hooks_root``)."""
        return self._hooks_root / name

    @property
    def hooks_root(self) -> Path:
        return self._hooks_root

    # -- teardown ------------------------------------------------------------

    def close(self, timeout: float = 15.0) -> int:
        """Reap the child; returns its return code.

        The wizard exits on its own after the last turn (success or failure),
        so no explicit /quit is needed.
        """
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            return self.proc.wait(timeout=5)
