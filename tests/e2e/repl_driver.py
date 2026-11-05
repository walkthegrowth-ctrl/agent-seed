"""Drive a real ``agent-repl`` under a pseudo-terminal (pty).

The command the user is expected to type is ``uv run agent-repl --url
<server>``. ``agent-repl`` uses ``prompt_toolkit.prompt()``, which reads from
``/dev/tty`` and therefore only works with a real terminal — so the REPL
subprocess is attached to a pty (the same mechanism ``expect``/``pexpect``
use). The pty / pump / settle / ansi-strip plumbing lives in
:mod:`tests.e2e.pty_common`; this module holds the REPL-specific parts:

- launch: ``uv run --no-sync agent-repl --url URL``;
- the REPL banner marker (``you>``);
- ``send`` / ``send_block`` for the REPL's prompt modes;
- ``close`` which sends ``/quit`` first for a clean shutdown.

A single background thread pumps the pty master into a shared buffer; tests
only ever read from that buffer (never the fd directly). Turns end on a
"quiet window" — no new bytes for ``settle`` seconds — a reliable
end-of-turn signal over loopback with the (instant) fake LLM.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Sequence

import httpx2

from .pty_common import (
    PtySession,
    PtyTimeout,
    base_env,
    find_uv,
)


def spawn_server(env: dict[str, str], project_root: str) -> subprocess.Popen:
    """Launch the real backend exactly the way a user would: ``uv run agent-seed``.

    (Kept at the module top level of this file because both ``test_e2e_repl``
    and ``test_e2e_real`` import it from here.)
    """
    full_env = base_env()
    full_env.update(env)
    cmd = [find_uv(), "run", "--no-sync", "agent-seed"]
    return subprocess.Popen(
        cmd,
        env=full_env,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def wait_until_healthy(url: str, timeout: float = 20.0) -> None:
    """Poll ``GET /health`` until the backend answers 200 (or time out)."""
    deadline = time.monotonic() + timeout
    last = "server did not start"
    while time.monotonic() < deadline:
        try:
            with httpx2.Client(timeout=2) as client:
                if client.get(f"{url}/health").status_code == 200:
                    return
        except httpx2.HTTPError as exc:
            last = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"backend at {url} not healthy within {timeout:.0f}s: {last}")


class ReplDriver(PtySession):
    """A live ``agent-repl`` attached to a pty, driven by the test.

    ``wait_ready`` consumes the banner up to the first prompt. ``send(line)``
    types one line and returns the ANSI-stripped text of the resulting turn.
    """

    MARKER = "you>"

    def __init__(
        self, url: str, cwd: str, project_root: str, rows: int = 50, cols: int = 160
    ) -> None:
        cmd = [find_uv(), "run", "--no-sync", "agent-repl", "--url", url]
        super().__init__(
            cmd=cmd,
            cwd=project_root,
            env=base_env(),
            rows=rows,
            cols=cols,
        )

    def wait_ready(self, timeout: float = 25.0) -> str:
        """Consume the startup banner up to the first prompt; returns the text."""
        out = self.read_until(self.MARKER, settle=1.2, timeout=timeout, what="startup banner")
        if self.MARKER not in out:
            from .pty_common import PtyTimeout

            raise PtyTimeout("you> prompt", timeout, out)
        return out

    def send(self, line: str, timeout: float = 30.0, settle: float = 1.2) -> str:
        """Type one line (Enter) and return the ANSI-stripped text of the turn."""
        return self.type(line, settle=settle, timeout=timeout)

    def _send_marked(
        self, line: str, marker: str, timeout: float = 30.0, settle: float = 1.2
    ) -> str:
        """Like :meth:`send` but settles on a different marker (e.g. ``...>``)."""
        return self.type_to(line, marker, settle=settle, timeout=timeout)

    def send_block(self, lines: Sequence[str], settle: float = 1.2, timeout: float = 30.0) -> str:
        """Open a multi-line block (a lone ``/``), type each line, then close
        it with another ``/`` (which sends the joined body). Returns the
        answer text for the joined message.

        In block mode the REPL prompts ``...> `` (not ``you> ``), so the
        intermediate steps settle on that marker; the closing ``/`` returns to
        normal mode and thus settles on ``you>``.
        """
        self._send_marked("/", "...>", settle=settle, timeout=timeout)  # open block
        for line in lines:
            self._send_marked(line, "...>", settle=settle, timeout=timeout)
        return self.send("/", settle=settle, timeout=timeout)  # close + send

    # -- teardown -----------------------------------------------------------

    def close(self, timeout: float = 15.0) -> int:
        """Send ``/quit`` and reap the process; returns its return code."""

        with contextlib.suppress(PtyTimeout, OSError):
            self.send("/quit", settle=1.2, timeout=timeout)
        try:
            return self.proc.wait(timeout=timeout)
        except Exception:
            self.kill()
            return self.proc.wait(timeout=5)
