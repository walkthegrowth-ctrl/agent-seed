"""Drive a real ``agent-repl`` under a pseudo-terminal (pty).

The command the user is expected to type is ``uv run agent-repl --url
<server>``, so the driver launches exactly that (via ``uv run``). ``agent-repl``
uses ``prompt_toolkit`` ``prompt()``, which reads from ``/dev/tty`` and therefore
only works with a real terminal — so the REPL subprocess is attached to a pty,
the same mechanism ``expect``/``pexpect`` use. The server subprocess needs no
tty and is launched plainly with pipes.

A single background thread pumps the pty master into a shared buffer; tests only
ever read from that buffer (never the fd directly). Turns end on a "quiet
window" — no new bytes for ``settle`` seconds — a reliable end-of-turn signal
over loopback with the (instant) fake LLM.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Sequence

if sys.platform == "win32":  # pragma: no cover - pty is POSIX-only
    raise RuntimeError("the e2e battery drives agent-repl under a pty (POSIX only)")


class REPLTimeout(TimeoutError):
    def __init__(self, what: str, timeout: float, captured: str) -> None:
        super().__init__(f"REPL produced no turn for {what!r} within {timeout:.0f}s")
        self.captured = captured


_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI sequences
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC sequences
    r"|\x1b[\x20-\x7e]"  # two-byte escapes (ESC + a byte in 0x20..0x7E)
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_ansi(text: str) -> str:
    """Normalize a raw pty capture into readable 'logical' text (for assertions)."""
    text = _ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "")
    text = _CONTROL_RE.sub("", text)
    return text


def _find_uv() -> str:
    uv = os.environ.get("UV_BIN")
    if not uv:
        out = subprocess.run(
            ["which", "uv"], capture_output=True, text=True, check=False
        ).stdout.strip()
        uv = out or None
    if not uv:
        raise RuntimeError("uv is required to run the e2e battery but was not found on PATH")
    return uv


def _base_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "COLUMNS": "160",
        "LINES": "50",
        "TERM": "dumb",  # line-input mode: no CPR request, prompt_toolkit replies instantly
    }


def spawn_server(env: dict[str, str], project_root: str) -> subprocess.Popen:
    """Launch the real backend exactly the way a user would: ``uv run agent-seed``."""
    full_env = _base_env()
    full_env.update(env)
    cmd = [_find_uv(), "run", "--no-sync", "agent-seed"]
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
    import httpx2

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


class ReplDriver:
    """A live ``agent-repl`` attached to a pty, driven by the test.

    ``wait_ready`` consumes the banner up to the first prompt. ``send(line)``
    types one line and returns the ANSI-stripped text of the resulting turn.
    """

    def __init__(
        self, url: str, cwd: str, project_root: str, rows: int = 50, cols: int = 160
    ) -> None:
        self.url = url
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

        cmd = [_find_uv(), "run", "--no-sync", "agent-repl", "--url", url]

        def preexec() -> None:
            os.setsid()
            fcntl.ioctl(self.slave, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=_base_env(),
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            preexec_fn=preexec,
            close_fds=True,
        )
        os.close(self.slave)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._eof = False
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    # -- single reader thread (tests never read the fd directly) ----------------

    def _pump(self) -> None:
        while True:
            r, _, _ = select.select([self.master], [], [], 0.1)
            if r:
                try:
                    data = os.read(self.master, 65536)
                except OSError:
                    break
                if not data:
                    break
                with self._lock:
                    self._buf.extend(data)
            elif self.proc.poll() is not None:
                if select.select([self.master], [], [], 0.1)[0]:
                    continue
                self._eof = True
                break

    def _take(self) -> str:
        with self._lock:
            text = bytes(self._buf).decode("utf-8", "replace")
            self._buf.clear()
        return text

    def _read_until_turn(self, marker: str, settle: float, total_timeout: float, what: str) -> str:
        """Wait for the ``marker`` *and* a stable quiet window; return all text
        captured since the last :meth:`send` (ANSI-stripped).

        A turn ends when the buffer has not changed for ``settle`` seconds AND
        the marker (the REPL's re-printed prompt) is present. If the REPL has
        already exited (e.g. after ``/quit``), settle on the first quiet check
        regardless of the marker — the marker may never re-render."""
        deadline = time.monotonic() + total_timeout
        last_len = -1
        last_change = time.monotonic()
        while True:
            with self._lock:
                current_len = len(self._buf)
                text = bytes(self._buf).decode("utf-8", "replace")
            now = time.monotonic()
            if current_len != last_len:
                last_len = current_len
                last_change = now
            quiet = (now - last_change) >= settle
            dead = self._eof or self.proc.poll() is not None
            if quiet and (dead or marker in text):
                break
            if now >= deadline:
                captured = strip_ansi(text)
                self._take()
                raise REPLTimeout(what, total_timeout, captured)
            time.sleep(0.05)
        return self._take_stripped()

    def _take_stripped(self) -> str:
        with self._lock:
            text = bytes(self._buf).decode("utf-8", "replace")
            self._buf.clear()
        return strip_ansi(text)

    def wait_ready(self, timeout: float = 25.0) -> str:
        """Consume the startup banner up to the first prompt; returns the text."""
        out = self._read_until_turn(
            "you>", settle=1.2, total_timeout=timeout, what="startup banner"
        )
        if "you>" not in out:
            raise REPLTimeout("you> prompt", timeout, out)
        return out

    def send(self, line: str, timeout: float = 30.0, settle: float = 1.2) -> str:
        """Type one line (Enter) and return the ANSI-stripped text of the turn."""
        self._take()
        os.write(self.master, (line + "\r").encode("utf-8"))
        return self._read_until_turn(
            "you>", settle=settle, total_timeout=timeout, what=f"turn for {line!r}"
        )

    def _send_marked(
        self, line: str, marker: str, timeout: float = 30.0, settle: float = 1.2
    ) -> str:
        """Like :meth:`send` but settles on a different marker (e.g. ``...>``)."""
        self._take()
        os.write(self.master, (line + "\r").encode("utf-8"))
        return self._read_until_turn(
            marker, settle=settle, total_timeout=timeout, what=f"turn for {line!r}"
        )

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

    # -- teardown ---------------------------------------------------------------

    def close(self, timeout: float = 15.0) -> int:
        """Send ``/quit`` and reap the process; returns its return code."""
        with contextlib.suppress(REPLTimeout, OSError):
            self.send("/quit", settle=1.2, timeout=timeout)
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill()
            return self.proc.wait(timeout=5)

    def _kill(self) -> None:
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, OSError):
                self.proc.kill()
        with contextlib.suppress(OSError):
            os.close(self.master)
