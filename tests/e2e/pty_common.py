"""POSIX pseudo-terminal helpers shared by the e2e battery.

The binaries the battery drives (``agent-repl`` uses ``prompt_toolkit.prompt``;
``agent-compose`` uses ``rich.prompt.Prompt.ask``) only work under a real
terminal — a pipe is not enough: line editing, history, and the ``isatty``
gates all require a pty. So every binary is launched under a pty, the same
mechanism ``expect``/``pexpect`` uses: open a master/slave pty pair, attach a
subprocess to the slave, pump the master into a shared buffer on a background
thread, and drive the session by writing to the master.

This module is POSIX-only (``pty``, ``fcntl``, ``termios``, ``select``). The
Windows test path is not a supported target for this repo, and the guard at
the top of the module makes it fail fast if someone tries.

The pty / pump / settle / ansi-strip logic is the same regardless of what the
child process is. The REPL-specific parts (``wait_ready`` for the ``you>``
banner, ``send_block`` for ``/``-open blocks, the ``--url`` argument
plumbing) and the composer-specific parts (the wizard's known prompt order,
the ``--root`` argument plumbing) are left to :class:`ReplDriver` and
:class:`ComposeDriver` respectively, which both inherit from
:class:`PtySession`.

The public contract is:

- :class:`PtySession` — the one live pty-driven subprocess.
- :func:`strip_ansi` — normalize a raw pty capture into readable text for
  assertions.
- :class:`PtyTimeout` — the "no turn within ``N`` seconds" failure.
- :func:`find_uv` / :func:`base_env` — the subprocess launcher.

Tests (``tests/test_e2e_repl.py`` and ``tests/test_composer_interactive.py``)
only ever read from the shared buffer, never the fd directly. Turns end on a
quiet window + a marker — the same end-of-turn signal the REPL's ``you>``
prompt has always provided, but generalized so a composer wizard (which
prompts in a known fixed order and exits on completion) can drive the same
plumbing with a different marker strategy.
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

if sys.platform == "win32":  # pragma: no cover - pty is POSIX-only
    raise RuntimeError("the e2e battery drives the binaries under a pty (POSIX only)")


ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI sequences
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC sequences
    r"|\x1b[\x20-\x7e]"  # two-byte escapes (ESC + a byte in 0x20..0x7E)
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class PtyTimeout(TimeoutError):
    """The child produced no turn for the named ``what`` within ``timeout`` seconds."""

    def __init__(self, what: str, timeout: float, captured: str) -> None:
        super().__init__(f"{what!r} produced no turn within {timeout:.0f}s")
        self.captured = captured


def strip_ansi(text: str) -> str:
    """Normalize a raw pty capture into readable "logical" text (for assertions).

    Drops CSI / OSC / two-byte escape sequences, collapses ``\\r\\n`` and bare
    ``\\r`` to ``\\n``, and removes the control bytes a terminal driver emits
    (``\\x07`` the bell, ``\\x0b`` vertical tab, ``\\x0c`` form-feed) that would
    otherwise leak into equality assertions.
    """
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "")
    text = CONTROL_RE.sub("", text)
    return text


def find_uv() -> str:
    """Resolve the ``uv`` binary for the battery.

    Honors the ``$UV_BIN`` override; otherwise looks it up on ``PATH``. Raises
    :class:`RuntimeError` if not found — the battery is a hard dependency on
    ``uv`` (the user runs ``uv run`` themselves, so the suite must do the same
    to be faithful to their environment).
    """
    uv = os.environ.get("UV_BIN")
    if not uv:
        out = subprocess.run(
            ["which", "uv"], capture_output=True, text=True, check=False
        ).stdout.strip()
        uv = out or None
    if not uv:
        raise RuntimeError("uv is required to run the e2e battery but was not found on PATH")
    return uv


def base_env() -> dict[str, str]:
    """A hermetic env for the pty child.

    Inherits the user's ``PATH`` and ``HOME`` (so the ``uv run`` resolves to
    the project's venv and the user's ``~/.config`` etc.); pins the locale so
    unicode output is stable; gives a wide-enough ``COLUMNS`` that ``rich``
    doesn't wrap prompts; and sets ``TERM=dumb`` so line-input mode behaves
    like plain ``readline`` (no CPR request).
    """
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "COLUMNS": "160",
        "LINES": "50",
        "TERM": "dumb",  # line-input mode: no CPR request, prompt replies instantly
    }


class PtySession:
    """A live subprocess attached to a pty, driven by the caller.

    The pump thread is the *only* reader of the pty master fd; callers only
    ever read from ``read_until`` / :meth:`type`, which pull from a shared
    buffer under a lock. A "turn" ends when the buffer has been quiet for
    ``settle`` seconds AND a marker is present — or immediately if the
    process has already exited (the marker may never re-render after a clean
    shutdown, so we settle on the first quiet check in that case).

    The session is POSIX-only; the top-of-module guard above makes the
    :mod:`ptysession` import fail fast on unsupported platforms.
    """

    def __init__(
        self,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        rows: int = 50,
        cols: int = 160,
    ) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.env = env

        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

        def preexec() -> None:
            os.setsid()
            fcntl.ioctl(self.slave, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
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

    # -- pump ---------------------------------------------------------------

    def _pump(self) -> None:
        """Read the pty master into a shared buffer until the child exits.

        Single reader thread; :meth:`type`, :meth:`read_until` and
        :meth:`close` all pull from the buffer under a lock. This is the
        same shape the REPL battery always used.
        """
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

    def _take_stripped(self) -> str:
        with self._lock:
            text = bytes(self._buf).decode("utf-8", "replace")
            self._buf.clear()
        return strip_ansi(text)

    # -- settle -------------------------------------------------------------

    def read_until(
        self,
        marker: str | tuple[str, ...],
        settle: float = 1.2,
        timeout: float = 30.0,
        what: str = "turn",
    ) -> str:
        """Wait for the ``marker`` *and* a stable quiet window; return the
        ANSI-stripped text captured since the previous drain.

        ``marker`` may be a single string or a tuple — when a tuple, the
        settle fires as soon as *any* of the markers is present. This is
        how the composer driver settles on either ``Created`` (success) or
        ``error:`` (failure path) after the final wizard prompt.

        A turn ends when the buffer has not changed for ``settle`` seconds AND
        the marker is present. If the child has already exited (``_eof`` or
        ``proc.poll() is not None``), settle on the first quiet check regardless
        of the marker — the marker may never re-render after a clean shutdown
        (e.g. after the REPL's ``/quit`` or the composer's final file write).

        Raises :class:`PtyTimeout` with whatever has been captured so far if
        the deadline is reached first.
        """
        deadline = time.monotonic() + timeout
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
            markers = (marker,) if isinstance(marker, str) else marker
            quiet = (now - last_change) >= settle
            dead = self._eof or self.proc.poll() is not None
            if quiet and (dead or any(m in text for m in markers)):
                break
            if now >= deadline:
                captured = strip_ansi(text)
                self._take()
                raise PtyTimeout(what, timeout, captured)
            time.sleep(0.05)
        return self._take_stripped()

    # -- drive --------------------------------------------------------------

    def type(self, line: str, settle: float, timeout: float) -> str:
        """Type one line (Enter) and return the ANSI-stripped text of the turn.

        Drains any prior output, writes ``line + \\r`` to the pty master, and
        settles via :meth:`read_until` with the session's default marker
        (overridable by subclasses setting ``self.MARKER``).
        """
        self._take()
        os.write(self.master, (line + "\r").encode("utf-8"))
        return self.read_until(
            self.MARKER, settle=settle, timeout=timeout, what=f"turn for {line!r}"
        )

    def type_to(
        self,
        line: str,
        marker: str | tuple[str, ...],
        settle: float,
        timeout: float,
        what: str | None = None,
    ) -> str:
        """Like :meth:`type` but settles on a different marker (a string *or*
        a tuple — the settle fires as soon as *any* is present).

        Used when a particular line produces a different prompt than the
        session default (e.g. the REPL's block-mode ``...>`` or the composer
        wizard's terminal ``Created``/``error:`` pair).
        """
        self._take()
        os.write(self.master, (line + "\r").encode("utf-8"))
        return self.read_until(
            marker, settle=settle, timeout=timeout, what=what or f"turn for {line!r}"
        )

    # -- teardown -----------------------------------------------------------

    def close(self, timeout: float = 15.0) -> int:
        """Reap the child; returns its return code.

        Best effort: if :meth:`wait` times out, escalate via
        :meth:`kill`. Subclasses that need a graceful shutdown (e.g. ``/quit``)
        should call :meth:`type` before this and then call
        :func:`super().close` (or just wait here).
        """
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            return self.proc.wait(timeout=5)

    def kill(self) -> None:
        """SIGKILL the child's whole process group; close the pty master."""
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, OSError):
                self.proc.kill()
        with contextlib.suppress(OSError):
            os.close(self.master)
