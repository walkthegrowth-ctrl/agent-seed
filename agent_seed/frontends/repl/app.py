"""REPL app: prompt for input, stream tokens, render tool events."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterator
from typing import Any

import httpx2
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel

from ...core import is_number
from ...discovery import RESERVED_AGENT_NAME

CONSOLE = Console()


class BackendError(RuntimeError):
    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"cannot reach backend at {url}: {detail}")


def unquote(value: str | None) -> str | None:
    """Strip one layer of matching surrounding quotes the user typed.

    The REPL tokenizes on whitespace itself, so a user who writes
    ``/model "qwen3.8:27b"`` produces the literal token ``"qwen3.8:27b"``
    (quotes included). Sent verbatim, that becomes an invalid model name on
    the backend (400). A single layer of matching quotes is therefore peeled
    off; anything else — including internal or unbalanced quotes — is kept
    exactly as typed.
    """
    if value is None or len(value) < 2:
        return value
    if value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def sse_events(raw: str) -> Iterator[tuple[str, str]]:
    """Yield (event, data_json_str) pairs from a fully-received SSE text body.

    Buffered variant of :func:`sse_events_streaming` (which parses a live line
    iterator). The REPL uses the streaming form; this is public/reference API
    (e.g. tests, scripts over an HTTP response that is already in memory).
    """
    event = "message"
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
    if data_lines:
        yield event, "\n".join(data_lines)


def sse_events_streaming(
    lines: Iterator[str],
) -> Iterator[tuple[str, str]]:
    """Yield (event, data) as each SSE event completes (blank line).

    Streaming-aware counterpart to sse_events: works on a live iterator
    of lines (e.g. httpx2 iter_lines()) so events surface as they arrive
    instead of after the whole body has been buffered.
    """
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
    # flush any trailing unterminated event (no closing blank line)
    if data_lines:
        yield event, "\n".join(data_lines)


class Session:
    def __init__(self, url: str, timeout: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session_id = uuid.uuid4().hex
        self.agent_name: str | None = None
        self.usage_session: dict[str, Any] = {}
        self.client = httpx2.Client(timeout=timeout)

    # -- backend calls ------------------------------------------------------

    def reset(self) -> None:
        self.client.post(f"{self.url}/reset", json={"session_id": self.session_id})
        self.session_id = uuid.uuid4().hex
        self.usage_session = {}

    def tools(self) -> list[dict[str, Any]]:
        """Public API: full OpenAI function schemas for the active agent.

        The REPL's ``/tools`` only calls :meth:`tools_names` (the compact list);
        this returns the full ``tools`` list for other clients / UIs.
        """
        params = {"agent_name": self.agent_name} if self.agent_name is not None else None
        response = self.client.get(f"{self.url}/tools", params=params)
        response.raise_for_status()
        return response.json().get("tools", [])

    def tools_names(self) -> list[str]:
        params = {"agent_name": self.agent_name} if self.agent_name is not None else None
        response = self.client.get(f"{self.url}/tools", params=params)
        response.raise_for_status()
        return response.json().get("names", [])

    def agents(self) -> list[str]:
        response = self.client.get(f"{self.url}/agents")
        response.raise_for_status()
        return response.json().get("agents", [])

    def set_agent(self, name: str | None) -> None:
        self.agent_name = name

    def model(self) -> dict[str, Any] | None:
        """This session's LLM config, falling back to the server default.

        /config is part of the documented protocol; the REPL treats it as
        optional: it shows the config only when the backend answers 200
        (older backends without the endpoint simply render nothing).
        """
        try:
            response = self.client.get(
                f"{self.url}/config", params={"session_id": self.session_id}
            )
            if response.status_code == 200:
                return response.json()
        except httpx2.HTTPError:
            pass
        return self._server_config()

    def _server_config(self) -> dict[str, Any] | None:
        try:
            response = self.client.get(f"{self.url}/config")
            if response.status_code == 200:
                return response.json()
        except httpx2.HTTPError:
            pass
        return None

    def set_model(self, model: str, base_url: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"session_id": self.session_id, "model": model}
        if base_url is not None:
            payload["base_url"] = base_url
        response = self.client.post(f"{self.url}/config", json=payload)
        response.raise_for_status()
        return response.json()

    def test_model(self, model: str, base_url: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"session_id": self.session_id, "model": model}
        if base_url is not None:
            payload["base_url"] = base_url
        response = self.client.post(f"{self.url}/config/test", json=payload)
        response.raise_for_status()
        return response.json()

    def chat_stream(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        payload: dict[str, Any] = {"session_id": self.session_id, "message": message}
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        with self.client.stream("POST", f"{self.url}/chat", json=payload) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", "replace")
                raise httpx2.HTTPStatusError(
                    f"backend returned {response.status_code}: {body}",
                    request=response.request,
                    response=response,
                )
            for event, data in sse_events_streaming(response.iter_lines()):
                try:
                    parsed = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    parsed = {"raw": data}
                yield event, parsed

    def close(self) -> None:
        self.client.close()


# -- rich renderers ---------------------------------------------------------


def render_tool_event(console: Console, name: str, arguments: Any, result: Any) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items())
    console.print(f"[bold cyan]⚙ {name}[/] [dim]({args})[/] [green]→[/] [magenta]{result}[/]")


def format_tokens(usage: dict[str, Any] | None) -> str:
    """A compact ``in=N out=M total=T`` string for a usage dict, or a dash.

    ``T`` is taken from ``total_tokens`` if present, otherwise derived as
    ``prompt_tokens + completion_tokens`` if both are present. A truthy non-empty
    dict with non-numeric values is rendered with ``str(usage)`` as a last
    resort; empty/None renders as ``—``.
    """
    if not usage:
        return "—"
    parts: list[str] = []
    for key, label in (("prompt_tokens", "in"), ("completion_tokens", "out")):
        if is_number(usage.get(key)):
            parts.append(f"{label}={int(usage[key])}")
    if is_number(usage.get("total_tokens")):
        parts.append(f"total={int(usage['total_tokens'])}")
    elif all(is_number(usage.get(k)) for k in ("prompt_tokens", "completion_tokens")):
        parts.append(f"total={int(usage['prompt_tokens']) + int(usage['completion_tokens'])}")
    return " ".join(parts) if parts else f"{usage}"


# -- REPL main loop ----------------------------------------------------------


def _agent_label(session: Session) -> str:
    """The on-screen agent label: the name, or ``(default)`` for the main agent."""
    return session.agent_name or "(default)"


def _help_layout() -> str:
    """The ``/help`` text, with command names padded so descriptions align.

    Every command row is built from ``(cmd, description)`` and padded to a
    common column (max command width + 3), so adding / renaming a command
    can't drift the alignment — the old hand-padded string left
    ``/model force <n>`` and ``/model test <n>`` two short against
    ``/model <n>``.
    """
    commands = [
        ("/help", "show this help"),
        ("/reset", "start a new session (clears server-side history)"),
        ("/agent [n]", "set the agent (a hooks subdirectory); no arg shows the current"),
        ("/agent default", "switch back to the main agent (the hooks root)"),
        ("/agents", "list the discoverable agent names on the backend"),
        ("/tools", "list the tools registered on the backend"),
        ("/usage", "session total; `on`/`off`/`toggle` for the per-turn report"),
        ("/model", "show the session's LLM config"),
        ("/model <n>", "switch the model (probes first; aborts if the model is unreachable)"),
        ("/model force <n>", "switch the model without a probe"),
        ("/model test <n>", "probe a model without switching"),
        ("/quit", "exit the REPL"),
        ("/", "(alone) open a multi-line block — see below"),
    ]
    width = max(len(cmd) for cmd, _ in commands)
    body = "\n".join(f"  {cmd:<{width + 3}}{desc}" for cmd, desc in commands)
    return (
        "[bold]commands[/]\n"
        + body
        + "\n\n[bold]multi-line mode[/]\n"
        + (
            "  Type [bold]/[/] and press Enter to open a block. Then type as many\n"
            "  lines as you want (blank lines are kept, slash commands are treated\n"
            "  as plain text). Close and send by typing [bold]/[/] alone on a line\n"
            "  and pressing Enter. The two / delimiters are stripped from the\n"
            "  message; the text between them is sent as-is. Ctrl-C cancels the\n"
            "  draft and returns to normal mode; Ctrl-D exits the REPL.\n\n"
            "Type a message and press Enter. Ctrl-D also exits the REPL."
        )
    )


HELP = _help_layout()


class Repl:
    def __init__(self, url: str, use_history: bool = True) -> None:
        self.session: Session | None = None
        self.url = url
        self.use_history = use_history
        # Per-turn usage report (the ``in=/out=/=`` line after each answer).
        # On by default; toggle with ``/usage on|off|toggle`` (see ``_command``).
        # Runtime REPL state, like ``use_history`` (not persisted across sessions).
        self.show_usage: bool = True
        # Whether tokens were streamed this turn and an answer line is still open
        # (so the next CONSOLE output — a tool line, the usage line, etc. — starts
        # on its own line instead of gluing onto the last streamed token).
        self._line_open: bool = False

    def _open_session(self) -> Session:
        try:
            session = Session(self.url)
            session.client.get(f"{session.url}/health", timeout=5)
        except httpx2.HTTPError as exc:
            raise BackendError(self.url, str(exc)) from exc
        return session

    def run(self) -> int:
        history = FileHistory(".repl_history") if self.use_history else None

        banner = Panel.fit(
            "[bold]agent-seed[/] REPL\n"
            f"backend: [cyan]{self.url}[/]\n"
            "agent: [cyan](default)[/]\n"
            "[dim]type /help for commands, /quit to exit[/]\n"
            "[dim]type / alone to open a multi-line block[/]",
            border_style="cyan",
        )
        CONSOLE.print(banner)

        session = self._open_session()
        self.session = session
        try:
            return self._loop(session, history)
        finally:
            session.close()

    def _loop(self, session: Session, history: Any) -> int:
        # mode is "normal" or "block"; buffer holds raw block-mode lines.
        mode = "normal"
        buffer: list[str] = []

        while True:
            try:
                if mode == "normal":
                    text = prompt("you> ", history=history)
                else:
                    # Block mode: scratch history so drafts stay out of
                    # .repl_history (the joined message is pushed once).
                    text = prompt("...> ", history=None)
            except KeyboardInterrupt:
                if mode == "normal":
                    CONSOLE.print("[dim](bye)[/]")
                    return 0
                CONSOLE.print("[dim](draft discarded)[/]")
                mode, buffer = "normal", []
                continue
            except EOFError:
                CONSOLE.print("[dim](bye)[/]")
                return 0

            if mode == "block":
                if text.strip() == "/":
                    body = "\n".join(buffer).strip()
                    mode, buffer = "normal", []
                    if body:
                        if history is not None:
                            history.append_string(body)
                        self._turn(session, body)
                # Any line (including blank, slash-like) is block content.
                else:
                    buffer.append(text)
                continue

            # -- normal mode -------------------------------------------------
            text = text.strip()
            if not text:
                continue

            if text in ("/quit", "/exit"):
                CONSOLE.print("[dim](bye)[/]")
                return 0

            if text == "/":
                mode, buffer = "block", []
                continue

            if text.startswith("/"):
                if not self._command(session, text):
                    return 0
                continue

            self._turn(session, text)

    # -- one chat turn -------------------------------------------------------

    def _close_line(self) -> None:
        """Terminate a streamed answer line so the next print starts on its own.

        Once tokens have been streamed this turn, the cursor sits mid-line after
        the last token (no newline yet). Anything printed before the turn ends —
        a tool line, the usage line, an error — would glue onto that last token.
        Calling this first drops a newline; it is idempotent (no-op once closed).
        """
        if self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_open = False

    def _turn(self, session: Session, message: str) -> None:
        self._line_open = False
        started = False
        try:
            for event, payload in session.chat_stream(message):
                if event == "token":
                    token = payload.get("text", "")
                    if not started:
                        CONSOLE.print("agent> ", end="")
                        started = True
                        self._line_open = True
                    sys.stdout.write(token)
                    sys.stdout.flush()
                elif event == "tool":
                    self._close_line()
                    render_tool_event(
                        CONSOLE,
                        payload.get("name", "?"),
                        payload.get("arguments", {}),
                        payload.get("result", ""),
                    )
                elif event == "error":
                    self._close_line()
                    CONSOLE.print(f"[bold red]error:[/] {payload.get('message')}")
                elif event == "done":
                    if payload.get("usage_session") is not None:
                        session.usage_session = dict(payload["usage_session"])
                    turn_usage = payload.get("usage")
                    if turn_usage and self.show_usage:
                        self._close_line()
                        CONSOLE.print(f"[dim]{format_tokens(turn_usage)}[/]")
                    # "done" is terminal; loop ends
            self._close_line()
        except httpx2.HTTPError as exc:
            self._close_line()
            CONSOLE.print(f"[bold red]backend error:[/] {exc}")

    def _switch_agent(self, session: Session, name: str) -> bool:
        """Validate ``name`` against the backend and commit the agent switch.

        ``default`` (a reserved name) always names the main agent. Everything else
        must appear in ``GET /agents`` — so a typo or a nonexistent folder is
        rejected here instead of silently falling back on the backend. Returns
        ``True`` if a switch was committed, ``False`` if rejected.
        """
        if name == RESERVED_AGENT_NAME:
            session.set_agent(None)
            return True
        try:
            valid = session.agents()
        except httpx2.HTTPError:
            valid = None
        if valid is not None and name not in valid:
            hint = ", ".join(valid) if valid else "(none on the backend)"
            CONSOLE.print(f"[red]unknown agent:[/] {name} (available: {hint})")
            CONSOLE.print("[dim]see /agents, or /agent default to use the main agent[/]")
            return False
        session.set_agent(name)
        return True

    # -- slash commands ------------------------------------------------------

    def _command(self, session: Session, text: str) -> bool:
        cmd = text.split(None, 1)[0]
        if cmd == "/help":
            CONSOLE.print(Panel(HELP, title="help", border_style="cyan"))
            return True
        if cmd == "/reset":
            session.reset()
            CONSOLE.print("[dim]session reset[/]")
            return True
        if cmd == "/agent":
            if text.strip() == cmd:
                CONSOLE.print(f"agent: [bold cyan]{_agent_label(session)}[/]")
                return True
            name = text.split(None, 1)[1].strip()
            if self._switch_agent(session, name):
                CONSOLE.print(f"agent: [bold cyan]{_agent_label(session)}[/]")
            return True
        if cmd == "/agents":
            names = session.agents()
            if not names:
                CONSOLE.print("[dim]no discoverable agents on the backend[/]")
            else:
                CONSOLE.print("[bold]agents:[/] " + ", ".join(names))
            CONSOLE.print("[dim]switch back to the main agent with /agent default[/]")
            return True
        if cmd == "/usage":
            parts = text.split(None, 1)
            arg = parts[1].strip().lower() if len(parts) > 1 else None
            if arg in ("on", "off", "toggle"):
                if arg == "toggle":
                    self.show_usage = not self.show_usage
                else:
                    self.show_usage = arg == "on"
                state = "on" if self.show_usage else "off"
                CONSOLE.print(f"[bold]usage:[/] per-turn report {state}")
                return True
            if arg is not None:
                CONSOLE.print(
                    "[red]usage:[/] /usage [on|off|toggle]  (with no arg: the session total)"
                )
                return True
            total = session.usage_session
            if not total:
                CONSOLE.print("[dim]no usage recorded yet (no turns, or backend sent none)[/]")
            else:
                CONSOLE.print(f"[bold]session:[/] {format_tokens(total)}")
            CONSOLE.print(
                f"[dim]per-turn report: {'on' if self.show_usage else 'off'} — "
                "toggle with /usage on|off|toggle; the total resets with /reset[/]"
            )
            return True
        if cmd == "/tools":
            names = session.tools_names()
            CONSOLE.print("[bold]tools:[/] " + ", ".join(names))
            return True
        if cmd == "/model":
            parts = text.split()
            if len(parts) == 1:
                config = session.model()
                if config is None:
                    CONSOLE.print("[dim]backend does not expose /config[/]")
                else:
                    CONSOLE.print(json.dumps({"llm": config.get("llm")}, indent=2, default=str))
            elif parts[1] == "test":
                if len(parts) < 3:
                    CONSOLE.print("[red]usage:[/] /model test <name> (optional base_url)")
                    return True
                self._model_probe(session, parts[2], parts[3] if len(parts) > 3 else None)
            elif parts[1] == "force":
                if len(parts) < 3:
                    CONSOLE.print("[red]usage:[/] /model force <name> (optional base_url)")
                    return True
                self._model_set(
                    session, parts[2], parts[3] if len(parts) > 3 else None, probe=False
                )
            else:
                if len(parts) < 2:
                    CONSOLE.print(
                        "[red]usage:[/] /model <name> (optional base_url)  "
                        "(or /model test <name>, /model force <name>)"
                    )
                    return True
                self._model_set(
                    session, parts[1], parts[2] if len(parts) > 2 else None, probe=True
                )
            return True
        CONSOLE.print(f"[red]unknown command:[/] {cmd} (try /help)")
        return True

    def _model_probe(self, session: Session, model: str, base_url: str | None) -> None:
        """Run a pure probe (``/model test``) and report the outcome."""
        try:
            result = session.test_model(unquote(model), unquote(base_url) if base_url else None)
        except httpx2.HTTPError as exc:
            CONSOLE.print(f"[bold red]error:[/] {exc}")
            return
        if result.get("ok"):
            CONSOLE.print(f"[green]ok:[/] {result.get('model')} ({result.get('latency_ms')} ms)")
        else:
            CONSOLE.print(f"[red]failed:[/] {result.get('error')}")

    def _model_set(self, session: Session, model: str, base_url: str | None, probe: bool) -> None:
        """Commit a model switch. If ``probe`` is set, ping the model first and
        abort (keeping the current model) when it cannot be reached — so a
        typo'd name never becomes the live model."""
        model = unquote(model)
        base_url = unquote(base_url) if base_url else None
        if probe:
            try:
                result = session.test_model(model, base_url)
            except httpx2.HTTPError as exc:
                CONSOLE.print(f"[bold red]error:[/] {exc}")
                return
            if not result.get("ok"):
                CONSOLE.print(
                    f"[red]not set:[/] probe failed for {model!r} ({result.get('error')})"
                )
                CONSOLE.print(
                    "[dim]current model unchanged — retry with /model force "
                    f"{model} to set it anyway[/]"
                )
                return
        try:
            outcome = session.set_model(model, base_url)
        except httpx2.HTTPError as exc:
            CONSOLE.print(f"[bold red]error:[/] {exc}")
            return
        effective = (outcome.get("llm") or {}).get("model", "?")
        CONSOLE.print(f"[green]model:[/] {effective}")
