"""REPL app: prompt for input, stream tokens, render tool events."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel

CONSOLE = Console()


class BackendError(RuntimeError):
    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"cannot reach backend at {url}: {detail}")


def sse_events(raw: str) -> Iterator[tuple[str, str]]:
    """Yield (event, data_json_str) pairs from an SSE text body."""
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
    of lines (e.g. httpx iter_lines()) so events surface as they arrive
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
        self.client = httpx.Client(timeout=timeout)

    # -- backend calls ------------------------------------------------------

    def reset(self) -> None:
        self.client.post(f"{self.url}/reset", json={"session_id": self.session_id})
        self.session_id = uuid.uuid4().hex

    def tools(self) -> list[dict[str, Any]]:
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
        # /config is part of the documented protocol, but the REPL treats it
        # as optional: it shows the config only when the backend answers 200
        # (older backends without the endpoint simply render nothing).
        try:
            response = self.client.get(f"{self.url}/config")
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        return None

    def chat_stream(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        payload: dict[str, Any] = {"session_id": self.session_id, "message": message}
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        with self.client.stream("POST", f"{self.url}/chat", json=payload) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", "replace")
                raise httpx.HTTPStatusError(
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


# -- REPL main loop ----------------------------------------------------------

HELP = """[bold]commands[/]
  /help        show this help
  /reset       start a new session (clears server-side history)
  /agent [n]   set the agent (a hooks subdirectory); no arg shows current
  /agents      list discoverable agent names from the backend
  /tools       list tools registered on the backend
  /model       show backend LLM config (if the backend exposes /config)
  /quit        exit
  /            (alone) open a multi-line block — see below

[bold]multi-line mode[/]
  Type [bold]/[/] and press Enter to open a block. Then type as many
  lines as you want (blank lines are kept, slash commands are treated
  as plain text). Close and send by typing [bold]/[/] alone on a line
  and pressing Enter. The two / delimiters are stripped from the
  message; the text between them is sent as-is. Ctrl-C cancels the
  draft and returns to normal mode; Ctrl-D exits the REPL.

Type a message and press Enter. Ctrl-D also exits."""


class Repl:
    def __init__(self, url: str, use_history: bool = True) -> None:
        self.session: Session | None = None
        self.url = url
        self.use_history = use_history

    def _open_session(self) -> Session:
        try:
            session = Session(self.url)
            session.client.get(f"{session.url}/health", timeout=5)
        except httpx.HTTPError as exc:
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

    def _turn(self, session: Session, message: str) -> None:
        first = True
        try:
            for event, payload in session.chat_stream(message):
                if event == "token":
                    token = payload.get("text", "")
                    if first:
                        CONSOLE.print("agent> ", end="")
                        first = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
                elif event == "tool":
                    render_tool_event(
                        CONSOLE,
                        payload.get("name", "?"),
                        payload.get("arguments", {}),
                        payload.get("result", ""),
                    )
                elif event == "error":
                    CONSOLE.print(f"[bold red]error:[/] {payload.get('message')}")
                # "done" is terminal; loop ends
            if not first:
                sys.stdout.write("\n")
                sys.stdout.flush()
        except httpx.HTTPError as exc:
            CONSOLE.print(f"[bold red]backend error:[/] {exc}")

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
            parts = text.split(None, 1)
            if len(parts) < 2:
                CONSOLE.print(f"agent: [bold cyan]{session.agent_name or '(default)'}[/]")
            else:
                session.set_agent(parts[1].strip() or None)
                CONSOLE.print(f"agent: [bold cyan]{session.agent_name or '(default)'}[/]")
            return True
        if cmd == "/agents":
            names = session.agents()
            if not names:
                CONSOLE.print("[dim]no discoverable agents on the backend[/]")
            else:
                CONSOLE.print("[bold]agents:[/] " + ", ".join(names))
            return True
        if cmd == "/tools":
            names = session.tools_names()
            CONSOLE.print("[bold]tools:[/] " + ", ".join(names))
            return True
        if cmd == "/model":
            config = session.model()
            if config is None:
                CONSOLE.print("[dim]backend does not expose /config[/]")
            else:
                CONSOLE.print(config)
            return True
        CONSOLE.print(f"[red]unknown command:[/] {cmd} (try /help)")
        return True
