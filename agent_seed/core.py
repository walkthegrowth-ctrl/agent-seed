"""Agent core: conversation state + the LLM/tool-calling loop.

Framework-agnostic: knows nothing about HTTP or terminals. The backend
wraps it in SSE; the REPL renders the same event stream.

Extensibility lives in :mod:`agent_seed.hooks`: optional, named hooks
attached to fixed points in the loop, all sharing one signature
``(ctx: RunContext) -> None``. See ``docs/manual.md``.

A single :class:`RunContext` is created once per ``Agent`` and owned by it.
The run-establishment hooks fire strictly once; the session-establishment
hooks fire when the active session changes (including the first one).
Everything else about a turn is per-request working state, rebuilt on each
LLM call from the current history.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from .discovery import get_hooks_for_agent
from .hooks import Hooks, RunContext
from .llm import LLMConfig, OpenAICompatClient
from .tools import ToolError, ToolRegistry

MAX_TOOL_ROUNDS = 8

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. "
    "Use the provided tools whenever they can produce an answer you "
    "are not certain about (e.g. arithmetic)."
)


class Agent:
    """One run, one ``Agent``.

    The run owns a single :class:`RunContext`. A run may switch which
    session it is driving (one at a time); the session-establishment hooks
    re-fire on each switch, the run-establishment hooks only on first use.
    A run may also switch its active agent (``agent_name`` → which hooks
    subdirectory is used). There is always an agent in place (the nameless
    main agent, ``agent_name=None``, is the default). The agent-identity
    boundary hooks ``enter_agent`` / ``exit_agent`` bracket each switch
    (``exit`` on the old agent, ``enter`` on the new one), and the
    establishment hooks re-fire afterwards (see :meth:`set_agent_name`).
    """

    def __init__(
        self,
        client: OpenAICompatClient,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        hooks: Hooks | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        # None -> auto-discovered from the hook folder (cached per process);
        # an explicit Hooks always wins over discovery.
        self._hooks_explicit = hooks is not None
        self.hooks = hooks if hooks is not None else get_hooks_for_agent(agent_name)
        self.history: list[dict[str, Any]] = []
        self._turn_count = 0
        self._run_established = False
        self._session_established = False

        # The run-scoped context. Owned by the Agent; created exactly once.
        ctx = RunContext()
        if run_id is not None:
            ctx.run_id = run_id
        if session_id is not None:
            ctx.session_id = session_id
        ctx.agent_name = agent_name
        # The base prompt is immutable for the run; the system prompt is the
        # current (potentially hook-modified) value.
        ctx.base_prompt = system_prompt
        ctx.system_prompt = system_prompt
        self._context = ctx

        # First activation: the agent on the driver seat takes over. There is
        # no prior agent to exit, so only enter_agent fires — even for the
        # main (nameless) agent. If it raises, Agent construction fails.
        if self.hooks.enter_agent is not None:
            self.hooks.enter_agent(ctx)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Agent:
        return cls(OpenAICompatClient(LLMConfig.from_env(env)))

    # ------------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------------

    @property
    def base_prompt(self) -> str:
        """The immutable system prompt established for this run."""
        return self._context.base_prompt

    @property
    def system_prompt(self) -> str:
        """The current system prompt (what the core sends as ``messages[0]``).

        A hook may change this via ``ctx.system_prompt``; it then persists
        across turns (it is sticky). See ``docs/manual.md``, "System prompt
        (dynamic)".
        """
        return self._context.system_prompt

    @property
    def run_id(self) -> str | None:
        return self._context.run_id

    @property
    def session_id(self) -> str | None:
        return self._context.session_id

    @property
    def agent_name(self) -> str | None:
        """The active agent name (``None`` = the main / root agent)."""
        return self._context.agent_name

    @property
    def context(self) -> RunContext:
        """The run-scoped :class:`RunContext` this run is operating on."""
        return self._context

    def reset(self) -> None:
        """Start a new turn sequence in the same session (and same run).

        Clears history and the turn counter. Run identity, base prompt, and
        any session state accumulated in ``extras`` are left alone.
        """
        self.history = []
        self._turn_count = 0

    def set_session(self, session_id: str) -> None:
        """Switch the active session of this run.

        If ``session_id`` is the current session, this is a no-op.
        Otherwise the session-establishment hooks (``initialize_session``
        then ``ready_session``) fire on the next ``chat()`` call. The
        previous session's ``extras`` remain available to the hook as a
        signal that the session changed.
        """
        if session_id == self._context.session_id:
            return
        self._context.session_id = session_id
        self._session_established = False

    def set_agent_name(self, agent_name: str | None) -> None:
        """Switch the active agent (which hooks / agent set this run uses).

        Fire on every *real* switch (``agent_name != current``); a re-select of
        the same agent is a no-op and fires nothing. There is always an agent
        in place (the nameless main agent when ``None``), so:

        - the **outgoing** agent's ``exit_agent`` fires first (its hooks are
          still active; ``ctx.agent_name`` is the *old* value). Raising it
          aborts the switch — no state is touched.
        - the switch is then committed (``ctx.agent_name`` updated; hooks
          re-resolved unless an explicit ``Hooks`` was supplied at
          construction, which always wins).
        - the **incoming** agent's ``enter_agent`` fires (``ctx.agent_name`` is
          now the *new* value).

        The new agent's establishment hooks (``initialize_run`` / ``ready_run``
        / ``initialize_session`` / ``ready_session``) then re-fire on the next
        ``chat()`` — so the full order is ``exit -> enter -> establishment``.
        History, ``run_id``, ``session_id``, and ``extras`` are left alone.
        """
        if agent_name == self._context.agent_name:
            return
        # Resolve the incoming hooks BEFORE touching state: a broken
        # hook_modules/<new>/ must abort the switch cleanly (DiscoveryError).
        new_hooks = self.hooks if self._hooks_explicit else get_hooks_for_agent(agent_name)
        # Outgoing agent takes the wheel off. If it raises, nothing is changed.
        if self.hooks.exit_agent is not None:
            self.hooks.exit_agent(self._context)
        # Commit the switch.
        self._context.agent_name = agent_name
        if not self._hooks_explicit:
            self.hooks = new_hooks
        # Incoming agent takes the wheel.
        if self.hooks.enter_agent is not None:
            self.hooks.enter_agent(self._context)
        self._run_established = False
        self._session_established = False

    # ------------------------------------------------------------------
    # internal establishment logic
    # ------------------------------------------------------------------

    def _establish(self) -> None:
        """Run the establishment hooks for the current state (once each)."""
        if not self._run_established:
            self._establish_run()
        if not self._session_established:
            self._establish_session()

    def _establish_run(self) -> None:
        ctx = self._context
        if self.hooks.initialize_run is not None:
            self.hooks.initialize_run(ctx)
        if ctx.run_id is None:
            ctx.run_id = uuid.uuid4().hex
        if self.hooks.ready_run is not None:
            self.hooks.ready_run(ctx)
        self._run_established = True

    def _establish_session(self) -> None:
        ctx = self._context
        if self.hooks.initialize_session is not None:
            self.hooks.initialize_session(ctx)
        if ctx.session_id is None:
            ctx.session_id = uuid.uuid4().hex
        if self.hooks.ready_session is not None:
            self.hooks.ready_session(ctx)
        self._session_established = True

    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        on_token: Callable[[str], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Send one user turn; run the tool loop; return the final text.

        Callbacks:
            on_token(token)   — each streamed content token on the final turn
            on_event(kind, payload) — side events, e.g. ("tool", {...})

        Hooks attached (or discovered) fire into the shared
        :class:`~agent_seed.hooks.RunContext` in the documented lifecycle
        order. See ``docs/manual.md``.
        """
        ctx = self._context
        self._turn_count += 1
        ctx.turn = self._turn_count
        self._establish()

        def call_hook(hook: Callable[[RunContext], None] | None) -> None:
            if hook is not None:
                hook(ctx)

        def emit(kind: str, **payload: Any) -> None:
            if on_event is not None:
                on_event(kind, payload)

        self.history.append({"role": "user", "content": message})

        for round_idx in range(MAX_TOOL_ROUNDS):
            ctx.tool_loop_round = round_idx
            # Per-request working state, rebuilt from the current session
            # history. ``system_prompt`` is the run's current (possibly
            # hook-modified) value; ``before_llm`` may further replace it —
            # see ``docs/manual.md``, "System prompt (dynamic)".
            ctx.messages = [{"role": "system", "content": ctx.system_prompt or ""}]
            ctx.messages.extend(self.history)
            ctx.tools = self.registry.schemas()
            call_hook(self.hooks.before_llm)
            # A hook may have set a new system_prompt; apply it to the
            # first (system) message. If the hook already edited messages[0]
            # directly, this overwrites — the canonical channel is
            # ``ctx.system_prompt``.
            if ctx.system_prompt is not None:
                ctx.messages[0]["content"] = ctx.system_prompt

            completion = self.client.chat(
                ctx.messages,
                tools=ctx.tools or None,
                on_token=on_token,
            )
            if completion.usage:
                ctx.usage.update(completion.usage)

            if not completion.tool_calls:
                return self._finish_turn(ctx, completion.content)

            # Record the assistant message with its tool_calls, then results.
            self.history.append(
                {
                    "role": "assistant",
                    "content": completion.content or None,
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in completion.tool_calls
                    ],
                }
            )

            for call in completion.tool_calls:
                call_name = call.name
                call_args = call.arguments
                vetoed = False
                ctx.tool_call = {"name": call.name, "arguments": call.arguments}
                call_hook(self.hooks.before_tool_call)
                if ctx.tool_call is None:
                    vetoed = True
                    content = f"error: tool '{call.name}' skipped by policy"
                else:
                    call_name = ctx.tool_call["name"]
                    call_args = ctx.tool_call["arguments"]
                    try:
                        ctx.tool_result = self.registry.call(call_name, call_args)
                        content = str(ctx.tool_result)
                    except (ToolError, TypeError) as exc:
                        ctx.tool_result = None
                        content = f"error: {exc}"
                if not vetoed:
                    call_hook(self.hooks.after_tool_call)
                    if ctx.tool_result is not None:
                        content = str(ctx.tool_result)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{uuid.uuid4().hex[:8]}",
                        "content": content,
                    }
                )
                emit(
                    "tool",
                    name=call_name,
                    arguments=call_args,
                    result=content,
                )
            call_hook(self.hooks.on_round)

        # Tool round-trips exhausted: ask for a final answer without tools.
        # (before_llm is not called here — this is the forced fallback.)
        ctx.tool_loop_round = MAX_TOOL_ROUNDS
        ctx.messages = [{"role": "system", "content": ctx.system_prompt or ""}]
        ctx.messages.extend(self.history)
        ctx.tools = []
        completion = self.client.chat(ctx.messages, tools=None, on_token=on_token)
        if completion.usage:
            ctx.usage.update(completion.usage)
        return self._finish_turn(ctx, completion.content)

    # ------------------------------------------------------------------

    def _finish_turn(self, ctx: RunContext, content: str) -> str:
        """Apply prepare/finish hooks, record history, return the answer."""
        ctx.final_answer = content
        hooks = self.hooks
        if hooks.prepare is not None:
            hooks.prepare(ctx)
        if hooks.finish is not None:
            hooks.finish(ctx)
        self.history.append({"role": "assistant", "content": ctx.final_answer})
        return ctx.final_answer
