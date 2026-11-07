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
import time
import uuid
from collections.abc import Callable
from typing import Any

from .discovery import get_hooks_for_agent
from .hooks import Hooks, RunContext
from .llm import LLMConfig, LLMProbe, OpenAICompatClient
from .prompting import compose_prompt, compose_system, normalize_examples
from .tools import ToolError, ToolRegistry
from .tools import registry as default_registry

MAX_TOOL_ROUNDS = 8

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. "
    "Use the provided tools whenever they can produce an answer you "
    "are not certain about (e.g. arithmetic)."
)


def _elapsed_ms(start: float) -> float:
    """Milliseconds since a monotonic clock reading (one decimal place)."""
    return round((time.monotonic() - start) * 1000.0, 1)


def is_number(value: Any) -> bool:
    """True for a real numeric type (int/float), excluding bool.

    Shared helper: numeric usage counters must be distinguishable from truthy
    non-numbers (a bare ``True`` is not a token count). Imported by the backend
    (session usage) and the REPL (usage rendering).
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sum_usage(total: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Add ``delta``'s numeric counters onto ``total`` (non-numeric keys: last-wins).

    ``dict.update`` is wrong here: a multi-call turn (tool round + final answer)
    would let the last completion's counts clobber the earlier ones. Summing the
    shared counters keeps the run-cumulative ``usage`` correct across calls.
    """
    out = dict(total)
    for key, value in delta.items():
        prev = out.get(key)
        if is_number(value) and is_number(prev):
            out[key] = prev + value
        else:
            out[key] = value
    return out


def _usage_delta(current: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Per-turn usage = the numeric counters that advanced since ``snapshot``."""
    out: dict[str, Any] = {}
    for key in set(current) | set(snapshot):
        if is_number(current.get(key)) or is_number(snapshot.get(key)):
            out[key] = (current.get(key) or 0) - (snapshot.get(key) or 0)
        elif key in current:
            out[key] = current[key]
    return out


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
        system_prompt: str | None = None,
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
        # current (potentially hook-modified) value. Precedence: the explicit
        # system_prompt argument > the agent's declared prompt (its Hooks.prompt,
        # loaded from prompt.md via discovery) > the built-in default. An
        # agent's enter_agent may still specialize ctx.system_prompt afterward.
        prompt_text = system_prompt or self.hooks.prompt or DEFAULT_SYSTEM_PROMPT
        ctx.base_prompt = prompt_text
        ctx.system_prompt = prompt_text
        # Establish the models served by this run. base_model is the
        # immutable reference (what the caller's client was built with);
        # current_model starts equal and is what the core actually calls.
        # enter_agent (below) may already rebind current_model — the
        # sub-agent-specialization case; base_model stays for restoring.
        client_cfg = getattr(client, "config", None)
        ctx.base_model = client_cfg
        ctx.current_model = client_cfg
        self._context = ctx

        # First activation: the agent on the driver seat takes over. There is
        # no prior agent to exit, so only enter_agent fires — even for the
        # main (nameless) agent. If it raises, Agent construction fails.
        if self.hooks.enter_agent is not None:
            self.hooks.enter_agent(ctx)

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        registry: ToolRegistry | None = None,
    ) -> Agent:
        """Build an agent from environment variables (``LLM_*``).

        Uses the shared default ``registry`` by default; pass a custom one to
        change the available tools. Tools are still subject to the strict
        per-agent whitelist (see ``docs/manual.md`` "Per-agent tool access").
        """
        return cls(OpenAICompatClient(LLMConfig.from_env(env)), registry or default_registry)

    @classmethod
    def test_model(cls, candidate: LLMConfig) -> LLMProbe:
        """Probe a candidate model config without touching any agent state.

        One minimal chat completion (one ping message, ``max_tokens=1``) on a
        throwaway client. A probe is a diagnostic: it does not go through
        ``on_model_change`` (a probe failing is a *finding*, not a *change*),
        so use :meth:`set_model` when a change should be gated by the active
        agent. Any failure — config error, HTTP error, timeout — is reported
        as ``ok=False`` with the error text, never raised.
        """
        start = time.monotonic()
        try:
            client = OpenAICompatClient(candidate)
            completion = client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception as exc:  # noqa: BLE001 - probe boundary
            return LLMProbe(ok=False, error=str(exc), latency_ms=_elapsed_ms(start))
        return LLMProbe(
            ok=True,
            latency_ms=_elapsed_ms(start),
            usage=completion.usage,
        )

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
    def model_config(self) -> LLMConfig | None:
        """The LLM this run currently calls (``RunContext.current_model``)."""
        return self._context.current_model

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
        - the **identity** (``ctx.base_prompt`` / ``ctx.system_prompt``) is
          re-seeded from the incoming agent's declared prompt (its
          ``Hooks.prompt`` loaded from ``prompt.md``, falling back to the
          built-in default when that file is absent). The outgoing agent's
          sticky value does **not** leak into the new one; an agent that
          wants a customized identity writes it in its own ``enter_agent``
          (or ``before_llm``), which runs *after* the re-seed and therefore
          wins.
        - the **incoming** agent's ``enter_agent`` fires (``ctx.agent_name`` is
          now the *new* value).

        The new agent's establishment hooks (``initialize_run`` / ``ready_run``
        / ``initialize_session`` / ``ready_session``) then re-fire on the next
        ``chat()`` — so the full order is ``exit -> enter -> establishment``.
        History, ``run_id``, ``session_id``, and ``extras`` are left alone.
        Unlike ``allowed_tools``, ``current_model`` is sticky across the
        switch: the core does not reset it. An agent that wants to specialize
        (or restore the default) must say so in its ``enter_agent`` /
        ``before_llm`` — e.g. ``ctx.current_model = ctx.base_model``.
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
        # A fresh agent starts with NO tools. The incoming agent's hooks
        # (enter_agent / before_llm) decide the whitelist from here on; an
        # agent that does not set allowed_tools offers nothing. This reset
        # prevents the outgoing agent's sticky value from leaking across the
        # shared RunContext.
        self._context.allowed_tools = None
        # The identity follows the active agent: a file-defined agent (an
        # agent folder that carries only prompt.md, no hooks) has no hook
        # that will write the identity for it, so the core seeds it here from
        # the incoming agent's declared prompt (Hooks.prompt -> built-in
        # default). The outgoing agent's sticky value is replaced, not
        # carried over. The incoming agent's enter_agent (below) fires
        # afterwards and may override/extend either — hooks stay in control.
        identity = self.hooks.prompt or DEFAULT_SYSTEM_PROMPT
        self._context.base_prompt = identity
        self._context.system_prompt = identity
        # Incoming agent takes the wheel.
        if self.hooks.enter_agent is not None:
            self.hooks.enter_agent(self._context)
        self._run_established = False
        self._session_established = False

    def set_model(self, candidate: LLMConfig) -> LLMConfig:
        """Switch this run's model (what the core calls from now on).

        The commit is gated by the active agent's ``on_model_change`` hook:

        - if ``candidate`` equals the current model, nothing happens (no-op).
        - otherwise ``ctx.current_model`` is set to ``candidate`` and the hook
          fires (its ``ctx`` already carries the new value). The hook may
          re-mutate ``ctx.current_model`` (substitute a model, restore
          ``ctx.base_model``) or **raise to veto** — which aborts the switch:
          ``current_model`` is rolled back and the exception re-raised.
        - ``ctx.base_model`` is never touched; restoring it is the hook's job
          (``ctx.current_model = ctx.base_model``).

        Returns the effective model after the hook.
        """
        previous = self._context.current_model
        if candidate == previous:
            return candidate
        self._context.current_model = candidate
        if self.hooks.on_model_change is not None:
            try:
                self.hooks.on_model_change(self._context)
            except Exception:
                self._context.current_model = previous
                raise
        return self._context.current_model

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
        # Snapshot the run-cumulative before this turn so we can report the
        # per-turn delta (ctx.turn_usage) without changing usage's semantics.
        usage_snapshot = dict(ctx.usage)
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
            # Per-round prompting inputs are non-sticky: seed them from the
            # active agent's declared baselines (Hooks.context / Hooks.examples)
            # every round so a hook's one-round tweak never carries forward,
            # then before_llm may adjust them (docs/manual.md "Prompting").
            ctx.context = self.hooks.context
            ctx.examples = normalize_examples(self.hooks.examples)
            ctx.addendum = None
            # Per-request working state, rebuilt from the current session
            # history. compose_prompt renders system (identity + context +
            # addendum) -> examples -> history. ``before_llm`` may override the
            # canonical fields (ctx.system_prompt / ctx.context / ctx.examples /
            # ctx.addendum) or touch ctx.messages directly (escape hatch).
            ctx.messages = compose_prompt(
                ctx.system_prompt,
                self.history,
                ctx.context,
                ctx.examples,
                ctx.addendum,
            )
            # Offer only what the active agent's whitelist allows (strict).
            # None/[] -> no tools offered. ``before_llm`` may still adjust
            # ``ctx.tools`` afterwards (existing escape hatch) or change
            # ``ctx.allowed_tools`` (carries to the next round and governs
            # the execute-path veto) — see ``docs/manual.md``.
            ctx.tools = self.registry.schemas_for(ctx.allowed_tools)
            call_hook(self.hooks.before_llm)
            # The canonical system-channel fields (system_prompt / context /
            # addendum) are applied to the first message after before_llm, so a
            # hook that rewrote them is reflected here (compose_system folds
            # them, dropping empties). A hook that edited messages[0] directly
            # is overwritten — the canonical channel is ctx.system_prompt + the
            # per-round fields.
            ctx.messages[0]["content"] = compose_system(
                ctx.system_prompt, ctx.context, ctx.addendum
            )

            completion = self.client.chat(
                ctx.messages,
                tools=ctx.tools or None,
                on_token=on_token,
                config=ctx.current_model,
            )
            if completion.usage:
                ctx.usage = _sum_usage(ctx.usage, completion.usage)

            if not completion.tool_calls:
                return self._finish_turn(ctx, completion.content, usage_snapshot)

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

            allowed_now = ctx.allowed_tools or []
            for call in completion.tool_calls:
                call_name = call.name
                call_args = call.arguments
                vetoed = False
                # Core tool-policy gate: the active agent's whitelist is a
                # hard boundary, enforced BEFORE before_tool_call (a hook
                # can narrow or rewrite a call, but cannot re-allow a name
                # the agent did not whitelist — strict semantics). The
                # core records a well-formed tool result so the next LLM
                # call stays valid. (Hooks' ``before_tool_call`` veto is
                # a separate, later gate for hook-level policy.)
                if call.name not in allowed_now:
                    vetoed = True
                    content = f"error: tool '{call.name}' not available to this agent"
                else:
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
        # (before_llm is not called here — this is the forced fallback; the
        # per-round prompting inputs were seeded in the loop above and remain set.)
        ctx.tool_loop_round = MAX_TOOL_ROUNDS
        ctx.messages = compose_prompt(
            ctx.system_prompt,
            self.history,
            ctx.context,
            ctx.examples,
            ctx.addendum,
        )
        ctx.tools = []
        completion = self.client.chat(
            ctx.messages,
            tools=None,
            on_token=on_token,
            config=ctx.current_model,
        )
        if completion.usage:
            ctx.usage = _sum_usage(ctx.usage, completion.usage)
        return self._finish_turn(ctx, completion.content, usage_snapshot)

    # ------------------------------------------------------------------

    def _finish_turn(self, ctx: RunContext, content: str, usage_snapshot: dict[str, Any]) -> str:
        """Apply prepare/finish hooks, record history, return the answer.

        ``usage_snapshot`` is the run-cumulative ``usage`` captured at the
        start of the turn; the delta (``ctx.usage - snapshot``) is stashed as
        ``ctx.turn_usage`` for the backend to surface on the ``done`` event.
        """
        ctx.final_answer = content
        ctx.turn_usage = _usage_delta(ctx.usage, usage_snapshot)
        hooks = self.hooks
        if hooks.prepare is not None:
            hooks.prepare(ctx)
        if hooks.finish is not None:
            hooks.finish(ctx)
        self.history.append({"role": "assistant", "content": ctx.final_answer})
        return ctx.final_answer
