"""Hook surface: the stable, extensible point in the agent core.

Two things are defined here:

* :class:`Hooks` — a small dataclass of named, optional callables. Each
  callable (a *hook*) is the point where an experimental module attaches
  itself to the agent loop. All hooks share one frozen signature::

      def my_hook(ctx: RunContext) -> None

  Hooks run in-place, synchronously, in the order shown by the core loop.
  They may inspect or mutate any field of :class:`RunContext` (including
  the open ``extras`` namespace), and may veto a tool call. There are two
  agent-boundary hooks — ``enter_agent`` / ``exit_agent`` — that run from
  the ``Agent`` lifetime (construction/switch) rather than inside
  ``chat()``; the rest run inside ``chat()`` in the listed order.

* :class:`RunContext` — the single shared object owned by one run (one
  ``Agent``, one ``RunContext``), passed to every hook. Its fields fall into
  three lifetimes:

  - *run-scoped* fields persist across turns (and survive ``reset()``):
    ``run_id``, ``session_id``, ``agent_name``, ``turn``, ``base_prompt``
    (immutable), ``system_prompt`` (sticky — a hook change holds until
    changed again), ``allowed_tools`` (sticky — a hook change holds until
    changed again), ``base_model`` (immutable) and ``current_model`` (sticky
    — the model the core actually calls; hooks own its value, and the core
     never resets it on an agent switch), cumulative ``usage``, and ``extras``.
  - *per-request* fields are rebuilt by the core on each LLM call:
    ``messages``, ``tools``, ``tool_loop_round``, and the per-round prompting
    inputs ``context`` / ``examples`` / ``addendum`` (seeded from the active
    agent's baseline each round, then ``before_llm`` may adjust them — the core
    honors those post-hook values: ``context`` / ``addendum`` are folded into the
    system message and ``examples`` are spliced between it and the hook's
    ``messages`` zone, see :mod:`agent_seed.prompting`).
    - *per-call* fields are set for each tool call or turn:
     ``tool_call``, ``tool_result``, ``final_answer``, ``turn_usage``.

What a hook is **not**:

* It is not a middleware chain. One slot holds at most one callable; if
  you need two, compose them inside a single callable.
* It is not an event bus. Hooks run in the turn's call stack, not
  asynchronously.
* It is not a subclassing scheme. You do not subclass ``Agent``; you
  attach hooks via the ``Hooks`` dataclass, or by dropping a hook module
  into the ``hook_modules/`` folder (auto-discovered once per process —
  see ``docs/manual.md``, "Promoting experiments").

See ``docs/manual.md`` for the full user manual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMConfig

# One hook is just a callable taking the shared run context and returning
# nothing: any work it wants to do happens via mutation of ``ctx``.
Hook = Callable[["RunContext"], None]


@dataclass
class RunContext:
    """Shared object passed to every hook in one agent interaction.

    Ownership is explicit per field (also listed in ``docs/manual.md``):

    * Core-managed — set by the core; hooks may read, most may modify.
    * Caller-supplied — optional ids the caller may pre-fill (``run_id``,
      ``session_id``, ``agent_name``); the core auto-generates the ids if
      left ``None`` by end of the *initialize* phase. ``agent_name`` has no
      auto-generation: ``None`` means the default (root) hooks.
    * Hook-owned — the ``extras`` namespace is a wall: the core never
      reads or writes it; hooks own it (flat keys).
    """

    # -- hierarchy ids (run-scoped, sticky) ----------------------------------
    # Both default to None. If a hook's initialize_run / initialize_session
    # (or the caller, via Agent(run_id=..., session_id=...)) leaves them
    # None, the core auto-generates a value at establishment and keeps it.
    # The session hooks re-fire whenever the active session changes.
    run_id: str | None = None
    session_id: str | None = None

    # -- agent identity (run-scoped, sticky) ---------------------------------
    # Identifies which agent (hooks subdirectory) is active:
    # hook_modules/<agent_name>/, or the hook_modules/ root when None.
    # There is always an agent in place — the nameless one (None) is the main
    # agent, not a special case. Caller-supplied via Agent(agent_name=...) or
    # Agent.set_agent_name(). On a switch the enter/exit boundary hooks and
    # the establishment hooks re-fire.
    agent_name: str | None = None

    # -- system prompt (run-scoped) ------------------------------------------
    # base_prompt: the prompt established for the run; immutable after
    # construction. Any hook can always see the original.
    # system_prompt: the CURRENT prompt — the sticky value the core sends as
    # messages[0]. A hook may extend/prepend/patch/replace it via
    # ctx.system_prompt; the change holds from that request onward (also
    # across turns) until changed again. See "System prompt (dynamic)".
    base_prompt: str | None = None
    system_prompt: str | None = None

    # -- tool policy (run-scoped, sticky) ------------------------------------
    # The whitelist of tool names this run may OFFER to the LLM and CALL.
    # Owned by the active agent: an agent's enter_agent (or before_llm) sets
    # it, the core reads it to (a) filter the offered schemas and (b) veto
    # any call to a name outside it. None or [] is the default and means NO
    # tools — tooling is opt-in per agent. The core resets this to None at
    # each agent switch (before the incoming agent's hooks fire) so a
    # switched-to agent that declares no set genuinely sees no tools. Sticky
    # like system_prompt: a change holds across turns until changed again.
    allowed_tools: list[str] | None = None

    # -- model selection (run-scoped) ----------------------------------------
    # base_model: the LLM established for the run (the one the Agent's client
    # was built with); immutable after construction. Any hook can always see
    # the original and restore it: ctx.current_model = ctx.base_model.
    # current_model: the LLM the core CALLS — the sticky value resolved from
    # ctx.current_model on every LLM call. Hook-owned: Agent.set_model commits
    # a change (firing on_model_change, which may mutate or veto it), the active
    # agent's enter_agent may bind its own, and before_llm may override per
    # round. It survives agent switches and reset() — the core never resets it;
    # a hook that wants a switch to mean "back to default" must say so itself.
    base_model: LLMConfig | None = None
    current_model: LLMConfig | None = None

    # -- core-managed counters -----------------------------------------------
    turn: int = 0  # user messages in this run (1-based; the core increments)
    tool_loop_round: int = 0  # LLM → tools iterations in this turn (per-turn)

    # -- per-request / per-call working state (rebuilt by the core) ----------
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Per-round prompting inputs (non-sticky): the core seeds these from the
    # active agent's baseline (its Hooks.prompt/context/examples) at the start
    # of each round, then before_llm may adjust them. compose_prompt reads them
    # to build the messages sent to the LLM — they are NOT persisted and do not
    # accumulate across rounds. See agent_seed/prompting.py and
    # docs/manual.md "Prompting".
    context: str | None = None  # per-round context block (e.g. RAG, date, state)
    # per-round few-shot [{"role","content"}]; settable in before_llm, spliced
    # into messages between the system message and the hook's message zone
    examples: list[dict[str, Any]] | None = None
    addendum: str | None = None  # per-turn note rendered last into the system message
    tool_call: dict[str, Any] | None = None  # set: {"name","arguments"} or None (veto)
    tool_result: Any = None  # core's return value; hooks may rewrite
    final_answer: str | None = None  # prepare/finish may rewrite
    usage: dict[str, Any] = field(default_factory=dict)  # cumulative across the run
    turn_usage: dict[str, Any] = field(default_factory=dict)  # this turn's tokens (per-turn)

    # -- hook-owned, flat keys (run-scoped) -----------------------------------
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hooks:
    """The full set of optional hooks, in fixed firing order.

    All slots share the signature ``(ctx: RunContext) -> None``. All slots
    default to ``None`` — a hook that is not set simply does not run.

    In addition to the callable slots, an agent may carry three optional
    *data* fields loaded from its hook folder by
    :func:`agent_seed.discovery.get_hooks`: ``prompt`` (identity, from
    ``prompt.md``), ``context`` (per-round context baseline, from
    ``context.md``), and ``examples`` (few-shot message list, from
    ``examples.jsonl``). They are declared in the active agent's hooks, used
    as the per-round seed for the matching :class:`RunContext` fields, and
    rendered by :func:`agent_seed.prompting.compose_prompt` — hooks still have
    full control (they override every one of these; see
    ``docs/manual.md``, "Prompting").

    ``enter_agent`` / ``exit_agent`` are the agent-identity boundary hooks:
    they fire from ``Agent.__init__`` / ``Agent.set_agent_name`` (not in
    ``chat()``) to bracket each agent becoming active / leaving. The remaining
    slots fire inside ``chat()`` in the order they are listed.
    """

    # Agent-identity boundary (meta-level; fire in the Agent ctor /
    # set_agent_name, NOT in chat()). There is always an agent in place: the
    # nameless one (agent_name=None, the hook_modules/ root) is the main
    # agent. enter_agent fires every time an agent becomes active (at
    # construction, even for the main agent, and as the incoming agent on a
    # switch); exit_agent fires when an agent is displaced (the outgoing
    # agent) — never at construction. On a switch: exit(old) -> enter(new).
    enter_agent: Hook | None = None  # incoming agent takes over; ctx.agent_name = NEW value
    exit_agent: Hook | None = None  # outgoing agent leaves; ctx.agent_name = OLD value

    # Model-selection boundary (Agent-lifetime, like enter_agent/exit_agent):
    # fires from Agent.set_model AFTER the new value is committed to
    # ctx.current_model. May mutate ctx.current_model (e.g. substitute a model,
    # restore ctx.base_model) or RAISE to veto the change — the core rolls back
    # to the previous value and re-raises, exactly like exit_agent aborting an
    # agent switch.
    on_model_change: Hook | None = None

    # Identity establishment (run before session, in this order).
    initialize_run: Hook | None = None  # may set ctx.run_id
    ready_run: Hook | None = None  # ctx.run_id guaranteed non-None
    initialize_session: Hook | None = None  # may set ctx.session_id
    ready_session: Hook | None = None  # ctx.session_id guaranteed non-None

    # Per tool_loop_round, in this order (before_llm once, then per tool call):
    before_llm: Hook | None = None  # mutate ctx.messages / ctx.tools
    before_tool_call: Hook | None = None  # may rewrite ctx.tool_call or veto (None)
    after_tool_call: Hook | None = None  # may rewrite ctx.tool_result
    on_round: Hook | None = None  # fire-only observability

    # Turn termination:
    prepare: Hook | None = None  # ctx.final_answer is set; validate/normalize
    finish: Hook | None = None  # LAST word; may rewrite ctx.final_answer

    # -- declared prompting content (data, not callables; optional) ----------
    # Loaded by discovery from the agent's hook folder (prompt.md / context.md
    # / examples.jsonl) or set explicitly. These are the *baselines* the core
    # seeds into the per-round RunContext fields (context/examples; prompt ->
    # base_prompt) before before_llm runs. They are data slots — no firing
    # order — and discovery merges them the same "last file wins" way as the
    # callable slots. Hooks can still override everything downstream of these.
    prompt: str | None = None  # agent identity (prompt.md); seeds base_prompt
    context: str | None = None  # per-round context baseline (context.md)
    examples: list[dict[str, Any]] | None = None  # few-shot baseline (examples.jsonl)
