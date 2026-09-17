"""Hook surface: the stable, extensible point in the agent core.

Two things are defined here:

* :class:`Hooks` — a small dataclass of named, optional callables. Each
  callable (a *hook*) is the point where an experimental module attaches
  itself to the agent loop. All hooks share one frozen signature::

      def my_hook(ctx: RunContext) -> None

  Hooks run in-place, synchronously, in the order shown by the core loop.
  They may inspect or mutate any field of :class:`RunContext` (including
  the open ``extras`` namespace), and may veto a tool call.

* :class:`RunContext` — the single shared object owned by one run (one
  ``Agent``, one ``RunContext``), passed to every hook. It has two
  lifetimes:

  - *run-scoped* fields persist across turns (and survive ``reset()``):
    ``run_id``, ``session_id``, ``turn``, ``base_prompt`` (immutable),
    ``system_prompt`` (sticky — a hook change holds until changed again),
    cumulative ``usage``, and ``extras``.
  - *per-request* fields are rebuilt by the core on each LLM call:
    ``messages``, ``tools``, ``tool_loop_round``; and *per-call* fields set
    for each tool call: ``tool_call``, ``tool_result``, ``final_answer``.

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

# One hook is just a callable taking the shared run context and returning
# nothing: any work it wants to do happens via mutation of ``ctx``.
Hook = Callable[["RunContext"], None]


@dataclass
class RunContext:
    """Shared object passed to every hook in one agent interaction.

    Ownership is explicit per field (also listed in ``docs/manual.md``):

    * Core-managed — set by the core; hooks may read, most may modify.
    * Caller-supplied — optional ids the caller may pre-fill; the core
      auto-generates if left ``None`` by end of the *initialize* phase.
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

    # -- system prompt (run-scoped) ------------------------------------------
    # base_prompt: the prompt established for the run; immutable after
    # construction. Any hook can always see the original.
    # system_prompt: the CURRENT prompt — the sticky value the core sends as
    # messages[0]. A hook may extend/prepend/patch/replace it via
    # ctx.system_prompt; the change holds from that request onward (also
    # across turns) until changed again. See "System prompt (dynamic)".
    base_prompt: str | None = None
    system_prompt: str | None = None

    # -- core-managed counters -----------------------------------------------
    turn: int = 0  # user messages in this run (1-based; the core increments)
    tool_loop_round: int = 0  # LLM → tools iterations in this turn (per-turn)

    # -- per-request / per-call working state (rebuilt by the core) ----------
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_call: dict[str, Any] | None = None  # set: {"name","arguments"} or None (veto)
    tool_result: Any = None  # core's return value; hooks may rewrite
    final_answer: str | None = None  # prepare/finish may rewrite
    usage: dict[str, Any] = field(default_factory=dict)  # cumulative across the run

    # -- hook-owned, flat keys (run-scoped) -----------------------------------
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hooks:
    """The full set of optional hooks, in fixed firing order.

    All slots share the signature ``(ctx: RunContext) -> None``. All slots
    default to ``None`` — a hook that is not set simply does not run.
    """

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
