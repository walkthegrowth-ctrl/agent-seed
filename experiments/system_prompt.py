"""Dynamic system-prompt experiment.

Demonstrates the dynamic-prompt contract on one hook:

* **Reads the base prompt** (``ctx.base_prompt``) so it can *extend* rather
  than clobber it.
* **Recomputes from base** every request — self-healing: the addendum
  never drifts or accumulates across turns.
* **Is dynamic** — the addendum varies with ``turn`` and ``session_id``.
* **Session-scoped state** — a per-session counter lives in ``extras``,
  reset (or re-seeded) when the core signals a new session via
  ``ready_session`` firing.

Promote with::

    cp experiments/system_prompt.py hook_modules/
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext


def before_llm(ctx: RunContext) -> None:
    base = ctx.base_prompt or ""
    addendum = f"\n\n[dynamic context: turn {ctx.turn} / session {ctx.session_id}]"
    # Recompute from base every time (not "append to current") so the
    # addendum stays a single block across turns.
    ctx.system_prompt = (base + addendum) if base else addendum.strip()


def ready_session(ctx: RunContext) -> None:
    # Fires when a session is (re-)established. A good place to seed
    # per-session state into extras (e.g. load prior context for this
    # session, start per-session counters, or react to a session change).
    prev = ctx.extras.get("session_id")
    if prev is not None and prev != ctx.session_id:
        ctx.extras["session_changed_from"] = prev
    ctx.extras["session_id"] = ctx.session_id


def make() -> Hooks:
    return Hooks(
        before_llm=before_llm,
        ready_session=ready_session,
    )
