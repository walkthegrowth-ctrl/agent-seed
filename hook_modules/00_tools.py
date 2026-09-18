"""Main agent's tool whitelist.

Declares, for the *main* (nameless / root) agent, which tools it may see and
call. Under the strict-whitelist policy the default of ``ctx.allowed_tools``
is ``None`` (= no tools), so the main agent has to declare what it wants —
this file is that declaration: ``calculator``.

Promotion / editing:
* Change the list here to add or remove tools for the main agent.
* A file named ``00_*`` sorts first so later files in the same folder can
  override it (discovery merge is "last file wins" on slot conflict).
* Delete this file to give the main agent no tools at all.

See ``experiments/tool_policy.py`` for the pattern and
``docs/manual.md`` ("Per-agent tool access (strict whitelist)") for the
semantics.
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext

ALLOWED_TOOLS: list[str] = ["calculator"]


def enter_agent(ctx: RunContext) -> None:
    """Fire when the main agent takes over: pin its tool whitelist."""
    ctx.allowed_tools = list(ALLOWED_TOOLS)


def make() -> Hooks:
    return Hooks(enter_agent=enter_agent)
