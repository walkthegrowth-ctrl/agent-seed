"""Experiment: per-agent tool whitelisting (``ctx.allowed_tools``).

The canonical pattern: ``enter_agent`` pins the run's tool whitelist the
moment the agent takes over. The core enforces it at BOTH the offer point
(schemas sent to the LLM) and the execute point (a call to a non-whitelisted
name is rejected with a well-formed ``role: tool`` error), so tool access is
opt-in per agent: a run with no ``allowed_tools`` set offers NO tools and
refuses every tool call.

Promotion model: put a file like this inside ``hook_modules/<agent_name>/``
and only that agent gets the whitelisted tools. The main agent can ship one
at the root too — see ``hook_modules/00_tools.py`` in this repo.

Attach directly:

    from agent_seed.core import Agent
    from experiments.tool_policy import make

    agent = Agent(client, registry, hooks=make())

Notes:

* ``enter_agent`` fires at construction (first activation) and on every
  switch INTO this agent — the right place to (re)declare its tools. The
  core resets ``allowed_tools`` to ``None`` on each switch BEFORE the
  incoming agent's hooks fire, so a switched-to agent that declares nothing
  genuinely sees no tools.
* ``allow_all()`` is a convenience for an agent that should keep the whole
  registered toolset — just don't restrict. See ``make_allow_all()``.
* Hook-level veto (``before_tool_call``) still works to *narrow* an allowed
  call further, but it never runs for a name the whitelist already rejected —
  the core settles that first.
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext

# Edit this per agent to declare what it may use. ``[]`` means NO tools.
ALLOWED_TOOLS: list[str] = ["calculator"]


def _enter_agent(ctx: RunContext) -> None:
    """Fire when this agent takes over: pin the run's whitelist."""
    ctx.allowed_tools = list(ALLOWED_TOOLS)


def make() -> Hooks:
    return Hooks(enter_agent=_enter_agent)


def make_allow_all(names: list[str]) -> Hooks:
    """A variant that whitelists an explicit set (e.g. the whole toolset).

    Usage: ``make_allow_all(["calculator", "weather"])``.
    """

    def _enter(ctx: RunContext) -> None:
        ctx.allowed_tools = list(names)

    return Hooks(enter_agent=_enter)
