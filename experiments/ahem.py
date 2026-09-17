"""Silly experiment: tack a nervous "(ahem)" onto every final answer.

Illustrates the ``finish`` hook slot — the very last point in a turn where a
hook may rewrite ``ctx.final_answer``. Attach with::

    from agent_seed.core import Agent
    from experiments.ahem import make

    agent = Agent(client, registry, hooks=make())
    print(agent.chat("what is 6 * 7?"))
    # 42
    # (ahem)
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext


def finish(ctx: RunContext) -> None:
    if ctx.final_answer is not None:
        ctx.final_answer = f"{ctx.final_answer}\n(ahem)"


def make() -> Hooks:
    # `make` is the canonical hook-module factory name; promoting this
    # experiment is a plain `cp experiments/ahem.py hook_modules/`.
    return Hooks(finish=finish)
