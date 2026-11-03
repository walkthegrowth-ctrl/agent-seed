"""End-to-end: a composed agent folder is discoverable and its tools callable.

Drives the *whole* path the runtime uses: catalog -> generate -> write a folder
into a temp HOOKS_ROOT -> discovery loads it (replace semantics) -> the inlined
reservoir tool is registered -> an Agent, scripted with a tool call, actually
invokes it and answers. No network; the LLM is FakeLLMClient.
"""

from __future__ import annotations

import pytest

from agent_seed.composer.catalog import list_tools
from agent_seed.composer.generator import ComposerError, Spec, generate, write_plan
from agent_seed.core import Agent
from agent_seed.discovery import get_hooks_for_agent, reset_cache
from agent_seed.hooks import RunContext
from agent_seed.tools import registry

from .conftest import FakeLLMClient, text_completion, tool_completion

RESERVOIR_TOOL = """
def double(n: int) -> int:
    \"\"\"Double a given number.\"\"\"
    return n * 2
"""


def _make_reservoir_tools(tmp_path) -> object:
    reservoir = tmp_path / "tools"
    reservoir.mkdir()
    (reservoir / "double.py").write_text(RESERVOIR_TOOL)
    return reservoir


@pytest.fixture()
def composed_agent(tmp_path, monkeypatch):
    """Compose a 'tutor' agent (reservoir tool 'double') into a temp HOOKS_ROOT."""
    reservoir = _make_reservoir_tools(tmp_path)

    # Compose.
    available = list_tools(tools_root=reservoir)
    spec = Spec(name="tutor", voice="You are a tutor.", tools=["double"])
    plan = generate("tutor", spec, available)

    hooks_root = tmp_path / "hook_modules"
    write_plan(plan, hooks_root, force=False)

    # Point runtime discovery at the composed folder and clear the cache.
    monkeypatch.setenv("HOOKS_ROOT", str(hooks_root))
    reset_cache()
    return hooks_root


def test_composed_agent_is_discoverable(composed_agent):
    hooks = get_hooks_for_agent("tutor")
    assert hooks is not None
    # Its hooks (enter_agent / sign) were loaded from the folder.
    assert hooks.enter_agent is not None


def test_composed_agent_registers_its_reservoir_tool(composed_agent):
    hooks = get_hooks_for_agent("tutor")
    ctx = RunContext(agent_name="tutor")
    hooks.enter_agent(ctx)
    assert ctx.allowed_tools == ["double"]
    # make() registered the inlined tool into the shared default registry.
    assert registry.has("double")


def test_composed_agent_calls_the_tool_end_to_end(composed_agent):
    reset_cache()
    hooks = get_hooks_for_agent("tutor")
    hooks.enter_agent(RunContext(agent_name="tutor"))  # ensure the tool is registered

    # Script: first call -> the LLM asks for double(21); second call -> the answer.
    client = FakeLLMClient(
        [
            tool_completion("double", {"n": 21}),
            text_completion("double(21) is 42"),
        ]
    )
    agent = Agent(client, registry, hooks=hooks, agent_name="tutor")
    answer = agent.chat("what is 21 times 2?")
    assert answer == "double(21) is 42"
    # History should record the successful tool result (the core fed "42" back).
    tool_results = [m for m in agent.history if m.get("role") == "tool"]
    assert any("42" in (m.get("content") or "") for m in tool_results)


def test_compose_unknown_tool_is_an_error(tmp_path):
    available = list_tools(tools_root=tmp_path / "nope")
    spec = Spec(name="x", tools=["does_not_exist"])
    with pytest.raises(ComposerError):
        generate("x", spec, available)
