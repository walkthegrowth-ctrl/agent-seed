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


def test_compose_agent_folder_reaches_the_llm_request(tmp_path, monkeypatch):
    """The composed voice / context / examples must land in the *sent*
    messages: the system message is identity + context folded together, and
    the few-shot examples ride between the system and the history.

    This is the "the composed agent actually *behaves* the way its files
    claim" assertion -- not just that the files exist or the hooks are loadable.
    """
    reservoir = _make_reservoir_tools(tmp_path)
    available = list_tools(tools_root=reservoir)

    spec = Spec(
        name="tutor",
        voice="You are X.",
        context="C-1.",
        examples=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        signoff="— T",
        tools=["double"],
    )
    plan = generate(spec.name, spec, available)
    hooks_root = tmp_path / "hook_modules"
    write_plan(plan, hooks_root, force=False)
    monkeypatch.setenv("HOOKS_ROOT", str(hooks_root))
    reset_cache()

    hooks = get_hooks_for_agent("tutor")
    hooks.enter_agent(RunContext(agent_name="tutor"))

    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, registry, hooks=hooks, agent_name="tutor")
    agent.chat("hi")

    request = client.requests[-1]
    messages = request["messages"]
    # The system message has both the voice and the context folded in.
    system = messages[0]
    assert system["role"] == "system"
    assert "You are X." in system["content"]
    assert "C-1." in system["content"]
    # The few-shot examples appear as the next two messages (between system
    # and the user turn).
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    # And the user's real turn is *after* the examples.
    assert messages[-1] == {"role": "user", "content": "hi"}
    # The reservoir tool is *offered* in the tool schema of the request.
    tool = request["tools"]
    assert tool, "the composed agent should offer its whitelisted tool"
    names = [t["function"]["name"] for t in tool]
    assert "double" in names, f"the request tools should include 'double': {names}"


def test_compose_agent_signoff_is_appended_to_final_answer(tmp_path, monkeypatch):
    """The composed signoff must be appended to the final answer (via the
    generated ``finish`` hook)."""
    reservoir = _make_reservoir_tools(tmp_path)
    available = list_tools(tools_root=reservoir)

    spec = Spec(name="tutor", voice="You are X.", signoff="— T", tools=["double"])
    plan = generate(spec.name, spec, available)
    hooks_root = tmp_path / "hook_modules"
    write_plan(plan, hooks_root, force=False)
    monkeypatch.setenv("HOOKS_ROOT", str(hooks_root))
    reset_cache()

    hooks = get_hooks_for_agent("tutor")
    hooks.enter_agent(RunContext(agent_name="tutor"))

    client = FakeLLMClient([text_completion("hello")])
    agent = Agent(client, registry, hooks=hooks, agent_name="tutor")
    answer = agent.chat("hi")
    assert answer == "hello\n— T"


def test_compose_agent_without_tools_offers_no_schemas(tmp_path, monkeypatch):
    """A composed agent without tools must not send *any* tool schema (the
    strict whitelist contract: default = no tools)."""
    reservoir = _make_reservoir_tools(tmp_path)
    available = list_tools(tools_root=reservoir)

    spec = Spec(name="tutor", voice="You are X.")  # no tools
    plan = generate(spec.name, spec, available)
    hooks_root = tmp_path / "hook_modules"
    write_plan(plan, hooks_root, force=False)
    monkeypatch.setenv("HOOKS_ROOT", str(hooks_root))
    reset_cache()

    hooks = get_hooks_for_agent("tutor")
    # A no-tools agent has no ``enter_agent`` to set the whitelist -- the core
    # defaults ``ctx.allowed_tools`` to ``None`` (strict: no tools).
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, registry, hooks=hooks, agent_name="tutor")
    agent.chat("hi")

    request = client.requests[-1]
    # No tools offered.
    assert not request["tools"], (
        f"a no-tools agent must offer no tool schemas, got: {request['tools']}"
    )
