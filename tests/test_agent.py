import pytest

from agent_seed.core import Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.tools import (
    ToolError,
    ToolRegistry,
    build_schema,
    calculator,
    type_to_schema,
)

from .conftest import FakeLLMClient, parse_arguments, text_completion, tool_completion


def _allow_all_tools(registry):
    def hook(ctx: RunContext) -> None:
        ctx.allowed_tools = list(registry.names())

    return hook


def test_calculator_schema():
    schema = build_schema(calculator)
    fn = schema["function"]
    assert fn["name"] == "calculator"
    assert fn["description"].startswith("Evaluate")
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "expression" in params["properties"]
    assert params["properties"]["expression"]["type"] == "string"
    assert params["required"] == ["expression"]


def test_registry_schemas_unique_and_stable():
    registry = ToolRegistry()
    registry.register(calculator)

    @registry.register
    def greet(name: str) -> str:
        """Say hi to someone."""
        return f"hi, {name}"

    schemas = registry.schemas()
    assert [s["function"]["name"] for s in schemas] == ["calculator", "greet"]
    # re-rendering is stable
    assert registry.schemas() == schemas


def test_registry_call_routes_kwargs():
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert registry.call("add", {"a": 1, "b": 2}) == 3


def test_registry_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.call("nope", {})


def test_type_to_schema_basics():
    import typing

    assert type_to_schema(str) == {"type": "string"}
    assert type_to_schema(int) == {"type": "number"}
    assert type_to_schema(bool) == {"type": "boolean"}
    assert type_to_schema(list[int]) == {
        "type": "array",
        "items": {"type": "number"},
    }
    assert type_to_schema(dict) == {"type": "object"}
    assert type_to_schema(typing.Any) == {}


# -- agent loop -------------------------------------------------------------


def test_agent_plain_answer_no_tools():
    registry = ToolRegistry()
    client = FakeLLMClient([text_completion("hello!"), text_completion("hi again")])
    agent = Agent(client, registry)

    assert agent.chat("hi") == "hello!"
    # first request carries the system prompt
    assert client.requests[0]["messages"][0]["role"] == "system"


def test_agent_uses_tool_then_answers():
    registry = ToolRegistry()
    registry.register(calculator)

    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "6 * 7"}),
            text_completion("42"),
        ]
    )
    # Strict tool policy: the agent must declare `calculator` as usable or the
    # core offers none and rejects the call.
    agent = Agent(client, registry, hooks=Hooks(enter_agent=_allow_all_tools(registry)))

    events = []
    answer = agent.chat(
        "what is 6 times 7?",
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    assert answer == "42"

    # the model must have been offered the tool schema
    assert client.requests[0]["tools"] == registry.schemas()

    # ...and the second round must include the tool result
    second = client.requests[1]["messages"]
    assert any(m.get("role") == "tool" and m["content"] == "42.0" for m in second)

    # ...and the assistant tool_call arguments were serialized as JSON
    assert parse_arguments(client.requests[1]) == {"expression": "6 * 7"}

    # the RE-facing event fired with the raw (non-string) result
    assert any(
        k == "tool" and p["name"] == "calculator" and p["result"] == "42.0" for k, p in events
    )


def test_agent_tool_error_is_reported_not_raised():
    registry = ToolRegistry()
    registry.register(calculator)

    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "1 / 0"}),
            text_completion("I cannot divide by zero"),
        ]
    )
    agent = Agent(client, registry)
    assert agent.chat("compute 1/0") == "I cannot divide by zero"

    second = client.requests[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("error:")


def test_agent_unknown_tool_is_rejected_when_not_in_whitelist():
    # Strict whitelist: a call to a tool the agent has NOT whitelisted is
    # rejected at the core (before any hook) with a well-formed role:tool
    # error — the LLM can adapt, the loop doesn't crash.
    registry = ToolRegistry()
    registry.register(calculator)
    client = FakeLLMClient(
        [
            tool_completion("nonexistent", {"x": 1}),
            text_completion("oops"),
        ]
    )
    agent = Agent(client, registry)  # allowed_tools unset (None) by default
    assert agent.chat("do the thing") == "oops"
    second = client.requests[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs and "not available to this agent" in tool_msgs[0]["content"]


# -- tool policy (strict whitelist) -------------------------------------------
#
# Default is NO tools: a run offers no tool schemas and rejects every tool
# call unless an agent (enter_agent) or hook (before_llm) sets
# ctx.allowed_tools. The core enforces it at BOTH the offer and execute
# points, and resets it to None on each agent switch.


def test_default_agent_offers_no_tools_and_rejects_calls():
    registry = ToolRegistry()
    registry.register(calculator)
    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "1+1"}),
            text_completion("nope"),
        ]
    )
    agent = Agent(client, registry)  # no tools declared
    assert agent.chat("q") == "nope"
    assert client.requests[0]["tools"] in (None, [])
    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "not available to this agent" in tool_msgs[0]["content"]


def test_enter_agent_whitelist_offers_only_listed_tools():
    registry = ToolRegistry()
    registry.register(calculator)

    @registry.register
    def other(x: int) -> int:
        """A second tool."""
        return x

    def _enter(ctx: RunContext) -> None:
        ctx.allowed_tools = ["calculator"]

    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, registry, hooks=Hooks(enter_agent=_enter))
    agent.chat("q")
    offered = [s["function"]["name"] for s in (client.requests[0]["tools"] or [])]
    assert offered == ["calculator"]  # `other` is registered but not offered


def test_allowed_tool_still_executes_and_fires_hooks():
    registry = ToolRegistry()
    registry.register(calculator)
    fired = {"before": 0, "after": 0}

    hooks = Hooks(
        enter_agent=lambda ctx: setattr(ctx, "allowed_tools", ["calculator"]),
        before_tool_call=lambda ctx: fired.__setitem__("before", fired["before"] + 1),
        after_tool_call=lambda ctx: fired.__setitem__("after", fired["after"] + 1),
    )
    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "6*7"}),
            text_completion("42"),
        ]
    )
    agent = Agent(client, registry, hooks=hooks)
    assert agent.chat("q") == "42"
    assert fired["before"] == 1 and fired["after"] == 1  # allowed -> hooks run
    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "42.0"


def test_rejected_tool_bypasses_all_tool_hooks():
    # A name outside the whitelist is settled by the core before
    # before_tool_call / after_tool_call get any chance.
    registry = ToolRegistry()
    registry.register(calculator)
    fired = {"before": 0, "after": 0}
    hooks = Hooks(
        before_tool_call=lambda ctx: fired.__setitem__("before", fired["before"] + 1),
        after_tool_call=lambda ctx: fired.__setitem__("after", fired["after"] + 1),
    )
    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "1"}),
            text_completion("blocked"),
        ]
    )
    agent = Agent(client, registry, hooks=hooks)  # no whitelist
    assert agent.chat("q") == "blocked"
    assert fired == {"before": 0, "after": 0}  # hooks never fired

    # ...whereas a declared allow-list does let the same tool through the
    # before/after hooks.
    hooks2 = Hooks(
        enter_agent=lambda ctx: setattr(ctx, "allowed_tools", ["calculator"]),
        before_tool_call=lambda ctx: fired.__setitem__("before", fired["before"] + 1),
        after_tool_call=lambda ctx: fired.__setitem__("after", fired["after"] + 1),
    )
    client2 = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "1"}),
            text_completion("ok"),
        ]
    )
    Agent(client2, registry, hooks=hooks2).chat("q")
    assert fired["before"] == 1 and fired["after"] == 1


def test_switch_to_agent_with_no_whitelist_resets_to_no_tools(tmp_path, monkeypatch):
    # Root agent whitelists calculator; 'bare' agent declares none. Switching
    # to 'bare' must reset allowed_tools to None (no stale leakage across the
    # shared RunContext).
    from agent_seed.discovery import reset_cache

    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))

    def _write(name: str, body: str) -> None:
        (tmp_path / name).write_text(body)

    _write(
        "root.py",
        "from agent_seed.hooks import Hooks\n"
        "def enter_agent(ctx): ctx.allowed_tools = ['calculator']\n"
        "def make() -> Hooks:\n    return Hooks(enter_agent=enter_agent)\n",
    )
    (tmp_path / "bare").mkdir()
    _write(
        "bare/b.py",
        "from agent_seed.hooks import Hooks\ndef make() -> Hooks:\n    return Hooks()\n",
    )
    reset_cache()

    registry = ToolRegistry()
    registry.register(calculator)
    client = FakeLLMClient([text_completion("a"), text_completion("b")])
    agent = Agent(client, registry)  # root/main agent: allowed_tools = ['calculator']
    assert agent.context.allowed_tools == ["calculator"]
    agent.set_agent_name("bare")  # bare declares nothing
    assert agent.context.allowed_tools is None  # reset, not inherited
    agent.chat("switched")
    offered = [s["function"]["name"] for s in (client.requests[0]["tools"] or [])]
    assert offered == []  # no tools offered after the switch


def test_before_llm_can_restrict_this_rounds_offer():
    # The offer is built from allowed_tools BEFORE before_llm (so a hook that
    # wants to change this round's offer edits ctx.tools directly — the
    # documented escape hatch, e.g. the existing ctx.tools.clear() pattern).
    registry = ToolRegistry()
    registry.register(calculator)

    @registry.register
    def other(x: int) -> int:
        """A second tool."""
        return x

    def _enter(ctx: RunContext) -> None:
        ctx.allowed_tools = list(registry.names())  # allow both at construction

    def _narrow(ctx: RunContext) -> None:
        ctx.tools = [s for s in ctx.tools if s["function"]["name"] == "calculator"]

    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, registry, hooks=Hooks(enter_agent=_enter, before_llm=_narrow))
    agent.chat("one")
    offered = [s["function"]["name"] for s in (client.requests[0]["tools"] or [])]
    assert offered == ["calculator"]


def test_allowed_tools_set_in_before_llm_governs_execute_veto():
    # Setting allowed_tools in before_llm is sticky: it governs the
    # execute-path veto for that round. A call to a tool the hook just
    # de-listed is rejected even though the (pre-hook) offer included it.
    registry = ToolRegistry()
    registry.register(calculator)

    def _revoke(ctx: RunContext) -> None:
        ctx.allowed_tools = []  # no tools allowed from this round on

    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "1"}),
            text_completion("blocked"),
        ]
    )
    agent = Agent(client, registry, hooks=Hooks(before_llm=_revoke))
    assert agent.chat("q") == "blocked"
    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert "not available to this agent" in tool_msgs[0]["content"]


def test_agent_respects_tool_round_cap():
    registry = ToolRegistry()
    registry.register(calculator)
    from agent_seed.core import MAX_TOOL_ROUNDS

    script = [tool_completion("calculator", {"expression": "1 + 1"})] * MAX_TOOL_ROUNDS
    script.append(text_completion("giving up gracefully"))
    client = FakeLLMClient(script)
    agent = Agent(client, registry)
    assert agent.chat("loop forever?") == "giving up gracefully"
    # MAX_TOOL_ROUNDS tool calls + 1 final answer = MAX_TOOL_ROUNDS + 1 requests
    assert len(client.requests) == MAX_TOOL_ROUNDS + 1
    # final request must NOT include tool schemas
    assert client.requests[-1]["tools"] is None


def test_agent_reset_clears_history():
    client = FakeLLMClient([text_completion("a"), text_completion("b")])
    agent = Agent(client, ToolRegistry())
    agent.chat("first")
    assert agent.history
    agent.reset()
    assert agent.history == []


def test_on_token_streams_only_final_content():
    registry = ToolRegistry()
    registry.register(calculator)
    client = FakeLLMClient(
        [
            tool_completion("calculator", {"expression": "2+2"}),
            text_completion("4"),
        ]
    )
    agent = Agent(client, registry)
    tokens: list[str] = []
    agent.chat("q", on_token=tokens.append)
    assert tokens == ["4"]
