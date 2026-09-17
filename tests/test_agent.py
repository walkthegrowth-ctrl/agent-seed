import pytest

from agent_seed.core import Agent
from agent_seed.tools import (
    ToolError,
    ToolRegistry,
    build_schema,
    calculator,
    type_to_schema,
)

from .conftest import FakeLLMClient, parse_arguments, text_completion, tool_completion


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
    agent = Agent(client, registry)

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


def test_agent_unknown_tool_error_is_reported():
    registry = ToolRegistry()
    client = FakeLLMClient(
        [
            tool_completion("nonexistent", {"x": 1}),
            text_completion("oops"),
        ]
    )
    agent = Agent(client, registry)
    assert agent.chat("do the thing") == "oops"
    second = client.requests[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs and "unknown tool" in tool_msgs[0]["content"]


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
