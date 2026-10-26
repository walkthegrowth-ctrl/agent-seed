"""Shared test fixtures: a fake LLM client driving the agent loop."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_seed.discovery import reset_cache
from agent_seed.llm import Completion, LLMConfig, ToolCall


@pytest.fixture(autouse=True)
def _hermetic_hook_discovery(tmp_path, monkeypatch):
    """Keep the test suite hermetic against any user hook_modules/.

    Agent(hooks=None) auto-discovers from hook_modules/; point the default
    folder at an empty temp dir so a real hooks/ folder can never feed the
    tests (e.g. a veto hook silently skipping calculator).
    """
    reset_cache()
    empty = tmp_path / "no_hooks"
    empty.mkdir()
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(empty))
    reset_cache()
    yield
    reset_cache()


class FakeLLMClient:
    """Yields scripted completions; records every request it receives."""

    def __init__(
        self,
        script: list[Completion],
        config: LLMConfig | None = None,
    ) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.config = config or LLMConfig(
            base_url="http://fake.local/v1", api_key="unused", model="fake-model"
        )

    def chat(
        self,
        messages,
        tools=None,
        on_token=None,
        config=None,
        max_tokens=None,
    ) -> Completion:
        self.requests.append({"messages": messages, "tools": tools, "config": config})
        if not self.script:
            raise AssertionError("FakeLLMClient: no scripted completion left")
        completion = self.script.pop(0)
        if on_token is not None and completion.content:
            on_token(completion.content)
        return completion


def tool_completion(
    name: str,
    arguments: dict[str, Any],
    content: str | None = None,
    usage: dict[str, Any] | None = None,
) -> Completion:
    return Completion(
        content=content,
        tool_calls=[ToolCall(id="call_1", name=name, arguments=arguments)],
        usage=usage,
    )


def text_completion(text: str, usage: dict[str, Any] | None = None) -> Completion:
    return Completion(content=text, usage=usage)


def parse_arguments(request: dict[str, Any], tool_call_index: int = 0) -> dict[str, Any]:
    """Recover tool arguments as sent from the last assistant message."""
    messages = request["messages"]
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            args = message["tool_calls"][tool_call_index]["function"]["arguments"]
            return json.loads(args)
    raise AssertionError("no assistant tool_call message found")
