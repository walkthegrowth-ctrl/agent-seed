"""Runtime model selection: set_model, on_model_change, stickiness, test_model."""

import pytest

from agent_seed.core import Agent
from agent_seed.hooks import Hooks
from agent_seed.llm import Completion, LLMConfig
from agent_seed.tools import ToolRegistry

from .conftest import FakeLLMClient, text_completion


def make_cfg(model: str = "fake-model", base_url: str = "http://a.local/v1") -> LLMConfig:
    return LLMConfig(base_url=base_url, api_key="unused", model=model)


def make_agent(hooks: Hooks | None = None) -> tuple[Agent, FakeLLMClient]:
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    return agent, client


def test_base_and_current_model_seed_from_client_config():
    agent, client = make_agent()
    assert agent.context.base_model == client.config
    assert agent.context.current_model == client.config
    assert agent.model_config == client.config


def test_chat_uses_current_model():
    agent, client = make_agent()
    agent.chat("q")
    assert client.requests[-1]["config"] is client.config


def test_set_model_switches_the_called_model():
    agent, client = make_agent()
    new = make_cfg("model-b")
    assert agent.set_model(new) == new
    agent.chat("q")
    assert client.requests[-1]["config"] == new
    # base_model is untouched and remains the restore target
    assert agent.context.base_model == client.config


def test_set_model_noop_when_unchanged():
    agent, client = make_agent()
    assert agent.set_model(agent.context.current_model) == agent.context.current_model


def test_set_model_noop_on_value_equal_distinct_instance():
    # core.py compares LLMConfig with `==` (frozen dataclass): a *different*
    # object that is value-equal to the current model must be a no-op.
    agent, client = make_agent()
    src = client.config
    duplicate = LLMConfig(base_url=src.base_url, api_key=src.api_key, model=src.model)
    assert duplicate is not src
    assert duplicate == src
    assert agent.set_model(duplicate) == duplicate
    agent.chat("q")
    assert client.requests[-1]["config"] == src


def test_before_llm_can_set_current_model_for_that_round():
    # Priority tier 4 (lowest, per-request): a before_llm hook wins at request
    # time and pins the model the core calls for that call.
    new = make_cfg("final-wins")

    def bind(ctx):
        ctx.current_model = new

    agent, client = make_agent(Hooks(before_llm=bind))
    agent.chat("q")
    assert client.requests[-1]["config"] == new


def test_on_model_change_can_mutate_the_committed_value():
    new = make_cfg("model-b")
    target = make_cfg("model-final")

    def on_model_change(ctx):
        ctx.current_model = target

    agent, client = make_agent(Hooks(on_model_change=on_model_change))
    assert agent.set_model(new) == target
    agent.chat("q")
    assert client.requests[-1]["config"] == target


def test_on_model_change_veto_rolls_back():
    def on_model_change(ctx):
        raise RuntimeError("policy: no model changes")

    agent, client = make_agent(Hooks(on_model_change=on_model_change))
    assert agent.context.current_model == client.config
    with pytest.raises(RuntimeError, match="policy"):
        agent.set_model(make_cfg("nope"))
    assert agent.context.current_model == client.config


def test_current_model_is_sticky_across_agent_switch():
    # enter_agent whitelists tools but never touches the model: a prior
    # set_model must survive the switch (the core does not reset it).
    def enter_agent(ctx):
        ctx.allowed_tools = ["calculator"]

    hooks = Hooks(enter_agent=enter_agent)
    agent, client = make_agent(hooks)
    agent.set_model(make_cfg("kept-model"))
    agent.set_agent_name("alpha")
    assert agent.context.current_model == make_cfg("kept-model")
    assert agent.context.base_model == client.config


def test_enter_agent_can_restore_base_model():
    # The canonical restore pattern: a hook that wants a fresh agent to
    # ignore the previous agent's choice resets to ctx.base_model.
    def enter_agent(ctx):
        ctx.allowed_tools = ["calculator"]
        ctx.current_model = ctx.base_model

    hooks = Hooks(enter_agent=enter_agent)
    agent, client = make_agent(hooks)
    agent.set_model(make_cfg("old-choice"))
    agent.set_agent_name("alpha")
    assert agent.context.current_model == client.config  # back to base
    agent.chat("q")
    assert client.requests[-1]["config"] == client.config


# -- Agent.test_model -----------------------------------------------------------


class _ProbeClient:
    """Records the probe call and yields a scripted outcome."""

    def __init__(self, config, outcome: Exception | Completion):
        self.config = config
        self.outcome = outcome

    def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
        self.seen_max_tokens = max_tokens
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_test_model_reports_success(monkeypatch):
    def factory(config):
        return _ProbeClient(
            config, Completion(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1})
        )

    monkeypatch.setattr("agent_seed.core.OpenAICompatClient", factory)
    probe = Agent.test_model(make_cfg("probe-model"))
    assert probe.ok
    assert probe.error is None
    assert probe.usage == {"prompt_tokens": 1, "completion_tokens": 1}
    assert probe.latency_ms is not None


def test_test_model_reports_failure(monkeypatch):
    def factory(config):
        return _ProbeClient(config, RuntimeError("connection refused"))

    monkeypatch.setattr("agent_seed.core.OpenAICompatClient", factory)
    probe = Agent.test_model(make_cfg("probe-model"))
    assert not probe.ok
    assert "connection refused" in probe.error


def test_test_model_sends_a_minimal_ping(monkeypatch):
    instance: dict = {}

    def factory(config):
        client = _ProbeClient(config, Completion(content="ok"))
        instance["client"] = client
        return client

    monkeypatch.setattr("agent_seed.core.OpenAICompatClient", factory)
    Agent.test_model(make_cfg("probe-model"))
    assert instance["client"].seen_max_tokens == 1
