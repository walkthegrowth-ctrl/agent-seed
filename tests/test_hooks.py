"""Tests for the hook surface (agent_seed/hooks.py) wired into Agent.chat().

Covers: firing order, ID establishment (provided / initialize-supplied /
auto-generated), turn counting, reset semantics, before_llm injection
without history pollution, tool vetoes and rewrites, result rewrites,
prepare/finish, extras coordination, and the ahem experiment.
"""

from typing import Any

from agent_seed.core import DEFAULT_SYSTEM_PROMPT, Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.tools import ToolRegistry, calculator

from .conftest import FakeLLMClient, text_completion, tool_completion


def _agent(
    script: list,
    hooks: Hooks | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> tuple[Agent, FakeLLMClient]:
    """Build an Agent with the calculator tool and a scripted LLM client.

    Under strict tool-whitelist policy this helper pins ``ctx.allowed_tools``
    to the whole registry at construction so existing tests that exercise
    tool execution keep working as before. Tests that want NO tools (or a
    specific subset) can override ``enter_agent`` in ``hooks``.
    """
    registry = ToolRegistry()
    registry.register(calculator)
    client = FakeLLMClient(script)

    from dataclasses import replace

    def _allow_all_tools(ctx: RunContext) -> None:
        ctx.allowed_tools = list(registry.names())

    if hooks is None:
        effective = Hooks(enter_agent=_allow_all_tools)
    elif hooks.enter_agent is None:
        effective = replace(hooks, enter_agent=_allow_all_tools)
    else:
        effective = hooks

    agent = Agent(
        client,
        registry,
        system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        hooks=effective,
        run_id=run_id,
        session_id=session_id,
    )
    return agent, client


# -- firing order -------------------------------------------------------------


def test_all_hooks_fire_in_documented_order():
    order: list[str] = []

    def tag(name: str):
        def hook(ctx: RunContext) -> None:
            order.append(name)

        return hook

    hooks = Hooks(
        initialize_run=tag("initialize_run"),
        ready_run=tag("ready_run"),
        initialize_session=tag("initialize_session"),
        ready_session=tag("ready_session"),
        before_llm=tag("before_llm"),
        before_tool_call=tag("before_tool_call"),
        after_tool_call=tag("after_tool_call"),
        on_round=tag("on_round"),
        prepare=tag("prepare"),
        finish=tag("finish"),
    )

    agent, _ = _agent(
        [tool_completion("calculator", {"expression": "1 + 1"}), text_completion("2")],
        hooks,
    )
    agent.chat("one plus one?")

    assert order == [
        "initialize_run",
        "ready_run",
        "initialize_session",
        "ready_session",
        "before_llm",
        "before_tool_call",
        "after_tool_call",
        "on_round",
        "before_llm",
        "prepare",
        "finish",
    ]


# -- id establishment ----------------------------------------------------------


def test_ids_auto_generated_are_stable_per_run():
    hooks = Hooks(
        ready_run=lambda ctx: ctx.extras.__setitem__("run_id", ctx.run_id),
        ready_session=lambda ctx: ctx.extras.__setitem__("session_id", ctx.session_id),
    )
    agent, _ = _agent([text_completion("hi"), text_completion("bye")], hooks)

    agent.chat("a")
    run_id = agent.run_id
    session_id = agent.session_id
    assert run_id and session_id
    assert agent.context.extras["run_id"] == run_id

    agent.chat("b")
    assert agent.run_id == run_id  # same across turns (one context)
    assert agent.session_id == session_id


def test_initialization_hooks_can_supply_ids():
    seen: dict[str, Any] = {}

    def init_run(ctx: RunContext) -> None:
        ctx.run_id = "run-supplied"

    def init_session(ctx: RunContext) -> None:
        ctx.session_id = "sess-supplied"

    def ready(ctx: RunContext) -> None:
        seen["run_id"] = ctx.run_id
        seen["session_id"] = ctx.session_id

    hooks = Hooks(
        initialize_run=init_run,
        ready_run=ready,
        initialize_session=init_session,
        ready_session=ready,
    )
    agent, _ = _agent([text_completion("ok")], hooks)
    agent.chat("hello")

    assert seen == {"run_id": "run-supplied", "session_id": "sess-supplied"}
    assert agent.run_id == "run-supplied"
    assert agent.session_id == "sess-supplied"


def test_caller_supplied_ids_are_respected():
    agent, _ = _agent(
        [text_completion("ok")],
        Hooks(),
        run_id="caller-run",
        session_id="caller-sess",
    )
    agent.chat("hello")
    assert agent.run_id == "caller-run"
    assert agent.session_id == "caller-sess"


def test_ready_hooks_see_non_none_ids():
    # nothing supplied: auto-gen must fill both before ready_* fires
    agent, _ = _agent(
        [text_completion("a")],
        Hooks(
            ready_run=lambda ctx: _assert_non_none("run_id", ctx.run_id),
            ready_session=lambda ctx: _assert_non_none("session_id", ctx.session_id),
        ),
    )
    agent.chat("x")


def _assert_non_none(name: str, value: Any) -> None:
    assert value is not None, f"{name} is None where it must be non-None"


def test_run_establishment_fires_exactly_once():
    order: list[str] = []
    hooks = Hooks(
        initialize_run=lambda ctx: order.append("init_run"),
        ready_run=lambda ctx: order.append("ready_run"),
        initialize_session=lambda ctx: order.append("init_session"),
        ready_session=lambda ctx: order.append("ready_session"),
        before_llm=lambda ctx: None,
    )
    agent, _ = _agent([text_completion("a"), text_completion("b"), text_completion("c")], hooks)
    agent.chat("1")
    agent.chat("2")
    agent.chat("3")
    assert order == ["init_run", "ready_run", "init_session", "ready_session"]


def test_session_change_refires_session_hooks_not_run_hooks():
    order: list[str] = []

    def tag(name: str):
        def h(ctx: RunContext) -> None:
            order.append(name)

        return h

    hooks = Hooks(
        initialize_run=tag("init_run"),
        ready_run=tag("ready_run"),
        initialize_session=tag("init_session"),
        ready_session=tag("ready_session"),
    )
    agent, _ = _agent(
        [text_completion("a"), text_completion("b")],
        hooks,
        session_id="first",
    )
    agent.chat("1")
    agent.set_session("second")
    agent.chat("2")
    assert order == [
        "init_run",
        "ready_run",
        "init_session",
        "ready_session",
        # second chat, new session: only the session hooks fire again
        "init_session",
        "ready_session",
    ]
    assert agent.session_id == "second"


def test_set_session_same_id_is_a_noop():
    order: list[str] = []

    def tag(name: str):
        def h(ctx: RunContext) -> None:
            order.append(name)

        return h

    hooks = Hooks(initialize_session=tag("init_session"), ready_session=tag("ready_session"))
    agent, _ = _agent([text_completion("a"), text_completion("b")], hooks, session_id="s")
    agent.chat("1")
    agent.set_session("s")
    agent.chat("2")
    assert order.count("ready_session") == 1


def test_reset_clears_turn_count_but_keeps_run_and_session():
    agent, _ = _agent([text_completion("a"), text_completion("b")])
    agent.chat("first")
    run_id, session_id = agent.run_id, agent.session_id
    assert agent.context.turn == 1

    agent.reset()
    assert agent.run_id == run_id
    assert agent.session_id == session_id


def test_extras_persist_across_resets():
    def stash(ctx: RunContext) -> None:
        ctx.extras["kept"] = ctx.extras.get("kept", 0) + 1

    agent, _ = _agent(
        [text_completion("a"), text_completion("b")],
        Hooks(before_llm=stash),
    )
    agent.chat("1")
    after_first = agent.context.extras["kept"]
    agent.reset()
    agent.chat("2")
    # extras is a run-scoped accumulator: reset() must not clear it
    assert agent.context.extras["kept"] == after_first + 1


def test_turn_increments_across_chats():
    hooks_seen: list[int] = []
    agent, _ = _agent(
        [text_completion("a"), text_completion("b"), text_completion("c"), text_completion("d")],
        Hooks(before_llm=lambda ctx: hooks_seen.append(ctx.turn)),
    )
    agent.chat("1")
    agent.chat("2")
    agent.chat("3")
    agent.reset()
    agent.chat("4")
    assert hooks_seen == [1, 2, 3, 1]


def test_tool_loop_round_is_set_per_iteration():
    rounds: list[int] = []
    agent, _ = _agent(
        [
            tool_completion("calculator", {"expression": "1+1"}),
            tool_completion("calculator", {"expression": "2+2"}),
            text_completion("done"),
        ],
        Hooks(before_llm=lambda ctx: rounds.append(ctx.tool_loop_round)),
    )
    agent.chat("q")
    assert rounds == [0, 1, 2]


# -- before_llm ----------------------------------------------------------------


def test_before_llm_injection_is_per_request_not_history():
    def inject(ctx: RunContext) -> None:
        ctx.messages.append({"role": "system", "content": "injected context"})

    agent, client = _agent([text_completion("hi")], Hooks(before_llm=inject))
    agent.chat("hello")

    sent = client.requests[0]["messages"]
    assert {"role": "system", "content": "injected context"} in sent
    # history must NOT contain the injected message
    assert not any(m.get("content") == "injected context" for m in agent.history)


def test_before_llm_can_reject_tools():
    agent, client = _agent(
        [text_completion("hi")], Hooks(before_llm=lambda ctx: ctx.tools.clear())
    )
    agent.chat("hi")
    assert client.requests[0]["tools"] in (None, [])


# -- tool call hooks -----------------------------------------------------------


def test_before_tool_call_rewrites_arguments():
    def rewrite(ctx: RunContext) -> None:
        ctx.tool_call["arguments"]["expression"] = "6 * 7"

    agent, client = _agent(
        [
            tool_completion("calculator", {"expression": "1 + 1"}),
            text_completion("42.0"),
        ],
        Hooks(before_tool_call=rewrite),
    )
    agent.chat("compute")

    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "42.0"


def test_before_tool_call_veto_records_policy_message_and_completes():
    def veto(ctx: RunContext) -> None:
        ctx.tool_call = None

    agent, _ = _agent(
        [
            tool_completion("calculator", {"expression": "1+1"}),
            text_completion("I cannot use tools"),
        ],
        Hooks(before_tool_call=veto),
    )
    assert agent.chat("q") == "I cannot use tools"

    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "skipped by policy" in tool_msgs[0]["content"]


def test_after_tool_call_rewrites_result():
    def rewrite(ctx: RunContext) -> None:
        ctx.tool_result = f"[redacted] {ctx.tool_result}"

    agent, _ = _agent(
        [
            tool_completion("calculator", {"expression": "6*7"}),
            text_completion("final"),
        ],
        Hooks(after_tool_call=rewrite),
    )
    agent.chat("q")

    tool_msgs = [m for m in agent.history if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "[redacted] 42.0"


# -- prepare / finish ----------------------------------------------------------


def test_prepare_sees_final_answer():
    seen: dict[str, str] = {}
    agent, _ = _agent(
        [text_completion("hello")],
        Hooks(prepare=lambda ctx: seen.__setitem__("answer", ctx.final_answer or "")),
    )
    agent.chat("q")
    assert seen == {"answer": "hello"}


def test_finish_can_rewrite_final_answer():
    agent, _ = _agent(
        [text_completion("hi")], Hooks(finish=lambda ctx: ctx.__setattr__("final_answer", "bye"))
    )
    assert agent.chat("q") == "bye"
    assert agent.history[-1] == {"role": "assistant", "content": "bye"}


def test_ahem_experiment_appends_to_final_answer():
    from experiments.ahem import make

    agent, _ = _agent([text_completion("42")], make())
    assert agent.chat("what is 6 * 7?") == "42\n(ahem)"


def test_ahem_also_applies_after_tool_rounds_and_cap():
    from agent_seed.core import MAX_TOOL_ROUNDS
    from experiments.ahem import make

    script = [tool_completion("calculator", {"expression": "1+1"})] * MAX_TOOL_ROUNDS
    script.append(text_completion("giving up"))
    agent, _ = _agent(script, make())
    assert agent.chat("loop") == "giving up\n(ahem)"


# -- extras coordination ---------------------------------------------------------


def test_extras_allows_inter_hook_coordination():
    def count_calls(ctx: RunContext) -> None:
        ctx.extras["llm_calls"] = ctx.extras.get("llm_calls", 0) + 1

    def budget_guard(ctx: RunContext) -> None:
        if ctx.extras.get("llm_calls", 0) > 10:
            raise AssertionError("budget exceeded")

    agent, _ = _agent(
        [
            tool_completion("calculator", {"expression": "1"}),
            tool_completion("calculator", {"expression": "2"}),
            text_completion("done"),
        ],
        Hooks(before_llm=count_calls, on_round=budget_guard, finish=budget_guard),
    )
    answer = agent.chat("q")
    assert answer == "done"


def test_usage_is_accumulated_from_client():
    class UsageClient(FakeLLMClient):
        def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
            completion = super().chat(
                messages, tools=tools, on_token=on_token, config=config, max_tokens=max_tokens
            )
            completion.usage = {"prompt_tokens": 10, "completion_tokens": 5}
            return completion

    client = UsageClient([text_completion("hi")])
    agent = Agent(client, ToolRegistry())
    agent.chat("q")
    assert agent.context.usage.get("prompt_tokens") == 10
    assert agent.context.usage.get("completion_tokens") == 5


# -- defaults / regression -------------------------------------------------------


def test_no_hooks_configured_behaves_exactly_as_before():
    agent, client = _agent([text_completion("hello!")])
    assert agent.chat("hi") == "hello!"
    assert client.requests[0]["messages"][0]["role"] == "system"


def test_hooks_default_hooks_dataclass_is_empty():
    h = Hooks()
    for name in (
        "enter_agent",
        "exit_agent",
        "initialize_run",
        "ready_run",
        "initialize_session",
        "ready_session",
        "before_llm",
        "before_tool_call",
        "after_tool_call",
        "on_round",
        "prepare",
        "finish",
    ):
        assert getattr(h, name) is None


def test_run_context_fields_have_sane_defaults():
    ctx = RunContext()
    assert ctx.run_id is None
    assert ctx.session_id is None
    assert ctx.agent_name is None
    assert ctx.turn == 0
    assert ctx.tool_loop_round == 0
    assert ctx.base_prompt is None
    assert ctx.system_prompt is None
    assert ctx.messages == []
    assert ctx.tools == []
    assert ctx.tool_call is None
    assert ctx.tool_result is None
    assert ctx.final_answer is None
    assert ctx.usage == {}
    assert ctx.extras == {}


# -- system prompt (dynamic, sticky) -------------------------------------------


def test_default_system_prompt_is_sent_and_is_base_prompt():
    base = "You are terse."
    agent, client = _agent([text_completion("hi")], system_prompt=base)
    agent.chat("q")
    assert agent.base_prompt == base
    assert client.requests[0]["messages"][0] == {"role": "system", "content": base}


def test_hook_may_extend_base_prompt_and_change_persists():
    agent, client = _agent([text_completion("a"), text_completion("b")])
    base = agent.base_prompt

    # Correct pattern: recompute from base (not "append to current") so the
    # change is stable across turns rather than accumulating.
    def extend(ctx: RunContext) -> None:
        ctx.system_prompt = (ctx.base_prompt or "") + " Be verbose."

    agent.hooks.before_llm = extend
    agent.chat("1")
    first = client.requests[0]["messages"][0]["content"]
    assert first == base + " Be verbose."

    agent.chat("2")
    second = client.requests[1]["messages"][0]["content"]
    # Recomputed the same way on turn 2 -> identical, no double-append.
    assert second == first
    assert second.count("Be verbose.") == 1


def test_accumulating_hook_is_hook_error_not_core():
    # Document that "append to the CURRENT prompt" accumulates (a hook bug).
    # The core stays faithful: it sends exactly what the hook left there.
    agent, client = _agent([text_completion("a"), text_completion("b")])

    def append_to_current(ctx: RunContext) -> None:
        ctx.system_prompt = (ctx.system_prompt or "") + "!"

    agent.hooks.before_llm = append_to_current
    agent.chat("1")
    agent.chat("2")
    assert client.requests[0]["messages"][0]["content"].endswith("!")
    assert client.requests[1]["messages"][0]["content"].endswith("!!")


def test_hook_may_replace_the_prompt_in_full():
    agent, client = _agent([text_completion("hi")])
    agent.hooks.before_llm = lambda ctx: ctx.__setattr__("system_prompt", "ROBOT MODE")
    agent.chat("q")
    sent = client.requests[0]["messages"][0]["content"]
    assert sent == "ROBOT MODE"
    # base is still recoverable
    assert agent.base_prompt != "ROBOT MODE"


def test_hook_can_restore_base_prompt_from_ctx():
    agent, client = _agent([text_completion("a"), text_completion("b")])
    base = agent.base_prompt

    def toggle(ctx: RunContext) -> None:
        # emulate a two-turn policy: add an addendum on turn 1, drop it on turn 2
        ctx.system_prompt = base + (" Addendum A." if ctx.turn == 1 else "")

    agent.hooks.before_llm = toggle
    agent.chat("1")
    agent.chat("2")
    assert "Addendum A." in client.requests[0]["messages"][0]["content"]
    assert "Addendum A." not in client.requests[1]["messages"][0]["content"]
    assert client.requests[1]["messages"][0]["content"] == base


def test_multi_hook_can_both_see_base_prompt():
    # Two hooks (simulating two modules) cooperate: 1 adds, 2 goes back to
    # base and adds its own. Without base, 2 would extend the wrong base.
    def first(ctx: RunContext) -> None:
        ctx.system_prompt = (ctx.base_prompt or "") + " +one"

    def second(ctx: RunContext) -> None:
        ctx.system_prompt = (ctx.base_prompt or "") + " +two"

    def composed(ctx: RunContext) -> None:
        first(ctx)
        second(ctx)

    agent, client = _agent([text_completion("hi")])
    base = agent.base_prompt
    agent.hooks.before_llm = composed
    agent.chat("q")
    assert client.requests[0]["messages"][0]["content"] == base + " +two"


def test_system_prompt_experiment_is_dynamic_and_stable_across_turns():
    from experiments.system_prompt import make

    agent, client = _agent([text_completion("a"), text_completion("b")])
    agent.hooks = make()
    agent.chat("1")
    p1 = client.requests[0]["messages"][0]["content"]
    agent.chat("2")
    p2 = client.requests[1]["messages"][0]["content"]

    # both turns carry the base AND the dynamic addendum
    assert agent.base_prompt in p1
    assert agent.base_prompt in p2
    assert f"turn {1}" in p1 and f"turn {2}" in p2
    # the dynamic context appears exactly once per turn (no drift)
    assert p1.count("[dynamic context:") == 1
    assert p2.count("[dynamic context:") == 1
    # extras were seeded by ready_session
    assert agent.context.extras.get("session_id") == agent.session_id
