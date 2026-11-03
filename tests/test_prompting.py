"""Tests for the prompting layer: compose_system, normalize_examples, compose_prompt,
and their wiring into Agent (per-round context/examples/addendum baselines, non-sticky
reset each round, priority of system_prompt vs hooks.prompt, and the hooks' full
override power over the composed message).

All tests use the hermetic hook-discovery fixture in conftest.py so no real
hook_modules/ folder leaks in.
"""

from agent_seed.core import DEFAULT_SYSTEM_PROMPT, Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.prompting import compose_prompt, compose_system, normalize_examples
from agent_seed.tools import ToolRegistry

from .conftest import FakeLLMClient, text_completion

# -- compose_system --------------------------------------------------------------


def test_compose_system_identity_only():
    assert compose_system("You are terse.") == "You are terse."


def test_compose_system_adds_context_block():
    out = compose_system("You are terse.", "Current: date=2026-10-27")
    assert out == "You are terse.\n\nCurrent: date=2026-10-27"


def test_compose_system_adds_addendum_last():
    out = compose_system("You are terse.", "ctx-block", "note-1")
    assert out == "You are terse.\n\nctx-block\n\nnote-1"


def test_compose_system_drops_blank_sections():
    out = compose_system("You are terse.", "  ", "", None)
    assert out == "You are terse."


def test_compose_system_all_blank_returns_empty():
    assert compose_system(None, "", "  ") == ""


def test_compose_system_strips_outer_whitespace_of_sections():
    out = compose_system("  identity  ", "  context  ", "  addendum  ")
    assert out == "identity\n\ncontext\n\naddendum"


# -- normalize_examples --------------------------------------------------------


def test_normalize_examples_none_and_empty():
    assert normalize_examples(None) == []
    assert normalize_examples([]) == []


def test_normalize_examples_identity_and_tool_calls_are_preserved():
    examples = [
        {"role": "user", "content": "what is 12*13?"},
        {"role": "assistant", "content": None, "tool_calls": [{"name": "calculator"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "156.0"},
        {"role": "assistant", "content": "It's 156."},
    ]
    out = normalize_examples(examples)
    assert out == examples
    # Deep-copy: mutating the returned list must not touch the input.
    out[0]["content"] = "mutated"
    assert examples[0]["content"] == "what is 12*13?"
    out[1]["tool_calls"][0]["name"] = "mutated"
    assert examples[1]["tool_calls"][0]["name"] == "calculator"


def test_normalize_examples_drops_items_without_role():
    examples = [
        {"role": "user", "content": "q"},
        {"content": "no role, dropped"},
        {"role": "assistant", "content": "a"},
    ]
    out = normalize_examples(examples)
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_normalize_examples_rejects_non_dict_items():
    out = normalize_examples(["a string", 42, {"role": "user", "content": "ok"}])
    assert out == [{"role": "user", "content": "ok"}]


# -- compose_prompt --------------------------------------------------------------


def test_compose_prompt_backward_compatible_no_optional_fields():
    """With no context/examples/addendum, compose_prompt returns the legacy
    [system, *history] shape — byte-identical to what the core sent before
    this layer existed."""
    identity = "You are terse."
    history = [{"role": "user", "content": "q"}]
    out = compose_prompt(identity, history)
    assert out == [{"role": "system", "content": identity}, {"role": "user", "content": "q"}]


def test_compose_prompt_with_context():
    out = compose_prompt("id", [], context="ctx-block")
    assert out == [{"role": "system", "content": "id\n\nctx-block"}]


def test_compose_prompt_with_examples_between_system_and_history():
    examples = [{"role": "user", "content": "e-user"}, {"role": "assistant", "content": "e-asst"}]
    history = [{"role": "user", "content": "q"}]
    out = compose_prompt("id", history, examples=examples)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in out] == ["id", "e-user", "e-asst", "q"]


def test_compose_prompt_full_stack():
    out = compose_prompt(
        "id",
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        context="ctx-block",
        examples=[{"role": "user", "content": "e"}],
        addendum="note-1",
    )
    assert out[0] == {"role": "system", "content": "id\n\nctx-block\n\nnote-1"}
    assert out[1:] == [
        {"role": "user", "content": "e"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_compose_prompt_copies_history():
    history = [{"role": "user", "content": "q"}]
    out = compose_prompt("id", history)
    out[-1]["content"] = "mutated"
    assert history[0]["content"] == "q"


# -- Agent integration: declarative baselines + per-round seed --------------------


def test_hooks_context_baseline_appears_every_round():
    hooks = Hooks(context="ctx-baseline")
    client = FakeLLMClient([text_completion("a"), text_completion("b")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("1")
    agent.chat("2")
    # No tools used, so one request per turn.
    for i, req in enumerate(client.requests):
        assert "ctx-baseline" in req["messages"][0]["content"], f"round {i}"
    # The baseline is present exactly once per round (no accumulation).
    assert client.requests[0]["messages"][0]["content"].count("ctx-baseline") == 1
    assert client.requests[1]["messages"][0]["content"].count("ctx-baseline") == 1


def test_hooks_examples_baseline_appears_between_system_and_history():
    hooks = Hooks(
        examples=[
            {"role": "user", "content": "e1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    client = FakeLLMClient([text_completion("hi")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("q")
    sent = client.requests[0]["messages"]
    # system, example pair, then the user's "q".
    assert sent[0]["role"] == "system"
    assert sent[1] == {"role": "user", "content": "e1"}
    assert sent[2] == {"role": "assistant", "content": "a1"}
    assert sent[3] == {"role": "user", "content": "q"}


def test_hooks_context_is_non_sticky_resets_to_baseline_each_round():
    """Canonical (system-block) fields set in before_llm reflect on the wire for
    that round, and are re-seeded from the agent baseline the next round — so
    one round's tweak never carries forward.
    """
    seeded_values: list[str | None] = []

    def peek_and_tweak(ctx: RunContext) -> None:
        seeded_values.append(ctx.context)
        if ctx.turn == 1:
            ctx.context = "turn1-injected"

    hooks = Hooks(before_llm=peek_and_tweak, context="baseline")
    client = FakeLLMClient([text_completion("a"), text_completion("b")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("1")
    agent.chat("2")

    # Both rounds begin with the agent's baseline as the seed value (even round
    # 1, where the hook then replaces it to "turn1-injected"). This is the
    # non-sticky reset: the core re-seeds from hooks.context each round.
    assert seeded_values == ["baseline", "baseline"]

    # Round 1's wire: the hook's value is what composes ("turn1-injected");
    # the baseline is not carried over as an additional section.
    r1 = client.requests[0]["messages"][0]["content"]
    assert "turn1-injected" in r1
    # Round 2's wire: the hook did NOT touch it, so the seed ("baseline") composes;
    # the round-1 injection is gone (non-sticky reset worked).
    r2 = client.requests[1]["messages"][0]["content"]
    assert "baseline" in r2
    assert "turn1-injected" not in r2


def test_addendum_from_before_llm_renders_last_into_system_message():
    def add_note(ctx: RunContext) -> None:
        ctx.addendum = "note-1"

    hooks = Hooks(before_llm=add_note, context="ctx-baseline")
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("q")
    sent = client.requests[0]["messages"][0]
    assert sent["content"] == DEFAULT_SYSTEM_PROMPT + "\n\nctx-baseline\n\nnote-1"


def test_before_llm_can_inject_examples_via_message_zone():
    """Dynamic per-cycle examples injection goes through ctx.messages (the hook
    zone), consistent with the existing test_before_llm_injection_is_per_request_not_history.
    The ctx.examples field is a compose-time input (read at composition); for
    per-round example tweaks, manipulate ctx.messages directly.
    """

    def inject(ctx: RunContext) -> None:
        # ctx.messages == [system, user(q)] at this point, so insert examples
        # between index 1 and index 2 to place them before history.
        ctx.messages.insert(1, {"role": "user", "content": "injected-example-q"})
        ctx.messages.insert(2, {"role": "assistant", "content": "injected-example-a"})

    hooks = Hooks(before_llm=inject)
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("q")
    sent = client.requests[0]["messages"]
    # system, injected pair, history user "q".
    assert [m.get("content") for m in sent] == [
        sent[0]["content"],
        "injected-example-q",
        "injected-example-a",
        "q",
    ]
    # None of the injected messages are in the stored history.
    assert not any(m.get("content") == "injected-example-q" for m in agent.history)


def test_before_llm_can_replace_ctx_system_prompt():
    def replace(ctx: RunContext) -> None:
        ctx.system_prompt = "ROBOT MODE"

    hooks = Hooks(before_llm=replace)
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("q")
    sent = client.requests[0]["messages"][0]
    assert sent["content"] == "ROBOT MODE"
    # base_prompt is still the original (immutable).
    assert agent.base_prompt == DEFAULT_SYSTEM_PROMPT


# -- precedence ------------------------------------------------------------------


def test_explicit_system_prompt_wins_over_hooks_prompt():
    hooks = Hooks(prompt="FROM HOOKS")
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), system_prompt="EXPLICIT", hooks=hooks)
    assert agent.base_prompt == "EXPLICIT"
    assert agent.system_prompt == "EXPLICIT"
    agent.chat("q")
    # The wire form carries the explicit value, not the hooks-declared one.
    assert "EXPLICIT" in client.requests[0]["messages"][0]["content"]
    assert "FROM HOOKS" not in client.requests[0]["messages"][0]["content"]


def test_hooks_prompt_overrides_default():
    hooks = Hooks(prompt="FROM HOOKS")
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    assert agent.base_prompt == "FROM HOOKS"
    agent.chat("q")
    sent = client.requests[0]["messages"][0]["content"]
    assert "FROM HOOKS" in sent
    assert DEFAULT_SYSTEM_PROMPT not in sent


def test_no_explicit_and_no_hooks_falls_to_default():
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=Hooks())
    assert agent.base_prompt == DEFAULT_SYSTEM_PROMPT


# -- baseline examples are deep-copied per round (J.3 regression) ---------------


def test_mutation_of_ctx_examples_does_not_corrupt_baseline():
    """The core seeds ``ctx.examples`` from the agent's declared ``hooks.examples``
    every round. If ``before_llm`` mutates the *contents* of an example in place,
    that change must NOT leak back into the stored baseline — the next round must
    re-seed a pristine copy. (Regression: a shallow ``list()`` copy would alias
    the dicts and corrupt the baseline.)"""

    baseline = [{"role": "user", "content": "BASE-Q"}, {"role": "assistant", "content": "BASE-A"}]
    hooks = Hooks(
        examples=baseline,
        before_llm=lambda ctx: ctx.examples[0].__setitem__("content", "MUTATED"),
    )
    client = FakeLLMClient([text_completion("a"), text_completion("b")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)

    # Round 1: before_llm mutates the seeded copy in place.
    agent.chat("1")
    # The baseline must be untouched by the in-place mutation.
    assert baseline[0]["content"] == "BASE-Q"
    # Round 2: re-seeded fresh; the mutation from round 1 must not carry over.
    agent.chat("2")
    assert baseline[0]["content"] == "BASE-Q"


# -- per-round fields are not persisted to history --------------------------------


def test_round_fields_do_not_leak_to_history():
    """context/addendum (canonical system-block fields) compose into the
    system message; neither appears in the stored history. Examples injected
    via ctx.messages stay in the wire form only.
    """

    def set_everything(ctx: RunContext) -> None:
        ctx.context = "round-ctx"
        ctx.addendum = "round-note"
        # Examples via the hook zone (ctx.messages), consistent with the
        # documented escape hatch.
        ctx.messages.insert(1, {"role": "user", "content": "round-e"})
        ctx.messages.insert(2, {"role": "assistant", "content": "round-a"})

    hooks = Hooks(before_llm=set_everything)
    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=hooks)
    agent.chat("q")
    sent = client.requests[0]["messages"]
    # system (identity + round-ctx + round-note), then the injected pair, then q
    assert "round-ctx" in sent[0]["content"]
    assert "round-note" in sent[0]["content"]
    assert {"role": "user", "content": "round-e"} in sent
    # History is unchanged: only user's "q" and the assistant's "ok".
    assert [m.get("content") for m in agent.history] == ["q", "ok"]
