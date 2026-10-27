"""Tests for hook-module discovery (agent_seed/discovery.py)."""

from pathlib import Path

import pytest

from agent_seed.core import Agent
from agent_seed.discovery import (
    DiscoveryError,
    _root_folder,
    discover,
    get_hooks,
    get_hooks_for_agent,
    list_agents,
    reset_cache,
    resolve_hooks_folder,
)
from agent_seed.hooks import Hooks, RunContext
from agent_seed.tools import ToolRegistry

from .conftest import FakeLLMClient, text_completion


def _write(folder: Path, name: str, body: str) -> Path:
    path = folder / name
    path.write_text(body)
    return path


# -- discovery basics -----------------------------------------------------------


def test_missing_folder_yields_empty_hooks(tmp_path):
    assert discover(tmp_path / "nope") == Hooks()


def test_folder_without_modules_yields_empty_hooks(tmp_path):
    reset_cache()
    (tmp_path / "hooks").mkdir()
    result = get_hooks(tmp_path / "hooks")
    assert result == Hooks()


def test_single_module_is_loaded(tmp_path):
    reset_cache()
    _write(
        tmp_path,
        "budget.py",
        "from agent_seed.hooks import Hooks, RunContext\n"
        "def before_llm(ctx):\n"
        "    ctx.extras['bumped'] = True\n"
        "def make() -> Hooks:\n"
        "    return Hooks(before_llm=before_llm)\n",
    )
    hooks = discover(tmp_path)
    assert hooks.before_llm is not None
    assert hooks.finish is None


def test_disjoint_modules_merge(tmp_path):
    reset_cache()
    _write(
        tmp_path,
        "a_alpha.py",
        "from agent_seed.hooks import Hooks"
        "\ndef one(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(before_llm=one)\n",
    )
    _write(
        tmp_path,
        "b_bravo.py",
        "from agent_seed.hooks import Hooks"
        "\ndef two(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(finish=two)\n",
    )
    hooks = discover(tmp_path)
    assert hooks.before_llm is not None
    assert hooks.finish is not None


def test_conflict_resolves_to_alphabetically_last(tmp_path):
    reset_cache()
    body = (
        "from agent_seed.hooks import Hooks\n"
        "def h(ctx):\n"
        "    ctx.extras['__m__'] = MARKER\n"
        "def make() -> Hooks:\n"
        "    return Hooks(finish=h)\n"
    )
    _write(tmp_path, "1_first.py", body.replace("MARKER", "'first'"))
    _write(tmp_path, "2_second.py", body.replace("MARKER", "'second'"))

    hooks = discover(tmp_path)
    assert hooks.finish is not None
    ctx = RunContext(final_answer="x")
    hooks.finish(ctx)
    assert ctx.extras["__m__"] == "second"


def test_cache_returns_same_object(tmp_path):
    reset_cache()
    _write(
        tmp_path,
        "one.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    first = get_hooks(tmp_path)
    second = get_hooks(tmp_path)
    assert first is second
    reset_cache()


# -- fail-fast -----------------------------------------------------------------


def test_module_without_make_fails(tmp_path):
    _write(tmp_path, "bad.py", "X = 1\n")
    with pytest.raises(DiscoveryError, match="must define"):
        discover(tmp_path)


def test_module_make_raising_fails(tmp_path):
    _write(tmp_path, "bad.py", "def make():\n    raise ValueError('boom')\n")
    with pytest.raises(DiscoveryError, match="make\\(\\) raised"):
        discover(tmp_path)


def test_module_make_wrong_type_fails(tmp_path):
    _write(tmp_path, "bad.py", "def make():\n    return None\n")
    with pytest.raises(DiscoveryError, match="must return a Hooks"):
        discover(tmp_path)


# -- prompt-content files (prompt.md / context.md / examples.jsonl) --------------


def test_loads_prompt_context_and_examples_files(tmp_path):
    (tmp_path / "prompt.md").write_text("You are alpha.\n")
    (tmp_path / "context.md").write_text("Current: date=2026-10-27\n")
    (tmp_path / "examples.jsonl").write_text(
        '{"role": "user", "content": "q1"}\n{"role": "assistant", "content": "a1"}\n\n'
    )
    hooks = discover(tmp_path)
    assert hooks.prompt == "You are alpha."
    assert hooks.context == "Current: date=2026-10-27"
    assert hooks.examples == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_prompt_files_are_optional(tmp_path):
    hooks = discover(tmp_path)
    assert hooks.prompt is None
    assert hooks.context is None
    assert hooks.examples is None


def test_prompt_md_overrides_module_prompt(tmp_path):
    (tmp_path / "hooks.py").write_text(
        "from agent_seed.hooks import Hooks\n"
        "def make() -> Hooks:\n    return Hooks(prompt='from module')\n"
    )
    (tmp_path / "prompt.md").write_text("from file\n")
    hooks = discover(tmp_path)
    assert hooks.prompt == "from file"


def test_examples_jsonl_invalid_json_raises_discoveryerror(tmp_path):
    (tmp_path / "examples.jsonl").write_text("not-json\n")
    with pytest.raises(DiscoveryError, match="examples file"):
        discover(tmp_path)


def test_examples_jsonl_requires_objects(tmp_path):
    (tmp_path / "examples.jsonl").write_text('["not", "an", "object"]\n')
    with pytest.raises(DiscoveryError, match="expected a JSON object"):
        discover(tmp_path)


def test_discovery_prompt_fields_are_merged_last_file_wins(tmp_path):
    (tmp_path / "a_first.py").write_text(
        "from agent_seed.hooks import Hooks\ndef make() -> Hooks:\n    return Hooks(context='a')\n"
    )
    (tmp_path / "b_second.py").write_text(
        "from agent_seed.hooks import Hooks\ndef make() -> Hooks:\n    return Hooks(context='b')\n"
    )
    hooks = discover(tmp_path)
    assert hooks.context == "b"


# -- prompt precedence through Agent ---------------------------------------------


def test_agent_no_explicit_prompt_uses_hooks_prompt_field(tmp_path):
    (tmp_path / "prompt.md").write_text("FILE IDENTITY\n")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()
    from agent_seed.core import DEFAULT_SYSTEM_PROMPT
    from agent_seed.tools import ToolRegistry as _Registry

    class _NoOpClient:
        config = None

        def chat(self, *a, **k):
            raise AssertionError("no chat expected")

    agent = Agent(_NoOpClient(), _Registry())
    assert agent.base_prompt == "FILE IDENTITY"
    assert agent.base_prompt != DEFAULT_SYSTEM_PROMPT
    monkeypatch.undo()


def test_import_error_fails(tmp_path):
    _write(tmp_path, "bad.py", "raise ImportError('nope')\n")
    with pytest.raises(DiscoveryError, match="failed to import"):
        discover(tmp_path)


def test_failure_prevents_any_merge(tmp_path):
    reset_cache()
    _write(
        tmp_path,
        "a_good.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    _write(tmp_path, "b_bad.py", "def make():\n    raise ValueError('boom')\n")
    with pytest.raises(DiscoveryError):
        discover(tmp_path)  # discover() is all-or-nothing: it never returns
    # a different folder is unaffected
    assert discover(tmp_path / "nope") == Hooks()


# -- Agent integration -----------------------------------------------------------


def test_agent_with_hooks_none_uses_discovered_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()
    _write(
        tmp_path,
        "ahemish.py",
        "from agent_seed.hooks import Hooks\n"
        "def finish(ctx):\n"
        "    ctx.final_answer = (ctx.final_answer or '') + '!'\n"
        "def make() -> Hooks:\n    return Hooks(finish=finish)\n",
    )

    agent = Agent(FakeLLMClient([text_completion("hi")]), ToolRegistry())
    assert agent.chat("q") == "hi!"


def test_explicit_hooks_override_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()
    _write(
        tmp_path,
        "ahemish.py",
        "from agent_seed.hooks import Hooks\n"
        "def finish(ctx):\n"
        "    ctx.final_answer = (ctx.final_answer or '') + '!'\n"
        "def make() -> Hooks:\n    return Hooks(finish=finish)\n",
    )

    agent = Agent(
        FakeLLMClient([text_completion("hi")]),
        ToolRegistry(),
        hooks=Hooks(),  # explicit empty hooks beat discovery
    )
    assert agent.chat("q") == "hi"


def test_discovered_hooks_run_full_turn_with_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()

    _write(
        tmp_path,
        "audit.py",
        "from agent_seed.hooks import Hooks, RunContext\n"
        "calls = []\n"
        "def before_tool_call(ctx):\n"
        "    calls.append(ctx.tool_call is not None)\n"
        "def make() -> Hooks:\n    return Hooks(before_tool_call=before_tool_call)\n",
    )

    from .conftest import tool_completion

    agent = Agent(
        FakeLLMClient(
            [tool_completion("calculator", {"expression": "2+2"}), text_completion("4")]
        ),
        ToolRegistry(),
    )
    from agent_seed.tools import calculator

    agent.registry.register(calculator)
    assert agent.chat("q") == "4"


def test_get_hooks_none_uses_default_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    reset_cache()
    _write(
        tmp_path,
        "one.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    hooks = get_hooks(None)
    assert hooks.finish is not None
    reset_cache()


def test_pycache_not_discovered(tmp_path):
    reset_cache()
    _write(
        tmp_path,
        "one.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): pass\n\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("junk")
    assert discover(tmp_path).finish is not None


# -- agents (subdirectory selection) ----------------------------------


def _root_and_agent(tmp_path: Path) -> Path:
    """A root hook (finish adds 'R') and agent 'alpha' (finish adds 'A')."""
    _write(
        tmp_path,
        "root.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): ctx.final_answer = (ctx.final_answer or '') + 'R'\n"
        "\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    (tmp_path / "alpha").mkdir()
    _write(
        tmp_path / "alpha",
        "a.py",
        "from agent_seed.hooks import Hooks"
        "\ndef h(ctx): ctx.final_answer = (ctx.final_answer or '') + 'A'\n"
        "\ndef make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    return tmp_path


def test_resolve_hooks_folder_none_uses_root(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    (tmp_path / "alpha").mkdir()
    assert resolve_hooks_folder(None) == tmp_path.resolve()


def test_resolve_hooks_folder_named_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    (tmp_path / "alpha").mkdir()
    assert resolve_hooks_folder("alpha") == (tmp_path / "alpha").resolve()


def test_resolve_hooks_folder_missing_falls_back_to_root(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    (tmp_path / "alpha").mkdir()
    assert resolve_hooks_folder("nope") == tmp_path.resolve()


def test_resolve_hooks_folder_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    (tmp_path / "alpha").mkdir()
    for bad in ("..", "../outside", "/etc", "a/.."):
        assert resolve_hooks_folder(bad) == tmp_path.resolve(), bad


def test_resolve_hooks_folder_subdir_must_be_under_root(tmp_path, monkeypatch):
    # A sibling dir with the same name must NOT be picked up.
    outside = tmp_path.parent / "alpha_outside"
    outside.mkdir()
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    assert resolve_hooks_folder("alpha_outside") == tmp_path.resolve()
    # now a real subdir of the root resolves correctly
    (tmp_path / "alpha_outside").mkdir()
    assert resolve_hooks_folder("alpha_outside") == (tmp_path / "alpha_outside").resolve()


def test_list_agents(tmp_path, monkeypatch):
    # Isolate from the autouse fixture (which leaves a no_hooks/ under tmp_path)
    # by using HOOKS_ROOT to point at a fresh directory.
    root = tmp_path / "hooks_root"
    (root / "zeta").mkdir(parents=True)
    (root / "alpha").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "a_file").write_text("not a dir")
    monkeypatch.setenv("HOOKS_ROOT", str(root))
    reset_cache()
    assert list_agents() == ["alpha", "zeta"]


def test_list_agents_no_root(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path / "no"))
    assert list_agents() == []


def test_hook_root_only_applies_to_default_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _root_and_agent(tmp_path)
    reset_cache()

    # default (None) gets the root's finish
    assert get_hooks_for_agent(None).finish is not None
    # a named agent replaces the root's hooks
    assert get_hooks_for_agent("alpha").finish is not None
    # but it is the SUBDIR's finish, not the root's: run each against a
    # context and assert only the agent's suffix appears.
    ctx = RunContext(final_answer="x")
    get_hooks_for_agent(None).finish(ctx)
    assert ctx.final_answer == "xR"
    ctx = RunContext(final_answer="x")
    get_hooks_for_agent("alpha").finish(ctx)
    assert ctx.final_answer == "xA"


def test_get_hooks_for_agent_missing_falls_back_to_root(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _root_and_agent(tmp_path)
    reset_cache()
    hooks = get_hooks_for_agent("nope")
    ctx = RunContext(final_answer="x")
    hooks.finish(ctx)
    assert ctx.final_answer == "xR"


def test_agent_at_construction_uses_subdir_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _root_and_agent(tmp_path)
    reset_cache()

    from .conftest import text_completion

    client = FakeLLMClient([text_completion("hi")])
    agent = Agent(client, ToolRegistry(), agent_name="alpha")
    assert agent.agent_name == "alpha"
    assert agent.chat("q") == "hiA"
    assert agent.context.agent_name == "alpha"


def test_set_agent_name_switches_hooks_and_refires_establishment(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _root_and_agent(tmp_path)
    reset_cache()

    from .conftest import text_completion

    client = FakeLLMClient([text_completion("a"), text_completion("b"), text_completion("c")])
    agent = Agent(client, ToolRegistry())
    assert agent.chat("1") == "aR"

    # Switch to the alpha subdirectory agent.
    agent.set_agent_name("alpha")
    assert agent.chat("2") == "bA"
    assert agent.agent_name == "alpha"
    # ...then back to the root (default agent).
    agent.set_agent_name(None)
    assert agent.chat("3") == "cR"
    assert agent.agent_name is None


def test_explicit_hooks_still_win_over_agent_name(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _root_and_agent(tmp_path)
    reset_cache()

    from .conftest import text_completion

    explicit = Hooks(finish=lambda ctx: None)

    def _e(ctx):
        ctx.final_answer = (ctx.final_answer or "") + "E"

    explicit.finish = _e
    client = FakeLLMClient([text_completion("a"), text_completion("b"), text_completion("c")])
    agent = Agent(client, ToolRegistry(), hooks=explicit)
    assert agent.chat("1") == "aE"  # explicit wins
    agent.set_agent_name("alpha")  # explicit still wins
    assert agent.chat("2") == "bE"
    agent.set_agent_name(None)  # ...and still wins
    assert agent.chat("3") == "cE"


def _boundary_module(tag: str) -> str:
    """A standalone hook module that logs every boundary/establishment hook.

    Each hook appends ``(f"{tag}:{label}", ctx.agent_name)`` to a shared
    ``ctx.extras["log"]`` list — a single ordered record spanning construction,
    both agents' enter/exit, and the active agent's establishment hooks.
    """
    slots = [
        ("enter_agent", "enter"),
        ("exit_agent", "exit"),
        ("initialize_run", "init_run"),
        ("ready_run", "ready_run"),
        ("initialize_session", "init_session"),
        ("ready_session", "ready_session"),
    ]
    lines = ["from agent_seed.hooks import Hooks"]
    for _, label in slots:
        lines.append(
            f'def _{label}(ctx): ctx.extras.setdefault("log", [])'
            f'.append(("{tag}:{label}", ctx.agent_name))'
        )
    args = ", ".join(f"{slot}=_{label}" for slot, label in slots)
    lines.append("def make() -> Hooks:")
    lines.append(f"    return Hooks({args})")
    return "\n".join(lines)


# -- enter_agent / exit_agent (the agent-identity boundary) ------------------


def test_ctor_fires_enter_not_exit(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    reset_cache()
    agent = Agent(FakeLLMClient([]), ToolRegistry())  # the main (nameless) agent
    log = agent.context.extras.get("log", [])
    assert ("R:enter", None) in log  # first activation: enter fires
    assert ("R:exit", None) not in log  # ...but never exit (nothing to leave)


def test_switch_fires_exit_old_then_enter_new(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    (tmp_path / "alpha").mkdir()
    _write(tmp_path / "alpha", "a.py", _boundary_module("A"))
    reset_cache()
    agent = Agent(FakeLLMClient([]), ToolRegistry())  # enter main
    agent.set_agent_name("alpha")  # exit main, enter alpha
    assert agent.context.extras.get("log") == [
        ("R:enter", None),
        ("R:exit", None),  # exit runs as the outgoing agent (old name)
        ("A:enter", "alpha"),  # enter runs as the incoming agent (new name)
    ]


def test_set_agent_name_same_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    reset_cache()
    agent = Agent(FakeLLMClient([]), ToolRegistry(), agent_name=None)
    ctx = agent.context
    assert ctx.extras.get("log") == [("R:enter", None)]
    agent.set_agent_name(None)  # re-select the same agent: no boundary hooks
    assert ctx.extras.get("log") == [("R:enter", None)]


def test_switch_aborts_cleanly_if_new_hooks_broken(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "bad.py").write_text(
        "from agent_seed.hooks import Hooks\ndef make():\n    raise ValueError('boom')\n"
    )
    reset_cache()
    agent = Agent(FakeLLMClient([]), ToolRegistry())  # main agent; enter fired
    ctx = agent.context
    assert agent.agent_name is None
    with pytest.raises(DiscoveryError):
        agent.set_agent_name("bad")  # new agent's hooks can't load
    # Atomic abort: the outgoing agent's exit was NOT fired, name unchanged.
    assert agent.agent_name is None
    assert ("R:exit", None) not in ctx.extras.get("log", [])
    assert ("R:enter", None) in ctx.extras.get("log", [])


def test_raii_chain_every_enter_after_first_has_a_matching_exit(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    (tmp_path / "alpha").mkdir()
    _write(tmp_path / "alpha", "a.py", _boundary_module("A"))
    reset_cache()
    agent = Agent(FakeLLMClient([]), ToolRegistry())  # enter main (no exit)
    agent.set_agent_name("alpha")  # exit main, enter alpha
    agent.set_agent_name(None)  # exit alpha, enter main
    assert agent.context.extras.get("log") == [
        ("R:enter", None),
        ("R:exit", None),
        ("A:enter", "alpha"),
        ("A:exit", "alpha"),
        ("R:enter", None),
    ]


def test_explicit_hooks_fire_enter_exit_on_switch(tmp_path, monkeypatch):
    logger = []
    explicit = Hooks(
        enter_agent=lambda ctx: logger.append("enter"),
        exit_agent=lambda ctx: logger.append("exit"),
    )
    agent = Agent(FakeLLMClient([]), ToolRegistry(), hooks=explicit)
    assert logger == ["enter"]  # ctor fires enter even with explicit hooks
    agent.set_agent_name("whatever")  # a real switch (name changed) fires both
    assert logger == ["enter", "exit", "enter"]
    agent.set_agent_name("whatever")  # no-op: nothing fires
    assert logger == ["enter", "exit", "enter"]


def test_enter_precedes_establishment_after_switch(tmp_path, monkeypatch):
    from .conftest import text_completion

    monkeypatch.setattr("agent_seed.discovery.DEFAULT_HOOKS_FOLDER", str(tmp_path))
    _write(tmp_path, "root.py", _boundary_module("R"))
    (tmp_path / "alpha").mkdir()
    _write(tmp_path / "alpha", "a.py", _boundary_module("A"))
    reset_cache()
    client = FakeLLMClient([text_completion("hi")])
    agent = Agent(client, ToolRegistry())  # enter main
    agent.set_agent_name("alpha")  # exit main, enter alpha; establishment deferred
    agent.chat("q")  # alpha's establishment hooks fire now
    assert agent.context.extras.get("log") == [
        ("R:enter", None),
        ("R:exit", None),
        ("A:enter", "alpha"),
        # the incoming agent's establishment runs AFTER its enter
        ("A:init_run", "alpha"),
        ("A:ready_run", "alpha"),
        ("A:init_session", "alpha"),
        ("A:ready_session", "alpha"),
    ]


def test_hooks_root_env_var_selects_folder(monkeypatch, tmp_path):
    # HOOKS_ROOT points at a fresh directory with its own agent subdir.
    root = tmp_path / "hooks_root"
    (root / "gamma").mkdir(parents=True)
    _write(
        root / "gamma",
        "g.py",
        "from agent_seed.hooks import Hooks\n"
        "def h(ctx): ctx.final_answer = (ctx.final_answer or '') + 'G'\n"
        "def make() -> Hooks:\n    return Hooks(finish=h)\n",
    )
    monkeypatch.setenv("HOOKS_ROOT", str(root))
    reset_cache()
    assert _root_folder() == root
    hooks = get_hooks_for_agent("gamma")
    ctx = RunContext(final_answer="x")
    hooks.finish(ctx)
    assert ctx.final_answer == "xG"
    reset_cache()
