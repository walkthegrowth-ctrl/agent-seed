"""Tests for hook-module discovery (agent_seed/discovery.py)."""

from pathlib import Path

import pytest

from agent_seed.core import Agent
from agent_seed.discovery import (
    DiscoveryError,
    discover,
    get_hooks,
    reset_cache,
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
