"""Tests for the tool catalog (agent_seed/composer/catalog.py)."""

from __future__ import annotations

from agent_seed.composer.catalog import list_tools


def test_lists_builtins_and_reservoir():
    tools = {t.name: t for t in list_tools()}
    assert "calculator" in tools
    assert tools["calculator"].in_registry is True
    # The seeded reservoir tools should be picked up from tools/*.py
    assert "current_date" in tools
    assert "random_int" in tools
    assert tools["current_date"].in_registry is False


def test_builtin_description_is_real():
    tools = {t.name: t for t in list_tools()}
    assert "arithmetic" in tools["calculator"].description.lower()


def test_reservoir_descriptions_are_real():
    tools = {t.name: t for t in list_tools()}
    assert "date" in tools["current_date"].description.lower()


def test_source_is_captured_for_reservoir():
    tools = {t.name: t for t in list_tools()}
    assert "def current_date" in tools["current_date"].source
    # The function body should survive the round-trip
    assert "strftime" in tools["current_date"].source


def test_missing_root_yields_builtins_only(tmp_path):
    tools = list_tools(tools_root=tmp_path / "nope")
    assert {t.name for t in tools} == {"calculator"}


def test_reservoir_file_with_broken_syntax_is_skipped(tmp_path):
    # A broken reservoir file should not stop the scan.
    (tmp_path / "broken.py").write_text("def not_a_valid_python(  ")
    (tmp_path / "good.py").write_text(
        "def good_tool(x: int) -> int:\n    '''A good tool.'''\n    return x\n"
    )
    tools = {t.name: t for t in list_tools(tools_root=tmp_path)}
    assert "good_tool" in tools
    assert "not_a_valid_python" not in tools


def test_underscores_are_skipped(tmp_path):
    (tmp_path / "helpers.py").write_text(
        "def _private_helper(x: int) -> int:\n    return x\n"
        "def public_tool(x: int) -> int:\n    return x\n"
    )
    tools = {t.name: t for t in list_tools(tools_root=tmp_path)}
    assert "_private_helper" not in tools
    assert "public_tool" in tools


def test_builtin_names_win_over_reservoir(tmp_path):
    (tmp_path / "shadow.py").write_text(
        "def calculator(expression: str) -> str:\n"
        "    '''shadows the built-in.'''\n"
        "    return 'no'\n"
    )
    tools = {t.name: t for t in list_tools(tools_root=tmp_path)}
    assert tools["calculator"].in_registry is True  # the built-in wins


def test_non_python_files_are_ignored(tmp_path):
    (tmp_path / "README.md").write_text("# not a tool\n")
    (tmp_path / "notpy.js").write_text("console.log('x')\n")
    tools = list_tools(tools_root=tmp_path)
    assert {t.name for t in tools} == {"calculator"}
