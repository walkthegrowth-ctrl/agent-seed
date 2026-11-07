"""Tests for the pure agent generator (agent_seed/composer/generator.py)."""

from __future__ import annotations

import pytest

from agent_seed.composer.catalog import Tool
from agent_seed.composer.generator import (
    ComposerError,
    Plan,
    Spec,
    generate,
    validate_name,
    write_plan,
)


def _tool(name: str, in_registry: bool = False) -> Tool:
    src = f'def {name}():\n    """Fake."""\n    pass\n'
    return Tool(name=name, description="Fake.", source=src, in_registry=in_registry)


AVAILABLE = [_tool("calculator", in_registry=True), _tool("weather")]


# -- name validation ---------------------------------------------------------


def test_validate_name_rejects_bad_names():
    for bad in ("", ".", "..", "a/b", "a\\b", "UPPER", "a.b", "has space", "has/slash"):
        with pytest.raises(ComposerError):
            validate_name(bad)


def test_validate_name_rejects_reserved_default():
    # `default` is reserved for the main agent: composing it is an error (it
    # would shadow the /agent default switch-back and agent_name="default").
    with pytest.raises(ComposerError, match="reserved"):
        validate_name("default")


def test_validate_name_accepts_good():
    validate_name("alpha")
    validate_name("tutor-1")
    validate_name("x9_")


# -- generate: basic ---------------------------------------------------------


def test_generate_voice_only():
    spec = Spec(name="n", voice="You are helpful.")
    plan = generate("n", spec, AVAILABLE)
    assert list(plan.files) == ["prompt.md"]
    assert plan.files["prompt.md"].strip() == "You are helpful."


def test_generate_no_channels_gives_skeleton():
    """A spec with no channels yields a neutral skeleton (not an empty folder):
    all five channel files present as placeholders, plus a README. Every file
    is loadable and changes no behaviour, so the skeleton agent works but is a
    place to grow from.
    """
    spec = Spec(name="otto")
    plan = generate("otto", spec, AVAILABLE)
    assert set(plan.files) == {
        "context.md",
        "examples.jsonl",
        "prompt.md",
        "README.md",
        "sign.py",
        "tools.py",
    }
    # Text channels are empty placeholders (0 bytes == discovery treats as unset).
    assert plan.files["prompt.md"] == ""
    assert plan.files["context.md"] == ""
    assert plan.files["examples.jsonl"] == ""
    # Python stubs wire the hook slot (make returns it) with a neutral body.
    for stub_name in ("tools.py", "sign.py"):
        stub = plan.files[stub_name]
        assert "def enter_agent" in stub or "def finish" in stub
        assert "def make() -> Hooks:" in stub
    # README orients the user on what to edit.
    assert "otto" in plan.files["README.md"]
    assert "prompt.md" in plan.files["README.md"]


def test_generate_tools_builtin_only():
    spec = Spec(name="n", tools=["calculator"])
    plan = generate("n", spec, AVAILABLE)
    assert "tools.py" in plan.files
    src = plan.files["tools.py"]
    # Built-in should NOT be inlined (no def calculator)
    assert "def calculator" not in src
    # It IS whitelisted
    assert '"calculator"' in src or "'calculator'" in src
    # No `registry.register` (it's already in the built-in registry)
    assert "registry.register" not in src


def test_generate_tools_reservoir_inlined():
    spec = Spec(name="n", tools=["weather"])
    plan = generate("n", spec, AVAILABLE)
    src = plan.files["tools.py"]
    # Reservoir tool IS inlined
    assert "def weather" in src
    # And registered in make()
    assert "registry.register(weather)" in src


def test_generate_all_channels():
    spec = Spec(
        name="tutor",
        voice="You are a tutor.",
        tools=["calculator", "weather"],
        context="Round 1.",
        examples=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        signoff="— tutor",
    )
    plan = generate("tutor", spec, AVAILABLE)
    assert sorted(plan.files) == [
        "context.md",
        "examples.jsonl",
        "prompt.md",
        "sign.py",
        "tools.py",
    ]


def test_generate_unknown_tool_raises():
    spec = Spec(name="n", tools=["nope"])
    with pytest.raises(ComposerError, match="nope"):
        generate("n", spec, AVAILABLE)


def test_generate_examples_invalid_role():
    spec = Spec(name="n", examples=[{"content": "no role"}])
    with pytest.raises(ComposerError, match="role"):
        generate("n", spec, AVAILABLE)


def test_generate_examples_valid_mapping_dropped():
    spec = Spec(name="n", examples=[{"not_role": "x"}])
    with pytest.raises(ComposerError, match="role"):
        generate("n", spec, AVAILABLE)


def test_rendered_includes_paths():
    spec = Spec(name="n", voice="v")
    plan = generate("n", spec, AVAILABLE)
    dump = plan.rendered()
    assert "n/prompt.md" in dump


def test_sign_file_has_signoff_literal():
    from agent_seed.composer.generator import _sign_file

    src = _sign_file("— team alpha 💡")
    assert "SIGNOFF" in src
    assert "def finish" in src
    # The literal should contain the text (escaped by json.dumps)
    assert "team" in src


# -- write_plan --------------------------------------------------------------


def test_write_plan_creates_folder(tmp_path):
    spec = Spec(name="tutor", voice="x")
    plan = generate("tutor", spec, AVAILABLE)
    written = write_plan(plan, tmp_path, force=False)
    assert (tmp_path / "tutor" / "prompt.md").exists()
    assert len(written) == 1


def test_write_plan_refuses_to_overwrite(tmp_path):
    (tmp_path / "tutor").mkdir()
    (tmp_path / "tutor" / "prompt.md").write_text("existing")
    spec = Spec(name="tutor", voice="x")
    p = generate(spec.name, spec, AVAILABLE)
    with pytest.raises(ComposerError, match="already exists"):
        write_plan(p, tmp_path, force=False)
    # File unchanged
    assert (tmp_path / "tutor" / "prompt.md").read_text() == "existing"


def test_write_plan_with_force(tmp_path):
    (tmp_path / "tutor").mkdir()
    (tmp_path / "tutor" / "prompt.md").write_text("old")
    plan = generate("tutor", Spec(name="tutor", voice="new"), AVAILABLE)
    write_plan(plan, tmp_path, force=True)
    assert (tmp_path / "tutor" / "prompt.md").read_text().strip() == "new"


def test_write_plan_empty_folder_creates_dir(tmp_path):
    plan = Plan(name="empty", files={})
    write_plan(plan, tmp_path, force=False)
    assert (tmp_path / "empty").is_dir()
