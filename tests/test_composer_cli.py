"""Tests for the ``agent compose`` CLI entry point
(``agent_seed/composer/cli.py``).

Covers the surface the docs promise but the rest of the suite doesn't exercise:
the ``--list-tools`` flag, interactive-name validation (via monkeypatched
``_is_interactive``), and the happy-path ``main`` invocation that writes a folder.
"""

from __future__ import annotations

from agent_seed.composer import cli as composer_cli
from agent_seed.composer.catalog import Tool


def test_list_tools_prints_builtins_and_reservoir(capsys, monkeypatch):
    fake_tools = [
        Tool(name="calculator", description="Arithmetic eval.", in_registry=True),
        Tool(name="roll_dice", description="Roll dice.", in_registry=False),
    ]
    monkeypatch.setattr(composer_cli, "list_tools", lambda: fake_tools)

    rc = composer_cli.main(["--list-tools"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "calculator" in out
    assert "built-in" in out
    assert "roll_dice" in out
    assert "reservoir" in out


def test_list_tools_empty_catalog_prints_hint(capsys, monkeypatch):
    monkeypatch.setattr(composer_cli, "list_tools", lambda: [])
    rc = composer_cli.main(["--list-tools"])
    assert rc == 0
    assert "(no tools available)" in capsys.readouterr().out


def test_interactive_requires_name(capsys, monkeypatch):
    """Non-interactive, no name → ComposerError (exit code 1). Interactive +
    blank name is handled by a Prompt and falls back to ComposerError here."""
    monkeypatch.setattr(composer_cli, "_is_interactive", lambda: False)
    # list_tools is called unconditionally; keep it hermetic.
    monkeypatch.setattr(
        composer_cli,
        "list_tools",
        lambda: [Tool(name="calculator", description="x", in_registry=True)],
    )
    rc = composer_cli.main([])
    assert rc == 1
    err = capsys.readouterr().err
    assert "agent name" in err.lower()


def test_dry_run_emits_path_without_writing(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(composer_cli, "_is_interactive", lambda: False)
    fake_tools = [Tool(name="calculator", description="Arithmetic.", in_registry=True)]
    monkeypatch.setattr(composer_cli, "list_tools", lambda: fake_tools)
    rc = composer_cli.main(
        ["tutor", "--voice", "You are a patient tutor.", "--dry-run"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "tutor/prompt.md" in out
    # --dry-run must not have created the folder.
    assert not (tmp_path / "tutor").exists()


def test_write_creates_folder_under_root(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(composer_cli, "_is_interactive", lambda: False)
    fake_tools = [Tool(name="calculator", description="Arithmetic.", in_registry=True)]
    monkeypatch.setattr(composer_cli, "list_tools", lambda: fake_tools)
    target = tmp_path / "hooks"
    rc = composer_cli.main(
        [
            "tutor",
            "--voice",
            "You are a patient tutor.",
            "--root",
            str(target),
        ],
    )
    assert rc == 0
    assert (target / "tutor" / "prompt.md").read_text().strip() == "You are a patient tutor."


def test_refuses_overwrite_without_force(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(composer_cli, "_is_interactive", lambda: False)
    fake_tools = [Tool(name="calculator", description="Arithmetic.", in_registry=True)]
    monkeypatch.setattr(composer_cli, "list_tools", lambda: fake_tools)

    target = tmp_path / "hooks"
    (target / "tutor").mkdir(parents=True)
    (target / "tutor" / "prompt.md").write_text("existing")

    rc = composer_cli.main(
        ["tutor", "--voice", "new", "--root", str(target)],
    )
    assert rc == 1  # ComposerError surfaced as exit 1
    assert "already exists" in capsys.readouterr().err
    assert (target / "tutor" / "prompt.md").read_text() == "existing"

    # Same call with --force overwrites.
    rc2 = composer_cli.main(
        ["tutor", "--voice", "new", "--root", str(target), "--force"],
    )
    assert rc2 == 0
    assert (target / "tutor" / "prompt.md").read_text().strip() == "new"


def test_invalid_example_role_is_rejected(capsys, monkeypatch):
    monkeypatch.setattr(composer_cli, "_is_interactive", lambda: False)
    fake_tools = [Tool(name="calculator", description="Arithmetic.", in_registry=True)]
    monkeypatch.setattr(composer_cli, "list_tools", lambda: fake_tools)

    rc = composer_cli.main(["tutor", "--example", "bogus=hello"])
    assert rc == 2
    assert "must be user/assistant/system/tool" in capsys.readouterr().err
