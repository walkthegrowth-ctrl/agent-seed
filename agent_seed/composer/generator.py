"""Agent generation: turn a Spec into a set of files, then write the agent folder.

The core logic is **pure**: :func:`generate` turns a :class:`Spec` (plus the catalog)
into an ordered ``{path: content}`` map with no side effects, so it is trivially
testable. :func:`write_plan` then writes that map under ``<root>/<name>/``. The CLI
drives both (validate the name, look the chosen tools up in the catalog,
:func:`generate`, and :func:`write_plan` only if not ``--dry-run``).

A generated agent is *just a folder* under the hooks root; runtime discovery
(``agent_seed.discovery``) does the rest (replace semantics: the folder is the
agent; delete it to remove the agent). The files emitted are the same shapes the
main agent already uses (``hook_modules/00_tools.py``, the howto ``sign.py``), so a
generated agent is portable and removable (``rm -rf hook_modules/<name>``).

Reservoir tools are **inlined** into the generated ``tools.py`` (their file's
imports plus their function source), so the agent folder is self-contained: it does
not depend on the CWD or on the reservoir staying present, matching "delete the
folder, delete the agent". Built-in tools (``calculator``, ...) already live in the
runtime registry, so they are only whitelisted — never inlined.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Tool

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ComposerError(ValueError):
    """Raised on a bad agent name or an invalid generation request."""


@dataclass
class Spec:
    """Everything a generated agent needs.

    ``name`` is required. The rest are optional channels; an unset channel does not
    produce a file (so a generated agent is exactly as big as asked). ``tools`` are
    tool *names* the agent may call; for each, a built-in is whitelisted only while
    a reservoir tool is inlined + whitelisted.
    """

    name: str
    voice: str | None = None
    tools: list[str] = field(default_factory=list)
    context: str | None = None
    examples: list[dict[str, Any]] | None = None
    signoff: str | None = None


@dataclass
class Plan:
    """The full file set for one generated agent (paths relative to the agent folder)."""

    name: str
    files: dict[str, str]

    def rendered(self) -> str:
        """A printable, path-annotated dump (for ``--dry-run``)."""
        parts: list[str] = [f"agent: {self.name}"]
        for rel, content in self.files.items():
            parts.append(f"\n===== {self.name}/{rel} =====\n{content}\n")
        return "\n".join(parts)


def validate_name(name: str) -> str:
    """Ensure ``name`` is a safe folder name (no separators, dots, or traversal)."""
    if (
        not name
        or "/" in name
        or "\\" in name
        or name in (".", "..")
        or not _NAME_RE.fullmatch(name)
    ):
        raise ComposerError(
            f"agent name {name!r} is not allowed; use [a-z0-9_-]+ with no slashes, "
            "dots, or path components"
        )
    return name


def validate_examples(examples: Iterable[Mapping[str, Any]]) -> None:
    """Each few-shot example must be a mapping with a non-empty ``role``."""
    for item in examples or []:
        if not isinstance(item, Mapping) or not item.get("role"):
            raise ComposerError(
                f"examples entry {item!r} must be a mapping with a non-empty 'role'"
            )


def _prompt_file(voice: str) -> str:
    return voice.strip() + "\n"


def _context_file(context: str) -> str:
    return context.strip() + "\n"


def _examples_file(examples: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(item), ensure_ascii=False) for item in examples) + "\n"


def _sign_file(signoff: str) -> str:
    literal = json.dumps(signoff, ensure_ascii=False)  # a safe, escaped string literal
    return (
        "from agent_seed.hooks import Hooks, RunContext\n"
        "\n"
        f"SIGNOFF = {literal}\n"
        "\n"
        "\n"
        "def finish(ctx: RunContext) -> None:\n"
        "    if ctx.final_answer is not None:\n"
        "        ctx.final_answer = ctx.final_answer.rstrip() + '\\n' + SIGNOFF\n"
        "\n"
        "\n"
        "def make() -> Hooks:\n"
        "    return Hooks(finish=finish)\n"
    )


def _tools_file(chosen: list[Tool]) -> str:
    """The generated ``tools.py``: whitelist (+ inline & register reservoir tools).

    * Built-in tools are already in the runtime registry -> whitelisted only.
    * Reservoir tools are inlined (their file's imports + their function source) and
      registered in ``make()`` (idempotent ``has()`` guard, the same pattern as
      ``experiments/memory.py``).
    """
    out: list[str] = ["from __future__ import annotations"]

    # De-duplicated imports from the reservoir tools (kept so the inlined functions
    # keep working even if their file needed them). The `from __future__ import
    # annotations` is already emitted above, so it is skipped when inlining.
    seen: set[str] = {"from __future__ import annotations"}
    imports: list[str] = []
    functions: list[str] = []
    for tool in chosen:
        if tool.in_registry:
            continue
        for imp in tool.imports:
            normalized = imp.strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            imports.append(normalized)
        functions.append(tool.source.strip())

    if imports:
        out.append("")
        out.extend(imports)
    out.append("")
    out.append("from agent_seed.hooks import Hooks, RunContext")
    if functions:
        out.append("")
        for fn in functions:
            out.append(fn)
        out.append("")

    out.append(
        f"ALLOWED_TOOLS: list[str] = {json.dumps([t.name for t in chosen], ensure_ascii=False)}"
    )
    out.append("")
    out.append("")
    out.append("def enter_agent(ctx: RunContext) -> None:")
    out.append("    ctx.allowed_tools = list(ALLOWED_TOOLS)")
    out.append("")
    out.append("")
    out.append("def make() -> Hooks:")

    reservoirs = [t for t in chosen if not t.in_registry]
    if reservoirs:
        out.append("    from agent_seed.tools import registry")
        for tool in reservoirs:
            out.append(f"    if not registry.has({tool.name!r}):")
            out.append(f"        registry.register({tool.name})")
    out.append("    return Hooks(enter_agent=enter_agent)")
    out.append("")
    return "\n".join(out)


def generate(name: str, spec: Spec, available: list[Tool]) -> Plan:
    """Build the file plan for ``spec`` against the ``available`` catalog.

    ``name`` is the agent name (also present on ``spec.name``); the two must agree,
    otherwise ``ComposerError``. ``available`` is the output of
    :func:`agent_seed.composer.catalog.list_tools`.
    """
    if spec.name != name:
        raise ComposerError(f"agent name mismatch: spec.name={spec.name!r} != {name!r}")
    validate_name(name)
    if spec.examples:
        validate_examples(spec.examples)

    # De-dup and order-preserving tool selection; fail fast on unknown names.
    by_name = {t.name: t for t in available}
    chosen_names: list[str] = []
    for tool_name in spec.tools:
        if tool_name not in chosen_names:
            chosen_names.append(tool_name)
    unknown = [n for n in chosen_names if n not in by_name]
    if unknown:
        have = ", ".join(t.name for t in available) or "(none)"
        raise ComposerError(
            f"tool(s) not found in the catalog: {', '.join(unknown)}. Available: {have}"
        )

    files: dict[str, str] = {}
    if spec.voice is not None:
        files["prompt.md"] = _prompt_file(spec.voice)
    if chosen_names:
        files["tools.py"] = _tools_file([by_name[n] for n in chosen_names])
    if spec.context is not None:
        files["context.md"] = _context_file(spec.context)
    if spec.examples:
        files["examples.jsonl"] = _examples_file(spec.examples)
    if spec.signoff is not None:
        files["sign.py"] = _sign_file(spec.signoff)
    return Plan(name=name, files=files)


def write_plan(plan: Plan, root: str | Path, force: bool = False) -> list[Path]:
    """Write ``plan`` under ``<root>/<name>/`` and return the files written.

    Refuses to touch an existing agent folder unless ``force`` is set: composing is
    additive, and clobbering a hand-written agent is not. Creates the folder even
    for an empty plan (a no-tool, no-voice agent is still a valid, empty agent).
    """
    target = Path(root) / plan.name
    if target.exists() and not force:
        existing = [p.name for p in target.iterdir() if p.is_file()]
        raise ComposerError(
            f"agent folder already exists: {target} "
            f"(contains: {', '.join(existing) or 'no files'}); pass --force to overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rel, content in plan.files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
