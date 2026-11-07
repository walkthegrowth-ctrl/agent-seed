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

from ..discovery import RESERVED_AGENT_NAME
from .catalog import Tool

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ComposerError(ValueError):
    """Raised on a bad agent name or an invalid generation request."""


@dataclass
class Spec:
    """Everything a generated agent needs.

    ``name`` is required. The rest are optional channels: for each channel that is
    set, a matching file is produced (voice → ``prompt.md``, context →
    ``context.md``, examples → ``examples.jsonl``, signoff → ``sign.py``;
    ``tools`` → ``tools.py`` with the whitelist built-in tools whitelisted only and
    reservoir tools inlined + whitelisted).

    A spec with **no channels set at all** is special: instead of an empty,
    zero-signal folder, :func:`generate` emits a neutral **skeleton** (the
    correct hook slots wired in with no-op bodies, empty text files, and a
    README) so the user gets a loadable-and-editable starting point rather than
    a blank folder.
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
    """Ensure ``name`` is a safe folder name (no separators, dots, or traversal).

    ``default`` is also rejected: it is reserved for the main agent (see
    :data:`agent_seed.discovery.RESERVED_AGENT_NAME`), so a hooks subdirectory may
    not bear it — composing ``default/`` would shadow the ``/agent default``
    switch-back surface and the API's ``agent_name="default"``.
    """
    if name == RESERVED_AGENT_NAME:
        raise ComposerError(
            f"agent name {name!r} is reserved for the (main) agent; choose another name"
        )
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


def _skeleton(name: str) -> dict[str, str]:
    """Neutral skeleton files for an agent composed with **no channels**.

    Instead of an empty (no-op, no-signal) folder, the composer emits a
    skeleton: every channel is present as an *editable placeholder*, but nothing
    changes behavior. The Python stubs wire the right hook slot (so the hook
    contract stays correct) with a ``pass`` body (``enter_agent`` leaves
    ``ctx.allowed_tools`` alone; ``finish`` leaves ``ctx.final_answer`` alone —
    i.e. the agent offers no tools and passes the LLM's answer through, the
    strict default). The text channels are empty (0 bytes) so they add nothing to
    the prompt. Each file is a place to grow from: edit it to turn that channel
    on, delete it to drop it. See the generated README.
    """
    return {
        "prompt.md": "",
        "context.md": "",
        "examples.jsonl": "",
        "tools.py": _stub_tools_file(),
        "sign.py": _stub_sign_file(),
        "README.md": _skeleton_readme(name),
    }


def _stub_tools_file() -> str:
    """A neutral ``tools.py`` skeleton: ``enter_agent`` wired in, ``pass`` body."""
    return '''"""This agent's tool policy — a neutral skeleton stub.

Generated by `agent compose` with **no tools specified**, so this is a
placeholder to grow from, not a live policy. The `enter_agent` hook is wired in
(so it fires when this agent takes over) but does nothing: it leaves
`ctx.allowed_tools` alone, which under the strict-whitelist policy means this
agent offers NO tools and every tool call is vetoed.

To give this agent tools, set the whitelist inside `enter_agent`:

    ctx.allowed_tools = ["calculator"]

(`agent compose --list-tools` lists the available tool names.)

docs: docs/manual.md — "Per-agent tool access (strict whitelist)".
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext


def enter_agent(ctx: RunContext) -> None:
    """Declare this agent's tool whitelist (strict; default = no tools).

    Neutral stub: leaves `ctx.allowed_tools` untouched (= the agent gets no
    tools). To allow tools, replace `pass` with, e.g.:

        ctx.allowed_tools = ["calculator"]
    """
    pass


def make() -> Hooks:
    return Hooks(enter_agent=enter_agent)
'''


def _stub_sign_file() -> str:
    """A neutral ``sign.py`` skeleton: ``finish`` wired in, ``pass`` body."""
    return '''"""This agent's sign-off — a neutral skeleton stub.

Generated by `agent compose` with **no sign-off specified**, so this is a
placeholder to grow from, not a live sign-off. The `finish` hook is wired in
(so it fires on the final answer) but does nothing: it leaves
`ctx.final_answer` untouched, so the LLM's answer passes through unchanged.

To sign off every answer, do it inside `finish` (the core sets
`ctx.final_answer` right before `finish`, so guard on it being non-None):

    if ctx.final_answer is not None:
        ctx.final_answer = ctx.final_answer.rstrip() + "\\n" + "- Your Team"

docs: docs/manual.md — "Turn termination".
"""

from __future__ import annotations

from agent_seed.hooks import Hooks, RunContext


def finish(ctx: RunContext) -> None:
    """Sign off the final answer.

    Neutral stub: leaves `ctx.final_answer` untouched. To append a sign-off,
    replace `pass` with an append guarded on `ctx.final_answer is not None`.
    """
    pass


def make() -> Hooks:
    return Hooks(finish=finish)
'''


def _skeleton_readme(name: str) -> str:
    """The README that orients a user to the neutral skeleton."""
    text = """# __NAME__

A **neutral skeleton** agent created by `agent compose` with no channels
specified. It **loads and works**, but changes nothing: every channel below is
present as an editable placeholder. Turn a channel on by editing its file; drop
it by deleting the file. Nothing here blocks you — grow it, keep it, or throw
it away.

## Channels
| file       | feeds                            | to use:                           |
|------------|----------------------------------|-----------------------------------|
| `prompt.md`| identity — the system prompt     | write the persona text            |
| `context.md`| per-round context (RAG, state)   | write the context block           |
| `examples.jsonl`| few-shot examples (JSON/line)  | add role/content lines            |
| `tools.py`   | tool whitelist (`enter_agent`)   | set `ctx.allowed_tools` in hook   |
| `sign.py`    | sign-off on answers (`finish`)   | rewrite `ctx.final_answer`        |

## Hook quick-reference
- `enter_agent(ctx)` — fires when this agent takes over (construction, and every
  switch into it). The canonical place to set `ctx.allowed_tools` (its tool
  whitelist; strict: default = no tools).
- `finish(ctx)` — fires on the final answer (the last hook before it is
  returned). The canonical place to rewrite `ctx.final_answer` (e.g. sign-off).

See `docs/manual.md` ("Per-agent tool access (strict whitelist)",
"Turn termination") and `docs/howto-craft-an-agent.md`.

## Removing
A folder under the hooks root **is** the agent: `rm -rf __NAME__`.
"""
    return text.replace("__NAME__", name)


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

    A spec with **no channels at all** (no voice, tools, context, examples, or
    sign-off) yields a neutral :func:`_skeleton` — the right hook slots wired in
    with no-op bodies plus empty text files and a README — instead of an empty
    folder: it loads and works, changes nothing, and is a place to grow from.
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

    # No channels at all -> a neutral skeleton (not an empty, no-signal folder).
    if (
        spec.voice is None
        and not chosen_names
        and spec.context is None
        and not spec.examples
        and spec.signoff is None
    ):
        return Plan(name=name, files=_skeleton(name))

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
    additive, and clobbering a hand-written agent is not.
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
