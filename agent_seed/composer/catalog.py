"""Tool catalog: the built-in tools plus the ready-made tool *reservoir*.

Two local, network-free sources (see ``docs/manual.md``, "Composing an agent"):

* the default ``agent_seed.tools.registry`` (ships ``calculator``) — the "project's
  own" tools; and
* a **reservoir** (a ``tools/`` folder, overridable via ``$TOOLS_ROOT``) of
  ready-made tools supplied from elsewhere.

Reservoir files are scanned with ``ast`` and **never imported at scan time**, so a
broken file, or one that imports an unavailable dependency, cannot stop the catalog
from being built. A reservoir tool function should be self-contained: it must not
depend on sibling helpers or module-level state (only, at most, top-level imports,
which the catalog also captures so a generated agent can inline them and stay
portable).

Name collisions: a reservoir tool redefining an existing built-in name is skipped
(built-ins are not shadowed) — use a new name.
"""

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tool:
    """One available tool.

    ``source`` is the tool function's source text (a generated agent inlines it so
    the agent folder is self-contained). ``imports`` are the top-level import
    statements the function's file declares (inlined too, so the tool keeps working
    even if its imports were needed). For built-in tools ``source``/``imports`` may
    be empty: the function already lives in the runtime registry, so a generated
    agent only needs to *whitelist* it.
    """

    name: str
    description: str
    source: str = ""
    imports: tuple[str, ...] = ()
    in_registry: bool = False


def _first_doc_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _reservoir_root(tools_root: str | Path | None = None) -> Path:
    return Path(tools_root or os.environ.get("TOOLS_ROOT") or "tools")


def _top_level_imports(text: str, root: ast.AST) -> tuple[str, ...]:
    """The file's top-level ``import X`` / ``from X import ...`` statements.

    (``root`` is the parsed module, kept only for API symmetry; ``ast.get_source_segment``
    takes just ``(source, node)``.)
    """
    out: list[str] = []
    for node in root.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(text, node)
            if seg is None:
                continue
            out.append("\n".join(line.strip() for line in seg.strip().splitlines()) + "\n")
    return tuple(out)


def list_tools(tools_root: str | Path | None = None) -> list[Tool]:
    """Built-in registry tools first, then reservoir tools (never imported)."""
    out: list[Tool] = []
    seen: set[str] = set()

    # 1) the default registry (calculator, ...).
    from agent_seed.tools import registry

    for name in registry.names():
        spec = registry.get(name)
        if spec is None:
            continue
        source = ""
        try:
            source = inspect.getsource(spec.fn)
        except (OSError, TypeError):
            source = ""
        out.append(Tool(name=name, description=spec.description, source=source, in_registry=True))
        seen.add(name)

    # 2) the reservoir folder (ready-made tools to attach). Never imported: only
    #    ast-parsed for its tool functions and their top-level imports.
    root = _reservoir_root(tools_root)
    if root.is_dir():
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (OSError, SyntaxError):
                continue  # a broken reservoir file must not stop the scan
            imports = _top_level_imports(text, tree)
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):  # skip private helpers
                    continue
                if node.name in seen:
                    continue  # a built-in of the same name wins
                source = ast.get_source_segment(text, node)
                if not source:
                    continue
                out.append(
                    Tool(
                        name=node.name,
                        description=_first_doc_line(ast.get_docstring(node)),
                        source=source.strip() + "\n",
                        imports=imports,
                        in_registry=False,
                    )
                )
                seen.add(node.name)

    return out
