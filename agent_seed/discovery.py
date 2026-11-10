"""Hook-module discovery: load promoted hooks from the ``hook_modules/`` folder.

A **hook module** is a standalone ``.py`` file that exposes one factory:

    def make() -> Hooks: ...

Public API
----------

* ``DEFAULT_HOOKS_FOLDER`` — the folder name relative to the CWD
  (``"hook_modules"``).
* ``_root_folder()`` — the hooks root: ``$HOOKS_ROOT`` (env) if set, else
  ``hook_modules`` relative to the CWD.
* ``get_hooks(folder=None)`` — the merged hooks for ``folder`` (default
  ``DEFAULT_HOOKS_FOLDER``), **cached by resolved path per process**.
* ``resolve_hooks_folder(agent_name=None)`` — map an ``agent_name`` to a
  folder: ``<root>/<agent_name>`` if that subdirectory exists, else the root
  (the "default agent"). Traversal (``..``) and names escaping the root are
  rejected and fall back to the root.
* ``get_hooks_for_agent(agent_name=None)`` — ``get_hooks(resolve_hooks_folder(
  agent_name))``; the merged hooks for the agent (root folder when ``None``),
  cached.
* ``list_agents()`` — the sorted names of subdirectories of the root, i.e. the
  discoverable ``agent_name`` values (an empty list when the root is absent or
  flat); the reserved ``default`` folder is excluded.
* ``RESERVED_AGENT_NAME`` — the name reserved for the main agent (``"default"``);
  a hooks subdirectory must not bear it, and ``agent_name="default"`` resolves to
  the root.
* ``find_reserved_subdir(folder=None)`` — the path of a stray ``default``
  subdirectory (a folder that must be renamed/removed), or ``None``.
* ``reset_cache()`` — drop all cached discoveries (for tests / hot reload).
  The cache is per resolved folder; changing ``$HOOKS_ROOT`` between calls
  requires ``reset_cache()`` to take effect.

Behavior
--------

``discover(folder)`` (called by ``get_hooks`` on a cache miss) scans the
folder, imports each ``*.py`` module under a unique name, validates it exposes
a ``make`` callable, and calls ``make()``. The resulting :class:`Hooks` are
merged in **alphabetical file order** (on a slot conflict, the file that sorts
last wins). Discovery happens **once per process per folder** and is cached:
the first use does one directory scan + import + merge, every later use is a
dict lookup.

Missing or empty folder -> ``Hooks()`` (no cost, old behavior).

Failures are **fail-fast**: a module that fails to import, that lacks
``make``, or whose ``make()`` raises / returns something other than
``Hooks`` raises a :class:`DiscoveryError` naming the offending file. Nothing
is half-merged. When an ``agent_name`` is in play, a broken
``hook_modules/<agent_name>/`` therefore surfaces the same way — as a
``DiscoveryError`` from ``get_hooks_for_agent`` (not a silent skip), so the
``Agent.set_agent_name`` switch aborts with nothing committed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from .hooks import Hooks

DEFAULT_HOOKS_FOLDER = "hook_modules"

#: The name reserved for the *main* agent (the hooks root). A hooks subdirectory
#: may not be named this: it would collide with the "switch back to default"
#: surface (the REPL's ``/agent default``), the composer (``agent compose default``),
#: and the API's ``agent_name="default"``. Enforced in the composer
#: (:func:`agent_seed.composer.generator.validate_name`) and surfaced by
#: :func:`find_reserved_subdir` (a scan that complains about an existing folder).
RESERVED_AGENT_NAME = "default"

_CACHE: dict[Path, Hooks] = {}


class DiscoveryError(RuntimeError):
    """A hook module failed to import, validate, or produce a Hooks."""


def reset_cache() -> None:
    """Drop all cached discoveries (mainly for tests / hot reload)."""
    _CACHE.clear()


def get_hooks(folder: str | Path | None = None) -> Hooks:
    """The merged hooks for ``folder`` (default ``hook_modules``), cached."""
    if folder is None:
        folder = DEFAULT_HOOKS_FOLDER
    key = Path(folder).resolve()
    if key not in _CACHE:
        _CACHE[key] = discover(key)
    return _CACHE[key]


def _root_folder() -> Path:
    """The hooks root: ``$HOOKS_ROOT`` if set, else ``hook_modules`` (CWD)."""
    env = os.environ.get("HOOKS_ROOT")
    return Path(env) if env else Path(DEFAULT_HOOKS_FOLDER)


def resolve_hooks_folder(agent_name: str | None = None) -> Path:
    """Map ``agent_name`` to a hooks folder.

    If ``agent_name`` names a subdirectory of the root (``hook_modules/<name>``)
    that directory is returned. Otherwise the root itself is returned (the
    "default agent"). A name equal to :data:`RESERVED_AGENT_NAME` always resolves
    to the root (the main agent) — it is never a literal folder. Traversal (``..``)
    and names escaping the root are rejected and fall back to the root.
    """
    root = _root_folder()
    root_resolved = root.resolve()
    if agent_name and agent_name != RESERVED_AGENT_NAME:
        candidate = root_resolved / agent_name
        try:
            real = candidate.resolve()
            if real.parent == root_resolved and real.is_dir():
                return real
        except (OSError, RuntimeError):  # broken symlinks / weird OS errors
            pass
    return root_resolved


def get_hooks_for_agent(agent_name: str | None = None) -> Hooks:
    """The merged hooks for ``agent_name`` (root folder when ``None``), cached."""
    return get_hooks(resolve_hooks_folder(agent_name))


def list_agents() -> list[str]:
    """Names of subdirectories of the root (potential ``agent_name`` values).

    The :data:`RESERVED_AGENT_NAME` folder (``default``), if present, is excluded:
    it names the main agent, not a selectable agent, so it must never be offered
    as one (the user switches back to it with ``/agent default``).
    """
    root = _root_folder()
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name != "__pycache__" and d.name != RESERVED_AGENT_NAME
    )


def find_reserved_subdir(folder: str | Path | None = None) -> Path | None:
    """The path of a ``default`` subdirectory under ``folder`` (default: the
    hooks root), or ``None`` if there is none.

    ``default`` is reserved for the main agent, so a ``hook_modules/default/``
    folder is anomalous: something created it before the reservation, or by
    hand. Callers (e.g. the composer) surface this as a complaint and direct the
    user to rename or delete it.
    """
    base = Path(folder) if folder is not None else _root_folder()
    candidate = base / RESERVED_AGENT_NAME
    return candidate if candidate.is_dir() else None


def discover(folder: Path) -> Hooks:
    """Load and merge every hook module and prompt file in ``folder`` (uncached).

    A folder is a set of hook modules (``*.py`` files exposing ``make()``) and
    optionally up to three prompt-content files with fixed names:

    * ``prompt.md``      -> ``Hooks.prompt``     (the agent's identity)
    * ``context.md``     -> ``Hooks.context``    (per-round context baseline)
    * ``examples.jsonl`` -> ``Hooks.examples``   (few-shot message list)

    Merge order for a data slot (lowest -> highest precedence):
    1. every ``*.py`` module's ``make()`` result (alphabetical, last non-None wins);
    2. the folder's own fixed-name file (wins over any module value).

    This is an extension of the existing "alphabetical last-wins" rule: a module
    that declares ``prompt`` is superseded by a ``prompt.md`` in the same folder
    (explicit file content beats code-declared content).
    """
    if not folder.is_dir():
        return Hooks()

    # Load every hook module once (fail-fast on any import / make() error),
    # keeping them in alphabetical order for the "last file wins" merges.
    modules: list[Hooks] = [_load_module(path) for path in sorted(folder.glob("*.py"))]

    merged = Hooks()
    for slot in (f.name for f in fields(Hooks)):
        winner: Any = None
        for hooks in modules:  # alphabetical: later file wins
            value = getattr(hooks, slot)
            if value is not None:
                winner = value
        setattr(merged, slot, winner)

    # Fixed-name prompt files override module-declared values (explicit file
    # content beats code-declared content).
    def _read_text(name: str) -> str | None:
        path = folder / name
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        return None

    prompt = _read_text("prompt.md")
    if prompt is not None:
        merged.prompt = prompt

    context = _read_text("context.md")
    if context is not None:
        merged.context = context

    examples_path = folder / "examples.jsonl"
    if examples_path.is_file():
        merged.examples = _load_examples(examples_path)

    return merged


def _load_examples(path: Path) -> list[dict[str, Any]]:
    """Parse an ``examples.jsonl`` file (one JSON object per non-blank line).

    Each line must be a JSON object; the result is a list of objects (deep-copied).
    An empty file yields ``[]``. A line that is not a valid JSON object raises
    ``DiscoveryError`` (fail-fast, same policy as a broken hook module).
    """
    out: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(
                f"examples file {path.name!r} line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DiscoveryError(
                f"examples file {path.name!r} line {lineno}: expected a JSON object, "
                f"got {type(value).__name__}"
            )
        out.append(value)
    return out


def _load_module(path: Path) -> Hooks:
    # A stable, per-path-unique name so that two hook folders that happen to
    # carry the same file stem (e.g. both have 00_tools.py) do not collide in
    # sys.modules. The stem keeps the name readable in tracebacks; a short
    # digest of the resolved absolute path guarantees uniqueness across folders.
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"agent_seed_hook_module_{path.stem}_{digest}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except DiscoveryError:
        raise
    except Exception as exc:  # noqa: BLE001 - report import failure with file name
        raise DiscoveryError(f"hook module {path.name!r} failed to import: {exc}") from exc

    make = getattr(module, "make", None)
    if not callable(make):
        raise DiscoveryError(f"hook module {path.name!r} must define `def make() -> Hooks`")
    try:
        hooks = make()
    except DiscoveryError:
        raise
    except Exception as exc:  # noqa: BLE001 - report make() failure with file name
        raise DiscoveryError(f"hook module {path.name!r}: make() raised: {exc}") from exc
    if not isinstance(hooks, Hooks):
        raise DiscoveryError(
            f"hook module {path.name!r}: make() must return a Hooks, got {type(hooks).__name__}"
        )
    return hooks
