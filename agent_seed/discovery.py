"""Hook-module discovery: load promoted hooks from the ``hook_modules/`` folder.

A **hook module** is a standalone ``.py`` file that exposes one factory:

    def make() -> Hooks: ...

``get_hooks()`` scans the folder, imports each module, calls ``make()``,
and merges the results in **alphabetical file order** (on a slot conflict,
the file that sorts last wins). Discovery happens **once per process per
folder** and is cached: the first use does one directory scan + import +
merge, every later use is a dict lookup.

Missing or empty folder -> ``Hooks()`` (no cost, old behavior).

Failures are **fail-fast**: a module that fails to import, that lacks
``make``, or whose ``make()`` raises / returns something other than
``Hooks`` raises a :class:`DiscoveryError` naming the offending file.
Nothing is half-merged.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

from .hooks import Hooks

DEFAULT_HOOKS_FOLDER = "hook_modules"

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
    "default agent"). Traversal (``..``) and names escaping the root are
    rejected and fall back to the root.
    """
    root = _root_folder()
    root_resolved = root.resolve()
    if agent_name:
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
    """Names of subdirectories of the root (potential ``agent_name`` values)."""
    root = _root_folder()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != "__pycache__")


def discover(folder: Path) -> Hooks:
    """Load and merge every hook module in ``folder`` (uncached)."""
    if not folder.is_dir():
        return Hooks()
    hooks_by_path: list[tuple[Path, Hooks]] = [
        (path, _load_module(path)) for path in sorted(folder.glob("*.py"))
    ]
    merged = Hooks()
    for slot in (f.name for f in fields(Hooks)):
        winner: Callable[[Any], None] | None = None
        for _, hooks in hooks_by_path:  # alphabetical: later file wins
            value = getattr(hooks, slot)
            if value is not None:
                winner = value
        setattr(merged, slot, winner)
    return merged


def _load_module(path: Path) -> Hooks:
    module_name = f"agent_seed_hook_module_{path.stem}"
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
