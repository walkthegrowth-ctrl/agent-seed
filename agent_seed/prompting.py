"""Prompt composition: turn (identity, context, examples, addendum) into messages.

The core builds the exact ``ctx.messages`` it sends each round by calling
:func:`compose_prompt`. It renders the agent's prompting inputs in a fixed,
documented order (see ``docs/manual.md``, "Prompting"):

    1. one ``system`` message = identity (``ctx.system_prompt``, the sticky
       value) with the optional ``ctx.context`` block and ``ctx.addendum``
       folded in as trailing sections (``compose_system``);
    2. the ``ctx.examples`` few-shot message pairs (role/content, or richer
       role messages as the model accepts);
    3. the stored ``history``.

Hooks keep full control: ``before_llm`` runs *after* ``ctx.messages`` is built,
so it may read or overwrite any of the per-round inputs (``ctx.context``,
``ctx.examples``, ``ctx.addendum``), rewrite ``ctx.system_prompt``, or replace
``ctx.messages`` outright. These functions are the *default composition*, not a
lock on the model's context.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

# A single "role message" is a Mapping with at least "role" and "content".
Message = Mapping[str, Any]


def compose_system(system_prompt: str | None, *sections: str | None) -> str:
    """Join the identity and its optional trailing sections with blank lines.

    Empty/whitespace-only sections are dropped, so a run with no context and
    no addendum returns the identity unchanged (the backward-compatible case).
    """
    parts = [section for section in (system_prompt, *sections) if section and section.strip()]
    return "\n\n".join(part.strip() for part in parts)


def normalize_examples(examples: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Coerce a declared example list into a clean list of role messages.

    Accepts the OpenAI role-message shape (``{"role", "content", ...}``) and
    passes each item through as a fresh dict (a deep copy, so a hook mutating
    one turn's ``ctx.examples`` cannot leak into the stored baseline). Items
    without a ``role`` are dropped; everything else is preserved verbatim.
    ``None`` / empty -> ``[]``.
    """
    out: list[dict[str, Any]] = []
    for item in examples or []:
        if isinstance(item, Mapping) and item.get("role"):
            out.append(copy.deepcopy(dict(item)))
    return out


def compose_prompt(
    system_prompt: str | None,
    history: Sequence[Message] | None,
    context: str | None = None,
    examples: Sequence[dict[str, Any]] | None = None,
    addendum: str | None = None,
) -> list[dict[str, Any]]:
    """Build the message list the LLM should see for one round.

    Render order: one ``system`` message (identity + context + addendum via
    :func:`compose_system`), then the few-shot ``examples``, then ``history``.
    When context, examples, and addendum are all empty the result is exactly
    ``[{"role":"system","content":system_prompt}, *history]`` — i.e. identical
    to what the core sent before this layer existed (backward compatible).

    In the live ``Agent.chat`` loop the core does **not** call this function:
    it lays down ``[system, *history]`` *before* ``before_llm`` so the hook can
    edit ``ctx.messages`` (its zone), then *after* ``before_llm`` it rebuilds
    the system message from the (possibly hook-changed) canonical fields and
    splices ``normalize_examples(ctx.examples)`` between the system message and
    ``ctx.messages[1:]`` — giving ``ctx.examples`` the same post-hook semantics
    as ``ctx.context`` / ``ctx.addendum``. This function renders the same final
    shape and is used by the ``MAX_TOOL_ROUNDS`` forced-final fallback and as a
    standalone utility (e.g. tests, scripts).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": compose_system(system_prompt, context, addendum)}
    ]
    messages.extend(normalize_examples(examples))
    messages.extend(
        copy.deepcopy(dict(msg)) for msg in (history or []) if isinstance(msg, Mapping)
    )
    return messages
