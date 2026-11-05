"""Console entry point: `agent compose` — create a ready-to-call agent folder.

Flags-first (scriptable); in a terminal it also offers a short wizard that prompts
for the values you did not pass (voice, tools, context, examples, sign-off). A
generated agent is a self-contained ``hook_modules/<name>/`` folder — discovery,
``GET /agents``, ``/agent <name>`` and ``POST /chat`` all work without any core
change. It is removable with ``rm -rf hook_modules/<name>``.
"""

from __future__ import annotations

import argparse
import os
import sys
from io import UnsupportedOperation

from .catalog import Tool, list_tools
from .generator import ComposerError, Plan, Spec, generate, validate_name, write_plan

DEFAULT_ROOT = os.environ.get("HOOKS_ROOT") or "hook_modules"

_VALID_ROLES = ("user", "assistant", "system", "tool")


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except UnsupportedOperation:
        return False


def unquote_once(s: str) -> str:
    """Strip whitespace; if the result begins and ends with the same quote, drop one outer pair.

    - ``'x'`` -> ``x``
    - ``"x"`` -> ``x``
    - ``x`` -> ``x`` (unchanged)
    - ``He said "x"`` -> ``He said "x"`` (outer chars differ; inner ``'x'`` stays)
    - ``'  padded  '`` -> ``padded`` (unquote, then strip)
    - ``'x`` (unbalanced) -> ``'x`` (unchanged)
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s


def parse_tools(raw: str) -> list[str]:
    """Parse a comma-separated tools string.

    Trim each token, drop empty ones, de-dup preserving first occurrence.
    Empty / whitespace-only input -> ``[]``.
    """
    out: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token and token not in out:
            out.append(token)
    return out


def resolve_tools(tools: list[str], available: list[Tool]) -> list[str]:
    """Check that every requested tool name exists in the catalog.

    Shared by the flag path and the wizard so an unknown tool is a clean
    ``error:`` message at either entry point (not a Python traceback, and not
    a silent "empty agents" surprise). Raises :class:`ComposerError`.
    """
    have_names = {t.name for t in available}
    have_hint = ", ".join(t.name for t in available) or "(none)"
    unknown = [n for n in tools if n not in have_names]
    if unknown:
        raise ComposerError(
            f"tool(s) not found in the catalog: {', '.join(unknown)}. Available: {have_hint}"
        )
    return list(tools)


def parse_examples_pairs(items: list[str]) -> list[dict[str, str]]:
    """Validate and normalise ``ROLE=CONTENT`` items into a message list.

    Shared by the ``--example`` flag and the interactive wizard so both
    entry points reject the same things. Each item must partition on its
    first ``=`` into a non-empty, well-known role (``user``/``assistant``/
    ``system``/``tool``); the content may be empty.
    """
    out: list[dict[str, str]] = []
    for item in items:
        role, sep, content = item.partition("=")
        if not sep:
            raise ComposerError(f"example expects ROLE=CONTENT, got {item!r}")
        role = role.strip()
        if role not in _VALID_ROLES:
            raise ComposerError(f"example role {role!r} must be user/assistant/system/tool")
        out.append({"role": role, "content": content})
    return out


def _build_spec(
    name: str | None,
    voice: str | None,
    tools_raw: list[str],
    context: str | None,
    examples: list[dict[str, str]],
    signoff: str | None,
    interactive: bool,
    available: list[Tool],
) -> Spec:
    """Resolve (prompt-if-interactive) the final values and build a Spec."""
    # Name: required.
    if not name:
        if interactive:
            from rich.prompt import Prompt

            while True:
                name = Prompt.ask("agent name", default=name or "")
                try:
                    validate_name(name)
                    break
                except ComposerError as exc:
                    print(f"  {exc}")
        else:
            raise ComposerError("an agent name is required (pass NAME or run interactively)")

    # Voice: optional (empty -> no prompt.md, uses the default system prompt).
    if voice is None and interactive:
        from rich.prompt import Prompt

        voice = Prompt.ask("voice / persona", default=voice or "")

    # Tools: optional, comma-separated names. Show what is available.
    tools = parse_tools(",".join(tools_raw))
    if not tools and interactive:
        from rich.prompt import Prompt
        from rich.text import Text

        hint = ", ".join(t.name for t in available) or "(none)"
        prompt_text = Text("tools (comma-separated, blank for none)")
        prompt_text.append(f"  [{hint}]")
        while True:
            raw = Prompt.ask(prompt_text, default="")
            tools = parse_tools(raw)
            if not tools:  # blank -> accept "no tools"
                break
            try:
                resolve_tools(tools, available)  # clean error on unknown name
                break
            except ComposerError as exc:
                print(f"  {exc}")
            if not tools:
                break
            try:
                resolve_tools(tools, available)
                break
            except ComposerError as exc:
                print(f"  {exc}")

    # Context: optional.
    if context is None and interactive:
        from rich.prompt import Prompt

        context = Prompt.ask("context block (blank for none)", default=context or "")

    # Examples: optional. The wizard shares parse_examples_pairs with the flag
    # so a bad role (e.g. bogus=hi) is rejected here exactly as it is for --example.
    if not examples and interactive:
        from rich.prompt import Prompt

        raw = Prompt.ask("examples (role=content pairs, blank for none)", default="")
        if raw.strip():
            # Blank tokens (e.g. "||" or a trailing "|") are not pairs; drop them so
            # parse_examples_pairs only sees actual ROLE=CONTENT items.
            items = [tok for tok in raw.split("|") if tok.strip()]
            examples.extend(parse_examples_pairs(items))

    # Sign-off: optional.
    if signoff is None and interactive:
        from rich.prompt import Prompt

        signoff = Prompt.ask("sign-off line (blank for none)", default=signoff or "")

    spec = Spec(
        name=name,
        voice=(unquote_once(voice) or None) if voice else None,
        tools=tools,
        context=(unquote_once(context) or None) if context else None,
        examples=examples or None,
        signoff=(unquote_once(signoff) or None) if signoff else None,
    )
    return spec


def _print_next_steps(plan: Plan, root: str, name: str) -> None:
    folder = f"{root.rstrip('/')}/{name}"
    print(f"\nCreated {len(plan.files)} file(s) in {folder}/:")
    for rel in plan.files:
        print(f"  {folder}/{rel}")
    print("\nNext steps:")
    print("  1. Restart the backend if it is running (hooks load once per process).")
    print(f'  2. Call it: /agent {name} (REPL), or POST /chat with "agent_name": "{name}"')
    print("     List agents:  /agents  or  GET /agents")
    print(f"\nTo remove this agent:  rm -rf {folder}")
    print("Full guide:  docs/howto-craft-an-agent.md  ·  docs/manual.md 'Composing an agent'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-compose",
        description="Compose an agent: create a hook_modules/<name>/ folder (voice + tools).",
    )
    parser.add_argument("name", nargs="?", help="agent name ([a-z0-9_-]+; no slashes/dots)")
    parser.add_argument("--voice", help="the agent's persona (its prompt.md)")
    parser.add_argument(
        "--tools",
        action="append",
        default=[],
        metavar="TOOLS",
        help="comma-separated tool names to whitelist (built-in or from the tools/ reservoir)",
    )
    parser.add_argument("--context", help="per-round context baseline (its context.md)")
    parser.add_argument(
        "--example", action="append", default=[], help="ROLE=CONTENT few-shot pair (repeatable)"
    )
    parser.add_argument(
        "--signoff", help="a finish-hook line appended to every answer (its sign.py)"
    )
    parser.add_argument("--root", help=f"hooks root folder (default: {DEFAULT_ROOT})")
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="print the generated files; write nothing"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="overwrite an existing agent folder"
    )
    parser.add_argument(
        "--list-tools", action="store_true", help="list the available tools and exit"
    )
    args = parser.parse_args(argv)

    try:
        interactive = _is_interactive()
        available = list_tools()

        if args.list_tools:
            if available:
                for t in available:
                    where = "built-in" if t.in_registry else "reservoir"
                    print(f"{t.name:20} [{where:9}] {t.description or ''}")
            else:
                print("(no tools available)")
            return 0

        try:
            examples = parse_examples_pairs(args.example)
        except ComposerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        spec = _build_spec(
            name=args.name,
            voice=args.voice,
            tools_raw=args.tools,
            context=args.context,
            examples=examples,
            signoff=args.signoff,
            interactive=interactive,
            available=available,
        )

        plan: Plan = generate(spec.name, spec, available)

        if args.dry_run:
            print(plan.rendered())
            return 0

        root = args.root or DEFAULT_ROOT
        write_plan(plan, root, force=args.force)
        _print_next_steps(plan, root, spec.name)
        return 0

    except ComposerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
