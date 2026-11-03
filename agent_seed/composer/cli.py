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


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except UnsupportedOperation:
        return False


def _parse_examples(raw: list[str]) -> list[dict[str, str]]:
    """`--example ROLE=CONTENT` (repeatable) -> role/content message list."""
    out: list[dict[str, str]] = []
    for item in raw or []:
        role, sep, content = item.partition("=")
        if not sep:
            raise ComposerError(f"--example expects ROLE=CONTENT, got {item!r}")
        role = role.strip()
        if role not in ("user", "assistant", "system", "tool"):
            raise ComposerError(f"--example role {role!r} must be user/assistant/system/tool")
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
    tools = [t.strip() for t in ",".join(tools_raw).split(",") if t.strip()]
    if not tools and interactive:
        from rich.prompt import Prompt

        hint = ", ".join(t.name for t in available) or "(none)"
        raw = Prompt.ask(
            "tools (comma-separated, blank for none)", default="", suffix=f"  [{hint}]"
        )
        tools = [t.strip() for t in raw.split(",") if t.strip()]

    # Context: optional.
    if context is None and interactive:
        from rich.prompt import Prompt

        context = Prompt.ask("context block (blank for none)", default=context or "")

    # Examples: optional.
    if not examples and interactive:
        from rich.prompt import Prompt

        raw = Prompt.ask("examples (role=content pairs, blank for none)", default="")
        if raw.strip():
            for pair in raw.split("|"):
                role, sep, content = pair.partition("=")
                if sep and role.strip():
                    examples.append({"role": role.strip(), "content": content})

    # Sign-off: optional.
    if signoff is None and interactive:
        from rich.prompt import Prompt

        signoff = Prompt.ask("sign-off line (blank for none)", default=signoff or "")

    spec = Spec(
        name=name,
        voice=(voice.strip() or None) if voice else None,
        tools=tools,
        context=(context.strip() or None) if context else None,
        examples=examples or None,
        signoff=(signoff.strip() or None) if signoff else None,
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
            examples = _parse_examples(args.example)
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
