"""Console entry point: `agent-repl` — talk to the agent-seed backend."""

from __future__ import annotations

import argparse
import os
import sys

from .app import BackendError, Repl

DEFAULT_URL = "http://127.0.0.1:8787"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-repl",
        description="REPL frontend for agent-seed.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("AGENT_BACKEND_URL", DEFAULT_URL),
        help=f"backend base URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args(argv)

    repl = Repl(args.url)
    try:
        return repl.run()
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
