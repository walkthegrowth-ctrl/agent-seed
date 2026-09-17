"""Console entry point: `agent-seed` — run the HTTP backend."""

from __future__ import annotations

import os

import uvicorn


def serve() -> None:
    host = os.environ.get("AGENT_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("AGENT_SERVER_PORT", "8787"))
    print(f"agent-seed backend listening on http://{host}:{port}")
    uvicorn.run(
        "agent_seed.backends.http.server:app",
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    serve()
