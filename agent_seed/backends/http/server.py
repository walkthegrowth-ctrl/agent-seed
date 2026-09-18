"""FastAPI app: SSE stream of agent events over HTTP.

Wire format (see docs/protocol.md):
    POST /chat  body {"session_id"?, "message"}  -> text/event-stream
    GET  /tools -> tool schemas
    GET  /health
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ... import __version__
from ...core import Agent
from ...discovery import list_agents
from ...hooks import Hooks
from ...llm import LLMConfig, LLMConfigError, OpenAICompatClient
from ...tools import ToolRegistry
from ...tools import registry as default_registry


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    agent_name: str | None = None


class SessionRequest(BaseModel):
    session_id: str


class SessionStore:
    """In-memory session_id -> Agent. Single-user by design."""

    def __init__(
        self,
        registry: ToolRegistry = default_registry,
        hooks: Hooks | None = None,  # None -> Agent auto-discovers from hook_modules/
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self._registry = registry
        self._hooks = hooks

    def get(self, session_id: str | None, agent_name: str | None = None) -> tuple[str, Agent]:
        session_id = session_id or uuid.uuid4().hex
        agent = self._agents.get(session_id)
        if agent is None:
            # One run per Agent. The run id is per-Agent (for debugging,
            # logging, and future parallel runs); the session id is the
            # conversation id and can later be switched via Agent.set_session.
            agent = Agent(
                OpenAICompatClient(LLMConfig.from_env()),
                self._registry,
                hooks=self._hooks,
                run_id=uuid.uuid4().hex,
                session_id=session_id,
                agent_name=agent_name,
            )
            self._agents[session_id] = agent
        elif agent_name is not None and agent.agent_name != agent_name:
            # The caller switched agents mid-session: re-resolve hooks and
            # let the establishment hooks re-fire on the next turn.
            agent.set_agent_name(agent_name)
        return session_id, agent

    def reset(self, session_id: str) -> None:
        agent = self._agents.get(session_id)
        if agent is not None:
            agent.reset()


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(
    registry: ToolRegistry = default_registry,
    hooks: Hooks | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``hooks=None`` (default) auto-discovers hook modules from the
    ``hook_modules/`` folder once per process. Pass an explicit
    ``Hooks(...)`` to override discovery for this app.
    """
    app = FastAPI(title="agent-seed", version=__version__)
    sessions = SessionStore(registry, hooks=hooks)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/config")
    def config() -> dict[str, Any]:
        try:
            llm = LLMConfig.from_env()
        except LLMConfigError:
            return {"llm": None}
        return {
            "llm": {
                "base_url": llm.base_url,
                "model": llm.model,
                "api_key_set": bool(llm.api_key and llm.api_key != "unset"),
            }
        }

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        return {"tools": registry.schemas(), "names": registry.names()}

    @app.get("/agents")
    def agents() -> dict[str, Any]:
        """List discoverable agent names (subdirs of the hooks root)."""
        return {"agents": list_agents()}

    @app.post("/reset")
    def reset(body: SessionRequest) -> dict[str, Any]:
        sessions.reset(body.session_id)
        return {"ok": True, "session_id": body.session_id}

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            session_id, agent = sessions.get(request.session_id, request.agent_name)
        except LLMConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

        async def stream():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def put(kind: str, payload: Any) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

            def producer() -> None:
                try:
                    agent.chat(
                        request.message,
                        on_token=lambda t: put("token", {"text": t}),
                        on_event=lambda kind, payload: put(kind, payload),
                    )
                except Exception as exc:  # noqa: BLE001 - stream boundary
                    put("error", {"message": str(exc)})
                finally:
                    put("__done__", None)

            loop.run_in_executor(None, producer)
            finished_ok = False
            while True:
                kind, payload = await queue.get()
                if kind == "__done__":
                    finished_ok = True
                    break
                if kind == "error":
                    yield sse_event("error", payload)
                    break
                yield sse_event(kind, payload)
            if finished_ok:
                yield sse_event("done", {"session_id": session_id})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Session-Id": session_id,
            },
        )

    return app


app = create_app()
