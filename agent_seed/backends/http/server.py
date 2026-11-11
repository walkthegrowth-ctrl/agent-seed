"""FastAPI app: SSE stream of agent events over HTTP.

Wire format (see docs/protocol.md):
    POST /chat    body {"session_id"?, "message", "agent_name"?} -> text/event-stream
    GET  /agents  -> discoverable agent names (subdirectories of the hooks root)
    GET  /config  -> server-side LLM config (llm | null)
    GET  /health  -> {"status": "ok", "version": ...}
    GET  /tools   -> this app's tool schemas (OpenAI function form) + names
                        (with ?agent_name=<n>: only the tools that agent's
                        whitelist allows; without: the app's whole catalog)
    GET  /prompt  -> the composed prompt (identity, context, addendum, examples);
                        scratch (deterministic) or live=1 with ?session_id=<id>
    POST /reset   body {"session_id"} -> clears that session's history
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
from ...core import DEFAULT_SYSTEM_PROMPT, Agent, is_number
from ...discovery import RESERVED_AGENT_NAME, get_hooks_for_agent, list_agents
from ...hooks import Hooks, RunContext
from ...llm import LLMConfig, LLMConfigError, OpenAICompatClient
from ...prompting import compose_system, normalize_examples
from ...tools import ToolRegistry
from ...tools import registry as default_registry


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    agent_name: str | None = None


class LLMUpdate(BaseModel):
    """Partial LLM config; at least one field must be supplied.

    The missing fields are merged over the current config (the session's
    ``current_model`` when a ``session_id`` is given, else the app default).
    """

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float | None = None


class ConfigUpdate(LLMUpdate):
    session_id: str | None = None


def llm_public(cfg: LLMConfig) -> dict[str, Any]:
    """A serializable view of an LLMConfig; the api key is never returned."""
    return {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_key_set": bool(cfg.api_key),
        "timeout": cfg.timeout,
    }


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
        # Per-session cumulative token usage (sum of each turn's per-turn usage).
        # Backend-managed: the core's ctx.usage is run-cumulative and survives
        # /reset, so the session-level total is tracked here and cleared by reset().
        self._session_usage: dict[str, dict[str, Any]] = {}
        # App-default LLM for NEW sessions (None -> env at session creation).
        # A POST /config without session_id updates this AND all live sessions.
        self._default_llm: LLMConfig | None = None
        # Configs set for a session_id that has not been created yet (no live
        # Agent). The frontends mint a session_id up front, so "/model <name>"
        # can legitimately arrive before the first /chat; the session adopts
        # this config as its base when it is created (see get()).
        self._pending: dict[str, LLMConfig] = {}

    def default_llm(self) -> LLMConfig:
        """The app-default LLM (the app-level setting if set, else env)."""
        if self._default_llm is not None:
            return self._default_llm
        return LLMConfig.from_env()

    def set_default_llm(self, cfg: LLMConfig) -> None:
        self._default_llm = cfg

    def pending(self, session_id: str) -> LLMConfig | None:
        """A config set for session_id before the session was created, or None."""
        return self._pending.get(session_id)

    def set_pending(self, session_id: str, cfg: LLMConfig) -> None:
        self._pending[session_id] = cfg

    def agent(self, session_id: str) -> Agent | None:
        return self._agents.get(session_id)

    @property
    def registry(self) -> ToolRegistry:
        """The tool registry this app serves (what `GET /tools` reports)."""
        return self._registry

    def get(self, session_id: str | None, agent_name: str | None = None) -> tuple[str, Agent]:
        session_id = session_id or uuid.uuid4().hex
        # The reserved name always names the main agent (None), never a folder
        # (discovery.py RESERVED_AGENT_NAME); normalize it once here so the
        # switch logic below treats it as a plain request for the main agent.
        if agent_name == RESERVED_AGENT_NAME:
            agent_name = None
        agent = self._agents.get(session_id)
        if agent is None:
            # One run per Agent. The run id is per-Agent (for debugging,
            # logging, and future parallel runs); the session id is the
            # conversation id and can later be switched via Agent.set_session.
            # A config pre-set for this session_id wins over the app default.
            cfg = self._pending.pop(session_id, None) or self.default_llm()
            # An app-default (not env-derived) LLM is passed explicitly so
            # Agent.base_model is whatever the app has been told to use.
            agent = Agent(
                OpenAICompatClient(cfg),
                self._registry,
                hooks=self._hooks,
                run_id=uuid.uuid4().hex,
                session_id=session_id,
                agent_name=agent_name,
            )
            self._agents[session_id] = agent
        elif agent.agent_name != agent_name:
            # A real switch — including back to the main agent (None). Re-resolve
            # hooks and let the establishment hooks re-fire on the next turn.
            # An omitted agent_name is a request for the main agent (protocol.md),
            # not "keep whatever is currently loaded".
            agent.set_agent_name(agent_name)
        return session_id, agent

    def accumulate_usage(
        self, session_id: str, turn_usage: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge one turn's per-turn usage into the session cumulative total.

        Returns the session total (summed numeric counters). Empty/None input is
        a no-op so a turn whose LLM sent no ``usage`` leaves the total untouched.
        """
        if not turn_usage:
            return dict(self._session_usage.get(session_id, {}))
        total = dict(self._session_usage.get(session_id, {}))
        for key, value in turn_usage.items():
            prev = total.get(key)
            if is_number(value) and is_number(prev):
                total[key] = prev + value
            else:
                total[key] = value
        self._session_usage[session_id] = total
        return total

    def reset(self, session_id: str) -> None:
        agent = self._agents.get(session_id)
        if agent is not None:
            agent.reset()
        self._session_usage.pop(session_id, None)


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
    def config(session_id: str | None = None) -> JSONResponse:
        if session_id is None:
            try:
                llm = sessions.default_llm()
            except LLMConfigError:
                return JSONResponse({"llm": None})
            return JSONResponse({"llm": llm_public(llm)})
        agent = sessions.agent(session_id)
        if agent is None:
            pending = sessions.pending(session_id)
            if pending is None:
                return JSONResponse({"error": f"no such session: {session_id}"}, status_code=404)
            # Session not created yet, but a config was set for it: report it.
            # The base is the app default; if that can't be determined (e.g. no
            # LLM_MODEL and no app default explicitly set), report base: null
            # rather than a 500 — the caller asked about a known session and
            # got a valid answer for the pending config.
            try:
                base_llm = llm_public(sessions.default_llm())
            except LLMConfigError:
                base_llm = None
            return JSONResponse({"llm": llm_public(pending), "base": base_llm})
        return JSONResponse(
            {
                "llm": llm_public(agent.context.current_model),
                "base": llm_public(agent.context.base_model),
            }
        )

    @app.post("/config")
    def set_config(body: ConfigUpdate) -> JSONResponse:
        if (
            body.model is None
            and body.base_url is None
            and body.api_key is None
            and body.timeout is None
        ):
            return JSONResponse(
                {"error": "at least one of model, base_url, api_key, timeout is required"},
                status_code=400,
            )

        def merge(base: LLMConfig) -> LLMConfig:
            return base.with_updates(
                base_url=body.base_url,
                api_key=body.api_key,
                model=body.model,
                timeout=body.timeout,
            )

        try:
            if body.session_id is not None:
                agent = sessions.agent(body.session_id)
                if agent is None:
                    # The session id is known to the caller but not yet created
                    # (frontends mint a session_id up front, so "/model <name>"
                    # can arrive before the first /chat). Record it as pending;
                    # SessionStore.get() adopts it when the session starts. The
                    # active agent's enter_agent still runs at creation and can
                    # override (sub-agent specialization) or restore base.
                    base = sessions.pending(body.session_id) or sessions.default_llm()
                    effective = merge(base)
                    sessions.set_pending(body.session_id, effective)
                    return JSONResponse(
                        {"ok": True, "session_id": body.session_id, "llm": llm_public(effective)}
                    )
                target = agent.context.current_model or sessions.default_llm()
                effective = merge(target)
                effective = agent.set_model(effective)  # on_model_change may veto (raise)
                return JSONResponse(
                    {"ok": True, "session_id": body.session_id, "llm": llm_public(effective)}
                )
            # No session: become the app default AND apply to every live session.
            # Atomic: if any live session's on_model_change vetoes, every live
            # session is restored to its prior current_model and the app default
            # is left unchanged (protocol.md: "vetoed → 409"). We apply the
            # sessions first and commit the app default only if they all accept,
            # so a partial commit is impossible.
            target = sessions.default_llm()
            effective = merge(target)
            # Snapshot each live session's prior current_model for rollback.
            snapshots = [
                (agent, agent.context.current_model) for agent in sessions._agents.values()
            ]
            try:
                for agent, _ in snapshots:
                    agent.set_model(effective)  # on_model_change may veto (raise)
            except Exception:
                # Restore every session to its pre-loop state. The rejecting
                # agent's set_model already rolled itself back; the others are
                # restored here. The app default is NOT committed.
                for agent, prior in snapshots:
                    agent.context.current_model = prior
                raise
            sessions.set_default_llm(effective)
            return JSONResponse({"ok": True, "llm": llm_public(effective)})
        except LLMConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - hook veto or client failure
            return JSONResponse({"error": str(exc)}, status_code=409)

    @app.post("/config/test")
    def test_config(body: ConfigUpdate) -> JSONResponse:
        # Merge over the most specific base available: the (pending or live)
        # session if given, else the app default.
        try:
            if body.session_id is not None:
                agent = sessions.agent(body.session_id)
                if agent is not None:
                    base = agent.context.current_model or sessions.default_llm()
                else:
                    base = sessions.pending(body.session_id) or sessions.default_llm()
            else:
                base = sessions.default_llm()
        except LLMConfigError as exc:
            # No base to merge over (e.g. LLM_MODEL unset and no pending/app
            # config). Do not silently probe a fabricated default that would
            # report a bogus model / false success — a client error instead.
            return JSONResponse(
                {
                    "ok": False,
                    "error": (
                        f"no base LLM config to probe against ({exc}); "
                        "set the app default (POST /config) or supply "
                        "base_url + api_key + model in the body"
                    ),
                },
                status_code=400,
            )
        try:
            candidate = base.with_updates(
                base_url=body.base_url,
                api_key=body.api_key,
                model=body.model,
                timeout=body.timeout,
            )
        except LLMConfigError as exc:
            return JSONResponse({"ok": False, "error": str(exc)})
        probe = Agent.test_model(candidate)
        return JSONResponse(
            {
                "ok": probe.ok,
                "error": probe.error,
                "latency_ms": probe.latency_ms,
                "usage": probe.usage,
                "model": candidate.model,
            }
        )

    @app.get("/tools")
    def tools(agent_name: str | None = None) -> JSONResponse:
        # Without agent_name: the whole catalog (what any agent COULD see —
        # useful for inventory / UI, not a live session's toolset).
        # With agent_name: only the tools that agent's whitelist allows.
        # We learn the declared set by running the agent's enter_agent hook
        # against a scratch RunContext (strict policy: None -> no tools).
        # A broken hook (one that raises) is a server-side fault (protocol.md:
        # "returns 500") — return a clean JSON body, not a traceback.
        if agent_name is None:
            tool_schemas = registry.schemas()
        else:
            hooks = get_hooks_for_agent(agent_name)
            allowed: list[str] | None = None
            if hooks is not None and hooks.enter_agent is not None:
                ctx = RunContext(agent_name=agent_name)
                try:
                    hooks.enter_agent(ctx)
                except Exception as exc:  # noqa: BLE001 - a broken agent hook
                    return JSONResponse({"error": str(exc)}, status_code=500)
                allowed = ctx.allowed_tools
            tool_schemas = registry.schemas_for(allowed) if allowed else []
        # names always reflects the tools actually returned (never a whitelist
        # name that is not in the app's registry).
        return JSONResponse(
            {"tools": tool_schemas, "names": [t["function"]["name"] for t in tool_schemas]}
        )

    @app.get("/agents")
    def agents() -> dict[str, Any]:
        """List discoverable agent names (subdirs of the hooks root)."""
        return {"agents": list_agents()}

    @app.get("/prompt")
    def prompt(
        session_id: str | None = None, agent_name: str | None = None, live: bool = False
    ) -> JSONResponse:
        """Read-only view of the composed prompt: identity, context, addendum, examples.

        ``live=true`` with a ``session_id`` reflects that session's last composed
        value (the sticky identity a hook may have set, plus that round's
        context/examples/addendum). Otherwise the scratch (deterministic) view is
        resolved from the selected agent's declared baselines — no live round, no
        hooks fired, mirroring ``GET /tools?agent_name=``.
        """
        if live:
            if session_id is None:
                return JSONResponse({"error": "live=true requires a session_id"}, status_code=400)
            agent = sessions.agent(session_id)
            if agent is None:
                return JSONResponse({"error": f"no such session: {session_id}"}, status_code=404)
            ctx = agent.context
            return JSONResponse(
                {
                    "agent_name": ctx.agent_name,
                    "session_id": session_id,
                    "live": True,
                    "system": compose_system(ctx.system_prompt, ctx.context, ctx.addendum),
                    "identity": ctx.system_prompt,
                    "context": ctx.context,
                    "addendum": ctx.addendum,
                    "examples": normalize_examples(ctx.examples),
                }
            )
        # Scratch: resolve the agent's hooks (fall back to the root) and use
        # the declared baselines as the per-round seed, exactly as the core does.
        hooks = get_hooks_for_agent(agent_name)
        identity = hooks.prompt or DEFAULT_SYSTEM_PROMPT
        return JSONResponse(
            {
                "agent_name": agent_name,
                "live": False,
                "system": compose_system(identity, hooks.context, None),
                "identity": identity,
                "context": hooks.context,
                "addendum": None,
                "examples": normalize_examples(hooks.examples),
            }
        )

    @app.post("/reset")
    def reset(body: SessionRequest) -> dict[str, Any]:
        sessions.reset(body.session_id)
        return {"ok": True, "session_id": body.session_id}

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            session_id, agent = sessions.get(request.session_id, request.agent_name)
        except Exception as exc:  # noqa: BLE001 - pre-stream setup failure
            # Anything that fails before the stream can start is a 500 with a
            # clean JSON body (protocol: "500 — configuration error before the
            # stream could start") — never a raw traceback. This covers the
            # config error (LLM_MODEL unset), a broken agent hook folder
            # (DiscoveryError from a mid-chat agent_name switch), and a hook
            # whose enter_agent raises (same documented-500 case as /tools).
            # The agent switch has already aborted if it failed (no state
            # was committed), so nothing is left half-set.
            return JSONResponse({"error": str(exc)}, status_code=500)

        async def stream():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def put(kind: str, payload: Any) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

            def producer() -> None:
                succeeded = False
                try:
                    agent.chat(
                        request.message,
                        on_token=lambda t: put("token", {"text": t}),
                        on_event=lambda kind, payload: put(kind, payload),
                    )
                    succeeded = True
                except Exception as exc:  # noqa: BLE001 - stream boundary
                    put("error", {"message": str(exc)})
                finally:
                    payload: dict[str, Any] = {}
                    if succeeded:
                        turn = dict(agent.context.turn_usage or {})
                        total = sessions.accumulate_usage(session_id, turn)
                        if turn:
                            payload["usage"] = turn
                        if total:
                            payload["usage_session"] = total
                    put("__done__", payload)

            loop.run_in_executor(None, producer)
            finished_ok = False
            done_payload: dict[str, Any] = {}
            while True:
                kind, payload = await queue.get()
                if kind == "__done__":
                    finished_ok = True
                    done_payload = payload or {}
                    break
                if kind == "error":
                    yield sse_event("error", payload)
                    break
                yield sse_event(kind, payload)
            if finished_ok:
                done_payload["session_id"] = session_id
                yield sse_event("done", done_payload)

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
