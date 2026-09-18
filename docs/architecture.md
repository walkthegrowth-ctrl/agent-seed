# Architecture

`agent-seed` is deliberately small. Everything fits in a handful of modules
and the whole system can be explained on one page.

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│  frontends/         (repl today; web, tui, ... in future)     │
│   - human- or machine-facing UIs                              │
│   - speak the HTTP/SSE protocol, nothing else                 │
├──────────────────────────────────────────────────────────────┤
│  backends/http      (FastAPI + SSE)                           │
│   - exposes the agent over HTTP, maintains sessions           │
├──────────────────────────────────────────────────────────────┤
│  hooks.py            Hooks + RunContext: the hook protocol    │
│  discovery.py        auto-loads promoted hook modules         │
│  core.py             Agent loop                               │
│   - owns history, drives LLM + tool round-trips               │
├──────────────────────────────────────────────────────────────┤
│  llm.py             OpenAI-compatible streaming client        │
│  tools.py           registry + schema gen + calculator tool   │
└──────────────────────────────────────────────────────────────┘
```

The only contract a frontend must honour is the HTTP/SSE protocol
(see `protocol.md`). The only contract a tool must honour is a Python
function with a docstring and type hints (see `manual.md`). The core
has no idea whether it is being driven by a terminal, a browser, or a curl
request.

## Data flow for one turn

1. Frontend `POST /chat {message, session_id?}`.
2. Backend looks up (or creates) the `Agent` for that session and streams.
3. `Agent.chat()` appends the user message and calls the LLM with the tool
   schemas.
4. If the LLM returns `tool_calls`, the core executes each via the registry,
   appends the `tool` role messages, and loops back to step 3.
5. When the LLM returns plain text, that is the final answer.
6. Each streamed content token → `event: token`, each executed tool →
   `event: tool`, termination → `event: done`.

## Module responsibilities

| path | responsibility |
|------|----------------|
| `agent_seed/core.py` | `Agent`: history, tool loop, round-trip cap. Talks to LLM + registry. No I/O besides the two interfaces it is given. |
| `agent_seed/hooks.py` | `Hooks` + `RunContext`: the stable, extensible point of the core. See `manual.md`. |
| `agent_seed/discovery.py` | Auto-loads promoted hook modules from `hook_modules/` (once per process, cached; fail-fast). Resolves an `agent_name` to a subdirectory (`hook_modules/<name>/`) with the root as the default. |
| `agent_seed/llm.py` | `OpenAICompatClient`: streaming `POST /chat/completions`, SSE line parsing, chunk folding. `LLMConfig.from_env`. |
| `agent_seed/tools.py` | `ToolRegistry`, `@tool`, JSON-schema generation from annotations, and the starter `calculator`. |
| `agent_seed/backends/http/server.py` | FastAPI app, session store, `POST /chat` SSE, `/tools`, `/config`, `/health`, `/reset`. |
| `agent_seed/backends/http/cli.py` | `agent-seed` entry point (uvicorn). |
| `agent_seed/frontends/repl/app.py` | `Session` (HTTP client), SSE parse + render, slash-commands. |
| `agent_seed/frontends/repl/cli.py` | `agent-repl` entry point. |

## Sessions

Identified by an opaque `session_id` string. The backend keeps an
in-memory `session_id → Agent` map. The client mints a UUID per run and
`/reset` asks the server to clear that session's history. No persistence,
no auth, single-user by design — see below.

## Design decisions & boundaries

- **Raw `httpx`, no LLM SDK.** One fewer dependency; works with any
  OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, OpenRouter, ...).
- **Tool schemas built from type hints.** No pydantic for tool input; a small
  `type_to_schema` maps `str/int/float/bool`, `list[T]`, `dict`, `Any`.
- **Errors are reported to the model, not raised to the user.** A tool
  exception becomes a `role: tool` message `error: ...` so the LLM can
  recover. Stream-level failures become an SSE `event: error`.
- **Tool round-trip cap** (`MAX_TOOL_ROUNDS = 8`) prevents runaway loops; the
  core then forces a final answer with tools disabled.
- **Deliberately out of scope:** persistence, auth, multi-user concurrency,
  rate limiting, config files, token accounting display, non-OpenAI tool
  dialects. Each is a planned extension point, not a bug.

## Extending

- New tool → register it (see `manual.md`); the schema, dispatch, and
  tests all fall out for free.
- New frontend → implement the protocol (see `protocol.md`); reuse
  `frontends/repl/app.py::Session` as a reference client or speak raw HTTP.
- New backend transport (e.g. WebSocket, gRPC) → expose `core.Agent` the same
  way `backends/http/server.py` does; `Agent.chat(message, on_token, on_event)`
  is the stable interface.
- New agent behavior (context, policy, budget, ...) → write a hook and attach
  it via `Hooks(...)` (see `manual.md`); promote a hook module (a
  `def make() -> Hooks` file) into `hook_modules/` to auto-load it in every
  `Agent`. Prototype in `experiments/` first — never auto-loaded. No core
  changes needed.
- A different *agent* (a distinct hook set) → put its modules in
  `hook_modules/<agent_name>/` and select it with `Agent(agent_name=...)`,
  `Agent.set_agent_name(...)`, or the `agent_name` field on `POST /chat`
  (see `protocol.md`, `manual.md`). The root `hook_modules/` is the default
  agent; a subdirectory replaces the root's hooks for that agent.
