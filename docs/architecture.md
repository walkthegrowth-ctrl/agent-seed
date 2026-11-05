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
│  prompting.py       compose identity/context/examples/addendum│
│                     into the round's messages (default render)│
└──────────────────────────────────────────────────────────────┘
```

The only contract a frontend must honour is the HTTP/SSE protocol
(see `protocol.md`). The only contract a tool must honour is a Python
function with a docstring and type hints (see `manual.md`). The core
has no idea whether it is being driven by a terminal, a browser, or a curl
request.

## Data flow for one turn

This is the one-sentence version. The full hook-by-hook lifecycle (establishment
hooks, the per-round tool loop, `prepare`/`finish`, and the per-agent boundary
hooks) lives in `manual.md` — treat it as the source of truth:

1. Frontend `POST /chat {message, session_id?, agent_name?}`.
2. Backend looks up (or creates) the `Agent` for that session and streams.
3. `Agent.chat()` appends the user message, then per round it (a) rebuilds
   the per-request `messages` / `tools` view from the run's history and the
   agent's whitelist, (b) runs the LLM, and (c) executes any tool calls the
   LLM returned, before looping back.
4. When the LLM returns plain text, `prepare` / `finish` run and the final
   text is returned.
5. The SSE mapping is the one thing the architecture cares about: each
   streamed content token → `event: token`, each executed tool →
   `event: tool`, termination → `event: done` (see `protocol.md`).

## Module responsibilities

| path | responsibility |
|------|----------------|
| `agent_seed/core.py` | `Agent`: history, tool loop, round-trip cap, prompt composition (seeds the per-round `context`/`examples`/`addendum`, calls `compose_prompt` to build each round's messages, and applies the canonical system block after `before_llm`). Talks to LLM + registry. No I/O besides the two interfaces it is given. Owns model selection: `set_model` (gated by `on_model_change`), `test_model`, `model_config`; passes `ctx.current_model` into every LLM call. |
| `agent_seed/hooks.py` | `Hooks` + `RunContext`: the stable, extensible point of the core (including the per-round `context`/`examples`/`addendum` fields and the declared `prompt`/`context`/`examples` data). See `manual.md`. |
| `agent_seed/discovery.py` | Auto-loads promoted hook modules **and the agent's prompt files** (`prompt.md` / `context.md` / `examples.jsonl`) from `hook_modules/` (once per process, cached; fail-fast). Resolves an `agent_name` to a subdirectory (`hook_modules/<name>/`) with the root as the default. |
| `agent_seed/llm.py` | `OpenAICompatClient`: streaming `POST /chat/completions`, SSE line parsing, chunk folding. `chat(..., config=None, max_tokens=None)` resolves a per-call config. `LLMConfig` (frozen) + `from_env` + `with_updates`; `LLMProbe`. |
| `agent_seed/tools.py` | `ToolRegistry`, `@tool`, JSON-schema generation from annotations, and the starter `calculator`. |
| `agent_seed/prompting.py` | `compose_prompt` / `compose_system` / `normalize_examples`: the default rendering of a round's prompt (identity + context + addendum as one system message, then few-shot examples, then history). A convenience surface — hooks keep full control. |
| `agent_seed/backends/http/server.py` | FastAPI app, session store, `POST /chat` SSE, `GET /tools`, `GET /agents`, `GET /prompt`, `GET /config` (+ `POST /config`, `POST /config/test` for runtime model selection), `GET /health`, `POST /reset`. |
| `agent_seed/backends/http/cli.py` | `agent-seed` entry point (uvicorn). |
| `agent_seed/frontends/repl/app.py` | `Session` (HTTP client), SSE parse + render, slash-commands. |
| `agent_seed/frontends/repl/cli.py` | `agent-repl` entry point. |
| `agent_seed/composer/catalog.py` | Tool catalog: built-in registry tools + the `tools/` "reservoir" (ast-scanned), as `Tool` dataclasses. |
| `agent_seed/composer/generator.py` | Pure `Spec → Plan → write_plan`: turns a compose spec (voice/tools/context/examples/sign-off) into an agent folder. Reservoir tools are inlined; built-ins whitelisted only. |
| `agent_seed/composer/cli.py` | `agent-compose` entry point (flags + interactive wizard; `--dry-run`, `--force`, `--list-tools`). |

## Sessions

Identified by an opaque `session_id` string. The backend keeps an
in-memory `session_id → Agent` map. The client mints a UUID per run and
`/reset` asks the server to clear that session's history. No persistence,
no auth, single-user by design — see below.

## Design decisions & boundaries

- **Raw `httpx2`, no LLM SDK.** One fewer dependency; works with any
  OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, OpenRouter, ...).
- **Tool schemas built from type hints.** No pydantic for tool input; a small
  `type_to_schema` maps `str/int/float/bool`, `list[T]`, `dict`, `Any`.
- **Errors are reported to the model, not raised to the user.** A tool
  exception becomes a `role: tool` message `error: ...` so the LLM can
  recover. Stream-level failures become an SSE `event: error`.
- **Tool round-trip cap** (`MAX_TOOL_ROUNDS = 8`) prevents runaway loops; the
  core then forces a final answer with tools disabled.
- **Prompt composition is a default render, not a lock.** Each round's messages
  are built by `agent_seed/prompting.py` from the run's identity (sticky
  `ctx.system_prompt`) plus the non-sticky per-round `context` / `examples` /
  `addendum` (seeded from the agent's declared `prompt.md` / `context.md` /
  `examples.jsonl`, then hooks may adjust). It is a convenience surface: hooks
  keep full control and win on every channel. See `manual.md`, "Prompting".
- **Deliberately out of scope:** persistence, auth, multi-user concurrency,
  rate limiting, config files, non-OpenAI tool
  dialects. Each is a planned extension point, not a bug. See also
  `docs/ideas.md` for smaller deferred items (e.g. the `examples.jsonl`
  `{"query","answer"}` sugar).

## Extending

Extension is a client-only or hook-only concern — the core is not modified.
`docs/manual.md` is the source of truth for tools, prompts, hooks, and agents;
the bullets below just point at the relevant section.

- New tool → `manual.md` "Adding tools".
- New agent prompts (identity, context, few-shot) → drop files in the agent's
  hook folder and/or a `before_llm` hook; `manual.md` "Prompting — the content layer".
- New agent behavior (policy, budget, ...) → a hook; `manual.md`
  "Hooks — the extension surface".
- A different *agent* (a distinct hook set) → `hook_modules/<agent_name>/`;
  `manual.md` "Agents: hook subdirectories".
- **Scaffolding an agent without hand-writing files** → `agent compose`
  (`agent_seed/composer/`: a pure `Spec → Plan → write_plan` generator plus a
  tool catalog over the built-in registry and the root `tools/` "reservoir");
  `manual.md` "Composing an agent (`agent compose`)".
- New frontend → `protocol.md`; `frontends/repl/app.py::Session` is the
  reference client.
- New backend transport (e.g. WebSocket, gRPC) → expose `core.Agent` the same
  way `backends/http/server.py` does; `Agent.chat(message, on_token, on_event)`
  is the stable interface.
