# agent-seed

Ultra minimalistic LLM agent with tool calling.

- **Backend** — small FastAPI server that runs the agent loop and exposes it as
  an HTTP + SSE stream.
- **Frontend** — a `prompt_toolkit` + `rich` REPL that talks to the backend
  over that protocol.
- **Core** — a framework-agnostic `Agent` (chat ↔ tool ↔ chat) plus an
  OpenAI-compatible streaming LLM client and a decorator-based tool registry.
- **Extensibility** — a frozen hook surface (`Hooks` + `RunContext`) that plugs
  behavior into the loop without subclassing, and a `hook_modules/` folder of
  promoted hooks that is auto-discovered once per process (see `docs/manual.md`).
  Want a tailor-made agent of your own? The 10-minute, no-code walkthrough is
  `docs/howto-craft-an-agent.md`.

The core has no idea whether it is being driven by a terminal, a browser, or
`curl`. The protocol is documented, so adding a web/TUI frontend is a
client-only task (see `docs/protocol.md`).

```
┌──────────────┐  HTTP/SSE   ┌──────────────┐  LLM (stream)  ┌─────────────┐
│   frontend   │ ──────────▶ │    backend   │ ──────────────▶│ LLM (any    │
│   (repl)     │ ◀────────── │   (FastAPI)  │ ◀──────────────│ OpenAI-     │
└──────────────┘             └──────┬───────┘                │ compat. box)│
                                    │ uses                   └─────────────┘
                                    ▼
                              ┌──────────┐
                              │   core   │  Agent loop + tools + LLM client
                              └──────────┘
```

## Dependencies

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Runtime libs:
`fastapi`, `uvicorn`, `httpx2`, `rich`, `prompt_toolkit`. Dev: `pytest`,
`pytest-asyncio`, `ruff`.

No LLM SDK — a thin `httpx2` client speaks the OpenAI chat-completions API
directly, so any OpenAI-compatible endpoint works (Ollama, LM Studio, vLLM,
OpenRouter, …).

## Setup

```bash
uv sync                      # create .venv, install runtime + dev deps (dev is the default group)
cp .env.example .env         # then set your LLM vars in .env
uv run pytest                # run the test suite (deterministic; one auto-skip real-LLM smoke)
```

## Configuration

All config is via environment variables (see `.env.example`):

| variable | default | meaning |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base (Ollama default) |
| `LLM_API_KEY` | `unused` | bearer token |
| `LLM_MODEL` | — *(required)* | model name (e.g. `qwen3:8b`) |
| `AGENT_SERVER_HOST` | `127.0.0.1` | bind host for the backend |
| `AGENT_SERVER_PORT` | `8787` | bind port for the backend |
| `HOOKS_ROOT` | `hook_modules` | hooks folder (root agent); agent subdirs live under it |

Note: there is no `LLM_TIMEOUT` env var; the client uses a 120 s default.
Change it at runtime with `POST /config` (the body accepts a `timeout`
field) — see `docs/protocol.md` — rather than a `.env` entry.

## Run

Terminal 1 — backend:

```bash
uv run agent-seed                 # or: LLM_MODEL=... uv run agent-seed
# → http://127.0.0.1:8787
```

Terminal 2 — REPL:

```bash
uv run agent-repl                 # defaults to http://127.0.0.1:8787
# uv run agent-repl --url http://other-host:8787
```

You can also **compose a new agent** (a `hook_modules/<name>/` folder with voice,
tools, context, examples, and a sign-off) without touching Python:

```bash
uv run agent-compose tutor --voice "You are a patient math tutor." --tools calculator
```

See `docs/howto-craft-an-agent.md` for the step-by-step walk-through.

REPL commands: `/help`, `/reset` (clear server-side session), `/tools`,
`/usage` (session token usage), `/agent [name]` / `/agents` (switch / list
agents), `/model` (show the session's LLM config), `/model <name>` (probe-then-switch),
`/model test <name>` (probe without switching), `/model force <name>` (switch
without a probe), `/quit` (or `/exit`). Type `/` alone to
open a multi-line block (close with `/` and Enter). Type `/quit` or press
`Ctrl-D` to exit.

### Example

```
you> what is 12 times 13?
you> ⚙ calculator(expression='12 * 13') → 156.0
agent> 12 times 13 is 156.

you> /tools
you> tools: calculator

you> /model
you> {
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "model": "qwen3:8b",
    "api_key_set": true,
    "timeout": 120.0
  }
}
```

## Using the backend directly

```bash
curl -N http://127.0.0.1:8787/chat \
  -H 'content-type: application/json' \
  -d '{"message": "what is 2 ** 10?"}'
```

Streams `event: tool`, `event: token`, and a final `event: done`
(see `docs/protocol.md`).

## Layout

```
agent_seed/
  core.py                 Agent: history + LLM/tool loop (framework-agnostic)
  hooks.py                Hooks + RunContext: the stable extension surface
  discovery.py            auto-loads promoted hooks from hook_modules/ (+ agent subdirs)
  llm.py                  OpenAI-compatible streaming client (httpx2)
  tools.py                @tool registry, JSON-schema gen, calculator
  prompting.py            compose identity/context/examples/addendum into a round
  composer/               `agent compose`: generator + tool reservoir catalog + CLI
    cli.py                `agent-compose` entry point (creates an agent folder)
  backends/
    http/
      server.py           FastAPI app + SSE + sessions
      cli.py              `agent-seed` entry point
  frontends/
    repl/
      app.py              REPL: Session client, SSE parse + render, /commands
      cli.py              `agent-repl` entry point
experiments/            prototype hook modules (e.g. ahem.py); never auto-loaded
hook_modules/           promoted hook modules (the default agent); subdirs are agents
tools/                  ready-made tool "reservoir" (`agent compose --tools` picks from here)
  date.py               sample tool: current_date
  random_int.py         sample tool: random_int
docs/
  architecture.md         layers, data flow, design decisions
  protocol.md             HTTP/SSE wire contract for frontends
  manual.md               user manual: adding tools + writing/promoting hooks
  howto-craft-an-agent.md no-code walk-through: make your own agent in minutes
tests/
  conftest.py             FakeLLMClient + script helpers
  test_calculator.py
  test_agent.py           agent loop (tool round-trips, errors, round cap)
  test_hooks.py           Hooks/RunContext defaults + slot wiring
  test_discovery.py       hook_modules scan, agent selection, enter/exit boundary
  test_llm.py             SSE parsing + chunk folding
  test_backend.py         real FastAPI app end-to-end (mocked LLM)
  test_model_selection.py runtime model selection (set_model / on_model_change / test_model)
  test_repl.py            REPL: SSE parse, Session client, /model & /model force & /model test
  test_repl_loop.py       REPL: command loop + slash commands
  test_e2e_repl.py        end-to-end: real backend+REPL vs a scripted in-loop LLM (pty, no network)
  test_e2e_real.py        end-to-end smoke vs a real LLM (auto-skips if none reachable)
  test_composer_generator.py  pure generator (name, channels, write_plan, --force)
  test_composer_catalog.py    tool catalog: built-ins + tools/ reservoir (ast scan)
  test_composer_e2e.py        a composed agent folder is discoverable + its tool callable
  e2e/                    e2e support: scripted LLM server (ThreadingHTTPServer) + pty REPL driver
```

## Design notes

- **Tool round-trip cap** (`MAX_TOOL_ROUNDS = 8`); the core then forces a
  final answer with tools disabled.
- **Tool errors are reported to the model** (`role: tool`, `error: ...`), not
  raised to the user.
- **Sessions are in-memory** and single-user by design — deliberately no
  auth, persistence, or rate limiting (see `docs/architecture.md` for the
  boundaries and extension points).
- **Extensibility via hooks, not subclassing.** New behavior (context, policy,
  budget, ...) attaches to fixed points in the loop through `Hooks`; promote a
  hook to `hook_modules/` to auto-load it everywhere (see `docs/manual.md`).
- **Agents.** `hook_modules/<name>/` is an alternate hook set selected by
  `agent_name` (`Agent(agent_name=...)`, `POST /chat`, REPL `/agent`); the root
  folder is the default agent (see `docs/manual.md`, and the no-code walk-through
  `docs/howto-craft-an-agent.md`). `agent compose` creates such a folder (voice,
  tools, context, examples, sign-off) in one step — see `docs/manual.md`,
  "Composing an agent".
- **`LLM_MODEL` is required.** The backend returns HTTP 500 if it is unset.

## License

See `LICENSE`.
