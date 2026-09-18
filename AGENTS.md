# AGENTS.md

Instructions for OpenCode (and other AI agents) working in this repo.

## Project

`agent-seed` — a minimal Python LLM chatbot with tool calling. An OpenAI-compatible
backend (FastAPI + SSE) runs a framework-agnostic agent loop; a terminal REPL frontend
talks to it over a documented HTTP/SSE protocol.

## Commands

- `uv sync` — create `.venv` and install runtime **and** dev/test tooling (`pytest`,
  `ruff`). `dev` is uv's default dependency group, so plain `uv sync` is enough;
  don't add `--extra`/`--group`.
- `uv run pytest` — full suite (no network, no external services).
- `uv run pytest tests/test_agent.py::test_agent_uses_tool_then_answers` — single test.
- `uv run ruff check agent_seed experiments tests` and `uv run ruff format` — lint +
  format (run both before considering a change done).
- `uv run agent-seed` — boot the backend (default `127.0.0.1:8787`).
- `uv run agent-repl [--url URL]` — REPL against the backend (default `http://127.0.0.1:8787`).

There is **no typechecker** configured (no mypy/pyright); rely on ruff + pytest + the
type hints in code.

## Layout

- `agent_seed/core.py` — `Agent` (history + LLM/tool loop). Framework-agnostic; the
  stable API is `Agent.chat(message, on_token, on_event)`. The run-scoped
  `RunContext` is internal to the `Agent` (`Agent.context`), not a `chat()` argument.
- `agent_seed/hooks.py` — `Hooks` + `RunContext`: the extensibility surface. See
  `docs/manual.md`.
- `agent_seed/discovery.py` — auto-loads hook modules from `hook_modules/` once per
  process (`get_hooks`, cached; fail-fast). `agent_name` → `hook_modules/<name>/`
  via `resolve_hooks_folder`/`get_hooks_for_agent`; `list_agents()` lists subdirs;
  root is `$HOOKS_ROOT` or CWD-relative `hook_modules/`.
- `agent_seed/llm.py` — streaming client + `LLMConfig.from_env`. **No LLM SDK**; raw
  `httpx` against OpenAI-compatible `/chat/completions`.
- `agent_seed/tools.py` — `ToolRegistry`, `@tool`, JSON-schema generation, and the
  `calculator` tool.
- `agent_seed/backends/http/` — FastAPI app, SSE, session store.
- `agent_seed/frontends/repl/` — the first (and only) frontend; web/tui should follow
  `docs/protocol.md` and may reuse `frontends/repl/app.py::Session`.
- `experiments/` — standalone hook modules (e.g. `ahem.py`); each file is self-contained,
  the core never imports from here, so delete to remove. **Never auto-loaded.**
- `hook_modules/` — promoted hook modules. Auto-discovered once per process: every `*.py`
  that defines `def make() -> Hooks` is merged in (alphabetical, last wins on conflicts).
  Promotion from `experiments/` is a plain `cp`. No `__init__.py` there on purpose.
  Subdirectories are **agents**: `hook_modules/<agent_name>/` is an alternate hook set,
  selected by `agent_name` (replace semantics — the root's `*.py` do not apply to that
  agent). The root is the default agent and the fallback for names with no subdirectory.
  `agent_name` is on `RunContext`, set via `Agent(agent_name=...)`,
  `Agent.set_agent_name(...)`, or the `agent_name` field on `POST /chat`.

## Gotchas

- **Tests must stay network-free.** LLM calls are faked via `tests/conftest.py`
  (`FakeLLMClient`) and `tests/test_backend.py` (monkeypatches `OpenAICompatClient`).
  Don't add tests that hit a real endpoint.
- **`.env` is auto-loaded** by `LLMConfig.from_env` (python-dotenv). `LLM_MODEL` is
  required — if unset the backend returns HTTP 500. For tests, guard against a stray
  `.env` providing `LLM_MODEL`.
- **Tool round-trips are capped** (`MAX_TOOL_ROUNDS` in `core.py`); tool exceptions are
  turned into `error:` messages fed back to the LLM, not raised to the caller.
- **`docs/protocol.md` is the contract** between backend and any frontend. Change the SSE
  event format there and in `backends/http/server.py` together.
- **Hook signatures are frozen** (`(ctx: RunContext) -> None`). All new core behavior
  goes on `RunContext` (append-only, with defaults) — never on hook signatures.
  `ctx.extras` is hooks-only: the core never reads or writes it.
- **`RunContext` has two lifetimes.** Run-scoped fields (`run_id`, `session_id`,
  `agent_name`, `base_prompt`, `system_prompt`, `turn`, `usage`, `extras`) are sticky
  across turns and survive `reset()`. Per-request fields (`messages`, `tools`,
  `tool_loop_round`) are rebuilt by the core before each `before_llm` — never rely on
  stale per-request values. Don't mutate stored history from a hook; use `ctx.system_prompt`
  (canonical channel) to change the system prompt, not `messages[0]` directly.
- **Agents are hook subdirectories, replace semantics.** `agent_name` selects
  `hook_modules/<name>/` *instead of* the root (the root's `*.py` do not apply), with the
  root as the default/fallback. A matching-name switch re-resolves hooks and re-fires all
  four establishment hooks (like `set_session`). `agent_name` is traversal-safe and
  overridable via `HOOKS_ROOT`.
- **There is always an agent in place — the main agent is the nameless root.**
  `None` is the main agent, not a "no agent" state.
- **Agent identity boundary: `enter_agent` / `exit_agent`.** `enter_agent` fires on
  first activation (Agent ctor, even for the main agent) and as the incoming agent on
  every switch; `exit_agent` fires as the outgoing agent on every switch, never at
  construction. A switch is atomic: resolve new hooks first (broken subdir aborts
  cleanly), then `exit` (old) → commit → `enter` (new); the new agent's establishment
  hooks re-fire on the next `chat()`. `exit_agent` raising aborts the switch; a
  re-select of the same agent fires nothing.
- **Establishment fires on need, not per chat.** The run-establishment hooks
  (`initialize_run`, `ready_run`) fire exactly once per `Agent`. The session-establishment
  hooks (`initialize_session`, `ready_session`) fire on the first chat and again whenever
  the active session changes (`Agent.set_session`). An `agent_name` switch
  (`Agent.set_agent_name`) re-fires both groups. Don't assume they fire every turn.

## Style

- Small, explicit, standard library where it fits.
- Tool schemas for new tools are derived from the function's name, first-line docstring,
  and type hints — keep functions annotated. See `docs/manual.md`.
- New agent behavior (context, policy, budget, ...) is a hook in `experiments/`,
  attached via `Agent(..., hooks=Hooks(...))` or `create_app(hooks=...)`; promote it
  to `hook_modules/` (a `def make() -> Hooks` file) to auto-load it everywhere.
  See `docs/manual.md` (user manual).
