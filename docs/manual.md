# agent-seed — user manual

This is the user manual for `agent-seed`, a framework-agnostic LLM agent loop
with tool calling. It covers the two things a user most often extends:

- **Adding tools** — plain Python functions that the LLM can call.
- **Writing hooks** — the stable extension surface for changing agent behavior
  without touching the core.

Companion documents:

- `docs/protocol.md` — the HTTP/SSE wire contract between any frontend and the
  backend.
- `docs/architecture.md` — layers, data flow, and design decisions.
- `README.md` — getting started: setup, config, and running.

## Contents

- [Adding tools](#adding-tools)
  - [How the schema is derived](#how-the-schema-is-derived)
  - [Type map](#type-map)
  - [Registration](#registration)
  - [Conventions &amp; gotchas](#conventions--gotchas)
  - [Testing tools](#testing-tools)
- [Hooks — the extension surface](#hooks--the-extension-surface)
  - [What a hook is](#what-a-hook-is)
  - [The full list of hooks (reference)](#the-full-list-of-hooks-reference)
  - [Scopes of one run and their fields](#scopes-of-one-run-and-their-fields)
  - [Reading the Agent's state](#reading-the-agents-state)
  - [The three lifetimes inside a `RunContext`](#the-three-lifetimes-inside-a-runcontext)
  - [Lifecycle](#lifecycle)
    - [`RunContext` reference](#runcontext-reference)
    - [`extras`: the walls](#extras-the-walls)
    - [System prompt (dynamic, sticky)](#system-prompt-dynamic-sticky)
  - [Hook modules and the two folders](#hook-modules-and-the-two-folders)
  - [Promoting experiments](#promoting-experiments)
    - [The contract: `make()`](#the-contract-make)
    - [Merge policies](#merge-policies)
    - [Overriding discovery](#overriding-discovery)
  - [Agents: hook subdirectories](#agents-hook-subdirectories)
    - [Your first agent (end to end)](#your-first-agent-end-to-end)
    - [Per-agent tool access (strict whitelist)](#per-agent-tool-access-strict-whitelist)
    - [The agent-identity boundary (enter / exit)](#the-agent-identity-boundary-enter--exit)
  - [Which hook to use, by goal](#which-hook-to-use-by-goal)
  - [Worked examples](#worked-examples)
  - [Testing hooks](#testing-hooks)
  - [Rules (the stability contract)](#rules-the-stability-contract)
  - [Extending the core with new slots](#extending-the-core-with-new-slots)
- [FAQ / troubleshooting](#faq--troubleshooting)
- [Glossary](#glossary)

---

## Adding tools

A tool is a plain Python function with a **one-line docstring** and **type
hints** on its parameters. That's it — the schema, dispatch, and (if you add
them) tests all follow from that.

### How the schema is derived

For a function `foo(a: str, b: int) -> float` with docstring `"Do the thing."`:

- `name` ← function name
- `description` ← first line of the docstring
- `parameters.properties` ← each annotated param mapped through `type_to_schema`
- `parameters.required` ← every annotated param (v0 treats all as required)

### Type map

| Python annotation | JSON schema |
|-------------------|-------------|
| `str`            | `{"type": "string"}` |
| `int`, `float`   | `{"type": "number"}` |
| `bool`           | `{"type": "boolean"}` |
| `list[T]`        | `{"type": "array", "items": <T>}` |
| `dict`           | `{"type": "object"}` |
| `Any`            | `{}` (no type constraint) |
| *unannotated*    | param is dropped from the schema (annotate it to expose it) |

### Registration

There are two equivalent ways to register:

```python
from agent_seed.tools import ToolRegistry

registry = ToolRegistry()


def echo(text: str) -> str:
    """Return the input unchanged."""
    return text


registry.register(echo)


# or the @tool decorator form (equivalent):
@registry.tool
def echo2(text: str) -> str:
    """Return the input unchanged."""
    return text
```

The default registry in `agent_seed/tools.py` already has `calculator`. If you
want your own toolset in a custom server, build a new `ToolRegistry` and pass
it to `create_app(registry)`:

```python
from agent_seed.backends.http.server import create_app
from agent_seed.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool
def weather(city: str) -> str:
    """Get the current weather for a city."""
    ...


app = create_app(registry)
```

The REPL/protocol does not change — `GET /tools` and tool calling both reflect
whatever registry the app was created with.

> **Registration ≠ availability.** A tool *registered* in the registry is only
> one step: under the **strict tool-whitelist policy** each agent must also
> *declare* which tools it may use via `ctx.allowed_tools` (in its
> `enter_agent` hook), or it offers **no** tools and rejects every tool call.
> See ["Per-agent tool access (strict whitelist)"](#per-agent-tool-access-strict-whitelist).

### Conventions &amp; gotchas

- **First docstring line is the description.** Put a blank line after it and
  the body is ignored for schema purposes.
- **Return a scalar or JSON-serialisable value.** The core does `str(result)`
  before handing it back to the model, so anything printable works, but return
  real data, not "ok", when the model needs to reason about it.
- **Raise to signal failure to the model.** Any `ToolError` (or `TypeError`
  from a bad argument) is caught by the core and re-shown to the LLM as
  `error: ...` so it can adapt. Don't swallow errors into a success string.
- **Unannotated parameters are dropped** from the schema. Always annotate.
- **Name collisions** — registering a second function with the same name
  overwrites the first; keep names unique.
- Tools run in the **server process** with its privileges. A tool like
  `calculator` is safe by design (AST-evaluated, no `eval`). If you add
  filesystem/shell tools, treat them as privileged and scope them accordingly.

### Testing tools

Tool logic is ordinary Python, so test it directly:

```python
# tests/test_weather.py
from agent_seed.tools import weather  # your new tool


def test_weather():
    assert weather("Berlin") == "sunny, 7C"
```

And, if you want to prove the agent actually invokes it, extend
`tests/test_agent.py` with a `FakeLLMClient` script that returns a
`tool_completion("weather", {"city": "Berlin"})` then a final text reply —
the same pattern already covers `calculator` (see
`test_agent_uses_tool_then_answers`).

---

## Hooks — the extension surface

A **hook** is a plain Python function you attach to a fixed point in the agent
loop's lifecycle. Every hook has exactly one signature, frozen forever:

```python
def my_hook(ctx: RunContext) -> None: ...
```

Hooks run **in place and synchronously**, in a fixed order. All of them except
the two agent-boundary hooks (`enter_agent` / `exit_agent`) run inside
`Agent.chat()`; those two run from the `Agent` lifetime (construction
`enter_agent`, and `exit_agent` → `enter_agent` on a switch). Every hook may
inspect or mutate any field of `RunContext` (including the open `extras`
namespace); the per-turn ones may veto a tool call and change the system
prompt. They do one thing: participate in the run.

What a hook is **not**:

- **not a middleware chain** — one slot holds at most one function. If you need
  two, compose them inside a single function; there is no priority list in a
  slot.
- **not an event bus** — hooks are not async, not decoupled from the turn; they
  run in the turn's call stack and their effects matter.
- **not subclassing** — you never subclass `Agent` to change behavior; you
  attach hooks to it, or promote a hook module (below).

Attaching is one line:

```python
from agent_seed.core import Agent
from agent_seed.hooks import Hooks, RunContext


def my_hook(ctx: RunContext) -> None:
    ...


agent = Agent(client, registry, hooks=Hooks(finish=my_hook))
```

`Hooks` slots all default to `None` — unset hooks simply don't run.

There is a second way that avoids this line entirely: put a **hook module** in
the `hook_modules/` folder and it is auto-discovered (see
["Promoting experiments"](#promoting-experiments)).

### The full list of hooks (reference)

Every slot at a glance — what fires it, who can trigger it, which agent runs
it, the `RunContext` fields it touches, its side effects, and a typical use.
Boundary hooks (top) fire from the `Agent` lifetime (ctor / `set_agent_name`);
all the rest fire inside `chat()`.

| Hook | What fires it | Who can request it | Runs as | `RunContext` fields involved | Side effects / guarantees | Typical use |
|------|---------------|--------------------|---------|------------------------------|---------------------------|-------------|
| `enter_agent` | `Agent(...)` construction (first activation) and on every switch to this agent | `Agent(agent_name=...)`, `set_agent_name()`, `POST /chat agent_name`, `/agent` | incoming agent | `agent_name`(new) | always an agent in place; RAII — pairs with a prior exit except at ctor; fires before that agent's establishment; fires for the main (None) agent too | initialize on takeover: load config, seed per-agent state |
| `exit_agent` | every switch away from this agent (never at construction) | `set_agent_name()`, `POST /chat agent_name`, `/agent` | outgoing agent | `agent_name`(old) | raises → switch aborts (atomic); never fires at ctor | clean up, persist, release |
| `on_model_change` | `Agent.set_model()` (never at construction) | `Agent.set_model()`, `POST /config` | active agent | `current_model`(new) | raises → change rolled back (atomic); fires after commit | gate / substitute / restore the model |
| `initialize_run` | first `chat()`, and again after any switch (once run not established) | core, on need | active agent | may set `run_id` | exactly once per (re)establishment | seed run identity / persistence key |
| `ready_run` | right after `initialize_run` (run guaranteed) | core, on need | active agent | read `run_id`, `extras` | once per (re)establishment | attach budget / memory |
| `initialize_session` | first `chat()`, `set_session`, and after any switch | core / caller on session change | active agent | may set `session_id` | on each session (re)establishment | seed / migrate per-session state |
| `ready_session` | right after `initialize_session` (session guaranteed) | core / caller on session change | active agent | read `session_id`, `extras` | on each session (re)establishment | load per-session external context |
| `before_llm` | every `tool_loop_round`, before each LLM call | core | active agent | `messages`, `tools`, `system_prompt` | per request only (rebuilt) | inject RAG/context/date, gate tools |
| `before_tool_call` | every tool call | core | active agent | `tool_call` (may rewrite/`None`) | `None` records a `"skipped by policy"` tool result | rewrite args, veto by policy |
| `after_tool_call` | after a successful tool exec | core | active agent | `tool_result` | fires only on success | redact / rewrite result |
| `on_round` | end of each `tool_loop_round` | core | active agent | any (read / `extras`) | fire-only, no required return | tracing / metrics |
| `prepare` | when `final_answer` is set, before `finish` | core | active agent | `final_answer` | before the answer is recorded to history | validate / normalize |
| `finish` | last word of the turn | core | active agent | `final_answer` | may rewrite; result recorded to history | transform / add signature |

Rules of thumb:

- **Boundary hooks are per-agent.** `enter_agent` / `exit_agent` run in the
  *agent that owns the slot*, so each agent's own lifecycle code is isolated.
- **Everything else is per-active-agent.** Once an agent is on the wheel, all
  the `chat()`-scope hooks run from its hook set.
- **Append-only.** All slots default to `None`; adding a slot is a minor
  change (existing hooks keep working). See ["Rules"](#rules-the-stability-contract).

### Scopes of one run and their fields

A **run** is one instantiation of `Agent` (one conversation driver). It is
synchronous, serves a single user, and owns **one** `RunContext` for its whole
life. A run drives **one active session at a time**; it may *switch* sessions
(see `Agent.set_session`), but never runs two at once. `run_id` exists for
debugging, logging, and future parallel runs — today it is 1:1 with the
`Agent`. A run scopes state at four granularities, mapping to these fields:

| scope         | field             | meaning                                        | set by                                            |
|---------------|-------------------|------------------------------------------------|---------------------------------------------------|
| **run**       | `run_id`          | one `Agent` lifetime                           | caller; else core auto-gens at establishment      |
| **run**       | `agent_name`      | which hooks subdirectory is active             | caller (ctor or `set_agent_name`); `None` = default |
| **session**   | `session_id`      | the conversation currently being driven        | caller; else core auto-gens at establishment      |
| **turn**      | `turn`            | one user message (1-based run counter)         | core                                              |
| **tool loop** | `tool_loop_round` | one LLM→tools iteration within a turn (0-based)| core                                              |

(The terms *run*, *session*, *turn*, *tool loop round* are defined in the
[Glossary](#glossary); this table is just their field mapping, so the section
name no longer implies a hierarchy the code does not have.)

### Reading the Agent's state

The run-scoped state is read back from the `Agent` as read-only surfaces
(change it through a hook's `RunContext`, or `set_session` / `set_agent_name`
for the session / agent — not by writing these):

| surface | what it gives you |
|---------|-------------------|
| `agent.agent_name`    | the active agent (`None` = the main / root agent) |
| `agent.run_id`        | the established run id (auto-generated if none was set) |
| `agent.session_id`    | the active session id |
| `agent.base_prompt`   | the immutable system prompt for the run |
| `agent.system_prompt` | the current system prompt (sticky; a hook may have changed it) |
| `agent.model_config`  | the model the core currently calls (= `ctx.current_model`); see [Model selection](#model-selection-dynamic-sticky) |
| `agent.history`       | the stored message list (every turn so far) |
| `agent.context`       | the full `RunContext` (run-scoped fields + the per-request fields of the last `chat()`) |

### The three categories of a `RunContext`

This is the most important part of the model. A `RunContext` falls into
**three groups**, and the core treats each differently (the same split
`hooks.py`'s docstring uses — see `agent_seed/hooks.py`):

| | **run-scoped** (stable across turns)   | **per-request** (rebuilt each LLM call)          | **working state** (set by the core / a tool call or turn) |
|---|----------------------------------------|--------------------------------------------------|------------------------------------------------------------|
| fields | `run_id`, `session_id`, `agent_name`, `base_prompt`, `system_prompt`, `allowed_tools`, `base_model`, `current_model`, `turn`, `usage`, `extras` | `messages`, `tools`, `tool_loop_round` | `tool_call`, `tool_result`, `final_answer` |
| core behavior | seeds once at establishment / keeps the hook's writes | rebuilds at the start of every round | `tool_call`/`tool_result` are set for one tool call; `final_answer` is set once per turn in `_finish_turn` |
| hook behavior | may read; may write once → the change **persists** ("sticky") until changed again | may read / rewrite *for this request only* | read the call's data; rewrite `tool_result` (via `after_tool_call`) or `final_answer` (via `prepare`/`finish`) — not persisted beyond the call / turn |

Consequences a hook can rely on:

- `ctx.session_id`, `ctx.system_prompt`, `ctx.allowed_tools`, and anything you
  put in `ctx.extras` **survive to the next turn** and survive
  `Agent.reset()`. That's the whole "sticky" story — you do not re-assert a
  changed value each turn.
- `ctx.messages`, `ctx.tools`, `ctx.tool_loop_round` are **rebuilt** before
  each `before_llm`. Do not rely on what you saw last request; recompute it.
- `ctx.usage` accumulates across the run.

### Lifecycle

Establishment runs **when needed** — once per run / per session, and **re-fires
when the active agent or session changes** — not on every `chat()`:

```
Agent is used for the first time (or after Agent.set_session)
├─ initialize_run(ctx)       # once per (re)establishment  →  hook may set ctx.run_id
│   [core: if still None → auto-generate]
├─ ready_run(ctx)            # once per (re)establishment  →  ctx.run_id guaranteed non-None
│
├─ initialize_session(ctx)   # on every session (re-)establishment
│   [core: if ctx.session_id is None → auto-generate]
└─ ready_session(ctx)        # on every session (re-)establishment;
                             #    ctx.session_id guaranteed non-None
```

Then for each `chat()` turn:

```
chat() begins  (turn N of the run)
 ┌─ per tool_loop_round (0 .. MAX_TOOL_ROUNDS-1), until the LLM answers:
 │   [core rebuilds ctx.messages from ctx.system_prompt + history;
 │    rebuilds ctx.tools from the registry ∩ ctx.allowed_tools (strict whitelist);
 │    sets ctx.tool_loop_round]
 │   ├─ before_llm(ctx)               # may rewrite ctx.messages / ctx.tools / ctx.system_prompt / ctx.allowed_tools
 │   [core applies ctx.system_prompt to messages[0]; sends the LLM request]
 │   │   if the LLM returns tool_calls:
 │   │     └─ per tool call:
 │   │         ├─ [core gate: is ctx.allowed_tools set AND does it contain
 │   │         │   this tool's name?]
 │   │         │    ├── NO → veto with "error: tool '<name>' not available
 │   │         │    │         to this agent"; no hook fires for this branch
 │   │         │    └── YES → continue:
 │   │         ├─ before_tool_call(ctx)     # rewrite ctx.tool_call or veto with None
 │   │         ├─ [core executes the tool]
 │   │         └─ after_tool_call(ctx)      # rewrite ctx.tool_result
 │   │   [core appends history entries]
 │   └─ on_round(ctx)                   # fire-only observability
 ├─ prepare(ctx)              # ctx.final_answer is set; validate / normalize
 └─ finish(ctx)               # LAST word; may rewrite ctx.final_answer
chat() ends —> the core returns ctx.final_answer
```

Notes:

- Establishment hooks fire **on need**: the run hooks once per (re)establishment,
  the session hooks on each active-session (re)establishment. Re-establishment
  happens on the first `chat()` and whenever the active session or agent changes
  (`set_session` / `set_agent_name`) — not on every `chat()`.
- `before_llm`, `on_round` fire once per round; `before_tool_call`,
  `after_tool_call` fire once per tool call (several possible per round).
- A vetoed tool call fires `before_tool_call` (setting `ctx.tool_call = None`)
  and records `role: tool` — "skipped by policy" — in history; it does not
  fire `after_tool_call`.
- The round-cap fallback (LLM kept calling tools after `MAX_TOOL_ROUNDS`)
  still fires `prepare` and `finish`; `before_llm` is not re-run for that
  forced final answer.

#### `RunContext` reference

```python
@dataclass
class RunContext:
    # run-scoped
    run_id: str | None          # established once per run; sticky
    session_id: str | None      # established per active session; sticky
    agent_name: str | None      # which hooks subdir is active; None = default
    base_prompt: str | None     # immutable for the run — the original system prompt
    system_prompt: str | None   # CURRENT prompt; sticky; hook-editable (see below)
    allowed_tools: list | None  # strict tool whitelist; sticky; hook-set (see below)
    base_model: LLMConfig | None  # immutable; the LLM the Agent was built with (restore target)
    current_model: LLMConfig | None  # sticky; the LLM the core actually calls (hook-owned)
    turn: int                   # core-managed, 1-based, reset by Agent.reset()
    usage: dict                 # cumulative from the LLM client across the run
    extras: dict                # HOOKS ONLY — flat keys, run-scoped

    # per-request (rebuilt by the core before each before_llm)
    tool_loop_round: int        # core (0-based)
    messages: list[dict]        # exact list the LLM will see THIS call
    tools: list[dict]           # tool schemas offered THIS call

    # per-call / per-turn working state
    tool_call: dict | None      # per-call: {"name","arguments"} or None (veto)
    tool_result: Any            # per-call: the tool's return value (or error string)
    final_answer: str | None    # per-turn: the final text (prepare/finish may rewrite)
```

#### `extras`: the walls

`extras` is the sanctioned space for inter-hook coordination and for any
experiment's own run state. **The core never reads or writes it.** It is
run-scoped — it (and any hook's write to the other run-scoped fields) survives
across turns and across `Agent.reset()`. Use flat, namespaced-style keys (e.g.
`ctx.extras["budget.calls"]`) so multiple experiments can share one context
without a formal namespace today.

```python
def count_calls(ctx: RunContext):
    ctx.extras["llm_calls"] = ctx.extras.get("llm_calls", 0) + 1


def budget_guard(ctx: RunContext):
    assert ctx.extras.get("llm_calls", 0) <= 10, "budget exceeded"
```

#### System prompt (dynamic, sticky)

The core sends a single `role: "system"` message as `messages[0]`; the text
comes from **`ctx.system_prompt`**, and it is **sticky**:

1. At construction, `Agent` seeds both `ctx.base_prompt` (immutable) and
   `ctx.system_prompt` from its `system_prompt` argument (default
   `DEFAULT_SYSTEM_PROMPT`).
2. Before each `before_llm`, the core has **already** put the current
   `system_prompt` at `messages[0]` and rebuilt `ctx.messages`. The hook may
   leave it alone or set `ctx.system_prompt` to something new — that new value
   is applied to `messages[0]` *after* the hook and is carried to the **next**
   request too (it's sticky; you don't re-assert it).
3. The original is always available at `ctx.base_prompt` — use it to extend,
   patch, or restore.

Four verbs, one channel:

```python
def extend(ctx: RunContext) -> None:     # base + addition
    ctx.system_prompt = (ctx.base_prompt or "") + "\nBe concise."


def prepend(ctx: RunContext) -> None:    # addition + base
    ctx.system_prompt = "You are a Growthcho agent." + (ctx.base_prompt or "")


def patch(ctx: RunContext) -> None:      # replace a slice of the base
    ctx.system_prompt = (ctx.base_prompt or "").replace(
        "e.g. arithmetic", "e.g. arithmetic, date math")


def replace(ctx: RunContext) -> None:    # the base is gone
    ctx.system_prompt = "You are terse. No filler."


def restore(ctx: RunContext) -> None:    # back to the original
    ctx.system_prompt = ctx.base_prompt
```

**Pattern to follow — recompute from `base_prompt`, not "append to current."**
"Append to current" accumulates across turns (`"!"` becomes `"!!"`) unless the
hook cleans up; recomputing from the base is stable and self-healing:

```python
# correct (stable):
def before_llm(ctx):
    # avoids drift on every turn, and can vary with ctx.session_id / ctx.turn / extras
    ctx.system_prompt = (ctx.base_prompt or "") + "\n[turn " + str(ctx.turn) + "]"
```

`experiments/system_prompt.py` is a working example (dynamic addendum + a
`ready_session` hook that seeds per-session state in `extras`).

**Who may write `ctx.system_prompt` — the same contract as
`ctx.base_prompt` → `ctx.system_prompt`.** Any hook can write the field (it
is a dataclass field, not a signature), but the core applies it *after*
`before_llm` on every LLM call — so:

- **`before_llm`** is the canonical *round* writer: the value you set there is
  the one that goes out on this request, and (because it's sticky) it carries
  to the next round until someone overwrites it.
- **`enter_agent`** is the canonical *switch-time* writer for an agent that
  specialises its prompt (the subagent pattern, mirroring the
  [Model selection](#model-selection-dynamic-sticky) example).
- **Any other slot** is legal but discouraged: the change survives turns, so
  a `finish` hook that sets the prompt is effectively a per-turn writer.

Later writers win. The "recompute from `ctx.base_prompt`" pattern (above) is the
stable, self-healing way to do per-turn work without drift.

#### Model selection (dynamic, sticky)

The core calls exactly one LLM per request: **`ctx.current_model`** — the
single source of truth, mirroring `ctx.system_prompt`. Its immutable reference,
`ctx.base_model`, is the config the `Agent` was built with (usually the
environment's). Both follow the base/current pattern:

```python
def restore_current_to_base(ctx: RunContext) -> None:
    ctx.current_model = ctx.base_model  # the canonical "back to default"
```

**Who may write `ctx.current_model` — a strict order of priority** (later
overrides earlier):

1. Environment — seeds `base_model` at construction only. No runtime authority.
2. The active agent's `enter_agent` — its specialization; it may rebind the
   model (see the subagent example below). Sticky until overwritten.
3. `Agent.set_model(cfg)` — the user channel (REPL/API). Commits the value,
   fires the active agent's `on_model_change`, which may mutate (substitute,
   restore to `base_model`) or **raise to veto** (rollback; the exception is
   re-raised to the caller). See `Agent.set_model` and `on_model_change` for
   the full contract.
4. `before_llm` — per round, unconditional. The hook always wins at request
   time; this is the "hooks never lose control" guarantee.

**Stickiness & agent switches.** `current_model` survives `set_agent_name` and
`reset()`. Unlike `allowed_tools`, the core **never resets it on a switch** —
the incoming agent's `enter_agent` decides if it wants to rebind.

```python
# agent specialization: a "math" agent that insists on a strong model
def enter_agent(ctx: RunContext) -> None:
    ctx.allowed_tools = ["calculator"]
    ctx.current_model = LLMConfig(base_url="http://...", model="qwen3:32b")
```

**`Agent.test_model(cfg)`** runs one minimal ping completion (`max_tokens=1`)
with a throwaway client and returns an `LLMProbe(ok, error, latency_ms,
usage)`. It is a diagnostic: a probe failure is a *finding*, not a *change*.
Use `set_model` for a gated change.

**HTTP/REPL surface.** `POST /config` (session-scoped or global),
`POST /config/test`, and the REPL's `/model` command (see `protocol.md` for
shapes).

### Hook modules and the two folders

Two top-level folders, both outside the `agent_seed` package, both loaded as
standalone files (never imported *by* the core):

| folder | meaning | loaded |
|--------|---------|--------|
| `experiments/`    | ideas, candidates, prototypes | **never** — only on explicit import |
| `hook_modules/`   | promoted hooks, known-good    | **automatically**, once per process |

`agent_seed/hooks.py` is the third, distinct thing: the stable *protocol*
(`RunContext` + the fixed slots) that both folders plug into.

Subdirectories of `hook_modules/` are *agents* — alternate hook sets selected
by `agent_name`. See ["Agents: hook subdirectories"](#agents-hook-subdirectories).

### Promoting experiments

The intended workflow:

1. Prototype in `experiments/` — self-contained files you test with explicit
   `hooks=...` and delete at will.
2. When one earns its keep, promote it: `cp experiments/foo.py hook_modules/`.
3. Done. Every `Agent` (direct, backend, REPL) picks it up — no registration
   anywhere.

#### The contract: `make()`

A hook module is a standalone `.py` file defining one factory:

```python
from agent_seed.hooks import Hooks, RunContext


def finish(ctx: RunContext) -> None:
    if ctx.final_answer is not None:
        ctx.final_answer = f"{ctx.final_answer} (v0.1)"


def make() -> Hooks:
    return Hooks(finish=finish)
```

Rules:

- `make` is the **single canonical factory name** — that is what makes
  promotion a plain copy. Parameterized hooks read their config from the
  environment inside `make()`, exactly like the backend does.
- Modules may import `agent_seed.*` and anything installed; they are executed
  with `importlib` under unique module names (no `sys.path` magic), so they
  stay standalone.
- `make()` runs exactly **once per process per folder** (the result is
  cached); first use does the scan + import + merge, later uses are a dict
  lookup.
- Failing to import, lacking `make`, or a `make()` that raises / returns
  something other than `Hooks` → `DiscoveryError` at first use. **Fail fast,
  all-or-nothing** — the `hook_modules/` folder is your production channel, so
  a broken file is a startup error, not a silent skip.

#### Merge policies

- Modules merge **alphabetically by filename**; on a slot conflict the file
  that sorts **last** wins (`b_audit.py` overrides `a_audit.py`).
- A slot no module provides stays `None`.
- `extras` is unaffected by merging (hooks coordinate through `ctx`, not
  through module state).

#### Overriding discovery

Passing an explicit `Hooks(...)` anywhere always beats discovery:

```python
Agent(client, registry, hooks=Hooks())            # empty hooks, no folder scan
create_app(registry, hooks=Hooks(finish=mine))    # this app: only mine
```

Library consumers: the default folder is `hook_modules/` relative to the
current working directory. In your own test suites, point
`agent_seed.discovery.DEFAULT_HOOKS_FOLDER` at an empty directory (see the
autouse fixture in this repo's `tests/conftest.py`) so a real
`hook_modules/` can never feed your tests.

### Agents: hook subdirectories

**There is always an agent in place.** The nameless one — the `hook_modules/`
root, `agent_name=None` — is the **main agent**, not a special case. You can
define additional, distinct agents by giving each its own subdirectory:

```
hook_modules/                 the main agent (also the fallback)
hook_modules/alpha/           alpha's own hooks (used instead of the root)
hook_modules/gamma/
```

The rules:

* **Replace, don't merge.** If `agent_name` names a subdirectory
  (`hook_modules/<agent_name>/`) that exists, *only* that directory is
  scanned. The root's `*.py` do **not** apply to that agent. The root folder
  is the **main agent's** hooks and the fallback for any name that has no
  subdirectory.
* **Fallback.** An `agent_name` with no matching subdirectory (or `None`)
  uses the root `hook_modules/`.
* **Traversal is rejected.** Names that would escape the root (`..`,
  absolute paths) fall back to the root.
* **Root location.** The search starts at `hook_modules/` relative to the
  CWD, or at `$HOOKS_ROOT` if that environment variable is set.

Select an agent at any of these layers:

```python
agent = Agent(client, registry, agent_name="alpha")   # at construction
agent.set_agent_name("gamma")                          # later (re-fires establishment)
agent.agent_name                                        # read it back

# backend:  POST /chat  { "message": "...", "agent_name": "alpha" }
# frontend: GET /agents  ->  { "agents": ["alpha", "gamma"] }
```

#### Your first agent (end to end)

An **agent** is a subdirectory of the hooks root holding ordinary hook modules
(a `def make() -> Hooks` per file, see ["The contract:
`make()`"](#the-contract-make)). It *replaces* the root's hooks for that agent
(the root stays the fallback). To add one: make the folder, drop a file, done.

```
hook_modules/             the main agent (unchanged; still the fallback)
hook_modules/alpha/
    sign.py               alpha's own hook(s)
```

```python
# hook_modules/alpha/sign.py
from agent_seed.hooks import Hooks, RunContext


def finish(ctx: RunContext) -> None:
    if ctx.final_answer is not None:
        ctx.final_answer = f"{ctx.final_answer}\n— alpha"


def make() -> Hooks:
    return Hooks(finish=finish)
```

A folder may hold several `make()` files; they merge alphabetically (last slot
wins), exactly like the root. `GET /agents` (or `/agents` in the REPL) lists
the subdirectories of the hooks root — `{"agents": ["alpha"]}` here; an empty
list means only the main agent exists. Now pick `alpha` (above); its hooks swap
in for the root's and its establishment hooks re-fire on the next `chat()`
(further below). To remove the agent, delete `hook_modules/alpha/`. No
registration anywhere.

#### Per-agent tool access (strict whitelist)

Tooling is **opt-in per agent**. The default `ctx.allowed_tools` is `None`,
which means **no tools** — the core offers no tool schemas to the LLM and
rejects every tool call with a well-formed `role: tool` error message so the
LLM can adapt. An agent that *wants* tools must declare them.

**The channel is `ctx.allowed_tools`.** A run-scoped, sticky list of tool
names the agent's hooks set (usually in `enter_agent`). The core reads it at
two points:

1. **Offer** — before each LLM call, `ctx.tools` is rebuilt as the intersection
   of the registry and `allowed_tools` (via `ToolRegistry.schemas_for`). An
   empty/None set means the LLM is offered *zero* tools.
2. **Execute** — if the LLM returns a `tool_calls` entry whose name is
   *outside* the whitelist, the core records a `role: tool` message
   `"error: tool '<name>' not available to this agent"` instead of executing.
   The hook-level `before_tool_call` veto gate does **not** fire in that
   branch — the core settles tool availability first, before any hook gets a
   chance to (wrongly) allow a disallowed name.

**Sticky semantics.** Like `system_prompt`, a change to `ctx.allowed_tools`
persists until overwritten. `enter_agent` is the canonical place to set it
(the right hook: it fires on first activation and on every switch *into* that
agent, and the core resets it to `None` on each switch *before* the incoming
agent's hooks fire — so a switched-to agent that declares nothing genuinely
sees no tools, and a switched-from agent's whitelisted set does not leak
across the shared `RunContext`). A hook that sets it in `before_llm` also
sticks from that round on (it governs the execute-path veto for the current
round). To narrow *this round only*, edit `ctx.tools` directly in
`before_llm` (the documented escape hatch, e.g. the existing
`ctx.tools.clear()` pattern) — that edit applies to the current offer but
does not carry to the next round.

**Declarative example** (`hook_modules/<agent>/tools.py`):

```python
from agent_seed.hooks import Hooks, RunContext

ALLOWED_TOOLS: list[str] = ["calculator", "weather"]


def enter_agent(ctx: RunContext) -> None:
    """Declare this agent's tools when it takes over."""
    ctx.allowed_tools = list(ALLOWED_TOOLS)


def make() -> Hooks:
    return Hooks(enter_agent=enter_agent)
```

- **Main agent, strict policy.** The nameless main agent (the `hook_modules/`
  root, `agent_name=None`) also has to declare its tools — the repo ships
  `hook_modules/00_tools.py` to allow `calculator` for it. Delete the file to
  remove tool access from the main agent.
- **A folder with no tool declaration = an agent with no tools.** There is no
  implicit inheritance or default allow set; an agent that wants nothing
  simply does not declare anything.
- **Strict whitelist, no "base set".** There is no "always-on" tool that
  bypasses the per-agent whitelist. If two agents both want `calculator`,
  both declare it (or reuse a shared hook module).
- **Slot-merge gotcha.** `enter_agent` is a hook slot; if two files in the
  same folder both define it, the alphabetically-last wins. Compose your
  tool declaration into the surviving hook.
- **Inspect with `GET /tools?agent_name=<n>`.** The backend runs the agent's
  `enter_agent` against a scratch `RunContext` and reports the resulting
  `allowed_tools` (plus matching tool schemas). Strict policy: a folder with
  no declaration returns `{"tools": [], "names": []}`.

**Worked example — an agent with no tools.** A folder with a hook module that
does not touch `allowed_tools`:

```
hook_modules/
    00_tools.py             # main agent: calculator
alpha/
    sign.py                 # alpha declares no tools
```

`alpha`'s `sign.py` is a plain hook (say, a `finish`). On a switch into
`alpha`, the core resets `allowed_tools` to `None` and `enter_agent` fires —
the root's `enter_agent` does not (alpha replaces the root), so alpha's tools
remain `None` and every tool call in the loop is rejected with
`"error: tool ... not available to this agent"`. That is the strict-policy
outcome for an undeclared agent.

#### The agent-identity boundary (enter / exit)

Because there is always an agent on the wheel, becoming active and leaving are
each an **event**. Two boundary hooks bracket every transition — `enter_agent`
(in the incoming agent's hooks) and `exit_agent` (in the outgoing agent's
hooks):

```
construction (first activation)         Agent 1 -> Agent 2              (repeatable)
┌─ enter_agent(agent 1)                 ┌─ exit_agent(agent 1)   # outgoing
│   (agent_name = 1; no prior agent)    ├─ [switch committed: agent_name -> 2]
│                                       ├─ enter_agent(agent 2)   # incoming
│      (then, on first chat:)           │      (then, on next chat:)
│   initialize_run · ready_run          │   initialize_run · ready_run
│   initialize_session · ready_session  │   initialize_session · ready_session
```

Rules:

* **Always an agent.** First activation fires `enter_agent` only — even for
  the main (nameless) agent, since there was no prior agent to exit. On
  *every* switch the order is `exit_agent` (old) → `enter_agent` (new), and
  then the active agent's establishment hooks re-fire on the next `chat()`.
* **RAII.** Every agent that takes over after the first one gets exactly one
  exit when it's displaced. A switch never leaves an agent "half-on."
* **Atomic.** The incoming agent's hooks are resolved *before* any state is
  touched, so a broken `hook_modules/<new>/` fails fast and the switch is
  aborted with **nothing** changed (no `exit`, no `enter`, name unchanged).
* **Runs as different agents.** `exit_agent` sees `ctx.agent_name` = the
  *old* value (outgoing); `enter_agent` sees it = the *new* value (incoming).
* **Scope.** They fire from `Agent.__init__` / `Agent.set_agent_name`, not
  inside `chat()`. History, `run_id`, `session_id`, and `extras` are not
  touched by the boundary itself.

This is where **clean-up / persistence** lives: `exit_agent` persists or
releases what the agent you're leaving needs, `enter_agent` initializes what
the agent you're taking over needs. Selecting an agent (via
`set_agent_name(...)`, `POST /chat`'s `agent_name`, or `/agent`) re-resolves
`Agent.hooks` (an explicit `Hooks(...)` from construction always wins) and
re-fires establishment on the next `chat()`, mirroring `set_session`.

### Which hook to use, by goal

| goal                                          | hook                | how |
|-----------------------------------------------|---------------------|-----|
| clean up / persist state when an agent is displaced | `exit_agent`    | runs as the outgoing agent; `ctx.agent_name` = old value (RAII pair with `enter_agent`) |
| initialize state when an agent takes over     | `enter_agent`       | runs as the incoming agent; `ctx.agent_name` = new value; fires on construction too |
| declare this agent's tools (strict whitelist) | `enter_agent`       | set `ctx.allowed_tools` (sticky; the core offers only these and vetoes the rest) |
| specialize this agent's model                 | `enter_agent`       | set `ctx.current_model` (sticky; survives switches unless a hook changes it) |
| restore the default model on takeover         | `enter_agent` / `before_llm` | `ctx.current_model = ctx.base_model` |
| gate / veto a model change (set_model)        | `on_model_change`   | mutate `ctx.current_model` or raise to veto (rolled back) |
| override / extend / patch the system prompt   | `before_llm`        | read `ctx.base_prompt` + set `ctx.system_prompt` (sticky) |
| inject RAG / context / date into the next LLM call only | `before_llm` | append to `ctx.messages` (per-request — history is not touched) |
| disable tools for one call                      | `before_llm`        | `ctx.tools.clear()` |
| supply a canonical `run_id`                    | `initialize_run`    | set `ctx.run_id` (fires once) |
| attach post-establishment run state (memory, budget) | `ready_run`    | read `ctx.run_id`, load into `ctx.extras` (fires once) |
| seed / migrate per-session state               | `initialize_session` | set `ctx.session_id` (fires on session change) |
| load per-session external context              | `ready_session`     | read `ctx.session_id` / detect change via `ctx.extras`, load into `ctx.extras` |
| audit or transform a tool call                 | `before_tool_call`  | read/rewrite `ctx.tool_call` |
| deny a tool call                               | `before_tool_call`  | `ctx.tool_call = None` |
| redact / rewrite a tool result                 | `after_tool_call`   | `ctx.tool_result = ...` |
| observability / tracing                        | `on_round`          | fire and forget (or log into `ctx.extras`) |
| validate / normalize the final answer          | `prepare`           | rewrite `ctx.final_answer` |
| transform the final answer                     | `finish`            | rewrite `ctx.final_answer` (e.g. a canned signature) |

### Worked examples

#### "ahem" — the last word (the `finish` hook)

See `experiments/ahem.py`:

```python
from agent_seed.hooks import Hooks, RunContext


def finish(ctx: RunContext) -> None:
    if ctx.final_answer is not None:
        ctx.final_answer = f"{ctx.final_answer}\n(ahem)"


def make() -> Hooks:
    return Hooks(finish=finish)
```

```python
from agent_seed.core import Agent
from experiments.ahem import make

agent = Agent(client, registry, hooks=make())
print(agent.chat("what is 6*7?"))
# 42
# (ahem)
```

#### Dynamic system prompt

See `experiments/system_prompt.py` — recomputes a `turn`-aware addendum from
`ctx.base_prompt` every request, and uses `ready_session` to seed per-session
state in `extras`.

#### Policy veto — deny a tool by name

```python
def veto_calc(ctx: RunContext) -> None:
    if ctx.tool_call and ctx.tool_call["name"] == "calculator":
        ctx.tool_call = None   # core records "skipped by policy" in history


hooks = Hooks(before_tool_call=veto_calc)
```

#### One-shot context injection (round 0 only)

```python
def inject_once(ctx: RunContext) -> None:
    if ctx.tool_loop_round == 0:
        ctx.messages.append({"role": "user", "content": "(project: agent-seed)"})
```

Per-request: `ctx.messages` is the exact list the LLM sees on this call, not
the agent's stored history.

#### Budget meter via `extras`

```python
def budget(ctx: RunContext, limit: int = 10) -> None:
    used = ctx.extras.get("calls", 0)
    ctx.extras["calls"] = used + 1
    if used >= limit:
        raise RuntimeError("LLM budget exceeded")


hooks = Hooks(before_llm=budget)
```

### Testing hooks

Tests in this repo must stay network-free. Use `tests/conftest.py::FakeLLMClient`.
Pattern:

```python
from agent_seed.core import Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.tools import ToolRegistry
from .conftest import FakeLLMClient, text_completion


client = FakeLLMClient([text_completion("hi")])
agent = Agent(
    client,
    ToolRegistry(),
    hooks=Hooks(finish=lambda ctx: ctx.__setattr__("final_answer", "bye")),
)
assert agent.chat("q") == "bye"
```

Assert on `client.requests[i]["messages"]` for what the LLM saw on call `i`,
`agent.history` for what was stored, `agent.context` for run-scoped state (ids,
prompt, `extras`), and the hook's captured lists for firing order.

### Rules (the stability contract)

Breaking any of these is a **major** change.

1. **Hook signatures are frozen.** All hooks are `(ctx: RunContext) -> None`.
   New behavior goes on `RunContext`, never on the signature.
2. **`extras` is a wall.** The core never reads or writes it. Hooks own it.
3. **`RunContext` core fields are append-only with defaults.** Adding a field
   (with a `None`/empty default) is a minor bump — existing hooks still work.
   Removing or renaming a field is a major break.
4. **Hook slots are append-only.** Adding a slot (`None`-defaulted) is a minor
   bump. Removing a slot is a major break.
5. **Run-scoped fields are sticky; per-request fields are rebuilt.**
   `system_prompt` persists across turns until changed; `messages`, `tools`,
   `tool_loop_round` are rebuilt before each `before_llm` — do not rely on
   stale per-request values.
6. **The canonical system-prompt channel is `ctx.system_prompt`.** The core
   applies it to `messages[0]` after `before_llm`. Directly editing `messages[0]`
   is also legal but will be overwritten by the core — prefer the field.
7. **Veto is explicit.** `ctx.tool_call = None` is the only way to stop a tool
   call. The core records a well-formed `role: tool` `"skipped by policy"`
   result so the next API call stays valid.
8. **No re-entry.** A hook must not call `agent.chat()` from inside itself.
9. **Hooks are synchronous.** The agent loop runs in a worker thread; async
   hooks are out of scope.
10. **Establishment fires on need.** Run hooks strictly once; session hooks on
     each (re)establishment. Do not assume establishment fires every `chat()`.
11. **The agent boundary is RAII and atomic.** There is always an agent in
    place (the main one when `agent_name` is `None`). `enter_agent` fires on
    first activation and on every incoming switch; `exit_agent` on every
    outgoing switch, never at construction. A switch resolves the new hooks
    first, so a broken target aborts cleanly — nothing is half-committed and
    `exit_agent` is not fired for a switch that didn't happen.
12. **Tool policy is opt-in per agent and strict.** The default
    `ctx.allowed_tools` is `None`/`[]` (*no tools*): the core offers no tool
    schemas and rejects every tool call with a well-formed `role: tool`
    `"error: tool '<name>' not available to this agent"`. An agent declares
    its tools by setting `ctx.allowed_tools` (canonically in `enter_agent`);
    that set is sticky and governs both the **offer** (schemas) and the
    **execute** (veto) paths. The core resets the field to `None` on each
    agent switch, so a switched-to agent that declares nothing sees no tools
    — there is no inheritance or "always-on" base set. A hook may *narrow* a
    call (`before_tool_call`) or change this round's offer (`ctx.tools`), but
    it can never *re-allow* a tool the whitelist already rejected.
13. **Model change is a gated, atomic commit.** The run's LLM is
    `ctx.current_model` (a sticky run-scoped field, seeded from
    `ctx.base_model` at construction). The only sanctioned way for a
    *user-facing* change to be committed is `Agent.set_model(candidate)`,
    which sets the value **then** fires `on_model_change` (the hook may
    substitute or `raise` to veto — `raise` rolls the value back and
    re-raises, mirroring the "atomic" boundary hook). A hook's own
    `ctx.current_model = ctx.base_model` in `enter_agent` / `before_llm` is
    a *specialisation / restore* pattern, not a user-facing commit — it
    bypasses `on_model_change` by construction (it is a per-agent binding,
    not a mid-turn switch). The core **never** resets `current_model` on a
    switch or `reset()`; only the incoming agent's hooks may.

### Extending the core with new slots

If none of the current slots cover you:

1. Add a new slot to `Hooks` (default `None`).
2. Add the `if hooks.<new> is not None: hooks.<new>(ctx)` guard at the right
   point in the core.
3. Document the slot in `docs/manual.md` and add a test in
   `tests/test_hooks.py`.

That's it. No core refactor, no signature change, and every existing hook keeps
working unchanged.

---

## FAQ / troubleshooting

**Q: My hook module in `hook_modules/` isn't being loaded.**
A: Check these in order:

1. The file defines `def make() -> Hooks` — that exact name and return type is
   required.
2. The file is a single `.py` at the top level of `hook_modules/`, not
   nested in a subfolder.
3. You're running from the directory that contains `hook_modules/` — the
   default search is relative to the current working directory.
4. You didn't pass an explicit `Hooks(...)` to `Agent` or `create_app` — an
   explicit value fully replaces discovery.

**Q: I put a hook in `hook_modules/agent/foo.py` — is it loaded?**
A: Not by default. Subdirectories of `hook_modules/` are *agents*: they are
loaded only when that agent is selected (i.e. `agent_name` = the directory
name) — the default agent is the `hook_modules/` root, so the subdirectory is
inert until you ask for it. See
["Agents: hook subdirectories"](#agents-hook-subdirectories).

**Q: I wrote two hooks for the same slot. Which one runs?**
A: Exactly one. Discovered modules merge alphabetically and the file that
sorts last wins the slot. If you wrote two hooks in *one* module, you must
combine them into a single function and pass that single function to the slot.

**Q: `ctx.messages` has one item one turn and three the next. What's going on?**
A: `ctx.messages` is the *per-request* list the LLM will see on this call only.
The core rebuilds it before every round from the stored history. Don't rely on
its previous-turn shape; read `agent.history` directly if you need the full
transcript.

**Q: My one turn should use a custom system prompt, and the rest of the
conversation the normal one. How do I make that one-shot?**
A: `ctx.system_prompt` is sticky, so instead set it per request in `before_llm`
and clear it just as often — or, more cleanly, recompute it each turn from the
canonical channel (see ["System prompt (dynamic, sticky)"](#system-prompt-dynamic-sticky)):

```python
def before_llm(ctx: RunContext) -> None:
    base = ctx.base_prompt or ""
    if ctx.turn == 3:                       # only turn 3
        ctx.system_prompt = base + "\nThis turn: be extra careful."
    else:
        ctx.system_prompt = base            # sticky value restored
```

Avoid directly editing `ctx.messages[0]` for this: the core applies
`ctx.system_prompt` to `messages[0]` after `before_llm`, so it would clobber
the direct edit.

**Q: I added a new tool but the LLM never calls it.**
A: In this order:

1. **Registered?** It must be in the `ToolRegistry` the `Agent` was created
   with. A tool registered only in `experiments/` is invisible unless you pass
   it into a `ToolRegistry` and give that registry to the `Agent` / `create_app`.
2. **Whitelisted for the active agent?** Under the strict tool-whitelist
   policy an agent must *declare* the tool by adding its name to
   `ctx.allowed_tools` (in `enter_agent`). If the current agent's hook doesn't
   list it, the core offers only the listed schemas (or none) — the LLM never
   sees the new tool. Add it to that agent's whitelist: `ctx.allowed_tools =
   ["calculator", "my_tool"]`.
3. **Described?** A clear first-line docstring drives the LLM's call decision.

**Q: My agent's tool call returns `"error: tool '<name>' not available to this
agent"`. What is going on?**
A: The active agent's `ctx.allowed_tools` does not include `<name>`. The core
rejected the call at the execute-path veto gate before `before_tool_call`
fired. Fix it by adding the name to the agent's whitelist in `enter_agent`
(e.g. `ctx.allowed_tools = ["calculator", "<name>"]`). If the agent is a
switched-in one and you expected it to inherit the previous agent's tools: it
does not — tool policy is opt-in per agent. Declare what you want.

**Q: How do I give agent X different tools than agent Y?**
A: Put each in its own folder and set `ctx.allowed_tools` in each `enter_agent`:

```python
# hook_modules/x/tools.py
from agent_seed.hooks import Hooks, RunContext

def enter_agent(ctx: RunContext) -> None:
    ctx.allowed_tools = ["calculator", "weather"]

def make() -> Hooks:
    return Hooks(enter_agent=enter_agent)

# hook_modules/y/tools.py
def enter_agent(ctx: RunContext) -> None:
    ctx.allowed_tools = ["search"]

def make() -> Hooks:
    return Hooks(enter_agent=enter_agent)
```

The core filters both the offered schemas and the allowed executions to exactly
the agent's set — no hook-level work needed. A switched-in agent that declares
nothing sees *no* tools (strict default). See
["Per-agent tool access (strict whitelist)"](#per-agent-tool-access-strict-whitelist).

**Q: How do I get per-session state without a database?**
A: Use `ctx.extras` in `ready_session`:

```python
def ready_session(ctx: RunContext) -> None:
    key = f"budget.{ctx.session_id}"
    if key not in ctx.extras:
        ctx.extras[key] = {"calls": 0}
```

Because `extras` is run-scoped and sticky, this survives across turns in the
same session. For multi-process backends this is per-process, not shared — the
design boundary is documented in `docs/architecture.md`.

**Q: How do I make `LLM_MODEL` optional in my own deployment?**
A: `LLMConfig.from_env` raises if it's unset; the backend surfaces that as HTTP
500. Provide it in the process environment (or `.env` if you load it manually)
before the server starts.

**Q: Can I run two frontends against one backend?**
A: Yes. Each frontend mints or supplies a `session_id`; the backend keys the
`session_id → Agent` map on it. There's no shared state between sessions beyond
the in-memory map.

**Q: Can I switch the model at runtime — mid-session, per-session, or for sub-agents?**
A: Yes. The model the core actually calls is `ctx.current_model` (a sticky
run-scoped field); its immutable reference is `ctx.base_model`. Three channels
write it, in ascending priority: the environment (boot seed only), the active
agent's `enter_agent` (specialization), and `Agent.set_model(cfg)` (user / API,
gated by the `on_model_change` hook, which may mutate it or raise to veto).
`before_llm` still wins per round, so hooks retain final control. Switches are
session-scoped: `current_model` is sticky across `set_agent_name` and
`reset()` — the core never resets it, an agent that wants the default re-runs
`ctx.current_model = ctx.base_model` itself. Use `Agent.test_model(cfg)` to
probe a candidate without committing (it is a diagnostic, not a change).
See "Model selection (dynamic, sticky)" and `protocol.md` (`POST /config`).

---

## Glossary

- **agent** — an active hook set. There is always an agent on the wheel: the
  **main agent** is the nameless `hook_modules/` root (`agent_name=None`); a
  named agent is a subdirectory `hook_modules/<agent_name>/` selected via
  `agent_name`. Selecting an agent brackets the boundary hooks and re-fires
  establishment.
- **agent loop** — the `Agent.chat()` cycle: send the message + tool schemas to
  the LLM, run any tool calls the LLM returned, and repeat until it returns
  plain text.
- **base prompt** — `ctx.base_prompt`: the immutable original system prompt for
  the run. Always available to hooks; use it as the source when computing a
  new `ctx.system_prompt`.
- **base model** — `ctx.base_model`: the immutable `LLMConfig` the `Agent` was
  built with. The canonical restore target
  (`ctx.current_model = ctx.base_model`).
- **current model** — `ctx.current_model`: the sticky `LLMConfig` the core
  actually calls; the single source of truth per request. Written by the
  active agent's `enter_agent`, `Agent.set_model` (gated by
  `on_model_change`), and `before_llm` (per round, final word). Survives
  agent switches and `reset()` — the core never resets it.
- **allowed_tools** — `ctx.allowed_tools`: the sticky strict whitelist of tool
  names the active agent may offer and call. Set by the agent's `enter_agent`
  hook (or any other hook; sticky until overwritten). The core enforces it at
  both the offer point (tool schemas) and the execute point (a non-allowed name
  is rejected before `before_tool_call` runs). `None`/`[]` is *no tools* —
  tooling is opt-in per agent. The core resets the field to `None` on each
  agent switch, so a switched-to agent that declares nothing sees no tools.
- **discovered hook** — a hook loaded automatically from `hook_modules/` via
  `make()` at first use.
- **establishment** — the one-time (per run) or per-session (per session)
  sequence of hooks that set up `run_id`, `session_id`, and any derived state.
  Fires on need, not every turn.
- **enter_agent / exit_agent** — the agent-identity boundary hooks. `enter_agent`
  fires when an agent becomes active (first activation + each incoming switch;
  runs as the incoming agent); `exit_agent` when an agent is displaced (each
  outgoing switch, never at construction; runs as the outgoing agent). A switch
  is atomic: `exit` → commit → `enter`.
- **main agent** — the nameless agent backed by the `hook_modules/` root
  (`agent_name=None`). The default and the agent on the wheel until one is
  selected; "no agent" is not a state.
- **on_model_change** — the model-selection boundary hook (Agent-lifetime, like
  `enter_agent` / `exit_agent`): fires from `Agent.set_model` *after* the new
  value is committed to `ctx.current_model`. May mutate it (substitute a model,
  restore `ctx.base_model`) or **raise to veto** — which rolls the change back
  and re-raises. A per-round override is `before_llm`'s job; this slot only
  guards the *commit*.
- **extras** — `ctx.extras`: a flat dict owned by hooks. The core never reads
  or writes it. Survives across turns and `Agent.reset()`.
- **hook** — a `(ctx: RunContext) -> None` function attached to a fixed slot in
  the agent loop's lifecycle.
- **hook module** — a standalone `.py` file with a single `def make() -> Hooks`
  factory, auto-discovered from `hook_modules/`.
- **per-request fields** — the `RunContext` fields (`messages`, `tools`,
  `tool_loop_round`) that the core rebuilds before each `before_llm`.
- **promote** — copy an `experiments/` file into `hook_modules/` so it is
  auto-loaded in every process.
- **ready_run / ready_session** — hooks that fire after the core has filled in
  `run_id` / `session_id` (i.e. the id is guaranteed non-None); the hook may
  seed derived state into `ctx.extras`.
- **run** — one instantiation of `Agent`. Synchronous, single-user, owns one
  `RunContext` for its whole life.
- **run-scoped fields** — the `RunContext` fields (`run_id`, `session_id`,
  `agent_name`, `base_prompt`, `system_prompt`, `turn`, `usage`, `extras`)
  that persist across turns.
- **session** — the active conversation within a run. A run can switch
  sessions; it never runs two at once.
- **sticky** — a run-scoped field's value persists to the next turn until
  explicitly changed.
- **system prompt** — the `role: "system"` message at `messages[0]`. The core
  pulls it from `ctx.system_prompt` and applies it *after* `before_llm`.
- **tool loop round** — one LLM→tools iteration within a turn (0-based). Capped
  by `MAX_TOOL_ROUNDS` (default 8).
- **turn** — one user message and the agent's processing of it (1-based counter).
- **veto** — a hook's way of stopping a tool call: set `ctx.tool_call = None`
  in `before_tool_call`. The core records a `"skipped by policy"` `role: tool`
  message so the next API call stays valid.
