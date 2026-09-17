# Hooks — user manual

How to write and attach hooks: the extension surface of the agent core.

## What a hook is

A **hook** is a plain Python function you attach to a fixed point in the
agent loop's lifecycle. Every hook has exactly one signature, frozen
forever:

```python
def my_hook(ctx: RunContext) -> None: ...
```

Hooks run **in place and synchronously** inside `Agent.chat()`, in a fixed
order. They may inspect or mutate any field of `RunContext` (including
the open `extras` namespace), may veto a tool call, and may change the
system prompt. They do one thing: participate in the run.

What a hook is **not**:

- **not a middleware chain** — one slot holds at most one function. If you
  need two, compose them inside a single function; there is no priority
  list in a slot.
- **not an event bus** — hooks are not async, not decoupled from the turn;
  they run in the turn's call stack and their effects matter.
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

There is a second way that avoids this line entirely: put a **hook module**
in the `hook_modules/` folder and it is auto-discovered (see
"Promoting experiments" below).

## The four levels

A **run** is one instantiation of `Agent` (one conversation driver). It is
synchronous, serves a single user, and owns **one** `RunContext` for its
whole life. A run drives **one active session at a time**; it may *switch*
sessions (see `Agent.set_session`), but never runs two at once. `run_id`
exists for debugging, logging, and future parallel runs — today it is
1:1 with the `Agent`.

| level | field | meaning | set by |
|---|---|---|---|
| **run** | `run_id` | one `Agent` lifetime | caller; else core auto-gens at establishment |
| **session** | `session_id` | the conversation currently being driven | caller; else core auto-gens at establishment |
| **turn** | `turn` | one user message (1-based run counter; reset by `Agent.reset()`) | core |
| **tool_loop_round** | `tool_loop_round` | one LLM→tools iteration within a turn (0-based) | core |

## The two lifetimes inside a `RunContext`

This is the most important part of the model. A `RunContext` has **two
kinds of fields**, and the core treats them differently:

| | **run-scoped** (stable across turns) | **per-request** (rebuilt each LLM call) |
|---|---|---|
| fields | `run_id`, `session_id`, `base_prompt`, `system_prompt`, `turn`, `usage`, `extras` | `messages`, `tools`, `tool_loop_round` |
| core behavior | seeds once at establishment / keeps the hook's writes | rebuilds at the start of every round |
| hook behavior | may read; may change once → the change **persists** ("sticky") until changed again | may read / rewrite *for this request only* |

Consequences a hook can rely on:

- `ctx.session_id`, `ctx.system_prompt`, and anything you put in
  `ctx.extras` **survive to the next turn** and survive `Agent.reset()`.
  That's the whole "sticky" story — you do not re-assert a changed value
  each turn.
- `ctx.messages`, `ctx.tools`, `ctx.tool_loop_round` are **rebuilt** before
  each `before_llm`. Do not rely on what you saw last request; recompute it.
- `ctx.usage` accumulates across the run.

## Lifecycle

Establishment runs **when needed**, not every turn:

```
Agent is used for the first time (or after Agent.set_session)
├─ initialize_run(ctx)       # exactly once per run  →  hook may set ctx.run_id
│   [core: if still None → auto-generate]
├─ ready_run(ctx)            # exactly once per run →  ctx.run_id guaranteed non-None
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
│    rebuilds ctx.tools; sets ctx.tool_loop_round]
│   ├─ before_llm(ctx)               # may rewrite ctx.messages / ctx.tools / ctx.system_prompt
│   [core applies ctx.system_prompt to messages[0]; sends the LLM request]
│   │   if the LLM returns tool_calls:
│   │     └─ per tool call:
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

- Establishment hooks fire **once** (run) or **on session (re-)establishment**
  (session) — not on every `chat()`.
- `before_llm`, `on_round` fire once per round; `before_tool_call`,
  `after_tool_call` fire once per tool call (several possible per round).
- A vetoed tool call fires `before_tool_call` (setting
  `ctx.tool_call = None`) and records `role: tool` — "skipped by policy" —
  in history; it does not fire `after_tool_call`.
- The round-cap fallback (LLM kept calling tools after `MAX_TOOL_ROUNDS`)
  still fires `prepare` and `finish`; `before_llm` is not re-run for that
  forced final answer.

## `RunContext` reference

```python
@dataclass
class RunContext:
    # run-scoped
    run_id: str | None          # established once per run; sticky
    session_id: str | None      # established per active session; sticky
    base_prompt: str | None     # immutable for the run — the original system prompt
    system_prompt: str | None   # CURRENT prompt; sticky; hook-editable (see below)
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

### `extras`: the walls

`extras` is the sanctioned space for inter-hook coordination and for any
experiment's own run state. **The core never reads or writes it.** It is
run-scoped — it (and any hook's write to the other run-scoped fields)
survives across turns and across `Agent.reset()`. Use flat,
namespaced-style keys (e.g. `ctx.extras["budget.calls"]`) so multiple
experiments can share one context without a formal namespace today.

```python
def count_calls(ctx: RunContext):
    ctx.extras["llm_calls"] = ctx.extras.get("llm_calls", 0) + 1

def budget_guard(ctx: RunContext):
    assert ctx.extras.get("llm_calls", 0) <= 10, "budget exceeded"
```

## System prompt (dynamic, sticky)

The core sends a single `role: "system"` message as `messages[0]`; the text
comes from **`ctx.system_prompt`**, and it is **sticky**:

1. At construction, `Agent` seeds both `ctx.base_prompt` (immutable) and
   `ctx.system_prompt` from its `system_prompt` argument (default
   `DEFAULT_SYSTEM_PROMPT`).
2. Before each `before_llm`, the core has **already** put the current
   `system_prompt` at `messages[0]` and rebuilt `ctx.messages`. The hook
   may leave it alone or set `ctx.system_prompt` to something new — that
   new value is applied to `messages[0]` *after* the hook and is carried to
   the **next** request too (it's sticky; you don't re-assert it).
3. The original is always available at `ctx.base_prompt` — use it to
   extend, patch, or restore.

Four verbs, one channel:

```python
def extend(ctx: RunContext) -> None:     # base + addition
    ctx.system_prompt = (ctx.base_prompt or "") + "\nBe concise."

def prepend(ctx: RunContext) -> None:    # addition + base
    ctx.system_prompt = "You are a Growthcho agent." + (ctx.base_prompt or "")

def patch(ctx: RunContext) -> None:      # replace a slice of the base
    ctx.system_prompt = (ctx.base_prompt or "").replace("e.g. arithmetic", "e.g. arithmetic, date math")

def replace(ctx: RunContext) -> None:    # the base is gone
    ctx.system_prompt = "You are terse. No filler."

def restore(ctx: RunContext) -> None:    # back to the original
    ctx.system_prompt = ctx.base_prompt
```

**Pattern to follow — recompute from `base_prompt`, not "append to
current".** "Append to current" accumulates across turns (`"!"` becomes
`"!!"`) unless the hook cleans up; recomputing from the base is stable and
self-healing:

```python
# correct (stable):
def before_llm(ctx):
    ctx.system_prompt = (ctx.base_prompt or "") + "\n[turn " + str(ctx.turn) + "]"

# avoids drift on every turn, and can vary with ctx.session_id / ctx.turn / extras
```

`experiments/system_prompt.py` is a working example (dynamic addendum + a
`ready_session` hook that seeds per-session state in `extras`).

## Hook modules and the two folders

Two top-level folders, both outside the `agent_seed` package, both loaded
as standalone files (never imported *by* the core):

| folder | meaning | loaded |
|---|---|---|
| `experiments/` | ideas, candidates, prototypes | **never** — only on explicit import |
| `hook_modules/` | promoted hooks, known-good | **automatically**, once per process |

`agent_seed/hooks.py` is the third, distinct thing: the stable *protocol*
(`RunContext` + the fixed slots) that both folders plug into.

## Promoting experiments

The intended workflow:

1. Prototype in `experiments/` — self-contained files you test with
   explicit `hooks=...` and delete at will.
2. When one earns its keep, promote it: `cp experiments/foo.py hook_modules/`.
3. Done. Every `Agent` (direct, backend, REPL) picks it up — no
   registration anywhere.

### The contract: `make()`

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
- Modules may import `agent_seed.*` and anything installed; they are
  executed with `importlib` under unique module names (no `sys.path`
  magic), so they stay standalone.
- `make()` runs exactly **once per process per folder** (the result is
  cached); first use does the scan + import + merge, later uses are a
  dict lookup.
- Failing to import, lacking `make`, or a `make()` that raises / returns
  something other than `Hooks` → `DiscoveryError` at first use. **Fail
  fast, all-or-nothing** — the `hook_modules/` folder is your
  production channel, so a broken file is a startup error, not a
  silent skip.

### Merge policies

- Modules merge **alphabetically by filename**; on a slot conflict the
  file that sorts **last** wins (`9_audit.py` overrides `a_audit.py`).
- A slot no module provides stays `None`.
- `extras` is unaffected by merging (hooks coordinate through `ctx`,
  not through module state).

### Overriding discovery

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

## Which hook to use, by goal

| goal | hook | how |
|---|---|---|
| override / extend / patch the system prompt | `before_llm` | read `ctx.base_prompt` + set `ctx.system_prompt` (sticky) |
| inject RAG / context / date into the next LLM call only | `before_llm` | append to `ctx.messages` (per-request — history is not touched) |
| disable tools for one call | `before_llm` | `ctx.tools.clear()` |
| supply a canonical `run_id` | `initialize_run` | set `ctx.run_id` (fires once) |
| attach post-establishment run state (memory, budget) | `ready_run` | read `ctx.run_id`, load into `ctx.extras` (fires once) |
| seed / migrate per-session state | `initialize_session` | set `ctx.session_id` (fires on session change) |
| load per-session external context | `ready_session` | read `ctx.session_id` / detect change via `ctx.extras`, load into `ctx.extras` |
| audit or transform a tool call | `before_tool_call` | read/rewrite `ctx.tool_call` |
| deny a tool call | `before_tool_call` | `ctx.tool_call = None` |
| redact / rewrite a tool result | `after_tool_call` | `ctx.tool_result = ...` |
| observability / tracing | `on_round` | fire and forget (or log into `ctx.extras`) |
| validate / normalize the final answer | `prepare` | rewrite `ctx.final_answer` |
| transform the final answer | `finish` | rewrite `ctx.final_answer` (e.g. a canned signature) |

## Worked examples

### "ahem" — the last word (the `finish` hook)

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

### Dynamic system prompt

See `experiments/system_prompt.py` — recomputes a `turn`-aware addendum
from `ctx.base_prompt` every request, and uses `ready_session` to seed
per-session state in `extras`.

### Policy veto — deny a tool by name

```python
def veto_calc(ctx: RunContext) -> None:
    if ctx.tool_call and ctx.tool_call["name"] == "calculator":
        ctx.tool_call = None   # core records "skipped by policy" in history

hooks = Hooks(before_tool_call=veto_calc)
```

### One-shot context injection (round 0 only)

```python
def inject_once(ctx: RunContext) -> None:
    if ctx.tool_loop_round == 0:
        ctx.messages.append({"role": "user", "content": "(project: agent-seed)"})
```

Per-request: `ctx.messages` is the exact list the LLM sees on this call,
not the agent's stored history.

### Budget meter via `extras`

```python
def budget(ctx: RunContext, limit: int = 10) -> None:
    used = ctx.extras.get("calls", 0)
    ctx.extras["calls"] = used + 1
    if used >= limit:
        raise RuntimeError("LLM budget exceeded")

hooks = Hooks(before_llm=budget)
```

## Testing hooks

Tests in this repo must stay network-free. Use
`tests/conftest.py::FakeLLMClient`. Pattern:

```python
from agent_seed.core import Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.tools import ToolRegistry
from .conftest import FakeLLMClient, text_completion

client = FakeLLMClient([text_completion("hi")])
agent = Agent(client, ToolRegistry(),
              hooks=Hooks(finish=lambda ctx: ctx.__setattr__("final_answer", "bye")))
assert agent.chat("q") == "bye"
```

Assert on `client.requests[i]["messages"]` for what the LLM saw on call
`i`, `agent.history` for what was stored, `agent.context` for run-scoped
state (ids, prompt, `extras`), and the hook's captured lists for firing
order.

## Rules (the stability contract)

Breaking any of these is a **major** change.

1. **Hook signatures are frozen.** All hooks are `(ctx: RunContext) -> None`.
   New behavior goes on `RunContext`, never on the signature.
2. **`extras` is a wall.** The core never reads or writes it. Hooks own it.
3. **`RunContext` core fields are append-only with defaults.** Adding a
   field (with a `None`/empty default) is a minor bump — existing hooks
   still work. Removing or renaming a field is a major break.
4. **Hook slots are append-only.** Adding a slot (`None`-defaulted) is a
   minor bump. Removing a slot is a major break.
5. **Run-scoped fields are sticky; per-request fields are rebuilt.**
   `system_prompt` persists across turns until changed; `messages`,
   `tools`, `tool_loop_round` are rebuilt before each `before_llm` — do not
   rely on stale per-request values.
6. **The canonical system-prompt channel is `ctx.system_prompt`.** The
   core applies it to `messages[0]` after `before_llm`. Directly editing
   `messages[0]` is also legal but will be overwritten by the core —
   prefer the field.
7. **Veto is explicit.** `ctx.tool_call = None` is the only way to stop a
   tool call. The core records a well-formed `role: tool`
   `"skipped by policy"` result so the next API call stays valid.
8. **No re-entry.** A hook must not call `agent.chat()` from inside
   itself.
9. **Hooks are synchronous.** The agent loop runs in a worker thread;
   async hooks are out of scope.
10. **Establishment fires on need.** Run hooks strictly once; session hooks
    on each (re)establishment. Do not assume establishment fires every
    `chat()`.

## Extending the core with new slots

If none of the current slots cover you:

1. Add a new slot to `Hooks` (default `None`).
2. Add the `if hooks.<new> is not None: hooks.<new>(ctx)` guard at the
   right point in the core.
3. Document the slot in `docs/hooks.md` and add a test in
   `tests/test_hooks.py`.

That's it. No core refactor, no signature change, and every existing hook
keeps working unchanged.
