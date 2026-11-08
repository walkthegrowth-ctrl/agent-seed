# HTTP/SSE protocol

This is the contract between any frontend (REPL today; web/tui later) and the
backend. If you build a new frontend, this is the only spec you need.

## Conventions

- Base URL: the backend's origin (e.g. `http://127.0.0.1:8787`).
- All endpoints are same-origin. No auth in v0.
- `session_id` is opaque; the server assigns or accepts a client-chosen one.

## `GET /health`

`200`

```json
{ "status": "ok", "version": "0.1.0" }
```

## `GET /config`

`200` — LLM configuration. Two forms:

Without `session_id` — the server's default LLM.

```json
{
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "model": "qwen3:8b",
    "api_key_set": true,
    "timeout": 120.0
  }
}
```

`llm` may be `null` if `LLM_MODEL` is unset.

With `session_id` — that session's effective LLM (what the core calls) and its
immutable reference (what the session started from):

```json
{
  "llm": { "base_url": "http://localhost:11434/v1", "model": "chosen-model", "api_key_set": true, "timeout": 120.0 },
  "base": { "base_url": "http://localhost:11434/v1", "model": "env-model", "api_key_set": true, "timeout": 120.0 }
}
```

A `session_id` is valid as soon as the **client** names it — frontends mint
their own id at startup. So this works even before the first `POST /chat`:

- If a config was already `POST /config`-ed for that id, the response echoes it
  (`llm` = the pending config, `base` = the app default).
- `404` only if the id was never touched (no live session and no pending config).

## `POST /config`

Set the LLM at runtime. Body (all fields optional; at least one required):

```json
{
  "session_id": "<optional>",
  "model": "new-model",
  "base_url": "http://...",
  "api_key": "...",
  "timeout": 60.0
}
```

- Without `session_id`, the change becomes the **app default** (used by all
  new sessions) and is applied to every live session.
- With `session_id`, only that session's LLM is changed.

A `session_id` is valid even before the first `POST /chat` (the client mints
its own id up front). For a not-yet-created session the config is stored as
**pending** and the session adopts it as its base at creation; the active
agent's `enter_agent`/`before_llm` may still override or restore.

The supplied fields are merged over the current config. `on_model_change` (a
hook veto) applies to *live* sessions only — a pending config records the
request verbatim, since no agent has taken over yet.

Status codes:

- `200` — applied or recorded. Body: `{ "ok": true, "session_id": "<id>", "llm": {...} }`
  (omit `session_id` when applied globally).
- `400` — no fields supplied / empty `model`.
- `409` — vetoed by `on_model_change`. Body: `{ "error": "<reason>" }`.

`api_key` is never returned (only `api_key_set`).

## `POST /config/test`

Probe a candidate LLM config (one minimal ping completion) without persisting
it. Body is the same partial shape as `POST /config` (any subset of
`model` / `base_url` / `api_key` / `timeout`); a `session_id` may be supplied
and is used to pick the merge base:

```json
{ "session_id": "<optional>", "model": "candidate-model", "base_url": "http://..." }
```

Fields are merged over the most specific base available — the session's current
(or pending) model if a `session_id` is supplied, else the app default. This is
a diagnostic: it does **not** go through `on_model_change` (a probe failing is
a *finding*, not a change — use `POST /config` for a gated change).

`200` — the probe ran to completion. Body carries `ok` (the finding):

```json
{ "ok": true, "error": null, "latency_ms": 42.1, "usage": {...}, "model": "candidate-model" }
```
```json
{ "ok": false, "error": "connection refused", "latency_ms": 12.0, "usage": null, "model": "candidate-model" }
```

`400` — the backend has no base LLM to merge over (e.g. `LLM_MODEL` unset, no
pending/app config, and the body doesn't supply enough of
`base_url` / `api_key` / `model`). The backend does **not** fabricate a default
to probe — a client error is returned instead.

## `GET /tools`

`200`

Optional query parameter:

- `agent_name` (string) — restrict the listing to the tools **that agent's
  whitelist allows** (resolved via `hook_modules/<agent_name>/`, falling back
  to `hook_modules/`). An agent that declares no tool set (no `enter_agent` /
  no `ctx.allowed_tools`) returns `{"tools": [], "names": []}`. Strict
  policy: an omitted set is *zero tools*, not *all tools*. To read an agent's
  whitelist the backend runs that agent's `enter_agent` against a scratch
  `RunContext`; if that hook **raises**, `GET /tools?agent_name=<n>` returns
  `500` — a broken agent hook, not a protocol error.

Without `agent_name`, the whole tool catalog (the union of every registered
tool) is returned. This is a *catalog* endpoint — it is not a live session
snapshot and does not mutate hook state.

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression (numbers, + - * / // % **).",
        "parameters": {
          "type": "object",
          "properties": { "expression": { "type": "string" } },
          "required": ["expression"]
        }
      }
    }
  ],
  "names": ["calculator"]
}
```

`tools` is in OpenAI function-calling schema form, ready to be forwarded.
`names` is the list of `tools[i].function.name`.

## `GET /agents`

`200` — the discoverable agent names (subdirectories of the hooks root),
sorted. Pass any of them as `agent_name` in `POST /chat` to switch the
agent's hooks to that subdirectory.

```json
{ "agents": ["alpha", "gamma"] }
```

An empty list (or no `/agents` support on an older backend) means only the
default agent (the `hook_modules/` root) is available.

## `GET /prompt`

`200` — a read-only, compositional view of the prompt the core is about to
send (or, with `live`, was just sending). This is a *diagnostic / debugging*
endpoint: it lets a frontend or a human see the composed system message, the
declared context and addendum section, and the few-shot example list.

Query parameters (all optional):

- `agent_name` (string) — which agent's hooks / baseline to resolve against.
  Without it the default (root) agent is used. Mirrors `GET /tools?agent_name=…`.
- `session_id` (string) — with `live=true`, the live session whose last composed
  value is reported. Without `live=true` this parameter is ignored.
- `live` (bool) — `true` for the session's last composed value (what was just
  sent, including any before_llm hook adjustments); the default `false` is the
  deterministic scratch view, resolved from the selected agent's declared
  baselines (its hook folder's `prompt.md` / `context.md` / `examples.jsonl`).

Response shape (scratch, the default — `live` omitted or false):

```json
{
  "agent_name": "alpha",           // null for the default agent
  "live": false,                   // scratch view
  "identity": "You are alpha.",    // the resolved identity (prompt.md or Hooks.prompt or default)
  "context": "state=on",           // the per-round context block (context.md baseline or null)
  "addendum": null,                // the per-turn addendum (live only; always null in scratch)
  "examples": [                    // the few-shot message list (canonical role/content shape)
    { "role": "user", "content": "e1" },
    { "role": "assistant", "content": "a1" }
  ],
  "system": "You are alpha.\n\nstate=on"   // compose_system(identity, context, addendum)
}
```

Response shape (live — `live=true` with a `session_id`): the session's
last composed value, including any `before_llm` hook adjustments:

```json
{
  "agent_name": "alpha",
  "session_id": "0f3a…",           // present only for live=true
  "live": true,
  "identity": "You are alpha.",
  "context": "state=on",
  "addendum": "round-note",        // the last round's hook-set addendum (null if none)
  "examples": [ /* same shape as scratch */ ],
  "system": "You are alpha.\n\nstate=on\n\nround-note"
}
```

- `system` is the exact text of the first message the core would send: identity
  folded with (optionally) context and addendum via
  `compose_system` (see `docs/manual.md`, "Prompting").
- `examples` is the canonical `[{"role","content"}, …]` shape (same as the
  OpenAI wire) — it is what `compose_prompt` places between the system message
  and the stored history.
- `addendum` is non-null only when `live=true` and the last round's hook set it.
- `agent_name` / `session_id` echo back what was resolved, so the caller knows
  which one of the two views it got.
- Read-only: this endpoint does not mutate the active agent's state, does not
  fire hooks, and does not change the LLM. Mirrors `GET /tools?agent_name=…`
  in that way (resolves the *agent's own* declared baselines against a scratch
  `RunContext` — no side-effect on the live agent).

Status codes:

- `200` — view resolved (scratch by default; `live` when the named session exists).
- `400` — `live=true` with no `session_id`.
- `404` — `live=true` with a `session_id` that has no live session.

## `POST /reset`

Body: `{ "session_id": "<id>" }` → `200 { "ok": true, "session_id": "<id>" }`.
Clears that session's message history. Client may also simply start using a new
`session_id` and treat the old one as dead.

## `POST /chat`

Body:

```json
{ "message": "what is 12 * 13?", "session_id": "<optional>", "agent_name": "<optional>" }
```

If `session_id` is omitted, the server assigns one and returns it in the
`X-Session-Id` response header and the final `done` event.

`agent_name` is optional. When provided, the agent's hooks are re-resolved
against `hook_modules/<agent_name>/` (falling back to `hook_modules/` if
that subdirectory does not exist) and the establishment hooks re-fire
(see `manual.md`). Omit it to keep / use the default agent.

Response: `200`, `Content-Type: text/event-stream`, `X-Session-Id: <id>`.

### Event types

Every event is a standard SSE frame `event: <type>` + `data: <json>`.

| type | payload | meaning |
|------|---------|---------|
| `tool` | `{ "name", "arguments", "result" }` | the agent invoked a tool and got a result (string). |
| `token` | `{ "text" }` | one streamed content token of the **final** answer, in order. |
| `done` | `{ "session_id", "usage"?, "usage_session"? }` | turn complete. `usage` = this turn's tokens; `usage_session` = the session cumulative total. Final event of a turn. |
| `error` | `{ "message" }` | unrecoverable failure; final event of a turn. |

A well-formed turn is: zero or more `tool` events, then one or more `token`
events (unless the final answer is empty), then exactly one `done` (or `error`).

### `done` usage fields

`done` always carries `session_id`. The two usage keys are **optional and
omitted** when the LLM endpoint reported no `usage` for the turn:

- `usage` — this turn's token counters, summed across every LLM call the turn
  made (a tool round + the final answer). Shape is whatever the LLM sent (the
  OpenAI convention is `prompt_tokens` / `completion_tokens` / `total_tokens`).
- `usage_session` — the **session** cumulative total (the sum of every turn's
  `usage` in this session). It is maintained by the backend and cleared by
  `POST /reset`. A fresh session's first turn has `usage_session` equal to
  `usage`.

Both are backend-computed (the backend sums the LLM's per-call counters; it does
not trust a client-supplied total). A frontend that ignores them (older
frontends) is unaffected — this is append-only.

### Example full response

```
event: tool
data: {"name": "calculator", "arguments": {"expression": "12 * 13"}, "result": "156.0"}

event: token
data: {"text": "12"}

event: token
data: {"text": " times"}

event: token
data: {"text": " 13"}

event: token
data: {"text": " is"}

event: token
data: {"text": " 156"}

event: token
data: {"text": "."}

event: done
data: {"session_id": "006477f9e62d4606ae4aa44cb3a20f7d", "usage": {"prompt_tokens": 21, "completion_tokens": 6, "total_tokens": 27}, "usage_session": {"prompt_tokens": 21, "completion_tokens": 6, "total_tokens": 27}}
```

### HTTP status codes

- `200` — stream started (errors arrive as an `error` event mid-stream).
- `500` — configuration error before the stream could start (e.g. `LLM_MODEL`
  unset). Body is a JSON error object, not an SSE stream.
- `4xx` — malformed request body.

## Client requirements

1. Accept `text/event-stream` and parse `event:`/`data:` frames.
2. Concatenate `token.text` in arrival order to render the answer.
3. Render `tool` events for observability (name, args, result).
4. Treat `done` as the end of the turn; retain `session_id` for the next turn.
5. On `error` or HTTP non-200, surface the message and stop the turn.
6. Read the stream to the `done` (or `error`) frame before assuming the turn
   is complete, and before issuing a request whose correctness depends on
   server state a turn sets. The turn runs on a backend worker; `done` is
   emitted only once that work has finished — **closing the stream before
   `done` does not mean the turn has finished**. A client that stops reading
   early may race against a producer thread that has not yet completed its
   LLM call and side effects (and any such state it later exposes, e.g. via
   `GET /config` or `GET /prompt?live=true`, is not guaranteed to reflect that
   turn). To abandon a turn you may close the connection, but you must not
   then rely on the turn having finished or its state being settled.

The reference implementation is `agent_seed/frontends/repl/app.py::Session`
(which always reads each turn's stream fully to `done` / `error`).
