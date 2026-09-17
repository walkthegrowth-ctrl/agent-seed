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

`200` — LLM configuration as understood by the server.

```json
{
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "model": "qwen3:8b",
    "api_key_set": true
  }
}
```

`llm` may be `null` if `LLM_MODEL` is unset.

## `GET /tools`

`200`

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

## `POST /reset`

Body: `{ "session_id": "<id>" }` → `200 { "ok": true, "session_id": "<id>" }`.
Clears that session's message history. Client may also simply start using a new
`session_id` and treat the old one as dead.

## `POST /chat`

Body:

```json
{ "message": "what is 12 * 13?", "session_id": "<optional>" }
```

If `session_id` is omitted, the server assigns one and returns it in the
`X-Session-Id` response header and the final `done` event.

Response: `200`, `Content-Type: text/event-stream`, `X-Session-Id: <id>`.

### Event types

Every event is a standard SSE frame `event: <type>` + `data: <json>`.

| type | payload | meaning |
|------|---------|---------|
| `tool` | `{ "name", "arguments", "result" }` | the agent invoked a tool and got a result (string). |
| `token` | `{ "text" }` | one streamed content token of the **final** answer, in order. |
| `done` | `{ "session_id" }` | turn complete. Final event of a turn. |
| `error` | `{ "message" }` | unrecoverable failure; final event of a turn. |

A well-formed turn is: zero or more `tool` events, then one or more `token`
events (unless the final answer is empty), then exactly one `done` (or `error`).

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
data: {"session_id": "006477f9e62d4606ae4aa44cb3a20f7d"}
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

The reference implementation is `agent_seed/frontends/repl/app.py::Session`.
