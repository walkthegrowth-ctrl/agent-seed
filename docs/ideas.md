# Ideas / backlog

A working list of **deferred** enhancements. Nothing here is implemented; each
entry is a *candidate*, not a commitment. These are recorded here so that the
conversation about them is not lost when the immediate work moves on.

Each entry is independent — pick any one of them in isolation and it should
add cleanly to the codebase without breaking the others.

## Prompting

### `{"query","answer"}` sugar for `examples.jsonl`

**State:** deferred (v1 ships the canonical shape only).

The examples file is currently one JSON object per line, in the
OpenAI-compatible role shape (`{"role","content"}`):

```jsonl
{"role": "user", "content": "what is 2+2?"}
{"role": "assistant", "content": "4"}
```

For simple question / answer pairs that is a little heavy — two lines to
say *"for Q, respond A"*. A plausible convenience shape would be one example
per line:

```jsonl
{"query": "what is 2+2?", "answer": "4"}
```

Normalization in `discovery.py` would expand each `{"query","answer"}` line to
the two canonical role messages, so the in-memory form is still exactly
`[{"role","content"}, …]` — a single internal representation, a two-shape
file input.

**Why it was deferred:**

1. The canonical shape is the same shape the LLM wire uses; hooks that
   compose their own examples in `before_llm` would have to know *both*
   shapes.
2. It cannot express the one shape the canonical form can: few-shot
   *tool-use* demonstrations (`assistant` `tool_calls` + `tool` result).
   For an agentic system that is not an edge case; it's the point of the
   file.
3. It adds one more parser branch and one more test to maintain, for a
   convenience that is ~one line of typing per example.

**If it is wanted later:**

- Add the parser branch to
  `agent_seed/discovery.py::_load_examples` (or a new `_normalize_example`
  helper it calls).
- Keep `normalize_examples` in
  `agent_seed/prompting.py` as the canonical internal form — it already
  drops / deep-copies the list; it should not know about sugar.
- Document the sugar in `manual.md` ("Declaring prompts per agent (the
  hook folder)") and in `examples.jsonl`'s own docstring if we add a
  `#`-comment style example block.
- Add a test to `tests/test_discovery.py` that mixes both shapes in one
  file (one `{"query","answer"}` line + one canonical `{"role","content"}`
  line) and asserts both normalize to the same output.

## Prompting (other)

### Prompt cost / token budgeting for examples

The core sums `usage` across the run / session; it does not account for the
*declared* `examples` content (which is static but still tokens). If a
frontend later wants to show "this prompt is about N tokens", `GET /prompt`
is the natural endpoint to add a `usage_estimate` field to (it already has
the composed strings; a length-based estimate is ~one line).

### RAG source integration

A RAG *source* (embedding store, vector search, document chunker) is not in
scope — the `ctx.context` channel is a hook's job to fill in from wherever
the chunks come from. We are deliberately not adding a "RAG provider"
interface; a hook that has access to the source and sets `ctx.context` per
round is the sanctioned integration.

### Prompt versioning / A/B

`prompt.md` / `context.md` / `examples.jsonl` are the *content* side of an
agent. A versioning layer (e.g. `prompt.v2.md` and a hook that picks the
active one) is a policy concern; it could be added as a hook module without
core changes.

## Robustness

### Cancellation of an in-flight turn

The producer thread in `server.py` does not respect a mid-stream client
disconnect. A `threading.Event` checked between LLM rounds and passed into
`OpenAICompatClient.chat` (as an optional `cancel_token`) would let a
hanging session be un-stuck.

### Context windowing / compaction

Very long sessions eventually overflow the model's context because
`ctx.messages` grows with the full history every round. The sanctioned
fix is a hook that trims / summarizes `ctx.messages` in `before_llm` using
the stored `agent.history`; a core-level compaction layer (e.g. a
summarizer that rewrites the system block) is not in scope today.

## Deployability

### Per-user isolation / auth

Currently the backend is single-user by design (`architecture.md` lists
this under "Deliberately out of scope"). Adding per-user isolation is the
right next step before any multi-tenant rollout; the natural extension is
a `user_id` on `RunContext` and a user-scoped `SessionStore`.

### `GET /prompt` on the REPL

`/prompt` (and optionally `/prompt live`) would render either the *scratch
view* for the current agent (its declared identity + context + examples) or,
with the `live` keyword, the last composed value for this session — the
same data the backend exposes at `GET /prompt?live=true`.
