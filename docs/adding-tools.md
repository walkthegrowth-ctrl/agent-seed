# Adding tools

A tool is a plain Python function with a **one-line docstring** and **type
hints** on its parameters. That's it — the schema, dispatch, and (if you add
them) tests all follow from that.

## How the schema is derived

For a function `foo(a: str, b: int) -> float` with docstring `"Do the thing."`:

- `name` ← function name
- `description` ← first line of the docstring
- `parameters.properties` ← each annotated param mapped through `type_to_schema`
- `parameters.required` ← every annotated param (v0 treats all as required)

Type map (`type_to_schema`):

| Python annotation | JSON schema |
|-------------------|-------------|
| `str` | `{"type": "string"}` |
| `int`, `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[T]` | `{"type": "array", "items": <T>}` |
| `dict` | `{"type": "object"}` |
| `Any` / unannotated | `{}` |

## Registering

There are three equivalent ways to register:

```python
from agent_seed.tools import ToolRegistry

registry = ToolRegistry()


# 1) explicit register
def echo(text: str) -> str:
    """Return the input unchanged."""
    return text


registry.register(echo)


# 2) the @tool decorator form
@registry.tool
def echo2(text: str) -> str:
    """Return the input unchanged."""
    return text


# 3) build it inline (same thing)
```

The default registry in `agent_seed/tools.py` already has `calculator`. If you
want your own toolset in a custom server, build a new `ToolRegistry` and pass
it to `server.create_app(registry)`:

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

## Conventions & gotchas

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

## Adding a test

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
