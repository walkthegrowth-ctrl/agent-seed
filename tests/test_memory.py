"""Hermetic tests for the durable-memory experiment (experiments/memory.py).

The suite never uses a real LLM or a network (tests must not *depend* on an
external LLM). We:

* redirect ``MEMORY_ROOT`` to a temp dir so all SQLite writes land in the
  test workspace, not the repo root;
* monkeypatch ``agent_seed.llm.OpenAICompatClient`` (the summarizer path)
  with a scripted client — the sanctioned pattern the other probe tests rely
  on (see ``test_model_selection.py::test_test_model_reports_*``);
* drive the agent's chat loop with ``FakeLLMClient`` (in ``conftest.py``),
  exactly like the other core tests.

Each test runs against an *isolated* store (a fresh per-test ``tmp_path``),
so tests are fully independent. The memory hooks are built via the
experiment's ``make()`` factory (the same way a real promoted deployment
would load them — discovery is exercised by the existing
``test_discovery.py`` suite, not re-tested here). ``memory.make()`` has a
documented side-effect (registering ``memory_recall`` on the global default
registry, idempotent); the ``mem_store`` fixture preserves and restores that
registry around each test so the suite stays hermetic regardless of ordering.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from agent_seed.core import Agent
from agent_seed.hooks import Hooks, RunContext
from agent_seed.llm import Completion, ToolCall
from agent_seed.tools import ToolRegistry
from agent_seed.tools import registry as global_registry

from .conftest import FakeLLMClient, text_completion

_REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_SRC = _REPO_ROOT / "experiments" / "memory.py"


def _load_module_under_test(src: Path, name: str):
    """Load a standalone file as a module (mirrors discovery's importlib path)."""

    spec = importlib.util.spec_from_file_location(name, src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# experiments/memory.py as a standalone module (like discovery loads it). The
# module name is arbitrary — the file content is what matters, and the module
# does not depend on its own name.
memory = _load_module_under_test(MEMORY_SRC, "_memory_under_test")


@pytest.fixture
def mem_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh, empty store; a *default* stubbed summarizer client (so no test
    can drive a real LLM); and restoration of the global default registry (so
    ``memory.make()``'s idempotent registration of ``memory_recall`` never
    leaks into other tests). Tests override the summarizer patch in-test when
    they need specific behaviour (the later monkeypatch wins)."""
    store = tmp_path / "mem"
    store.mkdir()
    monkeypatch.setenv("MEMORY_ROOT", str(store))
    monkeypatch.delenv("MEMORY_SUMMARY_WHEN", raising=False)
    monkeypatch.delenv("MEMORY_SUMMARY_MODEL", raising=False)
    memory._TOOL_SCOPE["agent_name"] = None
    memory._TOOL_SCOPE["session_id"] = None
    monkeypatch.setattr(
        "agent_seed.llm.OpenAICompatClient",
        lambda cfg: _StubSummaryClient(content="stub-summary"),
    )
    # Preserve the global registry's tool set; make() adds memory_recall, so
    # restore it on teardown to keep the suite hermetic regardless of order.
    saved_tools = dict(global_registry._tools)
    yield store
    global_registry._tools.clear()
    global_registry._tools.update(saved_tools)


class _StubSummaryClient:
    """Scripted stand-in for the summarizer's LLM client (monkeypatch target)."""

    def __init__(self, content: str | None, raise_exc: Exception | None = None):
        self._content = content
        self._raise = raise_exc

    def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
        if self._raise is not None:
            raise self._raise
        return Completion(content=self._content)


def _seed_row(agent: str, session: str, user: str, assistant: str) -> None:
    """Insert one turn into the store directly (for tests that seed state)."""
    memory._commit_turn(agent, session, user, assistant, tools=[], usage={})


def _summary_of(agent: str, session: str) -> tuple[str, int]:
    return memory._get_summary(agent, session)


def test_persist_then_reset_keeps_durable_store_and_reopens(mem_store):
    """``finish`` commits every turn to the store; ``Agent.reset()`` clears
    in-RAM history only. A fresh Agent for the same session still restores,
    and keeps committing to the same durable row stream."""
    client = FakeLLMClient([text_completion("a"), text_completion("b"), text_completion("c")])
    hooks = memory.make()
    agent = Agent(client, ToolRegistry(), hooks=hooks, session_id="s1")
    agent.chat("hi")
    agent.chat("how are you")
    agent.chat("thanks")

    path = mem_store / "main.db"
    assert path.exists()
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT user, assistant FROM memory WHERE session_id='s1' ORDER BY seq"
        ).fetchall()
    assert [r[0] for r in rows] == ["hi", "how are you", "thanks"]
    assert all(r[1] in ("a", "b", "c") for r in rows)

    # Reset is RAM-only.
    agent.reset()
    assert agent.history == []
    with sqlite3.connect(path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM memory WHERE session_id='s1'").fetchone()[0]
    assert count == 3

    # A fresh agent for the same session continues the same durable stream.
    client2 = FakeLLMClient([text_completion("d")])
    agent2 = Agent(client2, ToolRegistry(), hooks=hooks, session_id="s1")
    agent2.chat("again")
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT user FROM memory WHERE session_id='s1' ORDER BY seq"
        ).fetchall()
    assert [r[0] for r in rows] == ["hi", "how are you", "thanks", "again"]


def test_auto_restore_folds_summary_into_system_message(mem_store, monkeypatch):
    """The store's rolling summary for the active session is folded into the
    single system message (via ctx.context) for every round of every turn."""
    # Seed a summary in an isolated store.
    memory._save_summary("main", "s1", "The user is a chef working on the deploy plan.", 0)

    monkeypatch.setattr(
        "agent_seed.llm.OpenAICompatClient",
        lambda cfg: _StubSummaryClient(content="should not be called"),
    )

    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=memory.make(), session_id="s1")
    agent.chat("anything")

    system = client.requests[0]["messages"][0]
    assert system["role"] == "system"
    assert "The user is a chef working on the deploy plan." in system["content"]


def test_summary_refresh_is_fail_soft_when_llm_raises(mem_store, monkeypatch):
    """If the LLM for the summary call is unreachable / raises, the turn is
    still committed and the prior summary (if any) is kept. A broken store
    must never fail the user's turn."""
    # Prior summary to preserve.
    memory._save_summary("main", "s1", "prior-summary", 0)

    # Summarizer client raises (simulates provider outage).
    monkeypatch.setattr(
        "agent_seed.llm.OpenAICompatClient",
        lambda cfg: _StubSummaryClient(content=None, raise_exc=RuntimeError("conn refused")),
    )

    client = FakeLLMClient([text_completion("ok")])
    agent = Agent(client, ToolRegistry(), hooks=memory.make(), session_id="s1")
    answer = agent.chat("hello")
    assert answer == "ok"

    path = mem_store / "main.db"
    with sqlite3.connect(path) as conn:
        committed = conn.execute("SELECT COUNT(*) FROM memory WHERE session_id='s1'").fetchone()[0]
    assert committed == 1

    summary, seq = memory._get_summary("main", "s1")
    assert summary == "prior-summary"  # previous kept, never clobbered
    assert seq == 0


def test_summary_watermark_is_incremental(mem_store, monkeypatch):
    """The per-session watermark (last_seq) keeps the summarizer's input to
    only the turns added since the last refresh: 1 LLM call per committed
    turn, not per message."""

    calls: list[list[tuple[str, str]]] = []

    def capture(cfg):
        class Client:
            def chat(self, messages, tools=None, on_token=None, config=None, max_tokens=None):
                # Recover the 'New turns' block from the user prompt.
                prompt = messages[-1]["content"]
                new_block = prompt.split("New turns:\n", 1)[1].rsplit("\n\nUpdated", 1)[0]
                turns = []
                for pair in new_block.split("\n\n"):
                    lines = pair.split("\n")
                    turns.append(
                        (lines[0].removeprefix("user: "), lines[1].removeprefix("assistant: "))
                    )
                calls.append(turns)
                return Completion(content=f"fold-{len(calls)}")

        return Client()

    monkeypatch.setattr("agent_seed.llm.OpenAICompatClient", capture)

    client = FakeLLMClient([text_completion("a"), text_completion("b"), text_completion("c")])
    agent = Agent(client, ToolRegistry(), hooks=memory.make(), session_id="s1")
    agent.chat("first turn")
    agent.chat("second turn")
    agent.chat("third turn")

    # Exactly one summarizer call per committed turn (watermark prevents
    # redundant folds, e.g. when no new rows were added).
    assert len(calls) == 3
    assert calls[0] == [("first turn", "a")]
    assert calls[1] == [("second turn", "b")]
    assert calls[2] == [("third turn", "c")]
    summary, last_seq = _summary_of("main", "s1")
    assert summary == "fold-3"
    assert last_seq == 3


def test_memory_recall_tool_end_to_end(mem_store, monkeypatch):
    """The tool end-to-end: a scripted LLM that issues a ``memory_recall``
    tool call; the core dispatches it through our own registry (where we
    registered the real ``memory.memory_recall`` function); the result
    surfaces as a well-formed ``role:tool`` message the next LLM round sees."""

    # Seed one turn in the store that we'll search for.
    _seed_row("main", "s1", "I am a chef, working on the deploy plan.", "understood.")

    # A fresh registry with ONLY the real memory_recall function. Dispatching
    # through this stub (not the global default) means we do not depend on
    # which other tests have run, and never mutate the default registry.
    stub_registry = ToolRegistry()
    stub_registry.register(memory.memory_recall)

    # The summarizer must not be invoked (MEMORY_SUMMARY_WHEN stays at the
    # default "finish", but no LLM call is reachable anyway — the stub below
    # would succeed with no observable effect either way).
    monkeypatch.setattr(
        "agent_seed.llm.OpenAICompatClient",
        lambda cfg: _StubSummaryClient(content="irrelevant"),
    )

    # Whitelist the tool for this agent (strict policy: a tool only exists
    # once the agent's hooks declare it).
    def enter_agent(ctx: RunContext) -> None:
        ctx.allowed_tools = ["memory_recall"]

    hooks = Hooks(enter_agent=enter_agent)
    memory_hooks = memory.make()
    for slot in ("before_llm", "after_tool_call", "ready_session", "finish"):
        setattr(hooks, slot, getattr(memory_hooks, slot))

    client = FakeLLMClient(
        [
            Completion(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="memory_recall",
                        arguments={"query": "chef", "limit": 5},
                    )
                ],
            ),
            text_completion("You said you were a chef; deploying now."),
        ]
    )
    agent = Agent(client, stub_registry, hooks=hooks, session_id="s1")
    agent.chat("recall chef")

    # The dispatched tool result was a JSON array with the seeded row.
    messages = client.requests[-1]["messages"]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    rows = json.loads(tool_messages[0]["content"])
    assert rows and rows[0]["assistant"] == "understood."
    assert rows[0]["user"].startswith("I am a chef")


def test_search_fts5_or_like_fallback(mem_store):
    """FTS5 search (or the LIKE fallback) returns the matching row and
    supports session scoping."""
    _seed_row("main", "s1", "user asks about the deploy plan", "plan approved")
    _seed_row("main", "s2", "user asks about a different topic", "ok")
    memory._commit_turn("main", "s1", "user asks about a third row", "fine", tools=[], usage={})
    # Ensure the FTS index has rows (commit-turn inserts into fts if available).

    rows = memory.search("main", "deploy", session_id="s1", limit=5)
    assert any(r["user"].startswith("user asks about the deploy plan") for r in rows)

    scoped = memory.search("main", "different", session_id="s2", limit=5)
    assert any(r["assistant"] == "ok" for r in scoped) and len(scoped) == 1

    unscoped = memory.search("main", "third", limit=5)
    assert any(r["user"].endswith("third row") for r in unscoped)
