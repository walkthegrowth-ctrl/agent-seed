"""Durable per-agent memory: persistence + retrieval behind a hook.

What it does
------------

Gives an agent a durable store of its conversation turns that survives
``Agent.reset()`` and a process restart, and makes that memory available two
ways:

* **auto-restore** — when a session (re)starts, the session's rolling *summary*
  is folded into the system message for every round of every turn (the
  ``ctx.context`` channel, i.e. the "current state / RAG" slot). The agent
  picks up where it left off without being told to.
* **queryable transcript** — a ``memory_recall(query, session_id, limit)`` tool
  the agent can call to keyword/phrase-search the stored turns across sessions
  (SQLite FTS5, with a ``LIKE`` fallback where FTS5 is unavailable).

State lives in one SQLite file per agent, default ``agent_memory/<agent>.db``
in the CWD, overridable by ``$MEMORY_ROOT``. Only the standard library is used
(``sqlite3``, FTS5 ships with CPython). No core changes: every integration point
is a hook slot plus ``RunContext`` / ``ctx.extras``.

How the summary is made
-----------------------

A rolling per-session summary is rebuilt with **one extra LLM call** (the cost
we agreed is acceptable): it folds the *previous* summary plus the turns added
since the last update into a single updated summary. It runs in ``finish``
(after the turn is committed, so the turn's content is real data), gated by a
per-session *watermark* (``last_seq``) so only new work is summarized and the
call is a no-op when nothing changed. It is **fail-soft**: any error (config
missing, provider unreachable) keeps the prior summary and never disturbs the
user's turn.

Env knobs
---------

* ``MEMORY_ROOT``    — store directory (default ``./agent_memory``).
* ``MEMORY_SUMMARY_MODEL`` — optional model name for the summary call (default:
  ``ctx.current_model``; handy to point the fold at a cheaper model).
* ``MEMORY_SUMMARY_WHEN``  — ``finish`` (default), ``ready``, or ``both``.
* ``MEMORY_MAX_SUMMARY_CHARS`` — cap on the previous summary folded in
  (default ~600 chars; bounds the prompt, not the cost, of each fold).

Activation (per agent, strict-whitelist philosophy preserved)
-------------------------------------------------------------

    cp experiments/memory.py hook_modules/           # main (root) agent
    # then enable the tool by one line in hook_modules/00_tools.py:
    #     ALLOWED_TOOLS = ["calculator", "memory_recall"]
    # (named agent: cp into hook_modules/<name>/ and add the tool there — see
    #  experiments/tool_policy.py for the pattern)

Deactivation is ``rm hook_modules/memory.py`` (drop the tool name to fully
remove the tool). There is no ``enter_agent`` here on purpose: the strict
whitelist is the single gate that decides the agent's tools, so memory adds a
declaration (the tool) and never reaches into another module's hook slots.

Notes & gotchas (see docs/manual.md "Durable memory")
------------------------------------------------------

* ``memory.py`` defines ``before_llm`` / ``after_tool_call`` / ``finish`` /
  ``ready_session``. If the target hook folder *also* defines one of those
  slots, discovery's "alphabetically last wins" picks one — name this module
  ``zz_memory.py`` to make it the survivor, or consolidate the hooks.
* ``make()`` registers ``memory_recall`` on the default registry (idempotent).
  It is only *offered* to the LLM when the agent whitelists it — registration
  alone never grants access (strict policy).
* The tool resolves the acting agent from a per-process scope set by the hooks
  (tools do not receive ``ctx``). With several agents driving in one process,
  the most recently established agent is the tool's default context.
* ``POST /reset`` clears in-memory history but NOT this store — memory persists
  by design. Delete the ``.db`` to wipe an agent's memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from typing import Any

from agent_seed import llm as _llm
from agent_seed.hooks import Hooks, RunContext

# -- config -------------------------------------------------------------------

_DEFAULT_ROOT = os.path.join(".", "agent_memory")
_SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "600"))
_SUMMARIZER_SYSTEM = (
    "You are a memory summarizer. Given a previous summary and the new "
    "conversation turns, produce ONE updated summary that captures every "
    "important fact, decision, preference, and task the assistant knows about "
    "the user and the ongoing conversation. Be concise (1-6 sentences), keep "
    "it factual, and return ONLY the summary text."
)

# Tool scope (set by the hooks; read by memory_recall). Tools do not receive
# RunContext, so the acting agent is published here for the tool to resolve.
_TOOL_SCOPE: dict[str, Any] = {"agent_name": None, "session_id": None}


def _store_root() -> str:
    path = os.getenv("MEMORY_ROOT", _DEFAULT_ROOT)
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _db_path(agent_name: str | None) -> str:
    safe = (agent_name or "main").replace(os.sep, "_").replace("..", "_")
    return os.path.join(_store_root(), f"{safe}.db")


def _connect(agent_name: str | None) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(agent_name))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=2000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memory(
               id INTEGER PRIMARY KEY,
               session_id TEXT NOT NULL,
               ts TEXT NOT NULL,
               seq INTEGER NOT NULL,
               user TEXT NOT NULL,
               assistant TEXT NOT NULL,
               tools TEXT NOT NULL,
               usage TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS summary_state(
               session_id TEXT PRIMARY KEY,
               summary TEXT NOT NULL,
               last_seq INTEGER NOT NULL DEFAULT 0,
               updated_at TEXT NOT NULL
           )"""
    )
    # Optional full-text index (FTS5 ships with CPython). If the build lacks
    # it, skip silently and search degrades to LIKE (still stdlib).
    try:
        conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
               USING fts5(user, assistant, content='')"
            """
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return conn


def _fts_enabled(agent_name: str | None) -> bool:
    try:
        with _connect(agent_name) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


# -- store ops ----------------------------------------------------------------


def _commit_turn(
    agent_name: str | None,
    session_id: str,
    user: str,
    assistant: str,
    tools: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> int:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _connect(agent_name) as conn:
        (next_seq,) = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        cursor = conn.execute(
            "INSERT INTO memory(session_id, ts, seq, user, assistant, tools, usage)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                ts,
                next_seq,
                user,
                assistant,
                json.dumps(tools),
                json.dumps(usage or {}),
            ),
        )
        rowid = cursor.lastrowid
        if _fts_enabled(agent_name):
            conn.execute(
                "INSERT INTO memory_fts(rowid, user, assistant) VALUES (?,?,?)",
                (
                    rowid,
                    user,
                    assistant,
                ),
            )
        return next_seq


def _latest_rows(agent_name: str | None, session_id: str, last_seq: int) -> list[tuple[str, str]]:
    with _connect(agent_name) as conn:
        return [
            (user, assistant)
            for user, assistant in conn.execute(
                "SELECT user, assistant FROM memory"
                " WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
                (session_id, last_seq),
            )
        ]


def _get_summary(agent_name: str | None, session_id: str) -> tuple[str, int]:
    with _connect(agent_name) as conn:
        row = conn.execute(
            "SELECT summary, last_seq FROM summary_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return (row[0], row[1]) if row else ("", 0)


def _save_summary(agent_name: str | None, session_id: str, summary: str, last_seq: int) -> None:
    if not summary:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _connect(agent_name) as conn:
        conn.execute(
            """INSERT INTO summary_state(session_id, summary, last_seq, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id)
               DO UPDATE SET summary=excluded.summary,
                            last_seq=excluded.last_seq,
                            updated_at=excluded.updated_at""",
            (session_id, summary.strip(), last_seq, ts),
        )


def search(  # noqa: PLR0913 - explicit keyword surface
    agent_name: str | None,
    query: str,
    session_id: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    query = query.strip()
    limit = max(1, min(int(limit or 5), 25))
    rows: list[dict[str, Any]] = []
    with _connect(agent_name) as conn:
        if session_id and _fts_enabled(agent_name):
            rows = _fts_rows(conn, query, session_id, limit)
        elif _fts_enabled(agent_name):
            rows = _fts_rows(conn, query, None, limit)
        if not rows:
            rows = _like_rows(conn, query, session_id, limit)
    return rows


def _fts_rows(
    conn: sqlite3.Connection, query: str, session_id: str | None, limit: int
) -> list[dict[str, Any]]:
    """FTS5 MATCH, joined back to the memory rows for stable fields."""
    try:
        fts_query = '"' + query.replace('"', '""') + '"'
        if session_id:
            raw = conn.execute(
                """SELECT m.session_id, m.ts, m.seq, m.user, m.assistant
                   FROM memory_fts f
                   JOIN memory m ON m.id = f.rowid
                   WHERE memory_fts MATCH ? AND m.session_id = ?
                   ORDER BY rank LIMIT ?""",
                (fts_query, session_id, limit),
            ).fetchall()
        else:
            raw = conn.execute(
                """SELECT m.session_id, m.ts, m.seq, m.user, m.assistant
                   FROM memory_fts f
                   JOIN memory m ON m.id = f.rowid
                   WHERE memory_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.InterfaceError):
        return []
    return [
        {"session_id": s, "ts": ts, "seq": seq, "user": u, "assistant": a}
        for s, ts, seq, u, a in raw
    ]


def _like_rows(
    conn: sqlite3.Connection, query: str, session_id: str, limit: int
) -> list[dict[str, Any]]:
    like = f"%{query}%"
    if session_id:
        raw = conn.execute(
            "SELECT session_id, ts, seq, user, assistant FROM memory"
            " WHERE session_id = ? AND (user LIKE ? OR assistant LIKE ?)"
            " ORDER BY ts DESC LIMIT ?",
            (session_id, like, like, limit),
        ).fetchall()
    else:
        raw = conn.execute(
            "SELECT session_id, ts, seq, user, assistant FROM memory"
            " WHERE user LIKE ? OR assistant LIKE ?"
            " ORDER BY ts DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
    return [
        {"session_id": s, "ts": ts, "seq": seq, "user": u, "assistant": a}
        for s, ts, seq, u, a in raw
    ]


# -- LLM summary (module-attr access so tests can monkeypatch) -----------------


def _summarize(ctx: RunContext, prev: str, turns: list[tuple[str, str]]) -> str | None:
    cfg = ctx.current_model
    if cfg is None:
        return None
    override = os.getenv("MEMORY_SUMMARY_MODEL")
    if override:
        cfg = cfg.with_updates(model=override)
    client = _llm.OpenAICompatClient(cfg)
    turns_text = "\n\n".join(f"user: {u}\nassistant: {a}" for u, a in turns)
    prompt = (
        f"Previous summary:\n{prev or '(none)'}\n\nNew turns:\n{turns_text}\n\nUpdated summary:"
    )
    completion = client.chat(
        [
            {"role": "system", "content": _SUMMARIZER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    text = (completion.content or "").strip()
    return text or None


def _refresh_summary(ctx: RunContext, agent_name: str | None, session_id: str) -> None:
    """Fold new turns into the rolling summary. Fail-soft: never raises."""
    prev, last_seq = _get_summary(agent_name, session_id)
    new = _latest_rows(agent_name, session_id, last_seq)
    if not new:
        return
    prev = (prev or "")[:_SUMMARY_MAX_CHARS]
    try:
        text = _summarize(ctx, prev, new)
    except Exception:
        return  # fail-soft: keep the prior summary, leave the turn alone
    if text:
        # seq is per-session: the rows just folded run last_seq+1 .. last_seq+len(new)
        max_seq = last_seq + len(new)
        _save_summary(agent_name, session_id, text, max_seq)


# -- tool ---------------------------------------------------------------------


def memory_recall(query: str, session_id: str = "", limit: int = 5) -> str:
    """Search this agent's persistent memory across sessions.

    Args:
        query: a keyword phrase to find (quoted in FTS5; e.g. "deploy plan").
        session_id: optional; restrict results to one session (default: all).
        limit: max rows to return (1-25, default 5).

    Returns a JSON array of matching turns: [{session_id, ts, seq, user, assistant}].
    """
    agent_name = _TOOL_SCOPE.get("agent_name")
    results = search(agent_name, query, session_id=session_id, limit=limit)
    return json.dumps(results, ensure_ascii=False)


# -- hooks --------------------------------------------------------------------


def _set_tool_scope(ctx: RunContext) -> None:
    _TOOL_SCOPE["agent_name"] = ctx.agent_name
    _TOOL_SCOPE["session_id"] = ctx.session_id


def before_llm(ctx: RunContext) -> None:
    """Inject the session's rolling summary + capture the turn's user message.

    The summary goes through the canonical ``ctx.context`` channel (folded into
    the single system message after this hook, per the composition rules) and is
    re-read from the store each round, so it is non-sticky and never accumulates.
    """
    _set_tool_scope(ctx)
    if ctx.session_id is None:
        return
    try:
        summary, _seq = _get_summary(ctx.agent_name, ctx.session_id)
    except sqlite3.Error:
        summary = ""
    if summary:
        block = f"\n[Memory for this session]\n{summary.strip()}"
        ctx.context = f"{ctx.context}\n{block}".strip() if ctx.context else block.strip()
    # First user message of the composed prompt = this turn's user text.
    for message in reversed(ctx.messages or []):
        if message.get("role") == "user":
            ctx.extras["_mem_user"] = message.get("content") or ""
            break
    ctx.extras.setdefault("_mem_tools", [])


def after_tool_call(ctx: RunContext) -> None:
    """Record this tool call+result for the turn being committed."""
    if ctx.tool_call is None or ctx.tool_result is None:
        return
    bucket = dict(ctx.tool_call)
    bucket["result"] = str(ctx.tool_result)
    ctx.extras.setdefault("_mem_tools", []).append(bucket)


def ready_session(ctx: RunContext) -> None:
    """Fires on (re)establishment: publish the acting agent for the tool."""
    _set_tool_scope(ctx)
    when = os.getenv("MEMORY_SUMMARY_WHEN", "finish")
    if when in ("ready", "both") and ctx.session_id is not None:
        with contextlib.suppress(Exception):  # fail-soft at re-establishment too
            _refresh_summary(ctx, ctx.agent_name, ctx.session_id)


def finish(ctx: RunContext) -> None:
    """Commit the finished turn durably, then fold it into the rolling summary.

    ``finish`` is the last hook of the turn, so ``ctx.final_answer`` is final —
    the right point to persist (a ``prepare``/``finish`` rewrite is included).
    Summary refresh is fail-soft: any error keeps the prior summary and the
    turn's answer is never disturbed.
    """
    user = ctx.extras.pop("_mem_user", None)
    tools = ctx.extras.pop("_mem_tools", [])
    if ctx.session_id is None or user is None:
        return
    when = os.getenv("MEMORY_SUMMARY_WHEN", "finish")
    try:
        _commit_turn(
            ctx.agent_name,
            ctx.session_id,
            str(user),
            str(ctx.final_answer or ""),
            tools,
            dict(ctx.turn_usage or {}),
        )
    except sqlite3.Error:
        return  # a broken store must never fail the user's turn
    if when in ("finish", "both"):
        with contextlib.suppress(Exception):  # fail-soft: keep the prior summary
            _refresh_summary(ctx, ctx.agent_name, ctx.session_id)


def make() -> Hooks:
    from agent_seed.tools import registry as default_registry

    # Idempotent: registration is necessary but not sufficient (the strict
    # whitelist is the real gate); a hook folder that never whitelists the tool
    # simply never offers or calls it.
    if not default_registry.has("memory_recall"):
        default_registry.register(memory_recall)
    return Hooks(
        before_llm=before_llm,
        after_tool_call=after_tool_call,
        ready_session=ready_session,
        finish=finish,
    )
