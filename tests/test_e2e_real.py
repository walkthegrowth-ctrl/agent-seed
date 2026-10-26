"""End-to-end smoke against a *real* LLM.

Runs only when **both** are true, reusing the suite's own existing facility:

* ``LLM_MODEL`` / ``LLM_BASE_URL`` are configured (``LLMConfig.from_env()``), AND
* the endpoint is responsive to a minimal ``max_tokens=1`` ping
  (``OpenAICompatClient(cfg).chat(..., max_tokens=1)``).

Both checks are the same ones the rest of the suite's probes use; this module
just wraps them in an auto-skip so the battery stays green (and network-free)
when no real LLM is reachable, and exercises it when there is one. This is the
default "real" mode — a user with a live LLM gets a real LLM smoke for free;
a user without one gets the (skipped) smoke plus the deterministic
``test_e2e_repl.py`` coverage.

The tests only assert *shape* (event order, render markers), never *content*,
because a real LLM is non-deterministic. Everything else (the full REPL +
backend + tool + hook path) is covered deterministically by
``test_e2e_repl.py`` against a scripted fake LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.repl_driver import ReplDriver, spawn_server, wait_until_healthy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _real_llm_config() -> tuple[Any, Any]:
    """Return ``('cfg', LLMConfig)`` or ``('skip', reason_str_or_None)``."""
    from agent_seed.llm import LLMConfig, OpenAICompatClient

    try:
        cfg = LLMConfig.from_env()
    except Exception as exc:  # noqa: BLE001 - probe boundary
        return "skip", f"LLM_CONFIG not set or invalid: {exc}"
    try:
        OpenAICompatClient(cfg).chat([{"role": "user", "content": "ping"}], max_tokens=1)
    except Exception as exc:  # noqa: BLE001 - probe boundary
        return "skip", f"LLM at {cfg.base_url} not responsive: {exc}"
    return "cfg", cfg


@pytest.fixture(scope="module")
def real_llm() -> Iterator[Any]:
    kind, value = _real_llm_config()
    if kind == "skip":
        pytest.skip(f"real-LLM smoke skipped: {value}")
    yield value


@pytest.fixture(scope="module")
def real_backend(real_llm: Any) -> Iterator[dict[str, Any]]:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = {
        "LLM_MODEL": real_llm.model,
        "LLM_API_KEY": real_llm.api_key or "unused",
        "LLM_BASE_URL": real_llm.base_url,
        "AGENT_SERVER_HOST": "127.0.0.1",
        "AGENT_SERVER_PORT": str(port),
    }
    proc = spawn_server(env, str(PROJECT_ROOT))
    url = f"http://127.0.0.1:{port}"
    try:
        wait_until_healthy(url, timeout=25)
    except TimeoutError as exc:
        proc.terminate()
        out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        raise TimeoutError(f"backend failed to start:\n{out}\n{err}") from exc
    yield {"proc": proc, "url": url}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


@pytest.fixture
def repl(real_backend: dict[str, Any]) -> Iterator[ReplDriver]:
    driver = ReplDriver(real_backend["url"], cwd=str(PROJECT_ROOT), project_root=str(PROJECT_ROOT))
    try:
        driver.wait_ready(timeout=30)
        yield driver
    finally:
        try:
            driver.close(timeout=15)
        except Exception:
            driver._kill()


def test_help_command(repl: ReplDriver) -> None:
    out = repl.send("/help", timeout=30)
    for command in ("/help", "/reset", "/agent", "/tools", "/model"):
        assert command in out, f"/help output missing {command!r}:\n{out[:1200]}"


def test_chat_streams_a_token_and_ends(repl: ReplDriver) -> None:
    # Real LLM: we only assert the *shape* — tokens were streamed and the REPL
    # returned a prompt. Content is non-deterministic.
    out = repl.send("Reply with the single word: pong.", timeout=60)
    assert "agent>" in out, f"the REPL should render an agent turn:\n{out[:1200]}"
    assert "you>" in out, f"the REPL should return to the prompt:\n{out[:1200]}"


def test_model_shows_the_real_model(repl: ReplDriver, real_llm: Any) -> None:
    out = repl.send("/model", timeout=30)
    assert real_llm.model in out, (
        f"/model should show the configured model {real_llm.model!r}:\n{out[:1200]}"
    )
