"""Pure value semantics for the composer's CLI surface (agent_seed/composer/cli.py).

These helpers don't import ``rich`` or touch a TTY, so they're the right place to
pin *the rules* (quote-stripping, whitespace, de-duplication, role validation)
that the flag and wizard paths both rely on. L2 (the real terminal) verifies the
same values survive a pty round-trip; L3 (the real LLM request) verifies they
reach the outgoing request.
"""

from __future__ import annotations

import pytest

from agent_seed.composer.catalog import Tool
from agent_seed.composer.cli import (
    parse_examples_pairs,
    parse_tools,
    resolve_tools,
    unquote_once,
)
from agent_seed.composer.generator import ComposerError, validate_name

# -- unquote_once -------------------------------------------------------------
# The "quoted vs unquoted text must produce the same output" contract.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("'x'", "x"),  # single quotes
        ('"x"', "x"),  # double quotes
        ("'You are a tutor.'", "You are a tutor."),
        ('"You are a tutor."', "You are a tutor."),
        ("You are a tutor.", "You are a tutor."),  # unquoted: unchanged
        ('He said "x"', 'He said "x"'),  # outer chars differ
        ("He said 'x'", "He said 'x'"),  # outer chars differ
        ("'  padded  '", "padded"),  # unquote, then strip
        ('"  padded  "', "padded"),
        ("'x", "'x"),  # unbalanced: unchanged
        ('"x', '"x'),  # unbalanced: unchanged
        ("''", ""),  # two quotes, empty inside -> empty (then spec drops to None)
        ('""', ""),
        ("'a'b'", "a'b"),  # outer '...' pair stripped, inner ' kept
        ("'x'y'", "x'y"),  # outer '...' pair stripped, inner ' kept
        ("'\"x\"'", '"x"'),  # nested: outer single pair stripped, inner double kept
        ("'", "'"),  # single char, a quote: len 1 < 2, unchanged
        ('"', '"'),
        ("", ""),  # empty
        ("   ", ""),  # whitespace-only -> empty (rich also strips; belt-and-suspenders)
        ("  spaced value  ", "spaced value"),
        ("'spaced value'", "spaced value"),
    ],
)
def test_unquote_once(raw: str, expected: str) -> None:
    assert unquote_once(raw) == expected


def test_unquote_once_unicode_emoji() -> None:
    # The composed prompt must keep any unicode/emoji byte exactly (no mojibake),
    # whether or not it's wrapped in quotes.
    src = "Grüße — Team Alpha 💡"
    assert unquote_once(src) == src
    assert unquote_once(f"'{src}'") == src


# -- parse_tools --------------------------------------------------------------
# The contract: comma-separated, trimmed, empty tokens dropped, de-dup first-wins.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("   ", []),
        (",", []),
        (",,,", []),
        ("calculator", ["calculator"]),
        (" a ,  b ,c", ["a", "b", "c"]),
        ("a,b,c", ["a", "b", "c"]),
        ("a,a,b,a", ["a", "b"]),  # de-dup, preserve first occurrence
        ("a,b,a", ["a", "b"]),
        ("a,", ["a"]),  # trailing comma
        (",a", ["a"]),  # leading comma
        (" a , a ", ["a"]),  # two tokens that both strip to 'a' -> de-dup to one
        ("a,a", ["a"]),
        ("  a  b  ", ["a  b"]),  # whitespace inside a token is preserved
    ],
)
def test_parse_tools(raw: str, expected: list[str]) -> None:
    assert parse_tools(raw) == expected


# -- parse_examples_pairs -----------------------------------------------------
# The shared validator: the flag and the wizard must reject the same bad roles.


def test_parse_examples_pairs_basic() -> None:
    assert parse_examples_pairs(["user=hi"]) == [{"role": "user", "content": "hi"}]


def test_parse_examples_pairs_multiple() -> None:
    items = ["user=hi", "assistant=hello"]
    assert parse_examples_pairs(items) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_parse_examples_pairs_role_stripped_content_preserved() -> None:
    # The role is stripped (leading/trailing whitespace); the content is
    # whatever came after the first `=` -- verbatim, no further stripping.
    out = parse_examples_pairs(["  user =  hi  "])
    assert out == [{"role": "user", "content": "  hi  "}]


def test_parse_examples_pairs_equals_in_content() -> None:
    # Only the first `=` is the role/content separator; the content may contain `=`.
    assert parse_examples_pairs(["user=a=b"]) == [{"role": "user", "content": "a=b"}]


def test_parse_examples_pairs_pipe_in_content_when_single_item() -> None:
    # A single item may contain a `|` in its content. (The wizard's split by `|`
    # happens BEFORE calling this helper; here we treat the item atomically.)
    out = parse_examples_pairs(["user=hi|assistant=x"])
    assert out == [{"role": "user", "content": "hi|assistant=x"}]


def test_parse_examples_pairs_empty_content_allowed() -> None:
    assert parse_examples_pairs(["assistant="]) == [{"role": "assistant", "content": ""}]


def test_parse_examples_pairs_unicode() -> None:
    items = ["user=hi", "assistant=héllo — émoji 💡"]
    out = parse_examples_pairs(items)
    assert out[1]["content"] == "héllo — émoji 💡"


@pytest.mark.parametrize(
    "bad",
    [
        "bogus=hi",  # unknown role
        "USER=hi",  # wrong case
        "no-eq",  # no separator at all
        "=hi",  # empty role
    ],
)
def test_parse_examples_pairs_rejects_bad_roles(bad: str) -> None:
    with pytest.raises(ComposerError):
        parse_examples_pairs([bad])


def test_parse_examples_pairs_empty_content_is_allowed() -> None:
    # `user=` is valid (empty content is acceptable); only an empty *role* is bad.
    assert parse_examples_pairs(["user="]) == [{"role": "user", "content": ""}]


@pytest.mark.parametrize("bad_name", ["a..b", "_leads", "-", "UPPER", "..."])
def test_validate_name_rejects_gaps(bad_name: str) -> None:
    # The 9 existing cases in test_composer_generator.py plus the ones we
    # historically forgot: leading-underscore, dotted, dash, ellipsis.
    with pytest.raises(ComposerError):
        validate_name(bad_name)


@pytest.mark.parametrize("good_name", ["a", "x9_", "alpha-1", "a-b-c_d9"])
def test_validate_name_accepts_more(good_name: str) -> None:
    validate_name(good_name)


# -- resolve_tools ------------------------------------------------------------
# The shared "is this tool name in the catalog?" gate (flag AND wizard).


def _catalog(*names: str) -> list[Tool]:
    return [Tool(name=n, description=f"{n} tool.") for n in names]


def test_resolve_tools_accepts_all_known() -> None:
    available = _catalog("calculator", "double")
    assert resolve_tools(["calculator"], available) == ["calculator"]
    assert resolve_tools(["double"], available) == ["double"]
    assert resolve_tools(["calculator", "double"], available) == ["calculator", "double"]
    assert resolve_tools([], available) == []  # empty list is a no-op


def test_resolve_tools_rejects_unknown() -> None:
    available = _catalog("calculator")
    with pytest.raises(ComposerError, match="not_a_tool"):
        resolve_tools(["not_a_tool"], available)


def test_resolve_tools_message_lists_the_available_tools() -> None:
    # The user should see every available tool in the error so they can pick one.
    available = _catalog("calculator", "double")
    with pytest.raises(ComposerError) as exc_info:
        resolve_tools(["nope"], available)
    message = str(exc_info.value)
    assert "calculator" in message
    assert "double" in message
    assert "nope" in message  # and the bad name being rejected
