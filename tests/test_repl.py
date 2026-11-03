from agent_seed.frontends.repl.app import (
    Session,
    format_tokens,
    sse_events,
    sse_events_streaming,
    unquote,
)


def test_sse_events_streaming_yields_on_blank_line():
    lines = iter(
        [
            "event: token",
            'data: {"text": "a"}',
            "",
            "event: token",
            'data: {"text": "b"}',
            "",
            "event: done",
            'data: {"session_id": "s1"}',
            "",
        ]
    )
    events = list(sse_events_streaming(lines))
    assert events == [
        ("token", '{"text": "a"}'),
        ("token", '{"text": "b"}'),
        ("done", '{"session_id": "s1"}'),
    ]


def test_sse_events_streaming_multiline_data():
    # SSE data lines are stripped then joined with newlines.
    lines = iter(["event: tool", "data: line-one", "data: line-two", ""])
    events = list(sse_events_streaming(lines))
    assert events == [("tool", "line-one\nline-two")]


def test_sse_events_streaming_trailing_unterminated_event():
    lines = iter(["event: done", 'data: {"ok": true}'])
    assert list(sse_events_streaming(lines)) == [("done", '{"ok": true}')]


# -- sse_events (buffered variant, public API) -------------------------------


def test_sse_events_buffered_parses_full_body():
    raw = (
        "event: tool\n"
        'data: {"name":"calculator","arguments":{"expression":"1+1"},"result":"2.0"}\n'
        "\n"
        "event: token\n"
        'data: {"text": "2"}\n'
        "\n"
        "event: done\n"
        'data: {"session_id": "s1"}\n'
        "\n"
    )
    events = list(sse_events(raw))
    assert events == [
        (
            "tool",
            '{"name":"calculator","arguments":{"expression":"1+1"},"result":"2.0"}',
        ),
        ("token", '{"text": "2"}'),
        ("done", '{"session_id": "s1"}'),
    ]


def test_sse_events_buffered_trailing_unterminated_event():
    assert list(sse_events('event: done\ndata: {"ok": true}')) == [
        ("done", '{"ok": true}'),
    ]


# -- Session.reset (client-side rotation) ------------------------------------


def test_session_reset_posts_old_id_and_rotates_local():
    class _StubClient:
        def __init__(self):
            self.calls: list = []

        def post(self, url, json=None):
            self.calls.append((url, json))

        def close(self):
            pass

    session = Session("http://fake.invalid")
    stub = _StubClient()
    old_id = session.session_id
    session.client = stub
    session.reset()

    # the POST went out with the OLD id
    assert stub.calls == [("http://fake.invalid/reset", {"session_id": old_id})]
    # local id has rotated to a fresh UUID-hex
    assert session.session_id != old_id
    assert len(session.session_id) == 32
    session.close()


# -- unquote (peel one layer of matching quotes) ------------------------------


def test_unquote_peels_one_layer_of_matching_quotes():
    assert unquote('"qwen3.8:27b"') == "qwen3.8:27b"
    assert unquote("'single quoted'") == "single quoted"


def test_unquote_keeps_unbalanced_quotes_as_typed():
    # Only a *surrounding* pair is peeled; unbalanced quotes are left alone.
    assert unquote("plain") == "plain"
    assert unquote('"a') == '"a'  # leading quote, no closing -> unchanged
    assert unquote('a"') == 'a"'  # trailing quote, no leading -> unchanged
    assert unquote("'a") == "'a"  # single quote, unbalanced -> unchanged


def test_unquote_none_and_single_char():
    assert unquote(None) is None
    assert unquote('"') == '"'  # single char is not a pair
    assert unquote("") == ""


def test_unquote_peels_layer_even_with_inner_quotes():
    # First and last are a matching pair, so one outer layer is peeled
    # (the documented contract: "a single layer of matching quotes is peeled").
    assert unquote('"q"') == "q"
    assert unquote('"x y"') == "x y"


# -- Session model switching (payload + error surfacing) ----------------------


class _RecClient:
    """Records POST/GET calls and returns scripted responses."""

    def __init__(self, status=200, body=None):
        self.calls = []
        self.status = status
        self.body = body

    def post(self, url, json=None):
        self.calls.append((url, json))
        return _RecResp(self.status, self.body)

    def get(self, url, params=None):
        self.calls.append((url, params))
        return _RecResp(self.status, self.body)

    def close(self):
        pass


class _RecResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx2

            req = httpx2.Request("POST", "http://fake.invalid")
            resp = httpx2.Response(self.status_code, request=req, content=b"backend boom")
            raise httpx2.HTTPStatusError("backend error", request=req, response=resp)


def test_session_set_model_sends_session_and_model():
    session = Session("http://fake.invalid")
    rec = _RecClient(status=200, body={"ok": True})
    session.client = rec
    sid = session.session_id
    session.set_model("qwen3.8:27b")
    assert rec.calls == [
        ("http://fake.invalid/config", {"session_id": sid, "model": "qwen3.8:27b"})
    ]
    session.close()


def test_session_test_model_uses_test_endpoint_and_returns_body():
    session = Session("http://fake.invalid")
    rec = _RecClient(status=200, body={"ok": False, "error": "boom"})
    session.client = rec
    out = session.test_model("bad-model")
    assert rec.calls[0][0] == "http://fake.invalid/config/test"
    assert out == {"ok": False, "error": "boom"}
    session.close()


# -- format_tokens (REPL usage-line renderer) ---------------------------------


def test_format_tokens_none_and_empty():
    assert format_tokens(None) == "—"
    assert format_tokens({}) == "—"


def test_format_tokens_in_out_and_total():
    usage = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    assert format_tokens(usage) == "in=4 out=6 total=10"


def test_format_tokens_derives_total_when_missing():
    usage = {"prompt_tokens": 3, "completion_tokens": 5}
    assert format_tokens(usage) == "in=3 out=5 total=8"


def test_format_tokens_partial_counters_only():
    # Only prompt present: no in=/out= for the missing slot, no derived total.
    assert format_tokens({"prompt_tokens": 7}) == "in=7"
    # Only completion present.
    assert format_tokens({"completion_tokens": 9}) == "out=9"
    # Only total present.
    assert format_tokens({"total_tokens": 42}) == "total=42"


def test_format_tokens_bool_is_not_a_number():
    # `True` is an `int` in Python; it must not be treated as a token count.
    usage = {"prompt_tokens": True, "completion_tokens": False}
    assert format_tokens(usage) == "{'prompt_tokens': True, 'completion_tokens': False}"


def test_format_tokens_falls_back_to_str_for_non_numeric():
    assert format_tokens({"prompt_tokens": "lots"}) == "{'prompt_tokens': 'lots'}"


def test_format_tokens_prefers_explicit_total():
    # If both prompt+completion and an explicit total are present, use the
    # explicit one (do not re-derive).
    usage = {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 100,  # intentionally different from 3
    }
    assert format_tokens(usage) == "in=1 out=2 total=100"
