from agent_seed.frontends.repl.app import Session, sse_events, sse_events_streaming


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
