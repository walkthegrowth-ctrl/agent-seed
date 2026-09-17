from agent_seed.frontends.repl.app import sse_events_streaming


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
