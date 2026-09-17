from agent_seed.llm import Completion, fold_stream, parse_sse_line


def test_parse_sse_line():
    assert parse_sse_line(b"") is None
    assert parse_sse_line(b'{"a":1}') is None
    assert parse_sse_line(b"data: [DONE]") is None
    assert parse_sse_line(b'data: {"a":1}') == {"a": 1}
    assert parse_sse_line(b"  data:   \n") is None


def test_parse_sse_line_str():
    # httpx iter_lines() yields str; this is the runtime code path.
    assert parse_sse_line("") is None
    assert parse_sse_line('{"a":1}') is None
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line('data: {"a":1}') == {"a": 1}
    assert parse_sse_line("  data:   \n") is None


def test_fold_stream_content_and_usage():
    events = [
        {"choices": [{"delta": {"content": "he"}}], "usage": None},
        {"choices": [{"delta": {"content": "llo"}}]},
        {"choices": [], "usage": {"total_tokens": 7}},
    ]
    result = fold_stream(events)
    assert result.content == "hello"
    assert result.usage == {"total_tokens": 7}
    assert result.tool_calls == []


def test_fold_stream_tool_calls_split_across_chunks():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "calculator", "arguments": '{"exp'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'ression": "1+1"}'}}]
                    }
                }
            ]
        },
    ]
    result = fold_stream(events)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "calculator"
    assert call.arguments == {"expression": "1+1"}


def test_fold_stream_malformed_arguments_captured_raw():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c",
                                "function": {"name": "x", "arguments": "{not json"},
                            }
                        ]
                    }
                }
            ]
        }
    ]
    result = fold_stream(events)
    assert result.tool_calls[0].arguments == {"_raw": "{not json"}


def test_completion_default():
    completion = Completion()
    assert completion.content == ""
    assert completion.tool_calls == []
    assert completion.usage is None
