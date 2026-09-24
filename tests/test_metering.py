"""Tests for OpenAI-compatible token accounting (``serving/metering.py``).

Pure byte-level tests — no server, no event loop, no upstream. The contract
being pinned down is the one the invoice depends on:

* usage is read from the engine's own numbers wherever they exist;
* the injected ``include_usage`` chunk is suppressed so the client's stream is
  byte-identical to what it would have received without metering;
* when nothing authoritative is available the counts are flagged inexact
  rather than silently billed as real.
"""

from __future__ import annotations

import json

from repercep.serving.metering import (
    ResponseMeter,
    extract_usage,
    plan_request,
)


def _sse(*objs: object) -> bytes:
    out = b""
    for obj in objs:
        payload = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        out += b"data: " + payload + b"\n\n"
    return out


def _drive(meter: ResponseMeter, data: bytes, *, chunk: int = 4096) -> bytes:
    """Feed data through a meter in fixed-size slices, collecting output."""
    out = b""
    for i in range(0, len(data), chunk):
        out += meter.feed(data[i : i + chunk])
    return out + meter.finish()


# ---------------------------------------------------------------------------
# plan_request
# ---------------------------------------------------------------------------


def test_non_streaming_body_untouched() -> None:
    raw = json.dumps({"model": "qwen", "prompt": "hi"}).encode()
    plan = plan_request(raw)
    assert plan.body == raw
    assert plan.model == "qwen"
    assert plan.streaming is False
    assert plan.injected_usage is False


def test_streaming_gets_include_usage_injected() -> None:
    raw = json.dumps({"model": "qwen", "stream": True}).encode()
    plan = plan_request(raw)
    assert plan.streaming is True
    assert plan.injected_usage is True
    assert json.loads(plan.body)["stream_options"] == {"include_usage": True}


def test_explicit_client_stream_options_are_honoured() -> None:
    """A client that made its own choice must not have it overridden."""
    for client_choice in (True, False):
        raw = json.dumps(
            {"model": "q", "stream": True, "stream_options": {"include_usage": client_choice}}
        ).encode()
        plan = plan_request(raw)
        assert plan.injected_usage is False
        assert plan.body == raw


def test_streaming_preserves_sibling_stream_options() -> None:
    raw = json.dumps({"model": "q", "stream": True, "stream_options": {"other": 1}}).encode()
    plan = plan_request(raw)
    assert plan.injected_usage is True
    assert json.loads(plan.body)["stream_options"] == {"other": 1, "include_usage": True}


def test_malformed_body_passes_through_untouched() -> None:
    """Request validation belongs to the upstream, not the proxy."""
    for raw in (b"not json", b"[1,2,3]", b""):
        plan = plan_request(raw)
        assert plan.body == raw
        assert plan.injected_usage is False


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_extract_usage_from_body() -> None:
    body = json.dumps({"usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode()
    usage = extract_usage(body)
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 7, 18)


def test_extract_usage_missing_or_malformed() -> None:
    assert extract_usage(b"{}") is None
    assert extract_usage(b"not json") is None
    assert extract_usage(json.dumps({"usage": "nope"}).encode()) is None


def test_non_streaming_meter_forwards_verbatim_and_counts() -> None:
    body = json.dumps(
        {"choices": [{"message": {"content": "hi"}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    ).encode()
    meter = ResponseMeter(streaming=False, suppress_usage_chunk=False)
    assert _drive(meter, body, chunk=7) == body
    assert meter.exact is True
    assert meter.usage.total_tokens == 8


def test_non_streaming_without_usage_is_inexact() -> None:
    body = b'{"choices":[]}'
    meter = ResponseMeter(streaming=False, suppress_usage_chunk=False)
    assert _drive(meter, body) == body
    assert meter.exact is False


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_injected_usage_chunk_is_suppressed() -> None:
    """The client must see exactly what it would have without metering."""
    content = _sse(
        {"choices": [{"delta": {"content": "he"}}]},
        {"choices": [{"delta": {"content": "llo"}}]},
    )
    usage_chunk = _sse({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}})
    done = b"data: [DONE]\n\n"

    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    out = _drive(meter, content + usage_chunk + done)

    assert out == content + done
    assert meter.exact is True
    assert meter.usage.prompt_tokens == 9
    assert meter.usage.completion_tokens == 2


def test_client_requested_usage_chunk_is_forwarded() -> None:
    """When the client asked for usage, it still gets the chunk."""
    stream = _sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1}},
    ) + b"data: [DONE]\n\n"

    meter = ResponseMeter(streaming=True, suppress_usage_chunk=False)
    assert _drive(meter, stream) == stream
    assert meter.exact is True
    assert meter.usage.prompt_tokens == 4


def test_usage_attached_to_a_content_chunk_is_never_dropped() -> None:
    """Suppression must not eat a chunk that also carries the answer."""
    stream = _sse(
        {"choices": [{"delta": {"content": "hi"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    ) + b"data: [DONE]\n\n"

    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    assert _drive(meter, stream) == stream
    assert meter.usage.prompt_tokens == 3


def test_stream_survives_arbitrary_chunk_boundaries() -> None:
    """SSE lines split across network reads must reassemble byte-exactly."""
    content = _sse(
        {"choices": [{"delta": {"content": "alpha"}}]},
        {"choices": [{"delta": {"content": "beta"}}]},
    )
    usage_chunk = _sse({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}})
    done = b"data: [DONE]\n\n"
    stream = content + usage_chunk + done

    for size in (1, 2, 3, 5, 13, 64, 1024):
        meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
        assert _drive(meter, stream, chunk=size) == content + done, f"chunk size {size}"
        assert meter.usage.prompt_tokens == 12


def test_estimate_when_upstream_sends_no_usage() -> None:
    """Flagged inexact rather than billed as real."""
    stream = _sse(
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [{"delta": {"content": "c"}}]},
    ) + b"data: [DONE]\n\n"

    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    assert _drive(meter, stream) == stream
    assert meter.exact is False
    assert meter.usage.completion_tokens == 3
    # Prompt tokens are unknowable from the response alone — not invented.
    assert meter.usage.prompt_tokens == 0


def test_legacy_completions_text_deltas_are_counted() -> None:
    stream = _sse({"choices": [{"text": "x"}]}, {"choices": [{"text": "y"}]})
    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    _drive(meter, stream)
    assert meter.usage.completion_tokens == 2


def test_done_and_non_data_lines_pass_through() -> None:
    stream = b": keep-alive\n\ndata: [DONE]\n\n"
    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    assert _drive(meter, stream) == stream


def test_unterminated_tail_is_still_forwarded() -> None:
    """A truncated upstream must not silently lose the client's bytes."""
    stream = b'data: {"choices":[{"delta":{"content":"hi"}}]}'
    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    assert _drive(meter, stream) == stream


def test_non_json_data_line_is_forwarded_unchanged() -> None:
    stream = b"data: <<garbage>>\n\ndata: [DONE]\n\n"
    meter = ResponseMeter(streaming=True, suppress_usage_chunk=True)
    assert _drive(meter, stream) == stream


def test_finish_is_idempotent() -> None:
    """The proxy calls finish() again on the disconnect path."""
    body = json.dumps({"usage": {"prompt_tokens": 2, "completion_tokens": 2}}).encode()
    meter = ResponseMeter(streaming=False, suppress_usage_chunk=False)
    meter.feed(body)
    assert meter.finish() == b""
    assert meter.finish() == b""
    assert meter.usage.total_tokens == 4
