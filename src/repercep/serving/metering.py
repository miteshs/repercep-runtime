"""Token accounting for OpenAI-compatible traffic.

Everything here is pure and synchronous so it can be tested without a server,
an event loop, or a GPU — the proxy just feeds it bytes.

The problem this solves. Billing needs exact token counts; the gateway is a
passthrough that must not change what the client sees. Those pull against
each other on the streaming path:

* **Non-streaming** is easy. The upstream's JSON body already carries
  ``usage``. We accumulate the bytes we are forwarding anyway and read it.
* **Streaming (SSE)** is not. An OpenAI-compatible server emits ``usage``
  only when the request set ``stream_options.include_usage``. Most clients
  don't, so there is nothing to meter.

Three options existed and the trade is worth stating, because the choice is
visible in this file's complexity:

1. Estimate from chunk counts. Simple, and wrong by a few percent — which is
   fine for a dashboard and not fine for an invoice.
2. Inject ``include_usage`` and forward the extra chunk. Exact, but the
   client now receives a trailing chunk it did not ask for. Standard OpenAI
   behaviour, yet still a silent change to a response shape.
3. **Inject ``include_usage``, then suppress the one chunk the injection
   caused.** Exact billing, byte-identical stream from the client's side.

We do (3), and fall back to (1) — flagged ``exact=False`` in the ledger — when
the upstream sends no usage anyway. The cost is that the streaming path parses
SSE line-by-line instead of forwarding opaque bytes; :class:`ResponseMeter`
forwards each line the instant it completes, so time-to-first-token is
unaffected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

# Stop accumulating a non-streamed body past this. A response larger than this
# is not a chat completion, and an unbounded buffer in a proxy is a memory DoS.
_MAX_ACCUMULATE_BYTES: Final = 8 * 1024 * 1024

_SSE_DATA_PREFIX: Final = b"data:"
_SSE_DONE: Final = b"[DONE]"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Prompt/completion token counts for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """What the proxy learned from (and did to) the request body."""

    body: bytes
    """The body to send upstream — rewritten iff :attr:`injected_usage`."""
    model: str
    streaming: bool
    injected_usage: bool
    """True when we added ``stream_options.include_usage``; the response
    chunk it produces must be suppressed so the client sees no difference."""


def plan_request(raw: bytes) -> RequestPlan:
    """Inspect an OpenAI request body and prepare it for metering.

    A body that is not valid JSON, or not an object, is passed through
    untouched — the upstream owns request validation, and a proxy that
    rejects what the upstream would have accepted is a bug.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return RequestPlan(body=raw, model="", streaming=False, injected_usage=False)
    if not isinstance(parsed, dict):
        return RequestPlan(body=raw, model="", streaming=False, injected_usage=False)

    model = parsed.get("model")
    model_name = model if isinstance(model, str) else ""
    streaming = bool(parsed.get("stream", False))
    if not streaming:
        return RequestPlan(body=raw, model=model_name, streaming=False, injected_usage=False)

    opts = parsed.get("stream_options")
    if isinstance(opts, dict) and "include_usage" in opts:
        # The client made its own choice. Honour it and suppress nothing —
        # if they asked for usage they expect the chunk, and if they
        # explicitly asked for none we bill from an estimate rather than
        # override an explicit instruction.
        return RequestPlan(body=raw, model=model_name, streaming=True, injected_usage=False)

    merged = dict(opts) if isinstance(opts, dict) else {}
    merged["include_usage"] = True
    parsed["stream_options"] = merged
    return RequestPlan(
        body=json.dumps(parsed).encode("utf-8"),
        model=model_name,
        streaming=True,
        injected_usage=True,
    )


def _usage_from_obj(obj: Any) -> TokenUsage | None:
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    return TokenUsage(
        prompt_tokens=prompt if isinstance(prompt, int) else 0,
        completion_tokens=completion if isinstance(completion, int) else 0,
    )


def extract_usage(payload: bytes) -> TokenUsage | None:
    """Read ``usage`` out of a complete non-streamed response body."""
    try:
        return _usage_from_obj(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class ResponseMeter:
    """Tees an upstream response: forwards bytes, counts tokens.

    Feed it every chunk from the upstream and forward what :meth:`feed`
    returns — which is the input verbatim, except for the single injected
    usage chunk on the streaming path. Call :meth:`finish` at end of stream to
    flush, then read :attr:`usage` and :attr:`exact`.
    """

    def __init__(self, *, streaming: bool, suppress_usage_chunk: bool) -> None:
        self._streaming = streaming
        self._suppress = suppress_usage_chunk
        self._buffer = b""
        self._body = bytearray()
        self._truncated = False
        self._usage: TokenUsage | None = None
        self._delta_chunks = 0
        # Set when a suppressed data line was dropped, so the blank line that
        # terminates that SSE event is dropped with it rather than leaking
        # through as a stray event separator.
        self._drop_next_blank = False

    @property
    def usage(self) -> TokenUsage:
        """Measured usage, or the chunk-count estimate when none was sent."""
        if self._usage is not None:
            return self._usage
        # Fallback: one content delta ≈ one completion token. Prompt tokens
        # are unknowable from the response alone, so they stay 0 rather than
        # being invented.
        return TokenUsage(prompt_tokens=0, completion_tokens=self._delta_chunks)

    @property
    def exact(self) -> bool:
        """False when :attr:`usage` is the estimate, not the engine's count."""
        return self._usage is not None

    # -- byte plumbing ------------------------------------------------------

    def feed(self, chunk: bytes) -> bytes:
        """Consume an upstream chunk; return the bytes to forward onward."""
        if not self._streaming:
            if not self._truncated:
                if len(self._body) + len(chunk) > _MAX_ACCUMULATE_BYTES:
                    self._truncated = True
                    self._body.clear()
                else:
                    self._body.extend(chunk)
            return chunk
        return self._feed_sse(chunk)

    def finish(self) -> bytes:
        """Flush any buffered tail. Call once at end of stream."""
        if not self._streaming:
            if not self._truncated:
                self._usage = extract_usage(bytes(self._body))
            return b""
        tail, self._buffer = self._buffer, b""
        if not tail:
            return b""
        # An unterminated final line cannot be a well-formed SSE event, but it
        # is still the upstream's bytes: inspect it for usage, forward it
        # regardless rather than silently truncating the client's stream.
        self._inspect_sse_line(tail)
        return tail

    def _feed_sse(self, chunk: bytes) -> bytes:
        self._buffer += chunk
        out = bytearray()
        while True:
            idx = self._buffer.find(b"\n")
            if idx == -1:
                break
            line = self._buffer[: idx + 1]
            self._buffer = self._buffer[idx + 1 :]
            if self._forward_line(line):
                out += line
        return bytes(out)

    def _forward_line(self, line: bytes) -> bool:
        """Inspect one complete line (newline included). True to forward it."""
        stripped = line.strip()
        if not stripped:
            if self._drop_next_blank:
                self._drop_next_blank = False
                return False
            return True
        self._drop_next_blank = False
        suppress = self._inspect_sse_line(line)
        if suppress:
            self._drop_next_blank = True
            return False
        return True

    def _inspect_sse_line(self, line: bytes) -> bool:
        """Record usage from a data line. True if the line must be suppressed."""
        stripped = line.strip()
        if not stripped.startswith(_SSE_DATA_PREFIX):
            return False
        payload = stripped[len(_SSE_DATA_PREFIX) :].strip()
        if not payload or payload == _SSE_DONE:
            return False
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(obj, dict):
            return False

        found = _usage_from_obj(obj)
        if found is not None:
            self._usage = found
            # The chunk carrying usage is suppressible only when we caused it,
            # and only when it is the empty-choices terminal chunk. An engine
            # that attaches usage to a chunk that also carries content must
            # never have that content dropped.
            return bool(self._suppress and not obj.get("choices"))

        if _has_content_delta(obj):
            self._delta_chunks += 1
        return False


def _has_content_delta(obj: dict[str, Any]) -> bool:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return True
        # /v1/completions uses "text" rather than a chat "delta".
        if choice.get("text"):
            return True
    return False


__all__ = ["RequestPlan", "ResponseMeter", "TokenUsage", "extract_usage", "plan_request"]
