"""Cross-language tests for the per-request router.

Mirrors the Rust unit tests in `crates/repercep-router/src/lib.rs` at the Python
boundary. The scheduler is mocked with a tiny duck-typed class — the router
crate has no Cargo dep on `repercep-scheduler` and the Python wrapper similarly
accepts any object exposing `submit(id, priority)` and `cancel(id)`.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from repercep.runtime.router import FrameStream, Router, RouterError

# ---------------------------------------------------------------------------
# Mock scheduler (duck-typed; satisfies SchedulerHandle on the Rust side)
# ---------------------------------------------------------------------------


class _MockScheduler:
    """Records submit/cancel calls; optionally fails submit for negative tests."""

    def __init__(self, fail_submit: bool = False) -> None:
        self.submits: list[tuple[str, str]] = []
        self.cancels: list[str] = []
        self._fail_submit = fail_submit

    def submit(self, request_id: str, priority: str) -> None:
        self.submits.append((request_id, priority))
        if self._fail_submit:
            raise RuntimeError("mock scheduler forced failure")

    def cancel(self, request_id: str) -> None:
        self.cancels.append(request_id)


def _drain_stream(stream: FrameStream) -> list[dict[str, Any]]:
    """Synchronously drain an async frame stream to a list."""

    async def _go() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for frame in stream:
            out.append(frame)
        return out

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_accept_transitions_to_scheduled() -> None:
    sched = _MockScheduler()
    router = Router(sched)
    router.accept("a", "Normal", {"prompt": "hello"})
    assert router.state("a") == "Scheduled"
    assert sched.submits == [("a", "Normal")]


def test_duplicate_accept_raises() -> None:
    router = Router(_MockScheduler())
    router.accept("dup", "Normal", {})
    with pytest.raises(RouterError):
        router.accept("dup", "Normal", {})


def test_invalid_priority_raises() -> None:
    router = Router(_MockScheduler())
    with pytest.raises(RouterError):
        router.accept("a", "Critical", {})


def test_scheduler_failure_rolls_back() -> None:
    sched = _MockScheduler(fail_submit=True)
    router = Router(sched)
    with pytest.raises(RouterError):
        router.accept("a", "Normal", {})
    assert router.state("a") is None  # rolled back


def test_state_returns_none_for_unknown() -> None:
    router = Router(_MockScheduler())
    assert router.state("ghost") is None


def test_push_frames_in_order_and_subscribe_drains() -> None:
    router = Router(_MockScheduler(), per_request_queue_depth=8)
    router.accept("a", "Normal", {})
    stream = router.subscribe("a")

    async def push_then_drain() -> list[dict[str, Any]]:
        await router.push_frame("a", 0, b"frame-0", False)
        await router.push_frame("a", 1, b"frame-1", False)
        await router.push_frame("a", 2, b"frame-2", True)
        out: list[dict[str, Any]] = []
        async for frame in stream:
            out.append(frame)
        return out

    frames = asyncio.run(push_then_drain())
    assert [f["frame_index"] for f in frames] == [0, 1, 2]
    assert frames[-1]["is_final"] is True
    assert frames[0]["payload"] == b"frame-0"
    assert router.state("a") == "Complete"


def test_first_frame_transitions_to_streaming() -> None:
    router = Router(_MockScheduler(), per_request_queue_depth=4)
    router.accept("a", "Normal", {})

    async def _push() -> None:
        await router.push_frame("a", 0, b"x", False)

    asyncio.run(_push())
    assert router.state("a") == "Streaming"


def test_out_of_order_push_raises() -> None:
    router = Router(_MockScheduler(), per_request_queue_depth=4)
    router.accept("a", "Normal", {})

    async def _go() -> None:
        await router.push_frame("a", 0, b"a", False)
        await router.push_frame("a", 5, b"b", False)
        with pytest.raises(RouterError):
            await router.push_frame("a", 3, b"c", False)
        with pytest.raises(RouterError):
            await router.push_frame("a", 5, b"d", False)

    asyncio.run(_go())


def test_push_unknown_request_raises() -> None:
    router = Router(_MockScheduler())

    async def _go() -> None:
        with pytest.raises(RouterError):
            await router.push_frame("ghost", 0, b"x", False)

    asyncio.run(_go())


def test_push_after_complete_raises() -> None:
    router = Router(_MockScheduler(), per_request_queue_depth=4)
    router.accept("a", "Normal", {})

    async def _go() -> None:
        await router.push_frame("a", 0, b"x", True)
        with pytest.raises(RouterError):
            await router.push_frame("a", 1, b"y", False)

    asyncio.run(_go())


def test_backpressure_when_channel_full() -> None:
    router = Router(_MockScheduler(), per_request_queue_depth=2)
    router.accept("a", "Normal", {})

    # No subscriber → channel fills.
    async def _go() -> None:
        await router.push_frame("a", 0, b"x", False)
        await router.push_frame("a", 1, b"y", False)
        with pytest.raises(RouterError):
            await router.push_frame("a", 2, b"z", False)

    asyncio.run(_go())


def test_cancel_transitions_state_and_closes_stream() -> None:
    sched = _MockScheduler()
    router = Router(sched, per_request_queue_depth=4)
    router.accept("a", "Normal", {})
    stream = router.subscribe("a")

    async def _go() -> list[dict[str, Any]]:
        await router.push_frame("a", 0, b"x", False)
        # Drain the one frame first so it doesn't race with cancel.
        first = await stream.__anext__()
        assert first["frame_index"] == 0
        router.cancel("a")
        # Stream should now end.
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return [first]

    drained = asyncio.run(_go())
    assert drained[0]["payload"] == b"x"
    assert router.state("a") == "Cancelled"
    assert sched.cancels == ["a"]


def test_cancel_unknown_raises() -> None:
    router = Router(_MockScheduler())
    with pytest.raises(RouterError):
        router.cancel("nope")


def test_cancel_after_terminal_raises() -> None:
    router = Router(_MockScheduler())
    router.accept("a", "Normal", {})
    router.cancel("a")
    with pytest.raises(RouterError):
        router.cancel("a")


def test_subscribe_can_only_be_taken_once() -> None:
    router = Router(_MockScheduler())
    router.accept("a", "Normal", {})
    _ = router.subscribe("a")
    with pytest.raises(RouterError):
        router.subscribe("a")


def test_shutdown_blocks_new_ops() -> None:
    router = Router(_MockScheduler())
    router.shutdown()
    with pytest.raises(RouterError):
        router.accept("a", "Normal", {})


def test_state_strings_match_rust_variant_names() -> None:
    """Sanity-check the cross-language state vocabulary."""
    router = Router(_MockScheduler(), per_request_queue_depth=4)
    router.accept("a", "Normal", {})
    assert router.state("a") == "Scheduled"

    async def _go() -> None:
        await router.push_frame("a", 0, b"x", False)

    asyncio.run(_go())
    assert router.state("a") == "Streaming"

    async def _final() -> None:
        await router.push_frame("a", 1, b"y", True)

    # Need to drain to make room (queue depth 4 is plenty though).
    asyncio.run(_final())
    assert router.state("a") == "Complete"
