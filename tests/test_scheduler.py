"""Tests for the request scheduler at the Python-FFI boundary.

These mirror the Rust unit tests so we catch regressions that only show up
across the PyO3 boundary (string/dict marshaling, GIL release behavior,
exception type wiring). The Rust unit tests inside
``crates/repercep-scheduler/src/lib.rs`` cover the algorithmic correctness;
this file's job is to confirm the Python surface still says what we think.
"""

from __future__ import annotations

import threading
import time

import pytest

from repercep.runtime.scheduler import Scheduler, SchedulerError


def test_submit_and_next_roundtrip() -> None:
    s = Scheduler(capacity=4)
    s.submit("a", "normal", {"prompt": "hello"})
    got = s.next_blocking(timeout_ms=500)
    assert got is not None
    assert got["id"] == "a"
    assert got["priority"] == "normal"
    assert got["payload"] == {"prompt": "hello"}
    assert len(s) == 0


def test_priority_ordering_high_normal_low() -> None:
    s = Scheduler(capacity=16)
    s.submit("l1", "low", {})
    s.submit("n1", "normal", {})
    s.submit("h1", "high", {})
    s.submit("n2", "normal", {})
    s.submit("h2", "high", {})
    s.submit("l2", "low", {})
    order = [s.next_blocking(timeout_ms=500)["id"] for _ in range(6)]
    assert order == ["h1", "h2", "n1", "n2", "l1", "l2"]


def test_fifo_within_same_priority() -> None:
    s = Scheduler(capacity=8)
    for i in range(5):
        s.submit(f"r{i}", "normal", {"i": i})
    order = [s.next_blocking(timeout_ms=500)["id"] for _ in range(5)]
    assert order == ["r0", "r1", "r2", "r3", "r4"]


def test_cancel_skips_item() -> None:
    s = Scheduler(capacity=4)
    s.submit("a", "normal", {})
    s.submit("b", "normal", {})
    s.submit("c", "normal", {})
    s.cancel("b")
    assert len(s) == 2
    order = [s.next_blocking(timeout_ms=500)["id"] for _ in range(2)]
    assert order == ["a", "c"]
    assert len(s) == 0


def test_cancel_unknown_id_raises() -> None:
    s = Scheduler(capacity=4)
    with pytest.raises(SchedulerError):
        s.cancel("never-submitted")


def test_capacity_enforced() -> None:
    s = Scheduler(capacity=2)
    s.submit("a", "normal", {})
    s.submit("b", "high", {})
    with pytest.raises(SchedulerError):
        s.submit("c", "low", {})
    assert len(s) == 2
    assert s.capacity() == 2


def test_unknown_priority_raises() -> None:
    s = Scheduler(capacity=2)
    # An invalid priority maps to a plain RuntimeError from the FFI boundary,
    # not to SchedulerError (which is only for the four named scheduler-domain
    # variants). Pinning the narrower RuntimeError keeps the test honest.
    with pytest.raises(RuntimeError):
        s.submit("a", "urgent", {})


def test_shutdown_blocks_submit_and_drains_next() -> None:
    s = Scheduler(capacity=4)
    s.submit("a", "normal", {})
    s.submit("b", "high", {})
    s.shutdown()
    with pytest.raises(SchedulerError):
        s.submit("c", "low", {})
    # Drain
    r1 = s.next_blocking(timeout_ms=500)
    assert r1 is not None
    assert r1["id"] == "b"
    r2 = s.next_blocking(timeout_ms=500)
    assert r2 is not None
    assert r2["id"] == "a"
    # Once empty + shut down, next returns None.
    assert s.next_blocking(timeout_ms=500) is None


def test_next_blocking_timeout_returns_none_when_empty() -> None:
    s = Scheduler(capacity=4)
    t0 = time.monotonic()
    assert s.next_blocking(timeout_ms=50) is None
    elapsed_ms = (time.monotonic() - t0) * 1000
    # Allow generous slack: should be ~50ms, must be < 1s on any sane host.
    assert elapsed_ms < 1000


def test_next_blocking_wakes_on_submit_from_other_thread() -> None:
    s = Scheduler(capacity=4)
    received: list[dict[str, object]] = []

    def consumer() -> None:
        item = s.next_blocking(timeout_ms=2000)
        if item is not None:
            received.append(item)

    t = threading.Thread(target=consumer)
    t.start()
    # Give the consumer a chance to park inside next_blocking.
    time.sleep(0.05)
    s.submit("late", "high", {"hi": True})
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert len(received) == 1
    assert received[0]["id"] == "late"
    assert received[0]["payload"] == {"hi": True}


def test_len_and_capacity_reflect_state() -> None:
    s = Scheduler(capacity=3)
    assert len(s) == 0
    assert s.capacity() == 3
    s.submit("a", "normal", {})
    assert len(s) == 1
    s.submit("b", "high", {})
    assert len(s) == 2
    s.cancel("a")
    assert len(s) == 1


def test_cancel_after_shutdown_raises() -> None:
    s = Scheduler(capacity=2)
    s.submit("a", "normal", {})
    s.shutdown()
    with pytest.raises(SchedulerError):
        s.cancel("a")


def test_payload_round_trips_nested_json() -> None:
    s = Scheduler(capacity=2)
    payload = {
        "prompt": "a robot folding laundry",
        "params": {"num_frames": 121, "num_inference_steps": 35},
        "tags": ["cosmos", "mi300x"],
    }
    s.submit("x", "normal", payload)
    got = s.next_blocking(timeout_ms=500)
    assert got is not None
    assert got["payload"] == payload
