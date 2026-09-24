"""Request scheduler.

Production uses the Rust ``repercep_scheduler._native`` extension.  When the
native wheel has not been built, this module falls back to a small Python
implementation with the same v0 surface so unit tests and local development
do not require the Rust toolchain.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections import deque
from typing import Any


class _PythonSchedulerError(RuntimeError):
    """Raised for scheduler-domain errors."""


class _PythonScheduler:
    """Three-priority FIFO scheduler matching the PyO3 wrapper's API."""

    _PRIORITIES = ("high", "normal", "low")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._queues: dict[str, deque[dict[str, Any]]] = {
            priority: deque() for priority in self._PRIORITIES
        }
        self._known: set[str] = set()
        self._shutdown = False
        self._condition = threading.Condition()

    def submit(self, request_id: str, priority: str, payload: Any) -> None:
        parsed = priority.lower()
        if parsed not in self._queues:
            raise RuntimeError(f"invalid priority: {priority}")
        with self._condition:
            if self._shutdown:
                raise _PythonSchedulerError("scheduler is shutting down")
            if len(self) >= self._capacity:
                raise _PythonSchedulerError(f"queue is full: {self._capacity}")
            self._queues[parsed].append(
                {"id": request_id, "priority": parsed, "payload": payload}
            )
            self._known.add(request_id)
            self._condition.notify()

    def cancel(self, request_id: str) -> None:
        with self._condition:
            if self._shutdown:
                raise _PythonSchedulerError("scheduler is shutting down")
            if request_id not in self._known:
                raise _PythonSchedulerError(f"request not found: {request_id!r}")
            for queue in self._queues.values():
                for index, item in enumerate(queue):
                    if item["id"] == request_id:
                        del queue[index]
                        self._known.remove(request_id)
                        self._condition.notify()
                        return
            raise _PythonSchedulerError(f"request not found: {request_id!r}")

    def next_blocking(self, timeout_ms: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        with self._condition:
            while True:
                item = self._pop_next_locked()
                if item is not None:
                    return item
                if self._shutdown:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def _pop_next_locked(self) -> dict[str, Any] | None:
        for priority in self._PRIORITIES:
            queue = self._queues[priority]
            if queue:
                item = queue.popleft()
                self._known.discard(str(item["id"]))
                return item
        return None


def _load_native_scheduler() -> tuple[Any, Any] | None:
    try:
        native = importlib.import_module("repercep_scheduler._native")
    except Exception:
        return None
    return native.Scheduler, native.SchedulerError


_native_scheduler = _load_native_scheduler()

Scheduler: Any
SchedulerError: Any

if _native_scheduler is None:
    Scheduler = _PythonScheduler
    SchedulerError = _PythonSchedulerError
else:
    Scheduler, SchedulerError = _native_scheduler


__all__ = ["Scheduler", "SchedulerError"]
