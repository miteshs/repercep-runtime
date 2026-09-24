"""Per-request router.

Production uses the Rust ``repercep_router._native`` extension.  The Python
fallback below mirrors the v0 state machine closely enough for unit tests and
local serving development when the native wheel is not built.
"""

from __future__ import annotations

import asyncio
import importlib
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, ClassVar


class _PythonRouterError(RuntimeError):
    """Raised for router-domain errors."""


_SENTINEL = object()


@dataclass(slots=True)
class _Entry:
    state: str
    queue: asyncio.Queue[Any]
    subscribed: bool = False
    last_frame_index: int | None = None
    closed: bool = False


class _PythonFrameStream:
    def __init__(self, entry: _Entry) -> None:
        self._entry = entry

    def __aiter__(self) -> _PythonFrameStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._entry.closed and self._entry.queue.empty():
            raise StopAsyncIteration
        item = await self._entry.queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        assert isinstance(item, dict)
        return item


class _PythonRouter:
    """Request state machine plus bounded per-request frame queues."""

    _PRIORITIES: ClassVar[dict[str, str]] = {
        "low": "Low",
        "normal": "Normal",
        "high": "High",
    }

    def __init__(self, scheduler: Any, per_request_queue_depth: int = 64) -> None:
        if per_request_queue_depth < 1:
            raise ValueError("per_request_queue_depth must be >= 1")
        self._scheduler = scheduler
        self._queue_depth = per_request_queue_depth
        self._requests: dict[str, _Entry] = {}
        self._shutdown = False

    def accept(self, request_id: str, priority: str, payload: Any) -> None:
        self._ensure_open()
        canonical = self._parse_priority(priority)
        if request_id in self._requests:
            raise _PythonRouterError(f"duplicate request id: {request_id!r}")
        entry = _Entry(
            state="Scheduled",
            queue=asyncio.Queue(maxsize=self._queue_depth),
        )
        self._requests[request_id] = entry
        try:
            self._scheduler.submit(request_id, canonical)
        except Exception as exc:
            self._requests.pop(request_id, None)
            raise _PythonRouterError(f"scheduler error: {exc}") from exc
        del payload

    def subscribe(self, request_id: str) -> _PythonFrameStream:
        entry = self._entry(request_id)
        if entry.subscribed:
            raise _PythonRouterError(f"stream already subscribed: {request_id!r}")
        entry.subscribed = True
        return _PythonFrameStream(entry)

    async def push_frame(
        self, request_id: str, frame_index: int, payload: bytes, is_final: bool
    ) -> None:
        entry = self._entry(request_id)
        if entry.state in {"Complete", "Cancelled", "Failed"}:
            raise _PythonRouterError(
                f"invalid state transition: {entry.state} -> Streaming"
            )
        if (
            entry.last_frame_index is not None
            and frame_index <= entry.last_frame_index
        ):
            raise _PythonRouterError(
                f"out-of-order frame for {request_id!r}: got index {frame_index}, "
                f"expected > {entry.last_frame_index}"
            )
        if entry.queue.full():
            raise _PythonRouterError(
                f"backpressure: queue depth {self._queue_depth} for request {request_id!r}"
            )
        entry.last_frame_index = frame_index
        entry.state = "Complete" if is_final else "Streaming"
        if is_final:
            entry.closed = True
        entry.queue.put_nowait(
            {
                "request_id": request_id,
                "frame_index": frame_index,
                "payload": payload,
                "is_final": is_final,
            }
        )

    def cancel(self, request_id: str) -> None:
        entry = self._entry(request_id)
        if entry.state in {"Complete", "Cancelled", "Failed"}:
            raise _PythonRouterError(
                f"invalid state transition: {entry.state} -> Cancelled"
            )
        entry.state = "Cancelled"
        entry.closed = True
        with suppress(asyncio.QueueFull):
            entry.queue.put_nowait(_SENTINEL)
        with suppress(Exception):
            self._scheduler.cancel(request_id)

    def state(self, request_id: str) -> str | None:
        entry = self._requests.get(request_id)
        return None if entry is None else entry.state

    def shutdown(self) -> None:
        self._shutdown = True
        for entry in self._requests.values():
            entry.closed = True
            with suppress(asyncio.QueueFull):
                entry.queue.put_nowait(_SENTINEL)

    def _entry(self, request_id: str) -> _Entry:
        entry = self._requests.get(request_id)
        if entry is None:
            raise _PythonRouterError(f"unknown request: {request_id!r}")
        return entry

    def _ensure_open(self) -> None:
        if self._shutdown:
            raise _PythonRouterError("router is shutting down")

    def _parse_priority(self, priority: str) -> str:
        parsed = self._PRIORITIES.get(priority.lower())
        if parsed is None:
            raise _PythonRouterError(f"invalid priority: {priority}")
        return parsed


def _load_native_router() -> tuple[Any, Any, Any] | None:
    try:
        native = importlib.import_module("repercep_router._native")
    except Exception:
        return None
    return native.FrameStream, native.Router, native.RouterError


_native_router = _load_native_router()

FrameStream: Any
Router: Any
RouterError: Any

if _native_router is None:
    FrameStream = _PythonFrameStream
    Router = _PythonRouter
    RouterError = _PythonRouterError
else:
    FrameStream, Router, RouterError = _native_router


__all__ = ["FrameStream", "Router", "RouterError"]
