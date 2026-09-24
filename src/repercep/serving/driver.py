"""Engine driver — pulls scheduled requests and pushes frames back through the router.

This is the substance of the Stage-4 v2 path. The FastAPI handler accepts an
HTTP request, mints an id, stages the payload, calls ``Router.accept`` (which
in turn calls the real ``Scheduler.submit`` via :class:`_SchedulerAdapter`),
then streams frames to the wire by ``async for``-ing the router's
:class:`FrameStream`. The driver is the OTHER half: a single background
thread that loops ``scheduler.next_blocking``, runs the engine for each
popped request, and pushes the resulting frames back through the router.

## Why a thread, not an asyncio task

``Scheduler.next_blocking`` releases the GIL inside Rust (the crate uses
``py.allow_threads``), so the FastAPI event loop stays responsive while the
driver parks waiting for work. ``engine.generate`` is a synchronous
Python iterator that may take tens of seconds per request — we never want
that on the event loop. Putting the driver in a dedicated thread keeps the
boundary clean and lets us call ``run_coroutine_threadsafe`` to bounce
the async ``router.push_frame`` calls back onto the loop where the router's
tokio runtime is reachable.

## Why an adapter between router and scheduler

The router's PyO3 :class:`SchedulerHandle` duck-types ``submit(id, priority)``
— two args. The real :class:`repercep.runtime.scheduler.Scheduler` exposes
``submit(id, priority, payload)`` — three args. The router-side payload is
serialized but never forwarded to the scheduler in the current crate
boundary. We don't want to touch either crate, so :class:`_SchedulerAdapter`
bridges the two signatures: the HTTP handler stages the payload before
calling ``router.accept``, the router calls ``adapter.submit(id, priority)``,
and the adapter forwards to the real scheduler with the staged payload.

## Frame encoding

The router's ``push_frame`` takes opaque ``bytes``. We encode each frame as a
single NDJSON line (one frame per line, no trailing newline — the streaming
handler adds the newline). The line is the wire format: the handler can
write the bytes straight to the response body, no extra serialization.
Fields mirror :class:`repercep.runtime.types.FrameChunk` so a v1→v2 migration
is a URL swap; ``is_final`` and ``pixels_b64`` are the v2 additions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from repercep.runtime.types import GenerationRequest

if TYPE_CHECKING:
    from repercep.runtime.engine import WorldModelEngine
    from repercep.runtime.router import Router
    from repercep.runtime.scheduler import Scheduler

_LOG = logging.getLogger(__name__)


# How long the driver blocks inside next_blocking before re-checking the
# stop flag. Short enough that shutdown drains cleanly; long enough that
# an idle driver isn't spinning the CPU.
_NEXT_TIMEOUT_MS = 200

# Maximum time we'll wait for a single push_frame to land on the event
# loop. The router buffers a small number of frames per request; if a slow
# consumer makes us wait longer than this, we bail out on the request
# rather than wedge the driver.
_PUSH_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Scheduler adapter
# ---------------------------------------------------------------------------


class _SchedulerAdapter:
    """Bridges the router's 2-arg ``submit(id, priority)`` to the real 3-arg API.

    The router crate ships a :class:`PySchedulerHandle` that duck-types
    ``submit(id, priority)`` (see ``crates/repercep-router/src/lib.rs::529``).
    The real scheduler crate (and its Python wrapper) expects
    ``submit(id, priority, payload)``. Rather than touch either crate, we
    stage the payload here before calling ``router.accept``, and forward
    the staged payload when the router calls back.

    Single-process, no remoting — a plain dict guarded by a lock is enough.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        self._payloads: dict[str, Any] = {}
        self._lock = threading.Lock()

    def stage(self, request_id: str, payload: Any) -> None:
        """Stage a payload to be forwarded the next time the router calls submit."""
        with self._lock:
            self._payloads[request_id] = payload

    def discard(self, request_id: str) -> None:
        """Drop a staged payload (e.g. on accept rollback)."""
        with self._lock:
            self._payloads.pop(request_id, None)

    # ---- duck-typed surface the router will call ----

    def submit(self, request_id: str, priority: str) -> None:
        with self._lock:
            payload = self._payloads.pop(request_id, None)
        if payload is None:
            # Defensive: router calling submit for an unstaged id is a bug
            # on our side, surface it cleanly so accept rolls back.
            raise RuntimeError(
                f"scheduler adapter: no staged payload for request {request_id!r}"
            )
        # The scheduler wrapper's submit lowercases the priority itself, but
        # the router passes the canonical-case string ("Normal", etc.). The
        # scheduler accepts both via case-insensitive parsing.
        self._scheduler.submit(request_id, priority, payload)

    def cancel(self, request_id: str) -> None:
        # If we somehow stashed a payload for an id that never made it to the
        # scheduler, drop it here too. Best-effort.
        with self._lock:
            self._payloads.pop(request_id, None)
        try:
            self._scheduler.cancel(request_id)
        except Exception as exc:
            # Scheduler may not know the id if next_blocking already popped it.
            # The router's contract is best-effort cancel-notify, so we don't
            # raise — the router's own state machine still flipped to Cancelled.
            _LOG.debug("scheduler.cancel(%s) failed: %s", request_id, exc)


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------


def encode_frame_line(
    *,
    frame_index: int,
    total_frames: int,
    height: int,
    width: int,
    pixels_b64: str,
    latency_ms: float,
    is_final: bool,
) -> bytes:
    """Encode one frame as a single NDJSON line (no trailing newline).

    The handler appends the newline when writing to the wire. Field shape
    mirrors :class:`repercep.runtime.types.FrameChunk` so v1 and v2 consumers
    read the same fields; ``is_final`` and ``pixels_b64`` are the v2 adds.
    """
    obj = {
        "frame_index": frame_index,
        "total_frames": total_frames,
        "height": height,
        "width": width,
        "latency_ms": latency_ms,
        "is_final": is_final,
        "pixels_b64": pixels_b64,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def decode_frame_line(line: bytes) -> dict[str, Any]:
    """Inverse of :func:`encode_frame_line`. Kept symmetric for tests."""
    parsed = json.loads(line.decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


# ---------------------------------------------------------------------------
# The driver loop
# ---------------------------------------------------------------------------


class EngineDriver:
    """A single background thread that drains the scheduler and feeds the router.

    v0 is intentionally single-driver, single-engine — no concurrent
    generation. The seam is here for when we batch later; the loop body just
    needs to dispatch to N engines or batch K requests instead of running
    one at a time.
    """

    def __init__(
        self,
        *,
        engine: WorldModelEngine,
        scheduler: Scheduler,
        router: Router,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._engine = engine
        self._scheduler = scheduler
        self._router = router
        self._loop = loop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the driver thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="repercep-engine-driver", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the driver to stop and wait for the thread to exit."""
        self._stop.set()
        # Wake the driver if it's parked in next_blocking. shutdown() also
        # rejects new submits, which is the desired teardown order.
        try:
            self._scheduler.shutdown()
        except Exception as exc:
            _LOG.debug("scheduler.shutdown raised: %s", exc)
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    # ---- internals ----

    def _run(self) -> None:
        _LOG.info("engine driver started")
        while not self._stop.is_set():
            try:
                item = self._scheduler.next_blocking(timeout_ms=_NEXT_TIMEOUT_MS)
            except Exception as exc:
                _LOG.exception("scheduler.next_blocking raised: %s", exc)
                # Tight loops on a broken scheduler would just burn CPU; back
                # off briefly and retry. If the scheduler is permanently
                # broken the stop signal will fire eventually.
                time.sleep(0.05)
                continue
            if item is None:
                continue
            self._run_one(item)
        _LOG.info("engine driver stopped")

    def _run_one(self, item: dict[str, Any]) -> None:
        request_id = item["id"]
        payload = item["payload"]
        # Reconstruct the GenerationRequest. The HTTP handler wraps the
        # body in {"request": GenerationRequest.model_dump()} before staging,
        # so the engine driver only has to know one envelope shape.
        try:
            req_dict = payload["request"] if isinstance(payload, dict) else None
            if req_dict is None:
                raise ValueError(f"unexpected payload shape for {request_id!r}: {payload!r}")
            gen_req = GenerationRequest.model_validate(req_dict)
        except Exception as exc:
            _LOG.exception("dropping malformed payload for %s: %s", request_id, exc)
            self._cancel_silently(request_id)
            return

        started = time.perf_counter()
        index = -1
        total = gen_req.params.num_frames
        try:
            for frame in self._engine.generate(gen_req):
                index = frame.index
                if self._stop.is_set():
                    # Drop in-flight frames on shutdown; the router will close
                    # the channel either way when we stop subscribing.
                    self._cancel_silently(request_id)
                    return
                # Best-effort cancel check before each push.
                if self._router.state(request_id) == "Cancelled":
                    _LOG.info("request %s cancelled by client; aborting", request_id)
                    return
                is_final = (frame.index == frame.total - 1)
                line = self._encode(frame, started=started, is_final=is_final)
                if not self._push(request_id, frame.index, line, is_final):
                    return
            # Engine yielded all frames but the last one wasn't is_final
            # — defensive: push a synthetic final marker so the subscriber
            # closes cleanly. This shouldn't happen with a well-behaved
            # engine, but it's cheap insurance.
            if index >= 0 and index != total - 1:
                line = self._encode_marker(
                    frame_index=index + 1, total=total, started=started
                )
                self._push(request_id, index + 1, line, True)
        except Exception as exc:
            _LOG.exception("engine.generate failed for %s: %s", request_id, exc)
            self._cancel_silently(request_id)

    def _push(
        self, request_id: str, frame_index: int, payload: bytes, is_final: bool
    ) -> bool:
        """Push a frame back onto the router's event loop. Returns False on failure.

        ``router.push_frame`` is a PyO3 async fn that uses
        ``pyo3_async_runtimes::tokio::future_into_py`` — the awaitable factory
        requires a running asyncio loop on the calling thread. We're on a
        worker thread without a loop, so we wrap the call in a coroutine and
        schedule it on the FastAPI event loop. The coroutine evaluates
        ``push_frame(...)`` (which produces the awaitable) ONCE it's running
        on the right loop, then awaits it.
        """
        router = self._router

        async def _do_push() -> None:
            await router.push_frame(request_id, frame_index, payload, is_final)

        try:
            fut = asyncio.run_coroutine_threadsafe(_do_push(), self._loop)
            fut.result(timeout=_PUSH_TIMEOUT_S)
        except Exception as exc:
            _LOG.warning(
                "push_frame failed for %s frame %d: %s", request_id, frame_index, exc
            )
            self._cancel_silently(request_id)
            return False
        return True

    def _cancel_silently(self, request_id: str) -> None:
        # The router may already be in a terminal state for this request;
        # we swallow the InvalidTransition here to keep the driver going.
        try:
            self._router.cancel(request_id)
        except Exception as exc:
            _LOG.debug("router.cancel(%s) suppressed: %s", request_id, exc)

    def _encode(self, frame: Any, *, started: float, is_final: bool) -> bytes:
        pixels = frame.pixels
        height = int(pixels.shape[0])
        width = int(pixels.shape[1])
        # uint8 HxWxC → base64. Contiguous so .tobytes() reflects the visible
        # tensor; ROCm tensors are CPU here because StubEngine returns CPU
        # tensors. Real engines must call .cpu() before yielding (the contract
        # is the existing v1 path's contract — Frame.pixels travels in-process
        # but the wire form is base64 bytes).
        cpu = pixels.detach().cpu().contiguous()
        b64 = base64.b64encode(cpu.numpy().tobytes()).decode("ascii")
        return encode_frame_line(
            frame_index=int(frame.index),
            total_frames=int(frame.total),
            height=height,
            width=width,
            pixels_b64=b64,
            latency_ms=(time.perf_counter() - started) * 1e3,
            is_final=is_final,
        )

    def _encode_marker(
        self, *, frame_index: int, total: int, started: float
    ) -> bytes:
        # Defensive synthetic final-frame marker — no pixels.
        return encode_frame_line(
            frame_index=frame_index,
            total_frames=total,
            height=0,
            width=0,
            pixels_b64="",
            latency_ms=(time.perf_counter() - started) * 1e3,
            is_final=True,
        )


# ---------------------------------------------------------------------------
# Public helpers used by app.py
# ---------------------------------------------------------------------------


__all__ = [
    "EngineDriver",
    "_SchedulerAdapter",
    "decode_frame_line",
    "encode_frame_line",
]
