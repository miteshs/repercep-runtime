"""Frame-streaming HTTP API for the Repercep Runtime.

API-first: this module *is* the HTTP contract. The gRPC contract is the sibling
``proto/repercep.proto``, kept in lockstep with ``repercep.runtime.types``.

Two API surfaces live here:

* ``/v1/...`` — original synchronous path. The handler runs the engine
  inline and yields :class:`repercep.runtime.types.FrameChunk` over NDJSON.
  Untouched for backward compatibility.

* ``/v2/...`` — Stage-4 path that routes through the Rust core. The handler
  mints a request id, stages the body, calls ``Router.accept`` (which submits
  to the scheduler), and streams the router's frame channel back to the
  client. The engine itself runs in a single background driver thread that
  drains the scheduler — see :mod:`repercep.serving.driver`.

The driver thread, scheduler, and router are owned by the FastAPI app's
``lifespan`` context: created at startup, torn down on shutdown.
"""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Annotated, Literal

import anyio
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from repercep import __version__
from repercep.config import RuntimeConfig
from repercep.runtime.engine import EngineInfo
from repercep.runtime.router import Router, RouterError
from repercep.runtime.scheduler import Scheduler
from repercep.runtime.stub_engine import StubEngine

# These must stay runtime imports: FastAPI resolves route annotations at startup
# via get_type_hints, and the WebSocket session validates Action/ResetRequest
# and emits LatentStep at runtime (see per-file ruff ignore).
from repercep.runtime.types import (
    Action,
    FrameChunk,
    GenerationRequest,
    LatentStep,
    ResetRequest,
)
from repercep.serving.driver import EngineDriver, _SchedulerAdapter
from repercep.serving.llm_proxy import CallRecord, LlmProxy
from repercep.serving.tenancy import LEGACY_KEY_ID, KeyStore, Principal
from repercep.serving.usage import UsageEvent, UsageLedger, month_start

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from repercep.runtime.engine import WorldModelEngine
    from repercep.runtime.interactive import InteractiveWorldModel
    from repercep.runtime.types import WorldState


# Default scheduler capacity for the v2 path. Sized so a single-engine driver
# can absorb a small burst without rejecting; backpressure surfaces as a 503
# at the v2 endpoint when this fills.
_DEFAULT_SCHEDULER_CAPACITY = 64

# Per-request frame channel depth on the router. The driver pushes one frame
# at a time and the streaming handler drains continuously, so a small bound
# is enough; backpressure here would only fire if a client TCP-stalls.
_DEFAULT_FRAME_QUEUE_DEPTH = 64

# Concurrent /v2/world/session connections before a new one is refused. Each
# open session holds server-side engine state (e.g. a KV-cache entry) for
# its lifetime, so this is a memory bound, not just a fairness knob.
_DEFAULT_MAX_INTERACTIVE_SESSIONS = 16

# Close a /v2/world/session connection that sends nothing for this long.
_DEFAULT_SESSION_IDLE_TIMEOUT_S = 300.0

# Default per-request timeout when proxying to a co-located LLM upstream.
_DEFAULT_LLM_TIMEOUT_S = 600.0

# Attribution used when the gateway runs with no auth configured at all (the
# local-dev default). Usage is still metered so a single-tenant box has a
# working ledger the day it grows a second tenant.
_ANONYMOUS = Principal(key_id="anonymous", customer="anonymous")

# Attribution for the pre-tenancy shared bearer token.
_LEGACY = Principal(key_id=LEGACY_KEY_ID, customer="legacy")


# ---------------------------------------------------------------------------
# v2 request body
# ---------------------------------------------------------------------------


_PRIORITY = Literal["low", "normal", "high"]


class V2GenerationRequest(BaseModel):
    """v2 generation request — :class:`GenerationRequest` plus a priority hint.

    Composition over inheritance: keeping :class:`GenerationRequest` itself
    untouched means the v1 path's wire contract is bit-stable. The v2 body
    is the v1 body inside ``request`` plus an optional ``priority`` field.
    """

    model_config = ConfigDict(extra="forbid")

    request: GenerationRequest
    priority: _PRIORITY = "normal"


# ---------------------------------------------------------------------------
# Lifespan-owned state
# ---------------------------------------------------------------------------


class _V2State:
    """Holds the v2 path's owned objects so the lifespan can tear them down."""

    def __init__(
        self,
        *,
        engine: WorldModelEngine,
        scheduler: Scheduler,
        adapter: _SchedulerAdapter,
        router: Router,
        driver: EngineDriver,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.adapter = adapter
        self.router = router
        self.driver = driver


def _build_v2_state(
    engine: WorldModelEngine,
    *,
    scheduler_capacity: int,
    frame_queue_depth: int,
    loop: asyncio.AbstractEventLoop,
) -> _V2State:
    scheduler = Scheduler(capacity=scheduler_capacity)
    adapter = _SchedulerAdapter(scheduler)
    router = Router(adapter, per_request_queue_depth=frame_queue_depth)
    driver = EngineDriver(engine=engine, scheduler=scheduler, router=router, loop=loop)
    return _V2State(
        engine=engine,
        scheduler=scheduler,
        adapter=adapter,
        router=router,
        driver=driver,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    engine: WorldModelEngine | None = None,
    *,
    interactive_engine: InteractiveWorldModel | None = None,
    scheduler_capacity: int = _DEFAULT_SCHEDULER_CAPACITY,
    frame_queue_depth: int = _DEFAULT_FRAME_QUEUE_DEPTH,
    max_sessions: int = _DEFAULT_MAX_INTERACTIVE_SESSIONS,
    session_idle_timeout_s: float = _DEFAULT_SESSION_IDLE_TIMEOUT_S,
    api_token: str | None = None,
    key_store: KeyStore | None = None,
    usage_ledger: UsageLedger | None = None,
    llm_proxy: LlmProxy | None = None,
    llm_upstream_url: str | None = None,
    llm_upstream_timeout_s: float = _DEFAULT_LLM_TIMEOUT_S,
    llm_upstream_api_key: str | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        engine: the world-model engine to serve. Defaults to ``StubEngine`` so
            the API is runnable and testable before the Cosmos-Predict-7B
            engine lands.
        interactive_engine: optional action-conditioned world model served over
            the ``/v2/world/session`` WebSocket (ADR-0008). ``None`` disables it.
        scheduler_capacity: max in-flight requests on the v2 path before
            ``submit`` raises ``QueueFull`` (surfaces as HTTP 503).
        frame_queue_depth: per-request frame channel depth. Backpressure
            within the router fires when a client TCP-stalls beyond this.
        max_sessions: concurrent ``/v2/world/session`` connections before a
            new one is refused (session capacity, not the v2 generate path's
            ``scheduler_capacity``). Each open session holds server-side
            engine state (KV cache, etc.) for its lifetime.
        session_idle_timeout_s: close a ``/v2/world/session`` connection that
            sends nothing (no ``ResetRequest``, no ``Action``) for this long.
            Bounds how long a client that connects and goes silent can hold a
            session slot and its underlying engine state.
        api_token: when set, require ``Authorization: Bearer <api_token>`` on
            every HTTP endpoint except ``/health``, and on the
            ``/v2/world/session`` WebSocket (header or ``?token=`` query
            param, checked before ``accept()``). ``None`` (default) disables
            auth entirely — existing callers are unaffected. Still accepted
            alongside ``key_store``, so a deployment can migrate its callers
            to per-customer keys without a flag day.
        key_store: per-customer API keys
            (:class:`~repercep.serving.tenancy.KeyStore`). When set, a caller
            may authenticate with any live ``rpc_...`` key and every request
            is attributed to that key's customer. This is what makes the
            gateway multi-tenant; ``api_token`` alone cannot attribute or
            revoke per customer.
        usage_ledger: where per-call token usage is written
            (:class:`~repercep.serving.usage.UsageLedger`). Required for
            billing and for ``monthly_token_quota`` enforcement; without it
            the gateway serves fine and records nothing.
        llm_proxy: a pre-built
            :class:`~repercep.serving.llm_proxy.LlmProxy` (e.g. with an
            injected client for tests). Takes precedence over
            ``llm_upstream_url``.
        llm_upstream_url: base URL of a co-located OpenAI-compatible LLM
            server (vLLM/SGLang) to reverse-proxy. When set (and ``llm_proxy``
            is None), the OpenAI ``/v1/chat/completions``, ``/v1/completions``,
            ``/v1/models`` and a ``/v1/llm/health`` readiness route are mounted
            behind the same bearer auth. Traffic bypasses the world-model
            scheduler/driver by design (see ``docs/LLM_PROXY.md``). ``None``
            (default) disables the proxy.
        llm_upstream_timeout_s: per-request timeout when proxying upstream.
        llm_upstream_api_key: bearer injected on upstream calls for a secured
            upstream; the client's own Authorization is never forwarded.
    """
    active_engine: WorldModelEngine = engine if engine is not None else StubEngine()
    active_interactive = interactive_engine
    active_sessions = 0

    def _record_call(request: Request, record: CallRecord) -> None:
        """Metering sink for the LLM proxy: attribute a call and store it."""
        if usage_ledger is None:
            return
        principal: Principal = getattr(request.state, "principal", _ANONYMOUS)
        usage_ledger.record(
            UsageEvent(
                key_id=principal.key_id,
                customer=principal.customer,
                route=record.route,
                model=record.model,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                exact=record.exact,
                status=record.status,
                latency_ms=record.latency_ms,
            )
        )

    active_llm_proxy = llm_proxy
    if active_llm_proxy is None and llm_upstream_url is not None:
        active_llm_proxy = LlmProxy(
            upstream_url=llm_upstream_url,
            timeout_s=llm_upstream_timeout_s,
            api_key=llm_upstream_api_key,
        )
    if active_llm_proxy is not None:
        # Installed here rather than only at construction so an injected
        # proxy is metered too — the gateway, not the proxy's builder, owns
        # attribution.
        active_llm_proxy.attach_call_sink(_record_call)

    def _resolve(bearer: str | None) -> Principal | None:
        """Map a presented bearer credential to a principal.

        ``None`` means reject. Order matters: per-customer keys are tried
        first so a deployment that has both configured attributes traffic to
        real customers rather than collapsing it into ``legacy``.
        """
        if key_store is None and api_token is None:
            return _ANONYMOUS
        if bearer is None:
            return None
        if key_store is not None:
            principal = key_store.authenticate(bearer)
            if principal is not None:
                return principal
        if api_token is not None and hmac.compare_digest(bearer, api_token):
            return _LEGACY
        return None

    def _check_quota(principal: Principal) -> None:
        """Reject a caller that has spent its monthly token allowance.

        Enforced at call boundaries against *already-billed* usage, because
        a call's token cost is unknowable until it completes. A customer can
        therefore overshoot its quota by at most one call. That is the
        standard behaviour for token quotas and it is deliberate — the
        alternative (reserving a worst-case ``max_tokens`` up front) would
        reject calls that would have fitted.
        """
        if principal.monthly_token_quota is None or usage_ledger is None:
            return
        spent = usage_ledger.tokens_since(key_id=principal.key_id, since=month_start())
        if spent >= principal.monthly_token_quota:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"monthly token quota exhausted "
                    f"({spent}/{principal.monthly_token_quota})"
                ),
            )

    def _require_api_token(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> None:
        bearer = (
            authorization.removeprefix("Bearer ")
            if authorization is not None and authorization.startswith("Bearer ")
            else None
        )
        principal = _resolve(bearer)
        if principal is None:
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")
        _check_quota(principal)
        # Downstream handlers and the metering sink read attribution here.
        request.state.principal = principal

    _auth = [Depends(_require_api_token)]

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The driver thread needs a handle to *this* loop so it can bounce
        # router.push_frame back via run_coroutine_threadsafe.
        loop = asyncio.get_running_loop()
        state = _build_v2_state(
            active_engine,
            scheduler_capacity=scheduler_capacity,
            frame_queue_depth=frame_queue_depth,
            loop=loop,
        )
        _app.state.v2 = state
        state.driver.start()
        try:
            yield
        finally:
            # Order matters: stop the driver (which calls scheduler.shutdown()
            # internally, draining the queue and waking the thread), then
            # mark the router shutting down so any in-flight subscribers see
            # end-of-stream cleanly.
            state.driver.stop(timeout=5.0)
            with suppress(RouterError):
                state.router.shutdown()
            if active_llm_proxy is not None:
                with suppress(Exception):
                    await active_llm_proxy.aclose()

    app = FastAPI(title="Repercep Runtime", version=__version__, lifespan=lifespan)

    if active_llm_proxy is not None:
        # Mounted behind the same bearer dependency as /v1 and /v2. Traffic
        # bypasses the world-model scheduler/driver by design — see
        # docs/LLM_PROXY.md for why that is deliberate, not a TODO.
        app.include_router(active_llm_proxy.router, dependencies=_auth)
        app.state.llm_proxy = active_llm_proxy

    # -----------------------------------------------------------------------
    # v1 (untouched)
    # -----------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/info", dependencies=_auth)
    def info() -> EngineInfo:
        return active_engine.info()

    @app.get("/v1/usage", dependencies=_auth)
    def usage(request: Request) -> dict[str, object]:
        """The calling key's own month-to-date usage.

        Scoped to the caller deliberately: this is the endpoint a customer
        polls to see what they are spending, not an operator view. Key
        management and cross-customer reporting are CLI-only (``repercep
        keys`` / ``repercep usage``) so that no HTTP surface can mint or
        enumerate credentials.
        """
        principal: Principal = getattr(request.state, "principal", _ANONYMOUS)
        since = month_start()
        if usage_ledger is None:
            return {"key_id": principal.key_id, "metering": "disabled", "since": since}
        rows = [
            s for s in usage_ledger.summary(since=since) if s.key_id == principal.key_id
        ]
        spent = rows[0] if rows else None
        return {
            "key_id": principal.key_id,
            "customer": principal.customer,
            "since": since,
            "calls": spent.calls if spent else 0,
            "prompt_tokens": spent.prompt_tokens if spent else 0,
            "completion_tokens": spent.completion_tokens if spent else 0,
            "total_tokens": spent.total_tokens if spent else 0,
            "estimated_tokens": spent.estimated_tokens if spent else 0,
            "monthly_token_quota": principal.monthly_token_quota,
        }

    @app.post("/v1/generate/stream", dependencies=_auth)
    def generate_stream(request: GenerationRequest) -> StreamingResponse:
        """Stream frames as newline-delimited JSON ``FrameChunk`` records.

        The response begins as soon as the first frame is ready; the runtime
        does not buffer the whole clip. This is the frame-level streaming the
        implementation plan calls for.
        """

        def frames() -> Iterator[str]:
            started = time.perf_counter()
            for frame in active_engine.generate(request):
                chunk = FrameChunk(
                    frame_index=frame.index,
                    total_frames=frame.total,
                    height=int(frame.pixels.shape[0]),
                    width=int(frame.pixels.shape[1]),
                    latency_ms=(time.perf_counter() - started) * 1e3,
                )
                yield chunk.model_dump_json() + "\n"

        return StreamingResponse(frames(), media_type="application/x-ndjson")

    # -----------------------------------------------------------------------
    # v2 (Rust-router path)
    # -----------------------------------------------------------------------

    @app.post("/v2/generate/stream", dependencies=_auth)
    async def generate_stream_v2(
        body: Annotated[V2GenerationRequest, Field()],
    ) -> StreamingResponse:
        """Submit through the router/scheduler; stream NDJSON frames back.

        Each line is one frame as encoded by
        :func:`repercep.serving.driver.encode_frame_line` — the v1
        ``FrameChunk`` fields plus ``is_final`` and a base64-encoded pixel
        payload. Mirror the v1 format exactly so the migration is a URL swap.
        """
        state: _V2State = app.state.v2
        request_id = uuid.uuid4().hex
        # Stage the payload so the adapter has it when the router calls
        # submit. Wrap in a single-key envelope so the driver can future-
        # extend without re-flowing the wire shape.
        payload = {"request": body.request.model_dump()}
        state.adapter.stage(request_id, payload)
        try:
            state.router.accept(request_id, body.priority, payload)
        except RouterError as exc:
            # Roll back the staged payload — accept may have failed before
            # the adapter consumed it.
            state.adapter.discard(request_id)
            msg = str(exc).lower()
            # Map scheduler-side capacity errors to 503; everything else 400.
            if "queue is full" in msg or "queuefull" in msg:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stream = state.router.subscribe(request_id)

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for frame in stream:
                    # frame is the dict returned by the FrameStream:
                    # {request_id, frame_index, payload (bytes), is_final}.
                    yield bytes(frame["payload"]) + b"\n"
            except Exception:
                # Best-effort cancel on stream interruption (client disconnect
                # or driver failure). Avoid raising — the response is already
                # being streamed.
                with suppress(RouterError):
                    state.router.cancel(request_id)

        return StreamingResponse(body_iter(), media_type="application/x-ndjson")

    @app.post("/v2/generate/{request_id}/cancel", dependencies=_auth)
    async def cancel_v2(request_id: str) -> dict[str, str]:
        """Cancel an in-flight v2 request. 404 if the id is unknown."""
        state: _V2State = app.state.v2
        try:
            state.router.cancel(request_id)
        except RouterError as exc:
            msg = str(exc).lower()
            if "unknown request" in msg:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "cancelled", "request_id": request_id}

    @app.get("/v2/generate/{request_id}/state", dependencies=_auth)
    def state_v2(request_id: str) -> dict[str, str | None]:
        """Look up the router's current state for a request. ``None`` if unknown."""
        state: _V2State = app.state.v2
        return {"request_id": request_id, "state": state.router.state(request_id)}

    # -----------------------------------------------------------------------
    # v2 interactive world-model session (action-conditioned, closed-loop)
    # -----------------------------------------------------------------------

    @app.websocket("/v2/world/session")
    async def world_session(ws: WebSocket) -> None:
        """Bidirectional interactive world-model session.

        Protocol: the client sends a ``ResetRequest`` JSON to open the session,
        then one ``Action`` JSON per step; the server replies with a
        ``LatentStep`` JSON per step (``step_index`` 0 acknowledges the reset).
        State persists for the life of the connection. Engine calls run in a
        threadpool so the event loop stays free — the same rationale as the v2
        driver thread. See ADR-0008.

        Hardened for a design partner's first hour, not just the happy path:
        an optional bearer token (checked before ``accept()``, so an
        unauthenticated client never gets a live connection), a cap on
        concurrent sessions (each holds server-side engine state for its
        lifetime), an idle timeout (bounds how long a silent-but-connected
        client can hold a slot), and a release call on every exit path once
        a session has been opened (see :meth:`release` below).
        """
        nonlocal active_sessions

        if api_token is not None or key_store is not None:
            supplied = ws.query_params.get("token")
            if supplied is None:
                auth_header = ws.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    supplied = auth_header.removeprefix("Bearer ")
            if _resolve(supplied) is None:
                await ws.close(code=1008)
                return

        await ws.accept()
        engine = active_interactive
        if engine is None:
            await ws.send_json({"error": "no interactive engine configured"})
            await ws.close(code=1008)
            return

        if active_sessions >= max_sessions:
            await ws.send_json({"error": "session capacity reached"})
            await ws.close(code=1013)
            return
        active_sessions += 1

        # Tracked separately from the loop-local `state` (kept strictly
        # WorldState, matching engine.step/reset's Protocol signatures) so
        # the finally block can release whatever was last reached without
        # widening `state` itself to `WorldState | None` throughout the loop.
        opened_state: WorldState | None = None
        try:
            try:
                reset_raw = await asyncio.wait_for(
                    ws.receive_text(), timeout=session_idle_timeout_s
                )
            except TimeoutError:
                await ws.close(code=1001)
                return
            except WebSocketDisconnect:
                return
            try:
                reset = ResetRequest.model_validate_json(reset_raw)
            except ValidationError:
                await ws.send_json({"error": "first message must be a ResetRequest"})
                await ws.close(code=1008)
                return

            state = await run_in_threadpool(engine.reset, reset.conditioning, reset.params)
            opened_state = state
            await ws.send_text(LatentStep(step_index=0).model_dump_json())
            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.receive_text(), timeout=session_idle_timeout_s
                        )
                    except TimeoutError:
                        await ws.close(code=1001)
                        return
                    try:
                        action = Action.model_validate_json(raw)
                    except ValidationError:
                        await ws.send_json({"error": "invalid action"})
                        continue
                    state, latent_step = await run_in_threadpool(engine.step, state, action)
                    opened_state = state
                    await ws.send_text(latent_step.model_dump_json())
            except WebSocketDisconnect:
                return
        finally:
            active_sessions -= 1
            if opened_state is not None:
                release = getattr(engine, "release", None)
                if release is not None:
                    # Shielded: a disconnecting client cancels this handler's
                    # task (confirmed via a test client that tears down its
                    # side eagerly — some real ASGI servers do this too on an
                    # abrupt close, not just a clean one), and an unshielded
                    # await here gets a CancelledError before release() ever
                    # runs, silently reintroducing the leak this exists to
                    # fix. Shielding lets this one cleanup call finish even
                    # though the surrounding task is already being torn down.
                    with anyio.CancelScope(shield=True):
                        await run_in_threadpool(release, opened_state)

    return app


def create_app_from_config(config: RuntimeConfig | None = None) -> FastAPI:
    """Build the app from ``REPERCEP_*`` environment settings.

    The deployment entry point — ``uvicorn --factory
    repercep.serving.app:create_app_from_config`` — and what makes
    ``REPERCEP_LLM_ENABLED`` (and the other ``REPERCEP_LLM_*`` knobs) take
    effect. Tests construct :func:`create_app` directly with explicit args.
    """
    cfg = config if config is not None else RuntimeConfig()
    store = KeyStore(cfg.gateway_db) if cfg.gateway_db is not None else None
    ledger = UsageLedger(cfg.gateway_db) if cfg.gateway_db is not None else None
    return create_app(
        api_token=cfg.api_token,
        key_store=store,
        usage_ledger=ledger,
        llm_upstream_url=cfg.llm_upstream_url if cfg.llm_enabled else None,
        llm_upstream_timeout_s=cfg.llm_upstream_timeout_s,
        llm_upstream_api_key=cfg.llm_upstream_api_key,
    )
