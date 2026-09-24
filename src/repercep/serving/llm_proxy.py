"""OpenAI-compatible reverse-proxy to a co-located vLLM/SGLang server.

The runtime's *world-model* path (router → scheduler → driver) and this LLM
path are deliberately separate. An LLM upstream (vLLM, SGLang) already does
continuous batching + paged attention *internally*; routing its traffic
through Repercep's single-driver v2 scheduler (``serving/driver.py``) would
**serialize exactly what the upstream parallelizes** — strictly worse than
talking to it directly. So this module is a thin async passthrough that reuses
only the gateway's auth, metering and deployment surface. See
``docs/LLM_PROXY.md`` for the design rationale and co-location recipe.

Security note: the client's own ``Authorization`` header (the *gateway*
credential, meaningless upstream) is never forwarded. A configured ``api_key``
is injected instead, so a secured upstream (``vllm serve --api-key``) works
without leaking the caller's key.

Metering: when an ``on_call`` sink is supplied, every proxied completion is
token-counted via :mod:`repercep.serving.metering` and reported after the
response finishes. The proxy itself stays ignorant of who the caller is — it
hands the sink the :class:`~starlette.requests.Request` and lets the gateway
resolve attribution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from repercep.serving.metering import ResponseMeter, plan_request

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

_LOG = logging.getLogger(__name__)

# A connection-level failure to the upstream (down, wrong port, DNS) maps to
# 502 Bad Gateway — the gateway is up, the thing behind it isn't.
_UPSTREAM_UNREACHABLE = 502


@dataclass(frozen=True, slots=True)
class CallRecord:
    """What one proxied call cost, handed to the metering sink.

    Deliberately free of tenancy types: this module knows what was spent, not
    who spent it.
    """

    route: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    exact: bool
    status: int
    latency_ms: float


class LlmProxy:
    """Thin OpenAI-compatible reverse-proxy to one upstream LLM server.

    Construct once per app, mount :attr:`router` behind the gateway's auth
    dependency, and :meth:`aclose` on shutdown. Stateless beyond the pooled
    :class:`httpx.AsyncClient`, so it holds none of the world-model path's
    per-session state.
    """

    def __init__(
        self,
        *,
        upstream_url: str,
        timeout_s: float = 600.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        on_call: Callable[[Request, CallRecord], None] | None = None,
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._api_key = api_key
        # An injected client (tests use ``httpx.MockTransport``) carries its own
        # base_url; otherwise build one pinned to the upstream. Constructing an
        # AsyncClient outside a running loop is fine — it lazily opens the pool.
        self._client = client or httpx.AsyncClient(base_url=self._upstream_url, timeout=timeout_s)
        self._on_call = on_call
        self.router = self._build_router()

    async def aclose(self) -> None:
        """Close the pooled upstream client. Idempotent per httpx semantics."""
        await self._client.aclose()

    def attach_call_sink(self, sink: Callable[[Request, CallRecord], None]) -> None:
        """Add a metering sink to an already-constructed proxy.

        Exists because a proxy built by the caller and handed to
        ``create_app(llm_proxy=...)`` would otherwise never be metered — the
        gateway only got to pass ``on_call`` on the path where it built the
        proxy itself, so injecting one silently disabled billing.

        Sinks *chain* rather than replace. Losing revenue because a second
        sink was installed is a worse failure than running two.
        """
        existing = self._on_call
        if existing is None:
            self._on_call = sink
            return

        def _both(request: Request, record: CallRecord) -> None:
            existing(request, record)
            sink(request, record)

        self._on_call = _both

    # -- upstream request helpers -------------------------------------------

    def _upstream_headers(self, *, content_type: str, accept: str | None) -> dict[str, str]:
        headers = {"content-type": content_type}
        if accept is not None:
            headers["accept"] = accept
        if self._api_key is not None:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    def _emit(self, request: Request, record: CallRecord) -> None:
        """Report one call to the sink. A broken sink must not break serving."""
        if self._on_call is None:
            return
        try:
            self._on_call(request, record)
        except Exception:  # pragma: no cover - defensive
            _LOG.exception("usage sink failed for %s", record.route)

    async def _proxy_post_stream(self, path: str, request: Request) -> Response:
        """Forward a POST body upstream and stream the response back.

        The response is byte-identical to the upstream's, with one exception:
        on a streaming call where the client did not specify
        ``stream_options``, the proxy asks the upstream for a usage chunk and
        then suppresses that chunk. See :mod:`repercep.serving.metering` for
        why exact billing is worth that much machinery.
        """
        started = time.perf_counter()
        raw_body = await request.body()
        plan = plan_request(raw_body)
        headers = self._upstream_headers(
            content_type=request.headers.get("content-type", "application/json"),
            accept=request.headers.get("accept"),
        )
        upstream_req = self._client.build_request(
            "POST", path, content=plan.body, headers=headers
        )
        try:
            upstream = await self._client.send(upstream_req, stream=True)
        except httpx.RequestError as exc:
            self._emit(
                request,
                CallRecord(
                    route=path,
                    model=plan.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    exact=True,
                    status=_UPSTREAM_UNREACHABLE,
                    latency_ms=(time.perf_counter() - started) * 1e3,
                ),
            )
            raise HTTPException(
                status_code=_UPSTREAM_UNREACHABLE, detail=f"llm upstream unreachable: {exc}"
            ) from exc

        if upstream.status_code >= 400:
            # Surface the upstream's own error body/status rather than masking
            # it — a 400 from vLLM (bad params) should read as a 400 here.
            detail = (await upstream.aread()).decode(errors="replace")
            await upstream.aclose()
            self._emit(
                request,
                CallRecord(
                    route=path,
                    model=plan.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    exact=True,
                    status=upstream.status_code,
                    latency_ms=(time.perf_counter() - started) * 1e3,
                ),
            )
            raise HTTPException(status_code=upstream.status_code, detail=detail)

        meter = ResponseMeter(
            streaming=plan.streaming, suppress_usage_chunk=plan.injected_usage
        )
        status = upstream.status_code

        async def metered() -> AsyncIterator[bytes]:
            try:
                async for raw in upstream.aiter_raw():
                    out = meter.feed(raw)
                    if out:
                        yield out
                tail = meter.finish()
                if tail:
                    yield tail
            finally:
                await upstream.aclose()
                # Idempotent: on a client disconnect the normal finish() above
                # never ran, and usage would otherwise fall back to the
                # estimate for a call that did have real counts available.
                meter.finish()
                usage = meter.usage
                self._emit(
                    request,
                    CallRecord(
                        route=path,
                        model=plan.model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        exact=meter.exact,
                        status=status,
                        latency_ms=(time.perf_counter() - started) * 1e3,
                    ),
                )

        return StreamingResponse(
            metered(),
            status_code=status,
            media_type=upstream.headers.get("content-type"),
        )

    async def _proxy_get(self, path: str) -> Response:
        """Forward a GET upstream and return its (non-streamed) JSON body."""
        headers = (
            {"authorization": f"Bearer {self._api_key}"} if self._api_key is not None else None
        )
        try:
            upstream = await self._client.get(path, headers=headers)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=_UPSTREAM_UNREACHABLE, detail=f"llm upstream unreachable: {exc}"
            ) from exc
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def _readiness(self) -> Response:
        """Best-effort upstream reachability probe (readiness, not liveness).

        ``/v1/models`` is implemented by both vLLM and SGLang and needs the
        weights loaded to answer, so a 200 here is a genuine "ready to serve"
        signal — unlike the gateway's own always-200 ``/health`` liveness.
        """
        try:
            probe = await self._client.get("/v1/models")
            ok = probe.status_code == 200
        except httpx.RequestError:
            ok = False
        return JSONResponse(
            {"upstream": "ok" if ok else "unreachable", "url": self._upstream_url},
            status_code=200 if ok else 503,
        )

    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["llm"])

        @router.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Response:
            return await self._proxy_post_stream("/v1/chat/completions", request)

        @router.post("/v1/completions")
        async def completions(request: Request) -> Response:
            return await self._proxy_post_stream("/v1/completions", request)

        @router.get("/v1/models")
        async def models() -> Response:
            return await self._proxy_get("/v1/models")

        @router.get("/v1/llm/health")
        async def llm_health() -> Response:
            return await self._readiness()

        return router


__all__ = ["CallRecord", "LlmProxy"]
