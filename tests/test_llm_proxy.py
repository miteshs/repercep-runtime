"""Tests for the co-located LLM reverse-proxy (``serving/llm_proxy.py``).

The upstream (vLLM/SGLang) is faked with ``httpx.MockTransport`` — no real
LLM server, no GPU, no network. We assert the proxy's contract: it is OFF by
default, streams the upstream body through verbatim, propagates upstream
status/errors, maps a dead upstream to 502, reuses the gateway's bearer auth,
and never forwards the client's own Authorization header upstream.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from repercep.config import RuntimeConfig
from repercep.serving.app import create_app, create_app_from_config
from repercep.serving.llm_proxy import LlmProxy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

_UPSTREAM = "http://llm-upstream.test"


def _proxy(
    handler: Callable[[httpx.Request], httpx.Response], *, api_key: str | None = None
) -> LlmProxy:
    """An LlmProxy whose upstream is a MockTransport handler."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=_UPSTREAM)
    return LlmProxy(upstream_url=_UPSTREAM, api_key=api_key, client=client)


def _streamed(status: int, data: bytes, content_type: str) -> httpx.Response:
    """A MockTransport response backed by a real (unconsumed) async stream.

    ``httpx.Response(content=<bytes>)`` is marked stream-consumed at
    construction, so the proxy's ``aiter_raw()`` passthrough — the production
    path — cannot iterate it. Backing the response with an async generator
    reproduces a genuine streaming upstream, the way vLLM/SGLang actually reply.
    """

    async def _gen() -> AsyncIterator[bytes]:
        yield data

    return httpx.Response(status, content=_gen(), headers={"content-type": content_type})


def _json_stream(status: int, payload: object) -> httpx.Response:
    """A streamed JSON response — even non-``stream`` upstream replies are
    consumed by the proxy through ``aiter_raw``, so they must stream too."""
    return _streamed(status, json.dumps(payload).encode(), "application/json")


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_llm_routes_absent_by_default() -> None:
    """No upstream configured → the LLM endpoints do not exist (404)."""
    with TestClient(create_app()) as client:
        assert client.post("/v1/chat/completions", json={"model": "x"}).status_code == 404
        assert client.get("/v1/models").status_code == 404
        # The world-model path is untouched.
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Streaming passthrough
# ---------------------------------------------------------------------------


def test_chat_completions_streams_sse_through() -> None:
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return _streamed(200, sse, "text/event-stream")

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "m", "stream": True, "messages": []}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.content == sse


def test_chat_completions_non_stream_json_through() -> None:
    payload = {"id": "cmpl-1", "choices": [{"message": {"content": "hello"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_stream(200, payload)

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert resp.status_code == 200
        assert resp.json() == payload


def test_completions_passthrough() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        return _json_stream(200, {"choices": [{"text": "done"}]})

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.post("/v1/completions", json={"model": "m", "prompt": "hi"})
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "done"


def test_models_passthrough() -> None:
    listing = {"object": "list", "data": [{"id": "meta-llama/Llama-3.1-8B", "object": "model"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=listing)

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == listing


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_upstream_4xx_status_propagates() -> None:
    """A bad-request error from the upstream reads as the same status here."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "unknown model"})

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
        assert resp.status_code == 400
        assert "unknown model" in resp.json()["detail"]


def test_upstream_unreachable_is_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with TestClient(create_app(llm_proxy=_proxy(handler))) as client:
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert resp.status_code == 502
        assert "unreachable" in resp.json()["detail"]


def test_readiness_probe_reflects_upstream() -> None:
    def up(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    with TestClient(create_app(llm_proxy=_proxy(up))) as client:
        assert client.get("/v1/llm/health").json() == {"upstream": "ok", "url": _UPSTREAM}

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    with TestClient(create_app(llm_proxy=_proxy(down))) as client:
        resp = client.get("/v1/llm/health")
        assert resp.status_code == 503
        assert resp.json()["upstream"] == "unreachable"


# ---------------------------------------------------------------------------
# Auth: gateway token gates the routes; client auth is not forwarded upstream
# ---------------------------------------------------------------------------


def test_gateway_auth_gates_llm_routes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_stream(200, {"ok": True})

    app = create_app(llm_proxy=_proxy(handler), api_token="secret")
    with TestClient(app) as client:
        # No bearer → 401 (reuses the same dependency as /v1 and /v2).
        assert client.post("/v1/chat/completions", json={}).status_code == 401
        ok = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"authorization": "Bearer secret"},
        )
        assert ok.status_code == 200


def test_client_authorization_not_forwarded_upstream() -> None:
    """The gateway token must never leak upstream; the upstream key is used."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _json_stream(200, {"ok": True})

    app = create_app(llm_proxy=_proxy(handler, api_key="upstream-key"), api_token="gateway-tok")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"authorization": "Bearer gateway-tok"},
        )
        assert resp.status_code == 200
    # Upstream saw the configured upstream key, not the gateway token.
    assert seen["authorization"] == "Bearer upstream-key"


# ---------------------------------------------------------------------------
# Env-driven factory (deployment entry point)
# ---------------------------------------------------------------------------


def test_from_config_disabled_has_no_llm_routes() -> None:
    app = create_app_from_config(RuntimeConfig(llm_enabled=False))
    paths = app.openapi()["paths"]
    assert "/v1/chat/completions" not in paths
    assert "/v2/generate/stream" in paths  # world-model path still there


def test_from_config_enabled_registers_llm_routes() -> None:
    # A bogus upstream URL is fine — we assert the routes are wired, we don't call them.
    app = create_app_from_config(
        RuntimeConfig(llm_enabled=True, llm_upstream_url="http://127.0.0.1:9099")
    )
    paths = app.openapi()["paths"]
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
