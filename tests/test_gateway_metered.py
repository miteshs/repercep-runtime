"""End-to-end tests for the metered multi-tenant gateway.

The gateway, the key store, the usage ledger and the LLM proxy are wired
together exactly as a deployment wires them; only the LLM upstream is faked
(``httpx.MockTransport``). No GPU, no network, no real vLLM.

This is the P0 slice's acceptance test: hand two customers a key each, serve
them, and end up with a ledger you could invoice from.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from repercep.serving.app import create_app
from repercep.serving.llm_proxy import LlmProxy
from repercep.serving.tenancy import KeyStore
from repercep.serving.usage import UsageLedger, month_start

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from fastapi import FastAPI

_UPSTREAM = "http://llm-upstream.test"


def _streamed(status: int, data: bytes, content_type: str) -> httpx.Response:
    async def _gen() -> AsyncIterator[bytes]:
        yield data

    return httpx.Response(status, content=_gen(), headers={"content-type": content_type})


def _completion(prompt_tokens: int = 10, completion_tokens: int = 5) -> httpx.Response:
    return _streamed(
        200,
        json.dumps(
            {
                "id": "cmpl-1",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
        ).encode(),
        "application/json",
    )


def _default_handler(request: httpx.Request) -> httpx.Response:
    return _completion()


def _build(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response] = _default_handler,
    *,
    api_token: str | None = None,
) -> tuple[FastAPI, KeyStore, UsageLedger]:
    db = tmp_path / "gw.db"
    store = KeyStore(db)
    ledger = UsageLedger(db)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=_UPSTREAM)
    proxy = LlmProxy(upstream_url=_UPSTREAM, client=client)
    app = create_app(
        api_token=api_token,
        key_store=store,
        usage_ledger=ledger,
        llm_proxy=proxy,
    )
    return app, store, ledger


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_key_required_when_store_configured(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    with TestClient(app) as client:
        assert client.get("/v1/info").status_code == 401
        assert client.get("/v1/info", headers={"Authorization": "Bearer nope"}).status_code == 401
        # Liveness stays open — load balancers cannot hold a credential.
        assert client.get("/health").status_code == 200
    ledger.close()
    store.close()


def test_valid_key_is_accepted(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        r = client.get("/v1/info", headers={"Authorization": f"Bearer {full_key}"})
        assert r.status_code == 200
    ledger.close()
    store.close()


def test_revoked_key_stops_working_immediately(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    full_key, record = store.create(customer="acme")
    auth = {"Authorization": f"Bearer {full_key}"}
    with TestClient(app) as client:
        assert client.get("/v1/info", headers=auth).status_code == 200
        store.revoke(record.key_id)
        assert client.get("/v1/info", headers=auth).status_code == 401
    ledger.close()
    store.close()


def test_legacy_shared_token_still_works_alongside_keys(tmp_path: Path) -> None:
    """Migration must not need a flag day."""
    app, store, ledger = _build(tmp_path, api_token="legacy-secret")
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        assert client.get(
            "/v1/info", headers={"Authorization": "Bearer legacy-secret"}
        ).status_code == 200
        assert client.get(
            "/v1/info", headers={"Authorization": f"Bearer {full_key}"}
        ).status_code == 200
    ledger.close()
    store.close()


def test_open_gateway_unchanged_when_nothing_configured() -> None:
    """No store, no token → the pre-tenancy behaviour, unaffected."""
    with TestClient(create_app()) as client:
        assert client.get("/v1/info").status_code == 200


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------


def test_completion_is_metered_to_the_calling_customer(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    full_key, record = store.create(customer="acme")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"model": "qwen2.5-7b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200

    rows = ledger.summary()
    assert len(rows) == 1
    assert rows[0].key_id == record.key_id
    assert rows[0].customer == "acme"
    assert rows[0].calls == 1
    assert rows[0].prompt_tokens == 10
    assert rows[0].completion_tokens == 5
    assert rows[0].estimated_tokens == 0
    ledger.close()
    store.close()


def test_two_customers_are_billed_separately(tmp_path: Path) -> None:
    """The thing one shared token could never do."""
    app, store, ledger = _build(tmp_path)
    key_a, _ = store.create(customer="acme")
    key_b, _ = store.create(customer="globex")

    with TestClient(app) as client:
        for _ in range(3):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key_a}"},
                json={"model": "m", "messages": []},
            )
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key_b}"},
            json={"model": "m", "messages": []},
        )

    by_customer = {r.customer: r for r in ledger.summary()}
    assert by_customer["acme"].calls == 3
    assert by_customer["acme"].total_tokens == 45
    assert by_customer["globex"].calls == 1
    assert by_customer["globex"].total_tokens == 15
    ledger.close()
    store.close()


def test_streaming_call_is_metered_without_altering_the_stream(tmp_path: Path) -> None:
    """The client's bytes are what it would have got with no gateway at all."""
    content = (
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
    )
    usage = b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2}}\n\n'
    done = b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        # The proxy must have asked for usage on the client's behalf.
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return _streamed(200, content + usage + done, "text/event-stream")

    app, store, ledger = _build(tmp_path, handler)
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"model": "m", "messages": [], "stream": True},
        )
        assert r.status_code == 200
        assert r.content == content + done

    row = ledger.summary()[0]
    assert row.prompt_tokens == 8
    assert row.completion_tokens == 2
    assert row.estimated_tokens == 0
    ledger.close()
    store.close()


def test_streaming_without_upstream_usage_is_flagged_estimated(tmp_path: Path) -> None:
    stream = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _streamed(200, stream, "text/event-stream")

    app, store, ledger = _build(tmp_path, handler)
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"model": "m", "messages": [], "stream": True},
        )
        assert r.content == stream

    row = ledger.summary()[0]
    assert row.completion_tokens == 2
    assert row.estimated_tokens == 2, "must be disclosed as an estimate, not billed as exact"
    ledger.close()
    store.close()


def test_upstream_error_is_recorded_but_not_billed_as_tokens(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _streamed(400, b'{"error":"bad request"}', "application/json")

    app, store, ledger = _build(tmp_path, handler)
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {full_key}"},
            json={"model": "m", "messages": []},
        )
        assert r.status_code == 400

    rows = ledger.recent()
    assert len(rows) == 1
    assert rows[0]["status"] == 400
    assert rows[0]["prompt_tokens"] == 0
    assert rows[0]["completion_tokens"] == 0
    ledger.close()
    store.close()


def test_unauthenticated_call_is_not_metered(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    store.create(customer="acme")
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"model": "m"}).status_code == 401
    assert ledger.summary() == []
    ledger.close()
    store.close()


# ---------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------


def test_quota_blocks_once_exhausted(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    # Each call bills 15 tokens. A 15-token quota admits the first call
    # (nothing billed yet) and rejects the second.
    full_key, _ = store.create(customer="acme", monthly_token_quota=15)
    auth = {"Authorization": f"Bearer {full_key}"}
    body = {"model": "m", "messages": []}

    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", headers=auth, json=body).status_code == 200
        r = client.post("/v1/chat/completions", headers=auth, json=body)
        assert r.status_code == 429
        assert "quota" in r.json()["detail"]

    # The blocked call is rejected before reaching the upstream, so it costs
    # the customer nothing and does not appear as billed usage.
    assert ledger.summary()[0].calls == 1
    ledger.close()
    store.close()


def test_quota_may_be_overshot_by_at_most_one_call(tmp_path: Path) -> None:
    """A call's cost is unknowable until it completes, so the last admitted
    call can cross the line. Pinned down so the behaviour is a decision
    rather than a surprise on an invoice."""
    app, store, ledger = _build(tmp_path)
    full_key, _ = store.create(customer="acme", monthly_token_quota=20)
    auth = {"Authorization": f"Bearer {full_key}"}
    body = {"model": "m", "messages": []}

    with TestClient(app) as client:
        # 0 billed < 20 → admitted, bills 15.
        assert client.post("/v1/chat/completions", headers=auth, json=body).status_code == 200
        # 15 billed < 20 → admitted, bills 15 more and overshoots to 30.
        assert client.post("/v1/chat/completions", headers=auth, json=body).status_code == 200
        # 30 billed >= 20 → rejected.
        assert client.post("/v1/chat/completions", headers=auth, json=body).status_code == 429

    assert ledger.summary()[0].total_tokens == 30
    ledger.close()
    store.close()


def test_no_quota_means_unlimited(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    full_key, _ = store.create(customer="acme")
    auth = {"Authorization": f"Bearer {full_key}"}
    with TestClient(app) as client:
        for _ in range(5):
            r = client.post(
                "/v1/chat/completions", headers=auth, json={"model": "m", "messages": []}
            )
            assert r.status_code == 200
    assert ledger.summary()[0].calls == 5
    ledger.close()
    store.close()


def test_one_customers_quota_does_not_affect_another(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    capped, _ = store.create(customer="acme", monthly_token_quota=20)
    open_key, _ = store.create(customer="globex")
    body = {"model": "m", "messages": []}

    with TestClient(app) as client:
        for _ in range(2):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {capped}"},
                json=body,
            )
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {open_key}"},
            json=body,
        )
        assert r.status_code == 200
    ledger.close()
    store.close()


# ---------------------------------------------------------------------------
# /v1/usage
# ---------------------------------------------------------------------------


def test_usage_endpoint_reports_only_the_callers_own_spend(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    key_a, rec_a = store.create(customer="acme", monthly_token_quota=1_000)
    key_b, _ = store.create(customer="globex")
    body = {"model": "m", "messages": []}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions", headers={"Authorization": f"Bearer {key_a}"}, json=body
        )
        for _ in range(2):
            client.post(
                "/v1/chat/completions", headers={"Authorization": f"Bearer {key_b}"}, json=body
            )
        report = client.get(
            "/v1/usage", headers={"Authorization": f"Bearer {key_a}"}
        ).json()

    assert report["key_id"] == rec_a.key_id
    assert report["customer"] == "acme"
    assert report["calls"] == 1
    assert report["total_tokens"] == 15
    assert report["monthly_token_quota"] == 1_000
    assert report["since"] == month_start()
    ledger.close()
    store.close()


def test_usage_endpoint_reports_zero_before_any_traffic(tmp_path: Path) -> None:
    app, store, ledger = _build(tmp_path)
    full_key, _ = store.create(customer="acme")
    with TestClient(app) as client:
        report = client.get(
            "/v1/usage", headers={"Authorization": f"Bearer {full_key}"}
        ).json()
    assert report["calls"] == 0
    assert report["total_tokens"] == 0
    ledger.close()
    store.close()


def test_no_http_surface_can_mint_or_list_keys(tmp_path: Path) -> None:
    """Key management is CLI-only, so a stolen key cannot mint more."""
    app, store, ledger = _build(tmp_path)
    full_key, _ = store.create(customer="acme")
    auth = {"Authorization": f"Bearer {full_key}"}
    with TestClient(app) as client:
        for path in ("/v1/keys", "/v1/admin/keys", "/v1/admin/usage"):
            assert client.get(path, headers=auth).status_code == 404
            assert client.post(path, headers=auth, json={}).status_code == 404
    ledger.close()
    store.close()
