"""Tests for the HTTP serving layer, exercised against the StubEngine."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from repercep.serving.app import create_app


def test_health() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_info_reports_ready_engine() -> None:
    client = TestClient(create_app())
    resp = client.get("/v1/info")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_generate_stream_yields_ndjson_frames() -> None:
    client = TestClient(create_app())
    body = {
        "prompt": "a forklift moving a pallet across a warehouse",
        "params": {"num_frames": 4, "height": 64, "width": 64, "seed": 0},
    }
    resp = client.post("/v1/generate/stream", json=body)
    assert resp.status_code == 200

    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    assert len(lines) == 4

    first = json.loads(lines[0])
    assert first["frame_index"] == 0
    assert first["total_frames"] == 4
    assert first["height"] == 64
    assert first["width"] == 64
    # frames stream in order
    assert [json.loads(ln)["frame_index"] for ln in lines] == [0, 1, 2, 3]


def test_generate_stream_rejects_empty_prompt() -> None:
    client = TestClient(create_app())
    resp = client.post("/v1/generate/stream", json={"prompt": ""})
    assert resp.status_code == 422
