"""Tests for the ``/v2/world/session`` interactive WebSocket surface.

Drives the bidirectional session end-to-end through FastAPI's TestClient with a
stub action-conditioned engine. Gated on torch (the stub holds a latent tensor)
and fastapi; skips cleanly where either is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import (
    Action,
    ConditioningInput,
    LatentStep,
    ResetRequest,
    RolloutParams,
    WorldState,
)

if TYPE_CHECKING:
    import torch


class _StubInteractive:
    """Counts steps; echoes ``len(action.values)`` as the step energy."""

    def __init__(self) -> None:
        self.released: list[str] = []

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name="stub-ac", backend="cpu", device="cpu:0", dtype="float32", ready=True
        )

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        import torch

        return WorldState(context=torch.zeros(1, 2), step_index=0, session_id="sess")

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        nxt = WorldState(
            context=state.context, step_index=state.step_index + 1, session_id=state.session_id
        )
        return nxt, LatentStep(step_index=nxt.step_index, energy=float(len(action.values)))

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        return Action(values=[0.0])

    def release(self, state: WorldState) -> None:
        self.released.append(state.session_id)


def test_interactive_session_streams_steps() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    with (
        TestClient(create_app(interactive_engine=_StubInteractive())) as client,
        client.websocket_connect("/v2/world/session") as ws,
    ):
        ws.send_text(ResetRequest().model_dump_json())
        ack = ws.receive_json()
        assert ack["step_index"] == 0  # reset acknowledged

        for expected in (1, 2, 3):
            ws.send_text(Action(values=[0.5, -0.5]).model_dump_json())
            msg = ws.receive_json()
            assert msg["step_index"] == expected
            assert msg["energy"] == 2.0  # len(action.values)


def test_interactive_session_requires_engine() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/v2/world/session") as ws,
    ):
        assert "error" in ws.receive_json()


def test_interactive_session_releases_engine_state_on_disconnect() -> None:
    """The leak fix: engine.release() fires once the client goes away, not
    mid-session and not never."""
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    engine = _StubInteractive()
    # Deliberately nested, not combined (ruff SIM117): the assertions below
    # need to run at two different teardown points — right after the WS
    # closes (still inside the TestClient's `with`) and again after the
    # TestClient itself tears down — which a single combined `with` can't
    # express since it only has one exit point.
    with TestClient(create_app(interactive_engine=engine)) as client:  # noqa: SIM117
        with client.websocket_connect("/v2/world/session") as ws:
            ws.send_text(ResetRequest().model_dump_json())
            ws.receive_json()
            assert engine.released == []
        # Exiting the inner `with` only sends the client-side close frame; the
        # server-side handler's `finally` (which awaits release() in a
        # threadpool) may not have run yet. Exiting the OUTER TestClient
        # context guarantees the portal has drained all in-flight connection
        # coroutines, so check after that, not right after the disconnect.
    assert engine.released == ["sess"]


def test_interactive_session_rejects_over_capacity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    app = create_app(interactive_engine=_StubInteractive(), max_sessions=1)
    with TestClient(app) as client, client.websocket_connect("/v2/world/session") as first:
        first.send_text(ResetRequest().model_dump_json())
        first.receive_json()  # reset ack — first session now holds the one slot

        with client.websocket_connect("/v2/world/session") as second:
            msg = second.receive_json()
            assert "capacity" in msg["error"]


def test_interactive_session_closes_on_idle_timeout() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from repercep.serving.app import create_app

    app = create_app(interactive_engine=_StubInteractive(), session_idle_timeout_s=0.05)
    # Never send a ResetRequest — the server should close for silence, not
    # hang the slot open forever.
    with (
        TestClient(app) as client,
        client.websocket_connect("/v2/world/session") as ws,
        pytest.raises(WebSocketDisconnect),
    ):
        ws.receive_json()


def test_interactive_session_requires_token_when_configured() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from repercep.serving.app import create_app

    app = create_app(interactive_engine=_StubInteractive(), api_token="secret")
    with TestClient(app) as client:
        # No token: rejected before accept() -- never gets a live session.
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v2/world/session"):
            pass

        # Correct token via query param: works exactly like the untokened path.
        with client.websocket_connect("/v2/world/session?token=secret") as ws:
            ws.send_text(ResetRequest().model_dump_json())
            assert ws.receive_json()["step_index"] == 0

        # HTTP endpoints: 401 without the token, 200 with it.
        assert client.get("/v1/info").status_code == 401
        assert client.get("/v1/info", headers={"Authorization": "Bearer secret"}).status_code == 200

        # /health is deliberately exempt (infra liveness probes).
        assert client.get("/health").status_code == 200
