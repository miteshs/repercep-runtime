"""Tests for the Repercep FP8 backend registration with diffusers' dispatcher.

These tests are CPU-runnable for the registration / env-var bridge surface;
the correctness allclose test is GPU-gated (kernel availability + a Cosmos-DiT
shape that fits in a sensible bench but exercises the FP8 path).
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRITON = importlib.util.find_spec("triton") is not None
_HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None


pytestmark = pytest.mark.skipif(
    not _HAS_DIFFUSERS, reason="diffusers required for backend bridge tests"
)


@contextmanager
def _temporary_env(name: str, value: str | None) -> Iterator[None]:
    """Set or unset an env var only for the duration of the test."""
    prior = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior


@contextmanager
def _restore_active_backend() -> Iterator[None]:
    """Snapshot+restore the dispatcher's active backend (avoid test bleed)."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    prior = _AttentionBackendRegistry._active_backend
    try:
        yield
    finally:
        _AttentionBackendRegistry._active_backend = prior


def test_repercep_fp8_backend_is_registered() -> None:
    # Just importing ``repercep.attention`` should trigger the bridge module's
    # registration as a side effect — no opt-in API needed.
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    import repercep.attention  # noqa: F401

    member = AttentionBackendName("repercep_fp8")
    assert member in _AttentionBackendRegistry._backends
    # The backend function must accept the exact signature diffusers dispatch
    # introspects against. Missing any of these would cause the kwarg filter
    # in ``dispatch_attention_fn`` to drop required args.
    supported = _AttentionBackendRegistry._supported_arg_names[member]
    assert {"query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale"} <= supported


def test_register_is_idempotent() -> None:
    from repercep.attention.diffusers_backend import register_repercep_fp8_backend

    member_a = register_repercep_fp8_backend()
    member_b = register_repercep_fp8_backend()
    assert member_a is member_b


def test_attention_backend_context_manager_accepts_repercep_fp8() -> None:
    # Importing repercep.attention performs registration; the diffusers
    # ``attention_backend`` ctx manager then resolves the string by enum lookup.
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry,
        attention_backend,
    )

    import repercep.attention  # noqa: F401

    with _restore_active_backend(), attention_backend("repercep_fp8"):
        active = _AttentionBackendRegistry._active_backend
        assert str(active.value) == "repercep_fp8"


def test_env_var_bridge_activates_backend() -> None:
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    from repercep.attention.diffusers_backend import maybe_activate_from_env

    with _restore_active_backend(), _temporary_env("REPERCEP_FP8_ATTENTION", "1"):
        activated = maybe_activate_from_env()
        # ``activated`` is True the first time it switches; the dispatcher
        # may already be on repercep_fp8 from an earlier test.
        assert _AttentionBackendRegistry._active_backend.value == "repercep_fp8"
        del activated  # silence flake8/unused


def test_env_var_bridge_inactive_when_unset() -> None:
    # When REPERCEP_FP8_ATTENTION is absent, the bridge must NOT switch the
    # dispatcher — that preserves the default-safe path.
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    from repercep.attention.diffusers_backend import maybe_activate_from_env

    with _restore_active_backend(), _temporary_env("REPERCEP_FP8_ATTENTION", None):
        # Pin a known starting state.
        _AttentionBackendRegistry._active_backend = AttentionBackendName.NATIVE
        switched = maybe_activate_from_env()
        assert not switched
        assert _AttentionBackendRegistry._active_backend == AttentionBackendName.NATIVE


def test_backend_routes_short_seq_to_native() -> None:
    """At sub-crossover sequence lengths the FP8 backend must defer to SDPA.

    This is the safety guarantee: the kernel pays a dispatch tax that only
    amortizes at long S. Hitting it on a 256-token cross-attn would be a
    perf regression.
    """
    if not _HAS_TORCH:
        pytest.skip("torch not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")

    from repercep.attention.diffusers_backend import _repercep_fp8_attention

    # Short S → fallback path. We can't easily detect "fallback was taken"
    # from outside; instead, exercise the call and assert it matches SDPA
    # exactly (the fallback is literally SDPA).
    torch.manual_seed(0)
    b, s, h, d = 1, 256, 4, 64
    dtype = torch.bfloat16
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = _repercep_fp8_attention(q, k, v)
    # Reference: native SDPA on permuted tensors (the diffusers convention).
    q_bhsd = q.permute(0, 2, 1, 3)
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)
    ref = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
    ref = ref.permute(0, 2, 1, 3)
    # At short S we must be exactly the SDPA output (the fallback IS SDPA).
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_TRITON, reason="torch+triton required")
def test_backend_matches_native_on_cosmos_dit_shape() -> None:
    """At a Cosmos-DiT-like shape the FP8 backend must match native SDPA.

    Cosmos-DiT self-attention runs at S ≈ 109 k tokens, head_dim 128 in
    production; for CI we use head_dim 128 at a shorter S that still exceeds
    the FP8 routing threshold. Tolerance follows the existing FP8 tests in
    ``tests/test_attention_fp8.py``.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    # FP8TritonAttention is the AMD fp8e4b8 (fnuz) kernel; it does not
    # compile on NVIDIA Triton.  See F25 / ADR-0006; the H100 sibling
    # is exercised in tests/test_attention_cuda.py.
    if not torch.version.hip:
        pytest.skip("AMD-only FP8 kernel; H100 path lives in test_attention_cuda.py")

    from repercep.attention.diffusers_backend import (
        _FP8_MIN_SEQ_LEN,
        _repercep_fp8_attention,
    )
    from repercep.attention.fp8_triton import FP8TritonAttention

    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 kernel unavailable")

    torch.manual_seed(0)
    # Diffusers layout: (B, S, H, D). Use a small B / H to keep the test fast
    # but S well above the crossover so the FP8 branch is exercised.
    b, s, h, d = 1, _FP8_MIN_SEQ_LEN, 4, 128
    dtype = torch.bfloat16
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype) / 8.0
    k = torch.randn_like(q) / 8.0
    v = torch.randn_like(q)

    out = _repercep_fp8_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()

    # Reference: native SDPA on the same shape (BHSD permute → SDPA → back).
    q_bhsd = q.permute(0, 2, 1, 3)
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)
    ref = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
    ref = ref.permute(0, 2, 1, 3)

    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    # Same tolerance as ``test_fp8_triton_matches_sdpa_within_fp8_tolerance``.
    assert diff.mean().item() / ref_scale < 0.15


def test_cross_attention_shape_falls_back_to_native() -> None:
    """Q != K shape (cross-attention) must NEVER hit the kernel.

    The kernel asserts ``q.shape == k.shape == v.shape``; routing a cross-attn
    call to it would crash. Verifying the fallback path covers this.
    """
    if not _HAS_TORCH:
        pytest.skip("torch not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")

    from repercep.attention.diffusers_backend import _repercep_fp8_attention

    # Different KV seq_len than Q — classic cross-attention shape.
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(1, 8192, 4, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 512, 4, 128, device="cuda", dtype=dtype)
    v = torch.randn(1, 512, 4, 128, device="cuda", dtype=dtype)
    # Must not raise — the routing rejects, fallback executes.
    out = _repercep_fp8_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
