"""Tests for the custom AMX INT8 flash-attention wrapper.

Sibling of the BF16 flash-attention tests in ``tests/test_attention_cpu.py``.
Hosts without the ``amx_int8`` flag exit the kernel-call paths cleanly via
the wrapper's `available` probe; the native module is imported lazily and
``pytest.importorskip`` short-circuits the correctness test if it isn't
built.
"""

from __future__ import annotations

import platform

import pytest

from repercep.attention.amx_int8_flash import AMXInt8FlashAttention, _detect_amx_int8
from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="AMX detection via /proc/cpuinfo is Linux-only",
)


# --- Wrapper construction ---------------------------------------------------


def test_amx_int8_flash_constructs_cleanly() -> None:
    """Construction must succeed on every host, AMX-capable or not."""
    op = AMXInt8FlashAttention()
    assert op.name == "amx-int8-flash"
    # ``available`` returns a bool regardless of host capability.
    assert isinstance(op.available, bool)


@linux_only
def test_amx_int8_flash_unavailable_on_non_amx_host() -> None:
    """On a host without amx_int8 in /proc/cpuinfo, ``available`` is False
    and the import-error string is populated."""
    if _detect_amx_int8():
        pytest.skip(
            "Host advertises amx_int8; this is the negative-path test only."
        )

    op = AMXInt8FlashAttention()
    assert op.available is False
    assert op._import_error is not None
    assert "amx_int8" in op._import_error


# --- supports() shape/dtype gating ------------------------------------------


def test_amx_int8_flash_supports_rejects_when_unavailable() -> None:
    """``supports`` must be False whenever ``available`` is False, even
    on otherwise-valid (BF16, head_dim 64) shapes."""
    op = AMXInt8FlashAttention()
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64,
        kind=AttentionKind.FULL,
    )
    if not op.available:
        assert op.supports(shape, DType.BF16) is False


def test_amx_int8_flash_dtype_gating() -> None:
    """When the op IS available, only BF16 input is supported.  When it
    is NOT available, ``supports`` returns False for every dtype."""
    op = AMXInt8FlashAttention()
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64,
        kind=AttentionKind.FULL,
    )
    if op.available:
        assert op.supports(shape, DType.BF16) is True
        assert op.supports(shape, DType.FP16) is False
        assert op.supports(shape, DType.FP32) is False
        assert op.supports(shape, DType.INT8) is False
        assert op.supports(shape, DType.FP8_E4M3) is False
    else:
        for dt in (DType.BF16, DType.FP16, DType.FP32, DType.INT8):
            assert op.supports(shape, dt) is False


def test_amx_int8_flash_head_dim_gating() -> None:
    """Only head_dim in {64, 128} is supported.  Other head dims must be
    rejected even when the op is available."""
    op = AMXInt8FlashAttention()
    for hd in (32, 48, 80, 96, 256):
        shape = AttentionShape(
            batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=hd,
            kind=AttentionKind.FULL,
        )
        assert op.supports(shape, DType.BF16) is False


def test_amx_int8_flash_kind_gating() -> None:
    """FULL and CAUSAL are supported; NEIGHBORHOOD is not."""
    op = AMXInt8FlashAttention()
    nbh = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64,
        kind=AttentionKind.NEIGHBORHOOD,
    )
    assert op.supports(nbh, DType.BF16) is False
    if op.available:
        full = AttentionShape(
            batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64,
            kind=AttentionKind.FULL,
        )
        causal = AttentionShape(
            batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64,
            kind=AttentionKind.CAUSAL,
        )
        assert op.supports(full, DType.BF16) is True
        assert op.supports(causal, DType.BF16) is True


# --- Module surface mirrors the BF16 sibling --------------------------------


def test_amx_int8_flash_module_surface_matches_bf16_sibling() -> None:
    """The INT8 wrapper must expose the same attribute surface as the
    BF16 sibling so the registry can dispatch them interchangeably."""
    from repercep.attention import amx_flash, amx_int8_flash

    bf16_cls = amx_flash.AMXFlashAttention
    int8_cls = amx_int8_flash.AMXInt8FlashAttention

    bf16_op = bf16_cls()
    int8_op = int8_cls()

    # Class-level: both expose ``name`` as a string constant.
    assert isinstance(bf16_cls.name, str)
    assert isinstance(int8_cls.name, str)
    # The names disambiguate the two implementations.
    assert bf16_cls.name != int8_cls.name

    # Instance surface: ``available``, ``supports``, ``__call__``.
    for attr in ("available", "supports", "__call__"):
        assert hasattr(bf16_op, attr), f"BF16 sibling missing {attr}"
        assert hasattr(int8_op, attr), f"INT8 wrapper missing {attr}"

    # Internal state mirrors: ``_fn``, ``_import_error`` for the negative path.
    assert hasattr(int8_op, "_fn")
    assert hasattr(int8_op, "_import_error")
    assert hasattr(bf16_op, "_fn")
    assert hasattr(bf16_op, "_import_error")


# --- Native-module correctness path (skipped on non-AMX hosts) --------------


def test_amx_int8_flash_native_call_when_built() -> None:
    """On a host where the C++ extension is built, the kernel runs end to
    end and produces shape/dtype-correct output.  Skipped via
    ``pytest.importorskip`` everywhere else."""
    # Skip if the extension module isn't built; this is the common case
    # in CI and on the dev VM.  ``importorskip`` returns the module on
    # success and raises Skipped on failure -- both leave the test green.
    pytest.importorskip("kernels.cpu.amx_int8_attn._native")

    import torch

    op = AMXInt8FlashAttention()
    if not op.available:
        pytest.skip("amx_int8 not in /proc/cpuinfo on this host")

    torch.manual_seed(7)
    q = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
    out = op(q, k, v, causal=False, scale=None)
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
