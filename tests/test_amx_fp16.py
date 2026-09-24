"""Tests for the AMX_FP16 flash-attention wrapper (Granite Rapids kernel).

Sibling of the BF16 wrapper tests in ``tests/test_attention_cpu.py``.  The
end-state on the current dev hosts (SPR with masked AMX, no ``amx_fp16``)
is identical to the BF16 kernel today: the wrapper imports cleanly,
``available`` is ``False``, and ``supports`` short-circuits to ``False`` for
every shape/dtype.  When run on a real GNR host the kernel is built and
``available`` flips to ``True``.
"""

from __future__ import annotations

import platform

import pytest

from repercep.attention.amx_fp16_flash import (
    AMXFP16FlashAttention,
    _detect_amx_fp16,
)
from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="AMX detection via /proc/cpuinfo is Linux-only",
)


# --- Construction / availability --------------------------------------------


def test_construction_does_not_raise() -> None:
    """The wrapper must construct cleanly even when the native module is absent."""
    op = AMXFP16FlashAttention()
    assert op.name == "amx-fp16-flash"
    assert isinstance(op.available, bool)


@linux_only
def test_available_is_false_on_this_vm() -> None:
    """SPR + masked-AMX VM has no ``amx_fp16`` -> wrapper is unavailable.

    On a real Granite Rapids host this flips to True; on every CI host today
    (SPR, EMR, non-Intel) it stays False with a populated ``_import_error``.
    """
    op = AMXFP16FlashAttention()
    if not op.available:
        # Either the cpuinfo probe failed or the extension module is missing.
        # Both leave a human-readable string behind for diagnostics.
        assert op._import_error is not None
        assert op._import_error  # non-empty


@linux_only
def test_detect_amx_fp16_is_false_on_spr() -> None:
    """Probe-level cross-check: SPR/EMR must not advertise ``amx_fp16``."""
    # GNR is not yet shipping; any host running this suite today reports False.
    # If this ever flips True on CI, the dev hardware actually grew GNR -- in
    # which case the test should be amended to assert kernel buildability.
    assert _detect_amx_fp16() is False


# --- supports() contract ----------------------------------------------------


def _shape(head_dim: int = 64, kind: AttentionKind = AttentionKind.FULL) -> AttentionShape:
    return AttentionShape(
        batch=1,
        heads=4,
        seq_len_q=128,
        seq_len_kv=128,
        head_dim=head_dim,
        kind=kind,
    )


def test_supports_requires_available() -> None:
    """When the kernel is not built, ``supports`` is False for every input."""
    op = AMXFP16FlashAttention()
    if op.available:
        pytest.skip("kernel built; covered by GNR-only suite")
    assert op.supports(_shape(64), DType.FP16) is False
    assert op.supports(_shape(128), DType.FP16) is False


def test_supports_rejects_non_fp16_dtypes() -> None:
    """BF16 belongs to the sibling wrapper; FP32/INT8 belong to the SDPA floor."""
    op = AMXFP16FlashAttention()
    for dtype in (DType.FP32, DType.BF16, DType.INT8, DType.FP8_E4M3, DType.FP8_E5M2):
        # Regardless of availability, the dtype gate alone rules these out.
        assert op.supports(_shape(64), dtype) is False


def test_supports_rejects_unsupported_head_dim() -> None:
    """Only head_dim in {64, 128} hits the compiled tile geometry."""
    op = AMXFP16FlashAttention()
    for head_dim in (16, 32, 48, 96, 192, 256):
        assert op.supports(_shape(head_dim), DType.FP16) is False


def test_supports_rejects_neighborhood_kind() -> None:
    """NATTEN-style local attention has no AMX path here."""
    op = AMXFP16FlashAttention()
    assert (
        op.supports(_shape(64, AttentionKind.NEIGHBORHOOD), DType.FP16) is False
    )


# --- Module-surface parity with the BF16 sibling ----------------------------


def test_module_surface_parity_with_bf16_sibling() -> None:
    """The FP16 wrapper exposes the same public surface as the BF16 wrapper."""
    from repercep.attention import amx_flash, amx_fp16_flash

    # Class names diverge (BF16 vs FP16); we check the *attribute set* matches
    # so the registry can swap them transparently.
    bf16_cls = amx_flash.AMXFlashAttention
    fp16_cls = amx_fp16_flash.AMXFP16FlashAttention

    public_attrs = {
        a for a in dir(bf16_cls) if not a.startswith("_") or a in {"__call__"}
    }
    for attr in public_attrs:
        assert hasattr(fp16_cls, attr), (
            f"FP16 wrapper missing public attr `{attr}` present on BF16 sibling"
        )

    # Op names follow the same kebab-cased "amx-<dtype>-flash" pattern.
    assert bf16_cls.name == "amx-bf16-flash"
    assert fp16_cls.name == "amx-fp16-flash"
