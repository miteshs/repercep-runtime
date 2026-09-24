"""Tests for the CPU attention ops.

Mirrors ``tests/test_attention_cuda.py``.  GPU-specific tests skip on
CPU-only hosts; CPU tests run on every host.
"""

from __future__ import annotations

import platform

import pytest

from repercep.attention.amx_sdpa import AMXSDPAAttention
from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import SAPPHIRE_RAPIDS, DType, Vendor

linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="AMX detection via /proc/cpuinfo is Linux-only",
)


# --- AMXSDPAAttention -------------------------------------------------------


def test_amx_sdpa_available() -> None:
    """SDPA on CPU is the always-available floor when torch is importable."""
    op = AMXSDPAAttention()
    assert op.available is True
    assert op.name == "amx-sdpa"


def test_amx_sdpa_supports_expected_dtypes() -> None:
    """BF16 / FP16 / FP32 / INT8 all supported on the full kind set."""
    op = AMXSDPAAttention()
    full = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    causal = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.CAUSAL
    )
    for dtype in (DType.FP32, DType.FP16, DType.BF16, DType.INT8):
        assert op.supports(full, dtype) is True
        assert op.supports(causal, dtype) is True


def test_amx_sdpa_rejects_unsupported_dtype() -> None:
    """FP8 has no CPU ISA today, so the op disqualifies itself."""
    op = AMXSDPAAttention()
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    assert op.supports(shape, DType.FP8_E4M3) is False
    assert op.supports(shape, DType.FP8_E5M2) is False


def test_amx_sdpa_matches_torch_reference() -> None:
    """Output equals ``torch.nn.functional.scaled_dot_product_attention``."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    k = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    v = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    out = op(q, k, v, causal=False, scale=None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_amx_sdpa_causal_matches_reference() -> None:
    """Causal mask propagates through correctly."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(1)
    q = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    out = op(q, k, v, causal=True, scale=None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_amx_sdpa_bf16_roundtrip() -> None:
    """BF16 path runs end-to-end (oneDNN auto-dispatches to AMX on SPR+)."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(2)
    q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    out = op(q, k, v, causal=False, scale=None)
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape


# --- AMX flash attention (custom kernel) -------------------------------------


@linux_only
def test_amx_flash_unavailable_without_kernel_module() -> None:
    """When the C++ extension is not built, ``available`` is False cleanly."""
    from repercep.attention.amx_flash import AMXFlashAttention

    op = AMXFlashAttention()
    # If the kernel _native module isn't built (the common case in CI),
    # the op must declare itself unavailable rather than throwing on import.
    if not op.available:
        assert op._import_error is not None
    # If the kernel IS built (only on a dev host that ran `make kernels-cpu`),
    # the op is callable and produces correct output for a small shape.
    else:
        import torch

        torch.manual_seed(3)
        q = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        out = op(q, k, v, causal=False, scale=None)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # AMX BF16 has lower precision than the reference SDPA path; assert
        # a relaxed bound matching typical FA-2 vs SDPA tolerances.
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# --- IPEX flash attention ---------------------------------------------------


def test_ipex_flash_handles_missing_install() -> None:
    """When IPEX is not installed, ``available`` is False — no crash."""
    from repercep.attention.ipex_flash import IPEXFlashAttention

    op = IPEXFlashAttention()
    # The op must construct cleanly whether IPEX is present or not.  We do
    # not assert one way or the other on the result of ``available``; it's
    # environment-dependent.
    assert isinstance(op.available, bool)


# --- Registry routing -------------------------------------------------------


def test_registry_intel_branch_returns_supported_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Intel arch with AMX env unset, SDPA is the selected op."""
    monkeypatch.delenv("REPERCEP_AMX_ATTENTION", raising=False)
    from repercep.attention.registry import select_attention_op

    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    op = select_attention_op(SAPPHIRE_RAPIDS, shape, DType.BF16)
    # When the AMX flash kernel isn't built, the floor wins.  We accept
    # either "amx-sdpa" or the custom kernel name — both are valid INTEL
    # branch outcomes.
    assert op.name in ("amx-sdpa", "amx-bf16-flash", "ipex-flash")


def test_registry_unknown_intel_arch_still_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Generic AVX-512 host still gets a valid Intel-branch op."""
    monkeypatch.delenv("REPERCEP_AMX_ATTENTION", raising=False)
    from repercep.attention.registry import select_attention_op
    from repercep.hardware import DeviceArch

    skylake_avx512 = DeviceArch(Vendor.INTEL, "avx512", "Generic AVX-512")
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=64, seq_len_kv=64, head_dim=64, kind=AttentionKind.FULL
    )
    op = select_attention_op(skylake_avx512, shape, DType.BF16)
    assert op.name in ("amx-sdpa", "naive-sdpa")


def test_registry_int8_env_considers_int8_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REPERCEP_AMX_ATTENTION=int8`` lists the INT8 op first; on a host that
    lacks the built kernel it falls through to the SDPA floor cleanly."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "int8")
    # Re-import the registry module so the env read at module load picks up
    # the patched value (the env is captured at import time).
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    op = reg.select_attention_op(SAPPHIRE_RAPIDS, shape, DType.BF16)
    # Either the INT8 kernel wins (only on real AMX_INT8 silicon with the
    # extension built) or the chain falls through.  IPEX and BF16-flash are
    # excluded from the candidate list under env=int8.
    assert op.name in ("amx-int8-flash", "amx-sdpa", "naive-sdpa")


def test_registry_fp16_env_on_spr_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``=fp16`` on SPR (no amx_fp16): FP16 kernel disqualifies, falls through."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "fp16")
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    op = reg.select_attention_op(SAPPHIRE_RAPIDS, shape, DType.FP16)
    # SPR has no amx_fp16 flag, so AMXFP16FlashAttention disqualifies; under
    # env=fp16 the BF16 sibling and IPEX are also excluded.  Floor wins.
    assert op.name in ("amx-fp16-flash", "amx-sdpa", "naive-sdpa")


def test_registry_int8_env_routes_unsupported_dtype_to_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT8 kernel advertises BF16 input only; an FP32 call must still route."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "int8")
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    op = reg.select_attention_op(SAPPHIRE_RAPIDS, shape, DType.FP32)
    # INT8 disqualifies on FP32 input (supports() returns False); SDPA floor
    # accepts FP32 and wins.
    assert op.name in ("amx-sdpa", "naive-sdpa")


# --- Capability-gate monkeypatch routing ------------------------------------
#
# These tests verify the registry dispatch logic *independently* of whether the
# host CPU exposes the AMX flag or whether the C++ extension is built.  On the
# current dev VM both are masked, so without these monkeypatches a routing
# regression in the registry would slip through unnoticed.  The pattern: patch
# the wrapper's two availability gates (the /proc/cpuinfo detector AND the
# wrapper-class ``available`` property) so the wrapper claims it is callable;
# then assert the registry actually picks it.

_BF16_SHAPE = AttentionShape(
    batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
)


def _force_wrapper_available(monkeypatch: pytest.MonkeyPatch, module_name: str, cls_name: str,
                             detector_name: str) -> None:
    """Patch both gates so ``cls`` reports ``available`` True without the .so."""
    import importlib

    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)

    # Gate 1: the /proc/cpuinfo probe at __init__.
    monkeypatch.setattr(mod, detector_name, lambda: True)
    # Gate 2: the ``available`` property reads ``self._fn is not None``.  We
    # cannot set ``_fn`` before the instance exists, so override the property
    # at the class level to a plain True.  ``supports()`` reads ``available``
    # so this carries through into the registry's chain.
    monkeypatch.setattr(cls, "available", True)


def test_int8_routing_when_amx_int8_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMX_INT8 detected + env=int8 -> registry picks the INT8 kernel."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "int8")
    _force_wrapper_available(
        monkeypatch,
        "repercep.attention.amx_int8_flash",
        "AMXInt8FlashAttention",
        "_detect_amx_int8",
    )
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    op = reg.select_attention_op(SAPPHIRE_RAPIDS, _BF16_SHAPE, DType.BF16)
    assert op.name == "amx-int8-flash"


def test_fp16_routing_when_amx_fp16_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMX_FP16 detected + env=fp16 -> registry picks the FP16 kernel."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "fp16")
    _force_wrapper_available(
        monkeypatch,
        "repercep.attention.amx_fp16_flash",
        "AMXFP16FlashAttention",
        "_detect_amx_fp16",
    )
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    op = reg.select_attention_op(SAPPHIRE_RAPIDS, _BF16_SHAPE, DType.FP16)
    assert op.name == "amx-fp16-flash"


def test_bf16_routing_when_amx_bf16_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMX_BF16 detected + env=amx -> registry picks the BF16 kernel."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "amx")
    _force_wrapper_available(
        monkeypatch,
        "repercep.attention.amx_flash",
        "AMXFlashAttention",
        "_detect_amx_bf16",
    )
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    op = reg.select_attention_op(SAPPHIRE_RAPIDS, _BF16_SHAPE, DType.BF16)
    assert op.name == "amx-bf16-flash"


def test_int8_fallthrough_when_amx_int8_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """env=int8 but AMX_INT8 absent -> chain falls through to SDPA/naive."""
    monkeypatch.setenv("REPERCEP_AMX_ATTENTION", "int8")
    # Force the detector to report False (mirrors a non-AMX_INT8 host) — this
    # is the dev-VM state today; we make it explicit so the test is hermetic.
    import repercep.attention.amx_int8_flash as int8_mod

    monkeypatch.setattr(int8_mod, "_detect_amx_int8", lambda: False)
    import importlib

    import repercep.attention.registry as reg

    importlib.reload(reg)

    op = reg.select_attention_op(SAPPHIRE_RAPIDS, _BF16_SHAPE, DType.BF16)
    # INT8 disqualifies (available False because detector returned False), so
    # the floor wins.  IPEX is excluded from the candidate list under env=int8.
    assert op.name in ("amx-sdpa", "naive-sdpa")
