"""The NVIDIA CUDA backend — Hopper-first, parallel to ROCm.

PyTorch on CUDA exposes NVIDIA devices through ``torch.cuda``; the
distinguishing signal that this is real NVIDIA silicon (and not ROCm pretending
to be CUDA) is a non-empty ``torch.version.cuda`` *and* an empty
``torch.version.hip``.

This backend is the parallel of :class:`repercep.backend.rocm.ROCmBackend`.  AMD
remains the lead workload (ADR-0001); NVIDIA lands as a parallel track per
ADR-0006, closing the "Revisit if" clause of ADR-0001 and the system-vs-system
asymmetry called out in ``docs/METHODOLOGY.md`` §3.

Architecture mapping.  ``DeviceArch.gfx_id`` is the canonical micro-arch
identifier; on NVIDIA we encode the CUDA capability as ``sm<MAJOR><MINOR>``
optionally suffixed with ``a`` to mark the Hopper-specific ``a``-variant (the
one that exposes ``wgmma``/TMA and that Triton + CUTLASS target).  ``sm_90``
and ``sm_90a`` are treated as the same arch for selection purposes; the
``a`` suffix is added when the device is Hopper because that is what the
kernel-layer code expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.attention.registry import select_attention_op
from repercep.hardware import (
    H100,
    H200,
    BackendCapabilities,
    DeviceArch,
    DeviceSpec,
    DType,
    Vendor,
)

if TYPE_CHECKING:
    import torch

    from repercep.attention.protocol import AttentionOp
    from repercep.attention.types import AttentionShape


# Compute-capability → known arch.  Hopper variants carry ``a`` because the
# kernel-layer code (Triton, CUTLASS) compiles against ``sm_90a`` to unlock
# WGMMA + TMA — ``sm_90`` without the suffix is the strict-IEEE subset.  We
# treat the device as Hopper either way; selection is by gfx_id string so the
# canonical entries below use ``sm90a``.
#
# Ada (sm_89) and Ampere (sm_80, sm_86) are listed for completeness — they
# don't have FP8 MFMA but they support FA-2 and FA-3 (FA-3 dropped Ada/Ampere
# support upstream but the FA-2 fallback in HopperFlashAttention still wins on
# them).  Blackwell (sm_100, sm_120) will land here when public.
_SM_TO_ARCH: dict[str, DeviceArch] = {
    "sm_90": H100,
    "sm_90a": H100,
    # H200 shares the H100 sm_90a target; the differences are HBM size + BW.
    # We can't distinguish them from compute-capability alone, so we treat any
    # 80-GiB Hopper as H100 and any >80-GiB Hopper as H200.  See devices().
    "sm_80": DeviceArch(Vendor.NVIDIA, "sm80", "Ampere"),
    "sm_86": DeviceArch(Vendor.NVIDIA, "sm86", "Ampere"),
    "sm_89": DeviceArch(Vendor.NVIDIA, "sm89", "Ada"),
}


# Heuristic: H200's HBM3e is 141 GiB nominal; anything Hopper above ~96 GiB is
# H200, otherwise H100.  This is a heuristic for display, not for kernel
# selection (kernels select by sm_90a regardless).
_H200_MEMORY_THRESHOLD_GIB = 96.0


class CUDABackend:
    """NVIDIA CUDA backend.  Lead target: H100 / H200 (sm_90a, Hopper).

    Ampere and Ada are supported by attribute — the same Backend Protocol, the
    same attention dispatch, with FA-3 falling back to FA-2 and the FP8 Triton
    kernel disqualifying itself (no FP8 MFMA on those arches).  The lead
    measurement target is the Hopper H100 / H200 line per ADR-0006.
    """

    vendor: Vendor = Vendor.NVIDIA
    name: str = "cuda"

    def is_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        # ``torch.version.cuda`` is the CUDA toolkit the wheel was built
        # against; ``torch.version.hip`` is empty for genuine CUDA wheels.
        # ROCm wheels populate ``hip`` and use the ``torch.cuda`` namespace
        # as a compat layer — we exclude those here so a ROCm host doesn't
        # advertise both backends.
        return (
            torch.version.cuda is not None
            and not torch.version.hip
            and torch.cuda.is_available()
        )

    def devices(self) -> tuple[DeviceSpec, ...]:
        import torch

        specs: list[DeviceSpec] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # ``capability`` is (major, minor); torch exposes both via attrs.
            sm = f"sm_{props.major}{props.minor}"
            arch = _SM_TO_ARCH.get(sm)
            if arch is None:
                arch = DeviceArch(Vendor.NVIDIA, sm.removeprefix("sm_"), "unknown")
            # H100 vs H200 disambiguation by HBM size.  Both report sm_90 /
            # sm_90a; the only public signal is memory.
            elif arch is H100 and props.total_memory / 1024**3 > _H200_MEMORY_THRESHOLD_GIB:
                arch = H200
            specs.append(
                DeviceSpec(
                    index=i,
                    arch=arch,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    multi_processor_count=props.multi_processor_count,
                )
            )
        return tuple(specs)

    def capabilities(self) -> BackendCapabilities:
        # Build the attention-op order without importing the kernels themselves
        # — selection is by the registry, but capabilities() is consulted by
        # ``repercep info`` and tests as a declarative summary.
        from repercep.attention.hopper_flash import HopperFlashAttention

        flash = HopperFlashAttention()
        # Order matches the registry's NVIDIA branch.  TE goes last in the
        # display because it's optional + the install is heavy; the actual
        # selection promotes it when REPERCEP_FP8_ATTENTION=te is set.
        ops: list[str] = []
        if flash.available:
            ops.append(flash.name)
        ops.append("naive-sdpa")
        # FP8 ops are advertised when their kernels are importable; the
        # registry will only route to them under the REPERCEP_FP8_ATTENTION
        # env var.
        try:
            from repercep.attention.fp8_hopper_triton import FP8HopperTritonAttention

            if FP8HopperTritonAttention().available:
                ops.insert(0, "fp8-hopper-triton-flash")
        except ImportError:
            pass
        try:
            from repercep.attention.transformer_engine import TransformerEngineAttention

            if TransformerEngineAttention().available:
                ops.insert(0, "transformer-engine-fp8")
        except ImportError:
            pass

        # Hopper / Ada / Ampere all support BF16 + FP16 + FP32; only Hopper
        # and Ada have hardware FP8 (e4m3fn + e5m2, the IEEE-ish variants —
        # not the AMD FNUZ variants).  We expose FP8 dtypes only when the
        # device is actually Hopper/Ada.
        dtypes: set[DType] = {DType.FP32, DType.FP16, DType.BF16, DType.INT8}
        supports_fp8 = False
        for spec in self.devices():
            if spec.arch.gfx_id.startswith(("sm90", "sm89")):
                supports_fp8 = True
                dtypes.add(DType.FP8_E4M3)
                dtypes.add(DType.FP8_E5M2)
                break

        return BackendCapabilities(
            dtypes=frozenset(dtypes),
            supports_flash_attention=flash.available,
            supports_fp8=supports_fp8,
            supports_torch_compile=True,
            attention_ops=tuple(ops),
        )

    def torch_device(self, index: int = 0) -> torch.device:
        import torch

        return torch.device("cuda", index)

    def default_dtype(self) -> DType:
        return DType.BF16

    def attention_op(self, shape: AttentionShape, dtype: DType) -> AttentionOp:
        # If multiple NVIDIA devices are present they should share a uarch on
        # this host; pick the first.  If somehow they differ, ``select_attention_op``
        # is keyed on arch alone, and the strongest device's arch is the right
        # choice for kernel compilation.
        devices = self.devices()
        # No devices: registry will route to naive-sdpa via NVIDIA branch.
        arch = H100 if not devices else devices[0].arch
        return select_attention_op(arch, shape, dtype)
