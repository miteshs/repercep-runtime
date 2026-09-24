"""The AMD ROCm backend — Repercep's lead compute target.

PyTorch on ROCm exposes HIP devices through the ``torch.cuda`` namespace, so
device handles look CUDA-shaped; the distinguishing signal that this is really
AMD silicon is a non-empty ``torch.version.hip``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.attention.registry import select_attention_op
from repercep.attention.rocm_flash import ROCmFlashAttention
from repercep.hardware import (
    MI250X,
    MI300X,
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

# Map the base gfx target (gcnArchName before the first ':') to a known arch.
_GFX_TO_ARCH: dict[str, DeviceArch] = {
    "gfx942": MI300X,
    "gfx90a": MI250X,
}


class ROCmBackend:
    """AMD ROCm backend.  Lead target: MI300X (gfx942, CDNA3)."""

    vendor: Vendor = Vendor.AMD
    name: str = "rocm"

    def is_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.version.hip) and torch.cuda.is_available()

    def devices(self) -> tuple[DeviceSpec, ...]:
        import torch

        specs: list[DeviceSpec] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gfx = props.gcnArchName.split(":")[0]
            arch = _GFX_TO_ARCH.get(gfx, DeviceArch(Vendor.AMD, gfx, "unknown"))
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
        flash = ROCmFlashAttention()
        ops = ("rocm-ck-flash", "naive-sdpa") if flash.available else ("naive-sdpa",)
        return BackendCapabilities(
            # CDNA3 (gfx942) has native FP8 (OCP e4m3/e5m2) MFMA instructions.
            dtypes=frozenset(
                {
                    DType.FP32,
                    DType.FP16,
                    DType.BF16,
                    DType.FP8_E4M3,
                    DType.FP8_E5M2,
                    DType.INT8,
                }
            ),
            supports_flash_attention=flash.available,
            supports_fp8=True,
            supports_torch_compile=True,
            attention_ops=ops,
        )

    def torch_device(self, index: int = 0) -> torch.device:
        import torch

        return torch.device("cuda", index)

    def default_dtype(self) -> DType:
        return DType.BF16

    def attention_op(self, shape: AttentionShape, dtype: DType) -> AttentionOp:
        return select_attention_op(MI300X, shape, dtype)
