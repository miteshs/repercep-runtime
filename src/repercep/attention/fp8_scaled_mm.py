"""FP8 attention via ``torch._scaled_grouped_mm`` on CDNA3 (gfx942).

CDNA3 / MI300X has native ``mfma_*_fp8_fp8_*`` MFMA instructions (the
``V_MFMA_F32_*_FP8_FP8`` family).  PyTorch surfaces these on ROCm 7.2 through
``torch._scaled_grouped_mm``, the *batched* FP8 GEMM primitive that dispatches
to hipBLASLt under the hood — the same primitive a hand-written HIP kernel
would call.

This op decomposes attention into a pair of FP8 batched GEMMs around a BF16
softmax:

    scores = (Q_f8 @ K_f8.T) * (sQ * sK * scale)
    P      = softmax(scores)         # in BF16 — softmax is range-sensitive
    out    = (P_f8 @ V_f8) * (sP * sV)

The two big GEMMs run in FP8 (~2x math throughput on gfx942 vs BF16).  The
softmax stays in BF16 for accuracy.  This is the same tradeoff the
FlashAttention-3-FP8 paper uses.

ROCm note: MI300X's FP8 format is ``e4m3fnuz`` / ``e5m2fnuz`` (FNUZ = "finite,
no Z" — no infinities, no negative zero).  These are the *physical* types the
MFMA instructions consume.  We cast to ``e4m3fnuz`` for the data path; that is
not a configuration choice on this hardware.

Crossover with the fused kernel: this unfused path materializes the S^2
scores matrix and is bandwidth-bound at long S.  For the long Cosmos DiT
shapes (S up to ~109k) ``FP8TritonAttention`` (fused, online softmax) is
the right path; this module is the simpler default for shorter sequences
or hardware where Triton's FP8 path is not available.

See ADR-0002 and the kernel-layer note in kernels/README.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from repercep.attention.types import AttentionKind, AttentionShape
from repercep.hardware import DType

if TYPE_CHECKING:
    import torch


# torch._scaled_grouped_mm on ROCm wants M, N divisible by 16 (the MFMA tile
# width).  For shorter sequences we'd need to pad; that's a future-work item.
_MIN_DIM = 16

# head_dim values exercised in benchmarks / known-good.  Cosmos DiT uses 128.
_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})

# Kinds: only FULL and CAUSAL today.  NEIGHBORHOOD needs a mask in the
# scores matrix; future work.
_SUPPORTED_KINDS = frozenset({AttentionKind.FULL, AttentionKind.CAUSAL})


class FP8ScaledMMAttention:
    """FP8 attention via ``torch._scaled_grouped_mm`` (hipBLASLt FP8 GEMM)."""

    name = "fp8-scaled-mm"

    # On MI300X (CDNA3) FP8 is the FNUZ variant.  We carry the dtype as
    # ``torch.dtype`` when torch is importable, else ``None``.
    _FP8_E4M3: torch.dtype | None = None

    def __init__(self) -> None:
        try:
            import torch

            self._FP8_E4M3 = torch.float8_e4m3fnuz
            self._has_grouped_mm = hasattr(torch, "_scaled_grouped_mm")
            self._has_scaled_mm = hasattr(torch, "_scaled_mm")
        except ImportError:
            self._has_grouped_mm = False
            self._has_scaled_mm = False

    @property
    def available(self) -> bool:
        """True if FP8 dtypes + at least one scaled-mm op are present."""
        return (self._has_grouped_mm or self._has_scaled_mm) and self._FP8_E4M3 is not None

    def supports(self, shape: AttentionShape, dtype: DType) -> bool:
        accepted_dtypes = {DType.BF16, DType.FP16, DType.FP8_E4M3, DType.FP8_E5M2}
        # The unfused path materializes the (B, H, Sq, Skv) scores tensor in
        # BF16 + FP8.  At Cosmos-DiT scale (B=2, H=32, S=16k) that's >32 GiB,
        # which OOMs even on the 192 GiB MI300X after the model itself is
        # resident.  Gate the path by the materialized-scores byte budget;
        # the fused Triton kernel is the right path above this threshold.
        scores_bytes = shape.batch * shape.heads * shape.seq_len_q * shape.seq_len_kv * 2
        # 4 GiB cap is conservative: leaves room for the FP8 quantized inputs,
        # the P @ V output, and SDPA's own working set.
        max_scores_bytes = 4 * 1024**3
        return (
            self.available
            and dtype in accepted_dtypes
            and shape.head_dim in _SUPPORTED_HEAD_DIMS
            and shape.kind in _SUPPORTED_KINDS
            and shape.seq_len_q >= _MIN_DIM
            and shape.seq_len_kv >= _MIN_DIM
            and shape.seq_len_q % _MIN_DIM == 0
            and shape.seq_len_kv % _MIN_DIM == 0
            and scores_bytes <= max_scores_bytes
        )

    @staticmethod
    def _rowwise_quantize(
        x: torch.Tensor, fp8_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row FP8 quantization.

        Args:
            x: (BH, S, D) BF16 tensor.

        Returns:
            (x_fp8, dequant_scale)
            x_fp8:           (BH, S, D) FP8 in ``fp8_dtype``.
            dequant_scale:   (BH, S)    FP32 — multiplier to recover
                                          original magnitude from the FP8 value.
        """
        import torch

        fp8_max = 240.0  # gfx942 e4m3fnuz absolute max
        target = 0.95 * fp8_max  # leave a little headroom

        # amax over the head_dim axis only — keeps a separate scale per row.
        amax = x.abs().amax(dim=-1).clamp(min=1e-6)  # (BH, S)
        quant_mul = target / amax  # (BH, S) — multiply x by this before cast
        dequant = amax / target  # reciprocal — recover original magnitude

        x_scaled = (x.to(torch.float32) * quant_mul.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
        x_fp8 = x_scaled.to(fp8_dtype)
        return x_fp8, dequant.to(torch.float32)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        import torch

        if self._FP8_E4M3 is None:
            raise RuntimeError(
                "FP8ScaledMMAttention requires torch.float8_e4m3fnuz "
                "(gfx942-class). Pin attention_backend='naive-sdpa' on "
                "hardware without FP8."
            )
        fp8_dtype = self._FP8_E4M3

        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])

        # Layout: Repercep convention is (B, H, S, D).  Collapse BxH into one
        # leading group axis for _scaled_grouped_mm.
        b, h, sq, d = query.shape
        skv = key.shape[2]
        bh = b * h

        compute_dtype = torch.bfloat16
        q = query.to(compute_dtype).reshape(bh, sq, d)
        k = key.to(compute_dtype).reshape(bh, skv, d)
        v = value.to(compute_dtype).reshape(bh, skv, d)

        # Per-row FP8 quantization.
        q_fp8, q_scale = self._rowwise_quantize(q, fp8_dtype)  # q_scale: (bh, sq)
        k_fp8, k_scale = self._rowwise_quantize(k, fp8_dtype)  # k_scale: (bh, skv)
        # Note: V is re-quantized below in the (bh, d, skv) layout that
        # _scaled_grouped_mm requires for the PV matmul.

        # _scaled_grouped_mm wants mat2 physically allocated in
        # (BH, N, K) layout, then passed as .transpose(-1, -2).  Quantizing
        # K and V in this transposed orientation costs one extra reshape
        # but matches the hipBLASLt FP8 GEMM ABI.

        # First GEMM: scores = Q @ K^T.  M=Sq, N=Skv, K=D.
        # K is (BH, Skv, D) — that IS the (BH, N, K) layout already.
        scores_raw = torch._scaled_grouped_mm(
            q_fp8,  # (BH, Sq, D) — (BH, M, K)
            k_fp8.transpose(-1, -2),  # logical (BH, D, Skv), passed transposed
            scale_a=q_scale,  # (BH, Sq) — (BH, M)
            scale_b=k_scale,  # (BH, Skv) — (BH, N)
            out_dtype=compute_dtype,
        )  # (BH, Sq, Skv)

        # Apply softmax temperature.
        scores = scores_raw * scale

        if causal:
            mask = torch.ones(sq, skv, device=query.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax in BF16 (range-stable).
        p = torch.softmax(scores, dim=-1)

        # Second GEMM: out = P @ V.  M=Sq, N=D, K=Skv.
        # mat2 must be (BH, N, K) physical = (BH, D, Skv) physical.  Our V
        # is (BH, Skv, D) which is (BH, K, N) — wrong layout.  We need to
        # rematerialize V in (BH, D, Skv) layout, then pass with .transpose
        # to get the "mat2 transposed" form expected by the API.
        p_fp8, p_scale = self._rowwise_quantize(p, fp8_dtype)

        # Re-quantize V in the (BH, D, Skv) physical orientation.  Pre-
        # computing this once outside the loop would be ideal; for now we
        # accept the extra cost (it's a 4D->4D copy, dominated by the GEMM).
        v_reordered = v.transpose(-1, -2).contiguous()  # (bh, d, skv)
        v_fp8_nk, v_scale_n = self._rowwise_quantize(v_reordered, fp8_dtype)

        out = torch._scaled_grouped_mm(
            p_fp8,  # (bh, sq, skv)
            v_fp8_nk.transpose(-1, -2),  # logical (bh, skv, d), but physical (bh, d, skv)
            scale_a=p_scale,  # (bh, sq)
            scale_b=v_scale_n,  # (bh, d) — N of the result
            out_dtype=compute_dtype,
        )  # (bh, sq, d)

        out_4d = out.reshape(b, h, sq, d)
        return out_4d.to(query.dtype)
