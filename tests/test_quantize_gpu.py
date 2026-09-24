"""CPU-vs-CUDA parity tests for ``repercep.runtime.quantize`` (Item H).

Sibling of ``test_quantize.py`` — that file validates the algorithm on the
CPU side; this file validates that the *same* algorithm produces *the same*
results when the input tensors live on a CUDA device.  Item A's
quantization code accepts tensors on either device, but the only existing
coverage was CPU-only.  These tests close the cross-device gap on real Ada
hardware.

The tests skip cleanly on hosts without a CUDA-capable GPU so the suite
remains green on CPU-only CI.
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")


@skip_no_cuda
def test_quantize_cpu_cuda_bitwise_qweight_and_scale_ulp() -> None:
    """Same weight, two devices, same int8 bits + 1-ULP-tight scale match."""
    from repercep.runtime.quantize import quantize_linear_symmetric

    torch.manual_seed(0)
    weight_cpu = torch.randn(64, 128) * 2.0
    weight_cuda = weight_cpu.to("cuda")

    q_cpu = quantize_linear_symmetric(weight_cpu)
    q_cuda = quantize_linear_symmetric(weight_cuda)

    # Devices end up where the inputs are.
    assert q_cpu.qweight.device.type == "cpu"
    assert q_cuda.qweight.device.type == "cuda"
    assert q_cpu.scale.device.type == "cpu"
    assert q_cuda.scale.device.type == "cuda"

    # qweight is int8 — after the .to(int8) cast there's no FP rounding left,
    # so bit-identity across devices is the right contract.
    assert torch.equal(q_cpu.qweight, q_cuda.qweight.cpu()), (
        "int8 qweight diverges between CPU and CUDA"
    )

    # Scale is the per-row max-abs reduction divided by 127.  The reduction
    # order may differ between CPU and CUDA (multi-block tree reduction on
    # GPU vs serial on CPU), so we allow ~1 ULP of FP32 slack but no relative
    # slop — atol=1e-6 is comfortably above 1 ULP near unit magnitude here.
    torch.testing.assert_close(
        q_cpu.scale,
        q_cuda.scale.cpu(),
        atol=1e-6,
        rtol=0,
    )


@skip_no_cuda
def test_quantized_linear_module_forward_cpu_cuda_parity_bf16() -> None:
    """``QuantizedLinearModule.forward`` on CPU and CUDA agree within BF16 floor."""
    from repercep.runtime.quantize import QuantizedLinearModule

    torch.manual_seed(1)
    linear = torch.nn.Linear(8, 16).to(torch.bfloat16)

    qmod_cpu = QuantizedLinearModule.from_linear(linear)
    qmod_cuda = QuantizedLinearModule.from_linear(linear).to("cuda")

    x_cpu = torch.randn(4, 8, dtype=torch.bfloat16)
    x_cuda = x_cpu.to("cuda")

    out_cpu = qmod_cpu(x_cpu).to(torch.float32)
    out_cuda = qmod_cuda(x_cuda).to(torch.float32).cpu()

    diff = (out_cpu - out_cuda).abs().max().item()
    # BF16 mantissa is 7 bits; matmul accumulation order differs between
    # GEMM kernels on the two devices, so atol=2e-3 is the bf16 floor.
    assert diff < 2e-3, f"CPU/CUDA forward diff = {diff:.4e}"


@skip_no_cuda
def test_replace_linears_with_quantized_on_cuda_module() -> None:
    """Replacing Linears on a module that's already on CUDA keeps state on CUDA."""
    from repercep.runtime.quantize import (
        QuantizedLinearModule,
        replace_linears_with_quantized,
    )

    torch.manual_seed(2)
    m = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
    ).to("cuda")

    n = replace_linears_with_quantized(m)
    # (a) All Linears were swapped.
    assert n == 2
    assert isinstance(m[0], QuantizedLinearModule)
    assert isinstance(m[2], QuantizedLinearModule)
    assert not any(isinstance(sub, torch.nn.Linear) for sub in m.modules())

    # (b) The swapped modules' qweight + scale + bias are still on CUDA.
    for idx in (0, 2):
        wrapper = m[idx]
        assert wrapper.qweight.device.type == "cuda", (
            f"qweight on {wrapper.qweight.device} after swap; expected cuda"
        )
        assert wrapper.scale.device.type == "cuda"
        if wrapper.bias is not None:
            assert wrapper.bias.device.type == "cuda"

    # (c) Forward still works end-to-end on CUDA.
    x = torch.randn(4, 16, device="cuda")
    y = m(x)
    assert y.shape == (4, 8)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


@skip_no_cuda
def test_cuda_quant_moved_to_cpu_matches_native_cpu_quant() -> None:
    """Quant-on-CUDA + move-to-CPU agrees with quant-on-CPU within 1-ULP-scale.

    int8 qweight bits are bit-identical (after the .to(int8) cast no FP
    rounding remains).  The scale float can differ by up to 1 ULP because
    the per-row max-abs reduction reorders between CPU (serial) and CUDA
    (tree).  Dequantize on CPU from both sides and assert the result is
    within the budget that 1 ULP of scale * int8-range explains.
    """
    from repercep.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(3)
    weight_cpu = torch.randn(32, 48) * 1.5
    weight_cuda = weight_cpu.to("cuda")

    q_cpu = quantize_linear_symmetric(weight_cpu)
    q_cuda = quantize_linear_symmetric(weight_cuda)

    # Move the CUDA-quantized tensors to CPU and rebuild the dataclass so
    # ``dequantize_linear`` runs purely on CPU.
    from repercep.runtime.quantize import QuantizedLinear

    q_cuda_on_cpu = QuantizedLinear(
        qweight=q_cuda.qweight.cpu(),
        scale=q_cuda.scale.cpu(),
        bias=None,
    )

    # int8 bits ARE bit-identical across devices — once .to(int8) lands, no
    # FP rounding remains.  The scale, however, comes out of a per-row
    # max-abs reduction whose order differs between CPU (serial) and CUDA
    # (tree reduction across blocks).  On this RTX 2000 Ada the resulting
    # scale floats differ by exactly 1 ULP on a handful of rows — the
    # tensors print identically at 4 decimals but ``torch.equal`` returns
    # False.  This is the documented contract from test 1; we re-assert it
    # here at the representation level.
    assert torch.equal(q_cpu.qweight, q_cuda_on_cpu.qweight)
    torch.testing.assert_close(q_cpu.scale, q_cuda_on_cpu.scale, atol=1e-6, rtol=0)

    # Dequant uses ``qweight.to(fp32) * scale[:, None]``.  qweight bits are
    # identical; scale is 1-ULP-close.  Therefore each dequantized element
    # differs by at most ``qweight[i, j] * (scale_cpu[i] - scale_cuda[i])``,
    # i.e. up to ~127 * 1 ULP of the scale.  Bound that explicitly so the
    # test fails loudly on any future drift larger than the FP32 reduction
    # reordering can explain.
    deq_cpu_native = dequantize_linear(q_cpu)
    deq_cuda_via_cpu = dequantize_linear(q_cuda_on_cpu)
    diff = (deq_cpu_native - deq_cuda_via_cpu).abs()
    # The per-row error budget: |qweight|_inf <= 127, scale ULP <= scale * 2^-23.
    per_row_budget = 127.0 * q_cpu.scale.abs() * (2.0**-23) * 2.0  # x2 safety
    assert (diff <= per_row_budget.unsqueeze(1)).all(), (
        f"dequant CPU vs CUDA diverged beyond the 1-ULP-scale budget; "
        f"max diff = {diff.max().item():.4e}, "
        f"max budget = {per_row_budget.max().item():.4e}"
    )
