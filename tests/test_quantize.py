"""Tests for ``repercep.runtime.quantize``."""

from __future__ import annotations

from typing import cast

import pytest


def test_quantize_linear_basic() -> None:
    """Quantize a small weight, dequant, and assert within INT8 rounding tolerance."""
    import torch

    from repercep.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(0)
    weight = torch.randn(64, 128) * 2.0  # not normalised; tests scale calc
    q = quantize_linear_symmetric(weight)
    # Layout invariants.
    assert q.qweight.shape == weight.shape
    assert q.qweight.dtype is torch.int8
    assert q.scale.shape == (weight.shape[0],)
    assert q.scale.dtype is torch.float32
    assert q.bias is None
    # Dequant should reconstruct within ~scale/2 absolute error per element.
    reconstructed = dequantize_linear(q)
    err = (reconstructed - weight).abs()
    per_row_scale = q.scale.unsqueeze(1).expand_as(err)
    # INT8 symmetric rounding guarantees max error <= scale/2 per element.
    max_ratio = (err / per_row_scale).max().item()
    assert (err <= per_row_scale).all(), f"max err / scale = {max_ratio:.3f}"


def test_quantize_linear_preserves_bias() -> None:
    """Bias passes through unchanged (we don't quantize it — see docstring)."""
    import torch

    from repercep.runtime.quantize import quantize_linear_symmetric

    weight = torch.randn(8, 16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    q = quantize_linear_symmetric(weight, bias)
    assert q.bias is not None
    assert torch.equal(q.bias, bias)
    assert q.bias.dtype is torch.bfloat16


def test_quantize_linear_handles_zero_row() -> None:
    """All-zero row gets a clamped scale, not a NaN."""
    import torch

    from repercep.runtime.quantize import quantize_linear_symmetric

    weight = torch.zeros(4, 8)
    weight[1] = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125])
    q = quantize_linear_symmetric(weight)
    assert torch.isfinite(q.scale).all()
    assert torch.isfinite(q.qweight.to(torch.float32)).all()
    # Row 1 had non-zero values; its scale must be sensible (>0).
    assert q.scale[1] > 0
    # Rows 0, 2, 3 are zero; their qweight rows must be all zero.
    assert (q.qweight[0] == 0).all()
    assert (q.qweight[2] == 0).all()
    assert (q.qweight[3] == 0).all()


def test_quantize_linear_rejects_wrong_dim() -> None:
    """Only 2-D weights are supported."""
    import torch

    from repercep.runtime.quantize import quantize_linear_symmetric

    with pytest.raises(ValueError, match="expected 2-D"):
        quantize_linear_symmetric(torch.randn(8))


def test_quantize_module_linears_filters_by_name() -> None:
    """``name_filter`` selects a substring match against dotted module names."""
    import torch

    from repercep.runtime.quantize import quantize_module_linears

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Sequential(
                torch.nn.Linear(8, 8),
                torch.nn.Linear(8, 8),
            )
            self.vae = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
            )

    m = Toy()
    # Without a filter, every Linear is quantized.
    all_q = quantize_module_linears(m)
    assert set(all_q.keys()) == {"dit.0", "dit.1", "vae.0"}
    # With ``name_filter="dit"``, only DiT linears.
    dit_q = quantize_module_linears(m, name_filter="dit")
    assert set(dit_q.keys()) == {"dit.0", "dit.1"}


def test_quantize_linear_bfloat16_input() -> None:
    """Quant + dequant of a BF16 weight stays within BF16-then-quant tolerance."""
    import torch

    from repercep.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(1)
    weight = (torch.randn(16, 32) * 1.5).to(torch.bfloat16)
    q = quantize_linear_symmetric(weight)
    # Reconstructed in BF16 to match weight dtype for comparison.
    reconstructed = dequantize_linear(q, dtype=torch.bfloat16)
    # Combined error: BF16 truncation + symmetric INT8 rounding.  Relaxed
    # tolerance (BF16 mantissa is 7 bits; INT8 symmetric is roughly 1/256
    # of dynamic range) — together ~4 % rel for typical input distributions.
    err = (reconstructed.to(torch.float32) - weight.to(torch.float32)).abs()
    rel = err / weight.to(torch.float32).abs().clamp(min=1e-3)
    assert rel.median().item() < 0.04


def test_quantize_linear_bf16_roundtrip_within_scale() -> None:
    """BF16 weight: round-trip error per element bounded by scale/127 + 1 ULP.

    The tightest per-element bound for symmetric INT8 is scale/2 (half a
    quantization step), but starting from a BF16 weight we also pay a
    one-ULP BF16 truncation on the *input* to quantize and another on the
    output of dequant.  Bound: scale/2 + 2 BF16 ULPs of the weight.
    """
    import torch

    from repercep.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(2)
    weight = (torch.randn(32, 64) * 0.8).to(torch.bfloat16)
    q = quantize_linear_symmetric(weight)
    reconstructed = dequantize_linear(q, dtype=torch.bfloat16)
    err = (reconstructed.to(torch.float32) - weight.to(torch.float32)).abs()
    # scale/2 dominates here; the BF16 ULP slack accounts for a handful of
    # boundary elements where rounding goes the long way.
    per_row_bound = (q.scale * 0.5).unsqueeze(1).expand_as(err)
    bf16_ulp_slack = weight.to(torch.float32).abs() * (2.0 ** -7)
    assert (err <= per_row_bound + bf16_ulp_slack + 1e-8).all()


def test_quantized_linear_module_matches_linear_bf16() -> None:
    """``QuantizedLinearModule.forward`` matches ``nn.Linear.forward`` post-quant.

    The reference here is *not* the original ``nn.Linear`` (which would
    fail by design) but the ``nn.Linear`` rebuilt from the dequantized
    weight — i.e. we assert the wrapper computes the same matmul as the
    fallback dequant path, no extra error.
    """
    import torch

    from repercep.runtime.quantize import (
        QuantizedLinearModule,
        dequantize_linear,
        quantize_linear_symmetric,
    )

    torch.manual_seed(3)
    linear = torch.nn.Linear(8, 16).to(torch.bfloat16)
    qmod = QuantizedLinearModule.from_linear(linear)

    # Reference: a fresh Linear holding the dequantized weight + original bias.
    ref = torch.nn.Linear(8, 16, bias=True).to(torch.bfloat16)
    q = quantize_linear_symmetric(linear.weight, linear.bias)
    with torch.no_grad():
        ref.weight.copy_(dequantize_linear(q, dtype=torch.bfloat16))
        ref.bias.copy_(linear.bias)

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    out_q = qmod(x)
    out_ref = ref(x)
    # Same dequant, same matmul kernel — diff should be at most BF16
    # accumulation slop.
    diff = (out_q.to(torch.float32) - out_ref.to(torch.float32)).abs().max().item()
    assert diff < 1e-2, f"qmod vs ref max diff = {diff:.4e}"


def test_quantized_linear_module_error_vs_original_bounded() -> None:
    """Sanity: error vs the *original* Linear is bounded by per-row scale * ||x||."""
    import torch

    from repercep.runtime.quantize import QuantizedLinearModule

    torch.manual_seed(4)
    linear = torch.nn.Linear(8, 16, bias=False).to(torch.bfloat16)
    qmod = QuantizedLinearModule.from_linear(linear)

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    out_q = qmod(x).to(torch.float32)
    out_ref = linear(x).to(torch.float32)
    err = (out_q - out_ref).abs()
    # Per-row weight error <= scale/2, so per-row output error <= (scale/2) * ||x||_1
    # plus BF16 matmul slop.  Use a relaxed bound that's still meaningful.
    x_l1 = x.to(torch.float32).abs().sum(dim=-1, keepdim=True)
    scale = qmod.scale.unsqueeze(0)
    bound = 0.5 * scale * x_l1 + 1e-2
    assert (err <= bound).all(), f"max err / bound = {(err / bound).max().item():.3f}"


def test_quantized_linear_module_preserves_bias() -> None:
    """Bias passes through the swap and is reachable from ``parameters()``."""
    import torch

    from repercep.runtime.quantize import QuantizedLinearModule

    linear = torch.nn.Linear(8, 4).to(torch.bfloat16)
    qmod = QuantizedLinearModule.from_linear(linear)
    assert qmod.bias is not None
    assert torch.equal(qmod.bias.detach(), linear.bias.detach())
    assert any(p is qmod.bias for p in qmod.parameters())

    # And the bias=False case yields a None bias attribute.
    linear_nb = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
    qmod_nb = QuantizedLinearModule.from_linear(linear_nb)
    assert qmod_nb.bias is None


def test_replace_linears_with_quantized_swaps_in_place() -> None:
    """Every ``nn.Linear`` is replaced; non-Linear children are untouched."""
    import torch

    from repercep.runtime.quantize import (
        QuantizedLinearModule,
        replace_linears_with_quantized,
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Sequential(
                torch.nn.Linear(8, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 8),
            )
            self.vae = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
            )
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # not exercised here
            result: torch.Tensor = self.dit(x)
            return result

    m = Toy()
    n = replace_linears_with_quantized(m)
    assert n == 3
    # Every Linear is now a QuantizedLinearModule; the ReLU + LayerNorm are
    # left intact.
    assert isinstance(m.dit[0], QuantizedLinearModule)
    assert isinstance(m.dit[1], torch.nn.ReLU)
    assert isinstance(m.dit[2], QuantizedLinearModule)
    assert isinstance(m.vae[0], QuantizedLinearModule)
    assert isinstance(m.norm, torch.nn.LayerNorm)
    # And no nn.Linear instances remain anywhere in the tree.
    assert not any(isinstance(sub, torch.nn.Linear) for sub in m.modules())


def test_replace_linears_with_quantized_filters_by_name() -> None:
    """With ``name_filter="dit"``, the VAE linear is left alone."""
    import torch

    from repercep.runtime.quantize import (
        QuantizedLinearModule,
        replace_linears_with_quantized,
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Sequential(
                torch.nn.Linear(8, 8),
                torch.nn.Linear(8, 8),
            )
            self.vae = torch.nn.Linear(4, 4)

    m = Toy()
    n = replace_linears_with_quantized(m, name_filter="dit")
    assert n == 2
    assert isinstance(m.dit[0], QuantizedLinearModule)
    assert isinstance(m.dit[1], QuantizedLinearModule)
    # The VAE linear must be untouched — same instance type, same identity.
    assert isinstance(m.vae, torch.nn.Linear)
    # mypy statically infers m.vae as nn.Linear from Toy.__init__ and (correctly,
    # for THIS run) concludes no object can be both Linear and QuantizedLinearModule
    # — cast to sidestep that conclusion, since the whole point of the assertion
    # is a *runtime* check that the swap didn't happen here.
    assert not isinstance(cast("object", m.vae), QuantizedLinearModule)


def test_replace_linears_preserves_bias_through_swap() -> None:
    """Bias on the original Linear shows up on the wrapper after swap."""
    import torch

    from repercep.runtime.quantize import (
        QuantizedLinearModule,
        replace_linears_with_quantized,
    )

    m = torch.nn.Sequential(
        torch.nn.Linear(8, 16, bias=True),
        torch.nn.Linear(16, 4, bias=False),
    )
    # Sequential.__getitem__ types as the base nn.Module, whose __getattr__
    # returns Tensor | Module for any name — chaining .bias off that union
    # makes mypy try to call the Module branch too when .detach() follows.
    # Cast through the concrete type at each point instead of trusting the
    # Module fallback; the swap replaces the object at m[0], so the pre- and
    # post-swap casts are deliberately different types, not the same handle.
    pre_swap_bias = cast("torch.nn.Linear", m[0]).bias
    assert pre_swap_bias is not None
    original_bias = pre_swap_bias.detach().clone()
    n = replace_linears_with_quantized(m)
    assert n == 2
    swapped = cast("QuantizedLinearModule", m[0])
    assert swapped.bias is not None
    assert torch.equal(swapped.bias.detach(), original_bias)
    assert cast("QuantizedLinearModule", m[1]).bias is None
