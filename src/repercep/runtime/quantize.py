"""Weight quantization helpers — per-channel symmetric INT8.

The motivation for landing this on the CPU side first is the AMX_INT8
matmul lever: TDPBSSD on Sapphire Rapids does INT8 · INT8 -> INT32 at
twice the throughput of TDPBF16PS (BF16 · BF16 -> FP32) — *if* the
weights are pre-quantized.  This module owns the BF16/FP16 -> INT8
quantization path with per-output-channel scales, mirroring the
per-channel symmetric scheme used in oneDNN / IPEX's smooth_quant.

GPU sibling: ``torchao``-driven INT8 quant of the DiT linears.  We
deliberately implement this here rather than calling ``torchao`` because
(a) ``torchao`` requires a CUDA wheel that does not co-install cleanly
with our CPU torch wheel on the same host, and (b) the AMX kernels need
the quantized tensors in a specific tile-friendly layout that ``torchao``
doesn't produce.

The quantization is *static* (offline) — weights are quantized once at
load time, scales are stored alongside.  Activations are quantized
per-token on the fly inside the AMX kernel (per F23 the dynamic activation
quant cost is well below the matmul savings at the Cosmos DiT shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class QuantizedLinear:
    """A 2-D weight matrix quantized per output-channel to INT8 (symmetric).

    The original ``weight: (out_features, in_features)`` is decomposed into
    ``qweight: (out_features, in_features) int8`` and ``scale:
    (out_features,) float32``: ``weight ≈ qweight.to(float) * scale[:,
    None]``.  Bias (when present) stays in BF16/FP32 — quantizing bias to
    INT8 isn't worth the 0.001 % memory savings.

    Per-channel-symmetric is the right choice here for the same reason it
    is on GPU: each output channel of a Linear has its own scale, so a
    layer's dynamic range doesn't get collapsed into one outlier-driven
    scale.  Asymmetric would buy us nothing on these distributions
    (post-LayerNorm activations are already centered).
    """

    qweight: torch.Tensor  # (out, in) int8
    scale: torch.Tensor    # (out,) float32
    bias: torch.Tensor | None  # (out,) original dtype or None


def quantize_linear_symmetric(
    weight: torch.Tensor, bias: torch.Tensor | None = None
) -> QuantizedLinear:
    """Per-output-channel symmetric INT8 quantization of a 2-D weight.

    ``weight`` is expected ``(out_features, in_features)``, any float dtype.
    Returns a :class:`QuantizedLinear` with qweight + scale tensors live on
    the same device as ``weight``.

    Algorithm:
        for each output channel i:
            scale_i = max(|weight[i]|) / 127.0
            qweight[i] = round(weight[i] / scale_i).clamp(-128, 127).to(int8)

    Numerical caveat: weights with a single outlier per row produce a
    too-coarse scale for the rest of the row.  Mitigation (Session 16+):
    pre-process with ``smooth_quant``-style activation/weight rebalancing.
    """
    import torch

    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight; got shape {tuple(weight.shape)}")
    # Work in float32 for the scale computation regardless of input dtype —
    # at BF16 the per-row max can lose precision when the row's max
    # magnitude has lots of significant bits.
    w_fp32 = weight.detach().to(torch.float32)
    # Per-row max-abs.  Use ``unbiased=False`` semantics implicitly via the
    # max + abs ops; clamp to a small floor to avoid div-by-zero on
    # all-zero rows (which are rare but show up in pruned Linears).
    row_max = w_fp32.abs().amax(dim=1).clamp(min=1e-8)
    scale = (row_max / 127.0).to(torch.float32)
    qweight = (w_fp32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return QuantizedLinear(qweight=qweight, scale=scale, bias=bias)


def dequantize_linear(q: QuantizedLinear, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Reconstruct the weight matrix for reference/correctness checks.

    Useful in tests and in the kernel-fallback path when the AMX_INT8
    kernel isn't available (e.g. on pre-SPR hardware).  Defaults to
    float32 for fidelity; pass ``dtype=torch.bfloat16`` when comparing
    against an AMX BF16 reference.
    """
    import torch

    out = q.qweight.to(torch.float32) * q.scale.unsqueeze(1)
    if dtype is not None:
        out = out.to(dtype)
    return out


def quantize_module_linears(
    module: torch.nn.Module,
    *,
    name_filter: str | None = None,
) -> dict[str, QuantizedLinear]:
    """Walk ``module`` and quantize every ``nn.Linear`` whose name matches.

    Returns a dict mapping the dotted module path to its
    :class:`QuantizedLinear`.  Does NOT modify ``module`` — the caller is
    expected to wrap the linears with an INT8-aware forward (e.g. via
    :func:`replace_linears_with_quantized`).  Keeping quantization pure
    here lets the test suite assert exact values without side effects.

    The ``name_filter`` is a substring match against dotted names; pass
    ``"dit"`` to quantize only the diffusion transformer blocks, leaving
    the VAE in BF16 (the right default for Cosmos — the VAE is
    quantization-sensitive).
    """
    import torch

    out: dict[str, QuantizedLinear] = {}
    for name, sub in module.named_modules():
        if not isinstance(sub, torch.nn.Linear):
            continue
        if name_filter is not None and name_filter not in name:
            continue
        out[name] = quantize_linear_symmetric(sub.weight, sub.bias)
    return out


def replace_linears_with_quantized(
    module: torch.nn.Module,
    *,
    name_filter: str | None = None,
) -> int:
    """In-place swap every matching ``nn.Linear`` for a :class:`QuantizedLinearModule`.

    Returns the number of swaps performed.  The ``name_filter`` is a
    substring match against dotted names, with the same semantics as
    :func:`quantize_module_linears` — pass ``"dit"`` to quantize only the
    DiT blocks.

    This is the integration point the engine load path calls when the
    caller asks for ``quantize="int8-symmetric"``.  We mutate in place
    rather than returning a new module so the swap is invisible to
    downstream code that holds references to the parent — important
    because the DiT module graph is shared across timesteps inside the
    denoise loop.

    The traversal is parent-first so we can ``setattr`` on the immediate
    parent; using ``module.named_modules()`` directly would give us the
    target but not the parent.  ``named_children`` per-parent + recursion
    is the cleanest expression of that.
    """
    import torch

    # The class is built lazily inside _quantized_linear_module_class so it
    # isn't visible at module scope for mypy; Any here is honest, not lazy.
    cls: Any = _quantized_linear_module_class()
    count = 0

    def _recurse(parent: torch.nn.Module, prefix: str) -> None:
        nonlocal count
        for child_name, child in list(parent.named_children()):
            dotted = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch.nn.Linear) and (
                name_filter is None or name_filter in dotted
            ):
                setattr(parent, child_name, cls.from_linear(child))
                count += 1
                continue
            _recurse(child, dotted)

    _recurse(module, "")
    return count


# The QuantizedLinearModule class is built lazily on first access.  We
# can't define it at module-import time without forcing ``import torch``
# (the class has to inherit from ``nn.Module``), which would defeat the
# whole point of the lazy-torch pattern the rest of this file follows.
# PEP 562 ``__getattr__`` lets us hand out the class on demand while
# keeping the cold import path torch-free.
#
# A ``TYPE_CHECKING``-only mirror of the class's public shape below gives
# mypy a real module-level name to resolve `from ...quantize import
# QuantizedLinearModule` against, instead of falling back to
# ``__getattr__``'s ``object`` return type. Never executes at runtime (the
# whole point is to keep torch out of the cold-import path), so it costs
# nothing and can't drift silently — a shape change to the real nested
# class below must be mirrored here or callers lose type information again.
if TYPE_CHECKING:

    class QuantizedLinearModule(torch.nn.Module):
        qweight: torch.Tensor
        scale: torch.Tensor
        bias: torch.nn.Parameter | None
        out_features: int
        in_features: int

        def __init__(self, q: QuantizedLinear) -> None: ...
        @classmethod
        def from_linear(cls, linear: torch.nn.Linear) -> QuantizedLinearModule: ...
        def forward(self, x: torch.Tensor) -> torch.Tensor: ...


_QuantizedLinearModuleCls: type | None = None


def _quantized_linear_module_class() -> type:
    """Build (or fetch the cached) ``QuantizedLinearModule`` class."""
    global _QuantizedLinearModuleCls
    if _QuantizedLinearModuleCls is not None:
        return _QuantizedLinearModuleCls

    import torch

    class QuantizedLinearModule(torch.nn.Module):
        """An ``nn.Module`` wrapper around a :class:`QuantizedLinear`.

        Forward is the dequant-then-matmul fallback path — correct on any
        CPU and on GPU, but slower than the AMX_INT8 kernel that will
        eventually replace it.  Keeping the dequant path as the *default*
        forward (rather than calling the kernel) means the module is
        usable on pre-Sapphire-Rapids hardware, on AMX-disabled VMs (our
        CI today), and inside ``torch.compile`` tracing — none of which
        can call the AMX intrinsics.  When the kernel lands, it'll
        dispatch via the existing backend registry (see
        ``repercep.backend.cpu``) and this forward becomes the fallback
        branch under ``if not amx_available``.

        Numerically this is *not* the same as the original ``nn.Linear``:
        the weight has been round-tripped through INT8 symmetric
        quantization, so each row carries up to ~scale/2 of per-element
        error.  Tests bound this by the per-row scale, which is the
        tightest bound achievable without changing the quant scheme.
        """

        # PyTorch's nn.Module __getattr__ returns Tensor|Module for any
        # attribute, so without these annotations mypy can't follow the
        # buffer/parameter chain through arithmetic.  The runtime values are
        # set by register_buffer / Parameter assignment in __init__.
        qweight: torch.Tensor
        scale: torch.Tensor
        bias: torch.nn.Parameter | None

        def __init__(self, q: QuantizedLinear) -> None:
            super().__init__()
            # Register qweight + scale as buffers so ``.to(device)`` and
            # state-dict roundtrips work without surprise.  Bias goes in
            # as a Parameter when present so downstream code walking
            # ``parameters()`` still sees it (matches ``nn.Linear``).
            self.register_buffer("qweight", q.qweight)
            self.register_buffer("scale", q.scale)
            if q.bias is not None:
                self.bias = torch.nn.Parameter(
                    q.bias.detach().clone(), requires_grad=False
                )
            else:
                self.register_parameter("bias", None)
            self.out_features, self.in_features = q.qweight.shape

        @classmethod
        def from_linear(cls, linear: torch.nn.Linear) -> QuantizedLinearModule:
            """Quantize an existing ``nn.Linear`` and wrap the result."""
            q = quantize_linear_symmetric(linear.weight, linear.bias)
            return cls(q)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Dequant-then-linear: the SDPA-floor fallback path.

            Dequant happens in the input dtype so the matmul stays in the
            caller's precision (BF16 in / BF16 out for the DiT blocks).
            The AMX_INT8 kernel will replace this with an INT8 matmul
            that fuses the scale at the accumulator stage; the math is
            equivalent but ~2x faster on SPR.
            """
            w = (self.qweight.to(torch.float32) * self.scale.unsqueeze(1)).to(x.dtype)
            return torch.nn.functional.linear(x, w, self.bias)

    _QuantizedLinearModuleCls = QuantizedLinearModule
    return QuantizedLinearModule


def __getattr__(name: str) -> object:
    if name == "QuantizedLinearModule":
        return _quantized_linear_module_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
