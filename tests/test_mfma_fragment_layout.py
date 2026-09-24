"""Pins the gfx942 MFMA operand register layout.

This is the test that would have caught F48: ``kernels/hip/fp8_attn``
loaded its A and B fragments on only 16 of the 64 lanes and dropped the
K-group term, so three quarters of every ``K=32`` contraction was silently
zero.  Nothing called the kernel, so no published number moved -- but the
file claimed ISA authority in a comment while implementing a guess.

Ported from the same idea in Modular's open-source MAX kernels
(``max/kernels/test/gpu/structured_kernels/test_mfma_fragment_lane_mapping.mojo``,
Apache-2.0), whose docstring states the stakes plainly: *"If it fails, the
lane-mapping doc is wrong, and all FP8-cast logic downstream is built on bad
assumptions."*

Two layers, deliberately:

1. **Structural, no GPU.**  The lane mapping is a claim about a bijection
   between ``(lane, element)`` pairs and tile coordinates.  That claim is
   checkable on any host, in microseconds, with no ROCm.  This is where the
   regression pin lives -- CI runs it on every PR.
2. **Numerical, gfx942 only.**  Compile the HIP kernel and check it against a
   reference GEMM.  Skips cleanly off-hardware, per the repo convention.

The mapping under test, for an ``(M, N, K) = (16, 16, 32)`` FP8 tile on a
wave64 (see the derivation comment in ``kernels/hip/fp8_attn/fp8_gemm.hip``):

    A: m   = lane % 16, k = (lane // 16) * 8 + e,  e in [0, 8)
    B: n   = lane % 16, k = (lane // 16) * 8 + e,  e in [0, 8)
    C: col = lane % 16, row = (lane // 16) * 4 + i, i in [0, 4)
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None

# The tile under test: gfx942's native FP8 MFMA shape.
MFMA_M, MFMA_N, MFMA_K = 16, 16, 32
WAVE = 64

# Per-lane fragment sizes are *derived*, never hardcoded: the wave holds the
# whole tile, so each lane holds (tile elements / WAVE).  Modular's kernels
# express the same rule as `num_matrix_reg[dim_1, dim_2]() = dim_1*dim_2 //
# WARP_SIZE`; their FP8 16x16x32 intrinsic is gated on per-lane fragment
# lengths (8, 8, 4, 4), which is what these three expressions reproduce.
A_FRAG = (MFMA_M * MFMA_K) // WAVE
B_FRAG = (MFMA_K * MFMA_N) // WAVE
C_FRAG = (MFMA_M * MFMA_N) // WAVE

_HIP_SRC = Path(__file__).resolve().parent.parent / "kernels" / "hip" / "fp8_attn" / "fp8_gemm.hip"

# (lane, element) -> (row, col) within the tile.
CoordFn = Callable[[int, int], tuple[int, int]]


# --------------------------------------------------------------------------
# The mapping, as executable spec.
# --------------------------------------------------------------------------
def a_coord(lane: int, e: int) -> tuple[int, int]:
    """A-operand: lane/element -> (m, k) in the 16x32 A tile."""
    return (lane % MFMA_M, (lane // MFMA_M) * A_FRAG + e)


def b_coord(lane: int, e: int) -> tuple[int, int]:
    """B-operand: lane/element -> (k, n) in the 32x16 B tile."""
    return ((lane // MFMA_N) * B_FRAG + e, lane % MFMA_N)


def c_coord(lane: int, i: int) -> tuple[int, int]:
    """C-accumulator: lane/element -> (row, col) in the 16x16 C tile."""
    return ((lane // MFMA_N) * C_FRAG + i, lane % MFMA_N)


# --------------------------------------------------------------------------
# 1. Structural — runs everywhere, including CPU-only CI.
# --------------------------------------------------------------------------
def test_fragment_sizes_are_derived_not_guessed() -> None:
    # These are the (a, b, c, d) per-lane lengths the hardware intrinsic is
    # selected on.  If a future tile shape changes them, the derivation must
    # keep producing the right answer rather than a stale constant.
    assert (A_FRAG, B_FRAG, C_FRAG) == (8, 8, 4)


@pytest.mark.parametrize(
    ("name", "coord", "frag", "tile"),
    [
        ("A", a_coord, A_FRAG, (MFMA_M, MFMA_K)),
        ("B", b_coord, B_FRAG, (MFMA_K, MFMA_N)),
        ("C", c_coord, C_FRAG, (MFMA_M, MFMA_N)),
    ],
)
def test_mapping_is_a_bijection(
    name: str, coord: CoordFn, frag: int, tile: tuple[int, int]
) -> None:
    """Every tile element is owned by exactly one (lane, element) slot.

    This is the single assertion that catches the whole bug class: a mapping
    that skips lanes under-covers, and a mapping with a wrong stride
    double-covers.  Both fail here.
    """
    rows, cols = tile
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for lane in range(WAVE):
        for e in range(frag):
            rc = coord(lane, e)
            assert 0 <= rc[0] < rows and 0 <= rc[1] < cols, (
                f"{name}: lane {lane} elt {e} maps out of tile: {rc}"
            )
            assert rc not in seen, (
                f"{name}: {rc} owned by both lane {seen[rc][0]} and lane {lane}"
            )
            seen[rc] = (lane, e)
    assert len(seen) == rows * cols, (
        f"{name}: covered {len(seen)} of {rows * cols} elements -- "
        f"{rows * cols - len(seen)} would read as zero on hardware"
    )


@pytest.mark.parametrize(
    ("name", "coord", "frag", "tile"),
    [("A", a_coord, A_FRAG, (MFMA_M, MFMA_K)), ("B", b_coord, B_FRAG, (MFMA_K, MFMA_N))],
)
def test_every_lane_owns_a_distinct_full_fragment(
    name: str, coord: CoordFn, frag: int, tile: tuple[int, int]
) -> None:
    """All 64 lanes supply ``frag`` distinct in-tile elements.

    The direct pin for F48: the old kernel guarded its loads on
    ``lane < 16``, leaving 48 lanes feeding zeros into the MFMA.  Asserting
    per-lane coverage (rather than that the mapping merely returns something)
    is what makes this test able to fail.
    """
    rows, cols = tile
    for lane in range(WAVE):
        owned = {coord(lane, e) for e in range(frag)}
        assert len(owned) == frag, (
            f"{name}: lane {lane} owns {len(owned)} distinct elements, want {frag}"
        )
        for r, c in owned:
            assert 0 <= r < rows and 0 <= c < cols, f"{name}: lane {lane} maps outside the tile"


@pytest.mark.parametrize(
    ("name", "coord", "frag"), [("A", a_coord, A_FRAG), ("B", b_coord, B_FRAG)]
)
def test_lane_owns_contiguous_k(name: str, coord: CoordFn, frag: int) -> None:
    """Each lane's K elements are contiguous, so the load is one 8-byte read.

    Not just a perf property: the old B path walked K with stride 4
    (``krow = k_tile + kk * 4``), which reads the wrong elements entirely.
    """
    k_axis = 1 if name == "A" else 0
    for lane in range(WAVE):
        ks = [coord(lane, e)[k_axis] for e in range(frag)]
        assert ks == list(range(ks[0], ks[0] + frag)), (
            f"{name}: lane {lane} K not contiguous: {ks}"
        )
        assert ks[0] % frag == 0, f"{name}: lane {lane} K group not aligned: {ks[0]}"


def test_c_lane_owns_contiguous_rows_at_one_column() -> None:
    """C-fragment: 4 consecutive rows at a single column."""
    for lane in range(WAVE):
        coords = [c_coord(lane, i) for i in range(C_FRAG)]
        cols = {c for _, c in coords}
        rows = [r for r, _ in coords]
        assert len(cols) == 1, f"lane {lane} spans columns {cols}"
        assert rows == list(range(rows[0], rows[0] + C_FRAG))


# --- Negative controls: the mappings that were actually wrong. -------------
# The old kernel loaded nothing on lanes >= 16, so those lanes contributed a
# zero fragment rather than a tile coordinate.  Model that as "not in the
# covered set" — which is exactly why the bijection assertion above fails.
_OLD_ACTIVE_LANES = 16


def _old_a_covered() -> set[tuple[int, int]]:
    """The pre-2026-08 A mapping: guarded on lane < 16, no K-group term."""
    return {(lane, e) for lane in range(_OLD_ACTIVE_LANES) for e in range(A_FRAG)}


def _old_b_covered() -> set[tuple[int, int]]:
    """The pre-2026-08 B mapping: guarded on lane < 16, K walked with stride 4."""
    return {(e * 4, lane) for lane in range(_OLD_ACTIVE_LANES) for e in range(B_FRAG)}


def test_old_a_mapping_is_rejected() -> None:
    covered = _old_a_covered()
    # 16 lanes x 8 elements = 128 of the 512 A elements: 3/4 of every K=32
    # contraction fed the MFMA a zero.
    assert len(covered) == 128
    assert len(covered) < MFMA_M * MFMA_K


def test_old_b_mapping_is_rejected() -> None:
    covered = _old_b_covered()
    ks = {k for k, _ in covered}
    assert ks == {0, 4, 8, 12, 16, 20, 24, 28}, "old B walked K with stride 4"
    assert len(covered) < MFMA_K * MFMA_N


# --- The kernel source agrees with the spec above. ------------------------
def test_hip_kernel_declares_the_same_tile() -> None:
    """The .hip file's tile constants must match what this test pins.

    Cheap coupling: if someone retargets the kernel to another MFMA shape,
    this test starts lying unless it is updated too.
    """
    src = _HIP_SRC.read_text()
    found = {
        name: int(m.group(1))
        for name in ("MFMA_M", "MFMA_N", "MFMA_K", "WAVE")
        if (m := re.search(rf"constexpr int {name} = (\d+);", src))
    }
    assert found == {"MFMA_M": MFMA_M, "MFMA_N": MFMA_N, "MFMA_K": MFMA_K, "WAVE": WAVE}


def test_hip_kernel_does_not_reintroduce_the_lane_guard() -> None:
    """No ``lane < 16`` guard around the operand loads.

    Narrow on purpose: this is the exact shape of the bug, and the kernel is
    small enough that a textual pin is honest rather than brittle.
    """
    src = _HIP_SRC.read_text()
    assert not re.search(r"lane\s*<\s*16", src), (
        "operand loads must run on all 64 lanes; a lane<16 guard zeroes 3/4 of K"
    )


# --------------------------------------------------------------------------
# 2. Numerical — gfx942 only.
# --------------------------------------------------------------------------
def _gfx942_or_skip() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    if not torch.version.hip:
        pytest.skip("gfx942 FP8 MFMA path is AMD-only")


def _load_ext_or_skip() -> Any:
    try:
        from kernels.hip.fp8_attn.build import load_hip_fp8
    except ImportError:
        pytest.skip("kernels package not importable")
    try:
        return load_hip_fp8()
    except Exception as exc:  # hipcc missing, arch mismatch, ...
        pytest.skip(f"HIP extension did not build: {exc}")


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_mfma_all_ones_sums_to_k() -> None:
    """A = 1, B = 1  =>  C[m, n] == K, for every element.

    The sharpest possible probe of a dropped K-group: under the old mapping
    this returns K/4.  Chosen so every value is exact in both e4m3fnuz and
    the bf16 output.
    """
    import torch

    _gfx942_or_skip()
    ext = _load_ext_or_skip()

    dim_m = dim_n = 16
    dim_k = MFMA_K
    dev = torch.device("cuda", 0)
    a = torch.ones(dim_m, dim_k, device=dev).to(torch.float8_e4m3fnuz)
    b = torch.ones(dim_k, dim_n, device=dev).to(torch.float8_e4m3fnuz)

    c = ext.fp8_gemm(a, b, 1.0, 1.0).float()
    assert torch.equal(c, torch.full((dim_m, dim_n), float(dim_k), device=dev)), (
        f"expected every element == {dim_k}, got min={c.min().item()} max={c.max().item()}; "
        f"{dim_k // 4} means the K-group term is missing"
    )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_mfma_matches_reference_gemm_exactly() -> None:
    """Small-integer operands make the FP8 GEMM exact; compare bit-for-bit.

    Values in [-2, 2] are exact in e4m3fnuz, and |C| <= 2*2*32 = 128 is exact
    in bf16 -- so any mismatch is a layout error, not quantization.  A
    permuted-but-covering mapping passes the all-ones test above and fails
    this one.
    """
    import torch

    _gfx942_or_skip()
    ext = _load_ext_or_skip()

    # Multiple tiles in both dims and several K steps.
    dim_m, dim_n, dim_k = 32, 32, 64
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    a_i = torch.randint(-2, 3, (dim_m, dim_k), device=dev, dtype=torch.float32)
    b_i = torch.randint(-2, 3, (dim_k, dim_n), device=dev, dtype=torch.float32)

    ref = (a_i @ b_i).to(torch.bfloat16).float()
    got = ext.fp8_gemm(
        a_i.to(torch.float8_e4m3fnuz), b_i.to(torch.float8_e4m3fnuz), 1.0, 1.0
    ).float()

    assert torch.equal(got, ref), (
        f"max abs diff {(got - ref).abs().max().item()} -- operand layout mismatch"
    )
