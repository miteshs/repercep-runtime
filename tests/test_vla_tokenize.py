"""CPU tests for the VLA action (de)tokenizer math (``vla_tokenize``).

Pins the roundtrip and saturation properties any uniform action binning must
satisfy, so a Phase-1 GPU port that swaps in the exact OpenVLA index convention
has a convention-agnostic oracle to check against (docs/VLA_PORT_PLAN.md).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from repercep.models.vla_tokenize import discretize_actions, undiscretize_actions

# OpenVLA-like 7-DoF ranges (q01/q99), arbitrary but representative.
_LOW = torch.tensor([-1.0, -1.0, -1.0, -0.5, -0.5, -0.5, 0.0])
_HIGH = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0])


def test_roundtrip_within_one_bin() -> None:
    bins = 256
    g = torch.Generator().manual_seed(0)
    actions = _LOW + (_HIGH - _LOW) * torch.rand(64, 7, generator=g)
    decoded = undiscretize_actions(
        discretize_actions(actions, _LOW, _HIGH, bins), _LOW, _HIGH, bins
    )
    # Decode ∘ encode is identity to within one bucket's width per dimension.
    resolution = (_HIGH - _LOW) / (bins - 1)
    assert torch.all((decoded - actions).abs() <= resolution)


def test_endpoints_map_to_first_and_last_bucket() -> None:
    bins = 256
    lo_tok = discretize_actions(_LOW, _LOW, _HIGH, bins)
    hi_tok = discretize_actions(_HIGH, _LOW, _HIGH, bins)
    assert torch.all(lo_tok == 0)
    assert torch.all(hi_tok == bins - 1)


def test_out_of_range_saturates_not_wraps() -> None:
    bins = 256
    below = discretize_actions(_LOW - 10.0, _LOW, _HIGH, bins)
    above = discretize_actions(_HIGH + 10.0, _LOW, _HIGH, bins)
    assert torch.all(below == 0)
    assert torch.all(above == bins - 1)


def test_exact_small_grid() -> None:
    # bins=3 over [0,2] → buckets at 0,1,2; exact integer roundtrip.
    low, high = torch.tensor([0.0]), torch.tensor([2.0])
    vals = torch.tensor([[0.0], [1.0], [2.0]])
    tok = discretize_actions(vals, low, high, 3)
    assert tok.flatten().tolist() == [0, 1, 2]
    assert torch.equal(undiscretize_actions(tok, low, high, 3), vals)


def test_constant_channel_decodes_to_low() -> None:
    # A zero-width dim (q01 == q99) must not divide by zero; it decodes to low.
    low, high = torch.tensor([1.0, 5.0]), torch.tensor([1.0, 5.0])
    tok = discretize_actions(torch.tensor([[1.0, 5.0], [3.0, 9.0]]), low, high, 16)
    assert torch.all(tok == 0)
    decoded = undiscretize_actions(tok, low, high, 16)
    assert torch.equal(decoded, low.expand(2, 2))


def test_shape_preserved() -> None:
    actions = torch.zeros(3, 5, 7)
    tok = discretize_actions(actions, _LOW, _HIGH, 256)
    assert tok.shape == actions.shape
    assert undiscretize_actions(tok, _LOW, _HIGH, 256).shape == actions.shape


@pytest.mark.parametrize("bad_bins", [0, 1])
def test_rejects_degenerate_bins(bad_bins: int) -> None:
    with pytest.raises(ValueError, match="bins must be >= 2"):
        discretize_actions(torch.zeros(7), _LOW, _HIGH, bad_bins)
    with pytest.raises(ValueError, match="bins must be >= 2"):
        undiscretize_actions(torch.zeros(7, dtype=torch.long), _LOW, _HIGH, bad_bins)
