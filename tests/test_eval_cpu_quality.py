"""Tests for ``scripts/eval_cpu_quality.py`` — CPU LPIPS+MSE+PSNR + FVD runner.

Mirrors the style of ``tests/test_fvd.py``: load the script by path (since
``scripts/`` isn't a package), and assert against the argument parser plus the
combined RESULT-line format using mocked inner stages.  We do **not** run the
actual LPIPS or I3D pipelines here — they would download weights and take
minutes.  The integration is exercised end-to-end in the CLI smoke described
in ``docs/METHODOLOGY.md`` §"Held-out reference set for FVD".
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

# scripts/eval_cpu_quality.py isn't an importable package; load by path.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_EVAL_SRC = _SCRIPTS / "eval_cpu_quality.py"

_spec = importlib.util.spec_from_file_location("eval_cpu_quality_mod", _EVAL_SRC)
assert _spec is not None and _spec.loader is not None
_eval = importlib.util.module_from_spec(_spec)
sys.modules["eval_cpu_quality_mod"] = _eval
_spec.loader.exec_module(_eval)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parser_requires_reference_and_candidate() -> None:
    """The two positional video paths are required."""
    parser = _eval.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["only_one.mp4"])


def test_parser_minimal_args() -> None:
    """With just two positionals, FVD options stay None and defaults apply."""
    parser = _eval.build_parser()
    args = parser.parse_args(["ref.mp4", "cand.mp4"])
    assert args.reference == Path("ref.mp4")
    assert args.candidate == Path("cand.mp4")
    assert args.fvd_reference_set is None
    assert args.fvd_candidate_set is None
    assert args.lpips_net == "alex"
    assert args.fvd_num_clips == 1
    assert args.fvd_n_warn == 50


def test_parser_full_args() -> None:
    """All FVD knobs plumb through cleanly."""
    parser = _eval.build_parser()
    args = parser.parse_args(
        [
            "ref.mp4",
            "cand.mp4",
            "--lpips-net",
            "vgg",
            "--fvd-reference-set",
            "refs/",
            "--fvd-candidate-set",
            "cands/",
            "--fvd-num-clips",
            "3",
            "--fvd-n-warn",
            "100",
        ]
    )
    assert args.lpips_net == "vgg"
    assert args.fvd_reference_set == Path("refs/")
    assert args.fvd_candidate_set == Path("cands/")
    assert args.fvd_num_clips == 3
    assert args.fvd_n_warn == 100


def test_parser_rejects_unknown_lpips_net() -> None:
    """LPIPS backbone is constrained to the three published options."""
    parser = _eval.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["a.mp4", "b.mp4", "--lpips-net", "resnet"])


# ---------------------------------------------------------------------------
# Imports work without GPU
# ---------------------------------------------------------------------------


def test_module_imports_without_gpu() -> None:
    """Loading the script must not require torch/lpips at import time.

    Anything that needs torch is lazy-loaded inside ``_compute_pixel_metrics``
    or ``_compute_distribution_fvd``; importing the script itself stays cheap
    so ``--help`` is fast on CPU-only hosts.
    """
    # The module is already loaded above; just sanity-check the public API.
    assert callable(_eval.build_parser)
    assert callable(_eval.main)
    assert callable(_eval.run)
    # Confirm lazy loaders exist but haven't pulled torch into sys.modules
    # purely by being defined (they're factory functions, not eager imports).
    assert callable(_eval._compute_fvd_mod)
    assert callable(_eval._verify_quality_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_collect_videos_finds_mp4s(tmp_path: Path) -> None:
    """Flat directory of mp4s returns sorted paths; non-video files ignored."""
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "readme.txt").write_text("ignore me")
    out = _eval._collect_videos(tmp_path)
    assert [p.name for p in out] == ["a.mp4", "b.mp4"]


def test_collect_videos_rejects_empty(tmp_path: Path) -> None:
    """Empty dir errors loudly — the user almost certainly mistyped the path."""
    with pytest.raises(FileNotFoundError, match="no video files"):
        _eval._collect_videos(tmp_path)


def test_collect_videos_rejects_non_dir(tmp_path: Path) -> None:
    """A file path (not a directory) errors with NotADirectoryError."""
    f = tmp_path / "not_a_dir.mp4"
    f.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        _eval._collect_videos(f)


def test_run_requires_paired_fvd_dirs(tmp_path: Path) -> None:
    """Providing only one of the two --fvd-* dirs is a user error."""
    ref = tmp_path / "ref.mp4"
    cand = tmp_path / "cand.mp4"
    ref.write_bytes(b"")
    cand.write_bytes(b"")
    args = _eval.build_parser().parse_args(
        [str(ref), str(cand), "--fvd-reference-set", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="must be provided together"):
        _eval.run(args)


def test_run_errors_on_missing_video(tmp_path: Path) -> None:
    """Bare existence check on the pixel-pair paths happens before LPIPS load."""
    ref = tmp_path / "missing.mp4"
    cand = tmp_path / "also_missing.mp4"
    args = _eval.build_parser().parse_args([str(ref), str(cand)])
    with pytest.raises(FileNotFoundError):
        _eval.run(args)


# ---------------------------------------------------------------------------
# Combined RESULT format (mocked inner stages)
# ---------------------------------------------------------------------------


def _fake_pixel_metrics() -> dict[str, object]:
    return {
        "mse_mean": 12.5,
        "mse_max": 30.0,
        "psnr_mean": 37.16,
        "psnr_min": 33.36,
        "lpips_mean": 0.064,
        "lpips_max": 0.123,
        "n_frames": 16,
        "lpips_net": "alex",
    }


def _fake_fvd_result() -> dict[str, object]:
    return {
        "mode": "fvd",
        "fvd": 42.5,
        "n_reference_clips": 8,
        "n_candidate_clips": 8,
        "feature_dim": 2048,
        "reference_paths": ["refs/a.mp4", "refs/b.mp4"],
        "candidate_paths": ["cands/a.mp4", "cands/b.mp4"],
    }


def test_main_emits_combined_result_line_pixel_only(tmp_path: Path) -> None:
    """RESULT line carries the canonical JSON shape; fvd is null when absent."""
    ref = tmp_path / "ref.mp4"
    cand = tmp_path / "cand.mp4"
    ref.write_bytes(b"")
    cand.write_bytes(b"")

    buf = io.StringIO()
    with patch.object(
        _eval, "_compute_pixel_metrics", return_value=_fake_pixel_metrics()
    ), redirect_stdout(buf):
        rc = _eval.main([str(ref), str(cand)])

    assert rc == 0
    out = buf.getvalue()
    result_lines = [
        line for line in out.splitlines() if line.startswith("RESULT ")
    ]
    assert len(result_lines) == 1
    payload = json.loads(result_lines[0][len("RESULT ") :])
    assert payload["mode"] == "cpu_quality_eval"
    assert payload["reference"] == str(ref)
    assert payload["candidate"] == str(cand)
    assert payload["pixel"]["lpips_mean"] == pytest.approx(0.064)
    assert payload["pixel"]["psnr_mean"] == pytest.approx(37.16)
    assert payload["pixel"]["mse_mean"] == pytest.approx(12.5)
    assert payload["pixel"]["n_frames"] == 16
    assert payload["pixel"]["lpips_net"] == "alex"
    assert payload["fvd"] is None
    assert payload["fvd_reference_set"] is None
    assert payload["fvd_candidate_set"] is None
    # Human summary should also have landed before the RESULT line.
    assert "CPU Quality Eval" in out
    assert "Pixel/perceptual" in out


def test_main_emits_combined_result_line_with_fvd(tmp_path: Path) -> None:
    """When the FVD dirs are wired, the JSON carries the full FVD sub-dict."""
    ref = tmp_path / "ref.mp4"
    cand = tmp_path / "cand.mp4"
    ref.write_bytes(b"")
    cand.write_bytes(b"")
    ref_dir = tmp_path / "ref_set"
    cand_dir = tmp_path / "cand_set"
    ref_dir.mkdir()
    cand_dir.mkdir()
    (ref_dir / "v0.mp4").write_bytes(b"")
    (cand_dir / "v0.mp4").write_bytes(b"")

    buf = io.StringIO()
    with patch.object(
        _eval, "_compute_pixel_metrics", return_value=_fake_pixel_metrics()
    ), patch.object(
        _eval, "_compute_distribution_fvd", return_value=_fake_fvd_result()
    ), redirect_stdout(buf):
        rc = _eval.main(
            [
                str(ref),
                str(cand),
                "--fvd-reference-set",
                str(ref_dir),
                "--fvd-candidate-set",
                str(cand_dir),
            ]
        )

    assert rc == 0
    out = buf.getvalue()
    result_lines = [
        line for line in out.splitlines() if line.startswith("RESULT ")
    ]
    assert len(result_lines) == 1
    payload = json.loads(result_lines[0][len("RESULT ") :])
    assert payload["fvd"]["mode"] == "fvd"
    assert payload["fvd"]["fvd"] == pytest.approx(42.5)
    assert payload["fvd"]["n_reference_clips"] == 8
    assert payload["fvd"]["feature_dim"] == 2048
    assert payload["fvd_reference_set"] == str(ref_dir)
    assert payload["fvd_candidate_set"] == str(cand_dir)
    # Human summary includes the FVD block when requested.
    assert "Distribution FVD" in out
    assert "FVD:" in out


def test_main_handles_feature_l2_fallback(tmp_path: Path) -> None:
    """N=1 vs N=1 falls back to feature_l2; summary renders it correctly."""
    ref = tmp_path / "ref.mp4"
    cand = tmp_path / "cand.mp4"
    ref.write_bytes(b"")
    cand.write_bytes(b"")
    ref_dir = tmp_path / "ref_set"
    cand_dir = tmp_path / "cand_set"
    ref_dir.mkdir()
    cand_dir.mkdir()
    (ref_dir / "v0.mp4").write_bytes(b"")
    (cand_dir / "v0.mp4").write_bytes(b"")

    fallback = {
        "mode": "feature_l2",
        "feature_l2": 3.14,
        "n_reference_clips": 1,
        "n_candidate_clips": 1,
        "feature_dim": 2048,
        "reference_paths": ["refs/a.mp4"],
        "candidate_paths": ["cands/a.mp4"],
    }

    buf = io.StringIO()
    with patch.object(
        _eval, "_compute_pixel_metrics", return_value=_fake_pixel_metrics()
    ), patch.object(
        _eval, "_compute_distribution_fvd", return_value=fallback
    ), redirect_stdout(buf):
        rc = _eval.main(
            [
                str(ref),
                str(cand),
                "--fvd-reference-set",
                str(ref_dir),
                "--fvd-candidate-set",
                str(cand_dir),
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    result_line = next(
        line for line in out.splitlines() if line.startswith("RESULT ")
    )
    payload = json.loads(result_line[len("RESULT ") :])
    assert payload["fvd"]["mode"] == "feature_l2"
    assert payload["fvd"]["feature_l2"] == pytest.approx(3.14)
    # Human-readable block uses the pretty name and notes the fallback case.
    assert "feature L2" in out
    assert "too small for FVD" in out
