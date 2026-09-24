"""Tests for ``repercep.backend.cpu.CPUBackend``."""

from __future__ import annotations

import platform

import pytest

from repercep.backend.cpu import (
    CPUBackend,
    _detect_arch_and_name,
    _detect_dtypes_and_features,
    _detect_topology,
)
from repercep.backend.registry import _ALL_BACKENDS, available_backends, select_backend
from repercep.hardware import (
    EMERALD_RAPIDS,
    GRANITE_RAPIDS,
    SAPPHIRE_RAPIDS,
    BackendCapabilities,
    DType,
    Vendor,
)

# CPU detection reads /proc/cpuinfo, which exists on Linux only.  The rest of
# the suite still runs (the Protocol-level assertions don't read cpuinfo);
# this marker scopes the hardware-introspection tests.
linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="cpuinfo detection requires /proc/cpuinfo (Linux-only)",
)


def test_backend_satisfies_protocol() -> None:
    """Structural Protocol conformance — same shape as test_backend_cuda.py."""
    from repercep.backend.protocol import Backend

    backend: Backend = CPUBackend()
    assert backend.vendor is Vendor.INTEL
    assert backend.name == "cpu"
    # Protocol attributes must be reachable; runtime checks via isinstance
    # validate the duck-typed contract.
    assert isinstance(backend, Backend)


def test_backend_always_available() -> None:
    """``is_available()`` returns True when torch is importable."""
    assert CPUBackend().is_available() is True


def test_registry_includes_cpu_backend() -> None:
    """``_ALL_BACKENDS`` carries CPUBackend, after ROCm and CUDA."""
    names = [b.name for b in _ALL_BACKENDS]
    assert names == ["rocm", "cuda", "cpu"]


def test_select_backend_by_name() -> None:
    """Explicit ``prefer='cpu'`` returns the CPU backend even on a GPU host."""
    backend = select_backend(prefer="cpu")
    assert backend.name == "cpu"
    assert backend.vendor is Vendor.INTEL


def test_select_backend_unknown_raises() -> None:
    """Sentinel test — unknown backends still raise (cpu is now real)."""
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend(prefer="tpu")


def test_cpu_in_available_backends() -> None:
    """On any host with torch, CPU is present in ``available_backends()``."""
    avail = {b.name for b in available_backends()}
    assert "cpu" in avail


@linux_only
def test_devices_returns_single_spec() -> None:
    """CPU has exactly one DeviceSpec, with index 0."""
    specs = CPUBackend().devices()
    assert len(specs) == 1
    assert specs[0].index == 0
    assert specs[0].arch.vendor is Vendor.INTEL


@linux_only
def test_default_dtype_is_bf16() -> None:
    """BF16 — the AMX speed lever — is the default."""
    assert CPUBackend().default_dtype() is DType.BF16


@linux_only
def test_capabilities_advertises_sane_dtypes() -> None:
    """FP32, FP16, BF16, INT8 are always advertised; FP8 never is."""
    caps = CPUBackend().capabilities()
    assert isinstance(caps, BackendCapabilities)
    assert DType.FP32 in caps.dtypes
    assert DType.FP16 in caps.dtypes
    assert DType.BF16 in caps.dtypes
    assert DType.INT8 in caps.dtypes
    assert DType.FP8_E4M3 not in caps.dtypes
    assert DType.FP8_E5M2 not in caps.dtypes
    assert caps.supports_fp8 is False
    assert "naive-sdpa" in caps.attention_ops
    # The amx-sdpa floor must always be advertised (it's torch SDPA, always
    # importable on a torch-installed host).
    assert "amx-sdpa" in caps.attention_ops


@linux_only
def test_detect_arch_returns_intel_uarch_on_intel() -> None:
    """On any GenuineIntel host, arch.vendor is INTEL."""
    arch, name = _detect_arch_and_name()
    assert isinstance(name, str)
    assert arch.vendor is Vendor.INTEL


@linux_only
def test_detect_topology_returns_positive_counts() -> None:
    """Physical + logical core counts are both >= 1."""
    physical, logical = _detect_topology()
    assert physical >= 1
    assert logical >= physical


@linux_only
def test_detect_dtypes_and_features_returns_expected_keys() -> None:
    """The features dict carries the AMX + AVX-512 flag set the registry checks."""
    dtypes, features = _detect_dtypes_and_features()
    assert DType.BF16 in dtypes
    expected_flags = {
        "avx512f",
        "avx512_bf16",
        "avx512_fp16",
        "amx_tile",
        "amx_bf16",
        "amx_int8",
        "amx_fp16",
        "avx_vnni",
    }
    assert expected_flags.issubset(features.keys())


@linux_only
def test_torch_device_is_cpu() -> None:
    """``torch_device()`` returns a plain ``torch.device('cpu')``."""
    import torch

    dev = CPUBackend().torch_device()
    assert dev == torch.device("cpu")


@linux_only
def test_known_intel_uarchs_resolve_cleanly() -> None:
    """SPR/EMR/GNR must be distinguishable; same Vendor, distinct gfx_id."""
    assert SAPPHIRE_RAPIDS.vendor is Vendor.INTEL
    assert SAPPHIRE_RAPIDS.gfx_id == "spr"
    assert EMERALD_RAPIDS.gfx_id == "emr"
    assert GRANITE_RAPIDS.gfx_id == "gnr"
    # All three share the family ("Sapphire/Emerald/Granite Rapids" — uarch)
    # but the gfx_ids are distinct so the registry can target them
    # individually (e.g. AMX_FP16 only on Granite Rapids).
    assert len({SAPPHIRE_RAPIDS.gfx_id, EMERALD_RAPIDS.gfx_id, GRANITE_RAPIDS.gfx_id}) == 3
