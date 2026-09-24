"""Tests for the hardware domain types. No GPU required."""

from __future__ import annotations

from repercep.hardware import H100, MI300X, BackendCapabilities, DeviceSpec, DType, Vendor


def test_mi300x_arch() -> None:
    assert MI300X.vendor is Vendor.AMD
    assert MI300X.gfx_id == "gfx942"
    assert MI300X.family == "CDNA3"
    assert str(MI300X) == "amd:gfx942"


def test_arch_namespaces_do_not_collide() -> None:
    # gfx and sm targets live in separate vendor namespaces.
    assert MI300X.vendor is not H100.vendor
    assert MI300X != H100


def test_device_spec_memory_conversion() -> None:
    spec = DeviceSpec(
        index=0,
        arch=MI300X,
        name="AMD Instinct MI300X",
        total_memory_bytes=192 * 1024**3,
        multi_processor_count=304,
    )
    assert round(spec.total_memory_gib) == 192


def test_dtype_string_values() -> None:
    assert DType.BF16.value == "bf16"
    assert DType.FP8_E4M3.value == "fp8_e4m3"


def test_capabilities_is_frozen() -> None:
    caps = BackendCapabilities(
        dtypes=frozenset({DType.BF16}),
        supports_flash_attention=False,
        supports_fp8=True,
        supports_torch_compile=True,
        attention_ops=("naive-sdpa",),
    )
    assert DType.BF16 in caps.dtypes
