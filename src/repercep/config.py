"""Runtime configuration.

All settings are overridable via ``REPERCEP_*`` environment variables, e.g.
``REPERCEP_DTYPE=fp16`` or ``REPERCEP_DEVICE_INDEX=0``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from repercep.hardware import DType


def _default_weights_dir() -> Path:
    return Path.home() / ".cache" / "repercep" / "weights"


class RuntimeConfig(BaseSettings):
    """Top-level runtime configuration.

    Reads ``REPERCEP_*`` environment variables; unknown keys are rejected so a
    typo in an env var fails loud rather than being silently ignored.
    """

    model_config = SettingsConfigDict(env_prefix="REPERCEP_", extra="forbid")

    device_index: int = Field(0, ge=0, description="Which accelerator to bind to.")
    dtype: DType = Field(DType.BF16, description="Compute dtype for the model.")
    attention_backend: str = Field(
        "auto",
        description="'auto' lets the backend pick; or pin 'rocm-ck-flash' / 'naive-sdpa'.",
    )
    max_batch_size: int = Field(1, ge=1, description="Upper bound on concurrent requests.")
    latent_cache_gib: float = Field(8.0, gt=0, description="HBM budget for the paged latent cache.")
    weights_dir: Path = Field(
        default_factory=_default_weights_dir,
        description="Local directory for downloaded model weights.",
    )
    hf_token: str | None = Field(
        None,
        description="HuggingFace token for gated weights (Cosmos). Prefer REPERCEP_HF_TOKEN.",
    )

    # --- Gateway tenancy + metering (see docs/METERED_GATEWAY.md) ---
    # Off by default so a local/dev runtime stays a single-tenant box with no
    # auth and no ledger. Setting gateway_db is what turns the gateway
    # multi-tenant: per-customer API keys, per-call token metering, and
    # monthly quotas all read and write that one SQLite file.
    gateway_db: Path | None = Field(
        None,
        description="SQLite file holding API keys and the usage ledger. When set, the "
        "gateway requires a per-customer key and meters every LLM call. Manage keys "
        "with `repercep keys ...`.",
    )
    api_token: str | None = Field(
        None,
        description="Legacy single shared bearer token. Accepted alongside gateway_db "
        "for migration; usage under it cannot be attributed to a customer.",
    )

    # --- Co-located LLM proxy (see docs/LLM_PROXY.md) ---
    # The gateway can reverse-proxy an OpenAI-compatible LLM server (vLLM /
    # SGLang) running alongside the world-model runtime — same box, same auth,
    # same MI300X — WITHOUT routing that traffic through the world-model
    # scheduler/driver, which would serialize the upstream's own continuous
    # batching. Off by default: deployment coverage for an embodied stack's
    # language head, not a Repercep headline.
    llm_enabled: bool = Field(
        False, description="Expose the co-located OpenAI-compatible LLM proxy endpoints."
    )
    llm_upstream_url: str = Field(
        "http://127.0.0.1:8001",
        description="Base URL of the co-located vLLM/SGLang OpenAI server.",
    )
    llm_upstream_timeout_s: float = Field(
        600.0, gt=0, description="Per-request timeout when proxying to the LLM upstream."
    )
    llm_upstream_api_key: str | None = Field(
        None,
        description="Bearer token for a secured upstream (vLLM --api-key). Injected on "
        "upstream calls; the client's own Authorization header is never forwarded.",
    )
