# Repercep Runtime

**A vendor-neutral inference engine for every model class** — LLMs, VLAs,
video diffusion, and world models — with **AMD Instinct MI300X** (`gfx942`,
CDNA3) as the lead target rather than an afterthought.

## Why MI300X first

On MI300X there was no production-grade serving stack for these workloads
([ADR-0001](docs/adr/0001-mi300x-first-hardware-target.md)). The codebase
stays vendor-neutral (Protocol-based backends,
[ADR-0003](docs/adr/0003-vendor-neutral-backend-protocol.md)), so every result
below exists on both vendors and the NVIDIA path is not a port.

## Status

**Pre-alpha.** Six models across three serving regimes run end-to-end through
one engine on **AMD MI300X**, **NVIDIA H100**, and **Intel CPU (AMX)**, behind
one vendor-neutral backend Protocol. Every speedup here travels with the
workload it was measured on.

| Result | Measured | Where |
|---|---|---|
| Shared-prefix candidate batching (token-VLA decode) | **5.3–9.9×** both vendors, exact parity | `docs/VLA_ON_{H100,MI300X}.md` |
| Adaptive step-skip cache (video diffusion) | **3.2–3.8×** | `docs/COSMOS_ON_*.md` |
| Cosmos H100 / MI300X | 99.6 s (3.81× ref) / 142 s | `docs/COSMOS_ON_*.md` |
| Cosmos 3 Nano policy, MI300X | 3610 ms/chunk, ROCm gate passed | `docs/COSMOS3_ON_MI300X.md` |

Cosmos MI300X is, to our knowledge, the first publicly reported Cosmos
benchmark on any AMD GPU. Wan-2.2-A14B runs both 14B MoE experts resident in
the MI300X's 192 GiB — a config an 80 GiB H100 cannot hold without offload.
[ADR-0009](docs/adr/0009-kv-latent-reuse.md) records two GPU-verified negative
findings; negative results are published, not buried.

### What is built

- **Vendor-neutral backend layer** (ROCm / CUDA / CPU) with an attention
  abstraction, autotuned FP8 Triton flash kernels (gfx942 / Hopper / Ada), and
  CPU AMX kernels.
- **World-model path** — the native denoise loop with adaptive caching, the
  Cosmos/Wan engines, and an `InteractiveWorldModel` seam
  ([ADR-0008](docs/adr/0008-interactive-world-model-seam.md)) carrying V-JEPA
  2-AC energy-MPC planning, LingBot-VA 2.0, DreamZero, and Cosmos 3 Nano.
- **LLM path** — a co-located OpenAI-compatible reverse proxy to vLLM/SGLang.
  Deliberately not a reimplemented LLM engine: the upstream's continuous
  batching and paged attention are commodity.
- **Metered gateway** — per-tenant API keys, exact token metering, an
  append-only usage ledger, and monthly quotas.
- **Rust+PyO3 core** (paged latent cache, scheduler, request router) and a
  FastAPI serving surface (v1 sync, v2 router path, `/v2/world/session`).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the component map
- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — optimization strategy + measured ledger
- [`docs/TARGETS_AND_KERNELS.md`](docs/TARGETS_AND_KERNELS.md) — hardware targets and kernels
- [`docs/walkthroughs/`](docs/walkthroughs/) — line-by-line tour of Cosmos inference, outermost → kernel
- [`docs/walkthroughs-vjepa2-ac/`](docs/walkthroughs-vjepa2-ac/) — the same, for the interactive / energy-based path
- `docs/<MODEL>_ON_<HARDWARE>.md` — per-model, per-target bring-up and benchmark records

## Requirements

- An AMD GPU host with **ROCm 7.x** installed (`/dev/kfd` readable by your
  user), or an NVIDIA GPU with CUDA, or an Intel CPU with AMX.
- **Python 3.11+**.
- ~30 GB free disk for weights.

## Setup

`torch` must come from the ROCm wheel index that matches the system ROCm — the
default PyPI `torch` is CUDA-only and will not work on AMD.

```bash
pip install --user uv
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/rocm7.2
make install        # == uv pip install --python .venv -e ".[models,serving,dev]"
make check-gpu
make info
```

If `check-gpu` reports no GPU, the device nodes are likely not readable:

```bash
sudo usermod -aG render,video "$USER"   # then log out / back in
```

## Serving

```bash
repercep keys create --db gateway.db --customer acme --monthly-token-quota 5000000

REPERCEP_GATEWAY_DB=gateway.db \
REPERCEP_LLM_ENABLED=true \
REPERCEP_LLM_UPSTREAM_URL=http://127.0.0.1:8001 \
  uvicorn --factory repercep.serving.app:create_app_from_config

repercep usage --db gateway.db
```

## Developing

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest (GPU tests skip cleanly without a GPU)
```

## Layout

```
src/repercep/
  hardware.py      framework-agnostic device/dtype domain types
  config.py        runtime configuration (REPERCEP_* env vars)
  cli.py           `repercep info | keys | usage`
  backend/         vendor-neutral compute backends (rocm / cuda / cpu)
  attention/       attention ops behind the AttentionOp Protocol (+ FP8/AMX bridges)
  runtime/         denoise loop + adaptive cache, scheduler, router, types,
                   and the interactive (action-conditioned) seam
  models/          Cosmos, Wan-2.2, V-JEPA 2-AC, LingBot-VA, DreamZero,
                   Cosmos 3 Nano, and the token-VLA engine
  serving/         HTTP (v1/v2) + /v2/world/session WebSocket, the LLM proxy,
                   and the tenancy / metering / usage layer
  bench/           benchmark harness + per-stage profiler
crates/            Rust+PyO3 core — paged latent cache, scheduler, request router
kernels/           Triton (FP8 flash), HIP (FP8 GEMM), CPU AMX kernels
scripts/           model runners, GPU benchmark harnesses, diagnostics
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
