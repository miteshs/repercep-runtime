# ADR-0005 — Rust-core fork resolved in favor of Rust (Stage 3 unblocked)

- **Status:** Accepted
- **Date:** 2026-05-23
- **Resolves:** the open fork from the original implementation plan
  ("Language stack" section)
- **Supersedes:** the deferred posture in
  [ADR-0004](0004-polyglot-build-tooling.md) (`Rust-core fork deferred`)

## Context

The implementation plan spelled out a binary fork:

> If all of the first target workloads are datacenter, it's defensible to
> ship a Python-core (vLLM-style) datacenter runtime first and defer the Rust
> hot-path rewrite — ~2–3mo faster to v0.1, max Claude Code leverage. If even
> one early workload is closed-loop robotics (edge / sub-100ms), build
> Rust-core from day one.

The Python core works (Cosmos-Predict-7B on MI300X at 2.47× the H100
reference, 154 s on the 121 f / 36-step config). The early workload mix is not
yet known. By the plan's own logic, Python-core is the *defensible* default.

We are choosing Rust-core anyway, ahead of knowing the workload mix.

## Decision

Move three components of the Runtime — and only these three — to Rust crates
under `crates/`, with PyO3 bindings:

- `crates/repercep-cache` — paged latent-cache manager (port of
  `src/repercep/runtime/latent_cache.py`).
- `crates/repercep-scheduler` — request scheduler (greenfield).
- `crates/repercep-router` — per-request state machine + frame ordering
  (greenfield).

Everything else stays Python: model loading, the HF/Diffusers integration,
the denoise loop (where step-skip caching lives), the Cosmos engine, the
attention `Protocol` and SDPA→aotriton path, the FastAPI / gRPC serving
handlers, the CLI, the benchmark + profiler harnesses.

This is not a rewrite. It is a targeted extraction of the three components
that gain the most from no-GIL parallelism and deterministic latency.

## Rationale

1. **The 35-file Python core is at maximum tractability for this work.** Any
   month we wait, more code accretes and the extraction gets more expensive.
   The cost curve is monotonic up — there is no later moment when this is
   cheaper than now.

2. **The closed-loop robotics scenario is asymmetric.** If closed-loop robotics
   becomes an early workload, the plan projects a 6–9 month cold-start
   Rust rewrite. The cost of doing the targeted Rust work now (estimated
   ~2 weeks via parallel agent work) is much smaller than the option value
   it buys. Even if that workload never materializes, the cache/scheduler/router
   surface is a real ergonomic win — measurable latency tails on cache
   ops, GIL-free request orchestration.

3. **Scope is narrow and reversible.** Three crates, all named seams, all
   tested by existing Python pytest. If Phase 2 work (Wan-2.2, adaptive
   caching, FP8) slips by more than a month attributable to Rust eating
   cycles, the Python wrappers preserve the public API exactly — we can
   re-back them with pure-Python implementations in a day.

4. **The plan's "Highest-leverage move" already endorses interface discipline
   front-loaded.** mypy strict, Pydantic, Protocol-typed backends are already
   in place. PyO3 + workspace-pinned deps + crate-level `_native` convention
   extends that discipline, not replaces it.

5. **Aggressive use of Claude Code agents in parallel makes the cost
   atypical.** The plan's "1.2–1.5× on novel CUDA/MLIR" figure assumes
   a single human at the keyboard. With three crates dispatched to three
   parallel agents in isolated worktrees, the wall-clock cost is much
   lower than the engineering-month math implies.

## Consequences

- **Python wrappers preserve every existing public API.** `from
  repercep.runtime.latent_cache import PagedLatentCache` keeps working
  byte-for-byte; the wrapper just re-exports the PyO3-bound class.
- **The pre-existing test suite IS the acceptance criterion for the cache.**
  Agent A's mandate: `pytest tests/test_runtime.py` passes unchanged.
- **maturin lands as a dev-only build tool** (`pyproject.toml` `[dev]`
  extras). `make rust-install` runs `maturin develop --release` over the
  workspace; this is the dev loop. CI builds release wheels via maturin
  for distribution.
- **Cargo.lock is committed** starting at the Stage 3 commit (workspace now
  produces Python extension modules — reproducible builds are the right
  default).
- **clippy gate goes from "skip cleanly" to "deny warnings"** the moment
  Stage 3 lands. The Makefile target was already wired in Stage 1.

## Revisit if

- **Phase 2 ships >1 month late** and the attribution traces to Rust work:
  swap the PyO3 wrappers back to pure-Python implementations behind the
  same public API. The seam exists; the cost of unwinding is ~1–2 days
  per crate.
- **Operational experience suggests a different decomposition** of
  which components want to be in Rust. Adjust crate boundaries; the
  workspace + maturin + wrapper pattern is reusable.
- **A second hardware backend (NVIDIA) reveals that the actual hot path
  is in the kernels not the orchestration**, in which case Rust-core was
  defensible but not load-bearing. That doesn't reverse this decision
  but it does change where we invest next.
