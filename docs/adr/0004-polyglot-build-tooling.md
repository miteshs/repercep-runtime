# ADR-0004 — Polyglot build tooling scaffolded; Rust-core fork deferred

- **Status:** Accepted (scaffold only — no Rust code yet)
- **Date:** 2026-05-23
- **Relates to:** the original implementation plan's "Language stack"
  section; ADR-0003 (vendor-neutral backend Protocol)

## Context

The implementation plan commits to a five-language stack with non-overlapping jobs:

- **Python** — user-facing API, model loading, HF/Diffusers/PyTorch integration
- **Rust** — Runtime core (scheduler, request router, paged latent-cache manager)
- **Triton + CUDA / HIP C++** — kernels
- **C++ (MLIR)** — kernel synthesizer's dialect and lowering passes (Phase 4+)
- **TypeScript + React** — Studio dashboard (Phase 3+)

with Cargo + CMake + uv as the build trio (resist Bazel until a real polyglot
monorepo coordination problem appears).

The plan also names a specific fork to resolve before writing the Rust
core: **if any early target workload is closed-loop robotics (sub-100ms edge),
Rust-core from day one; if they are all datacenter, Python-core is defensible
and the Rust hot-path rewrite can be deferred.**

Today the Runtime core is Python (vLLM-style), the MI300X path runs at 2.47×
the H100 reference at 121f / 36 steps, and the early workload mix is not yet known.

## Decision

Scaffold the polyglot tooling now, with **no Rust code yet**:

1. A virtual Cargo workspace at the repo root (`Cargo.toml`, `members = []`).
2. `crates/` for future Rust workspace members.
3. `kernels/` for future Triton / CUDA / HIP — deliberately at the repo root,
   not under `src/`, so it carries a different review standard.
4. `rust-toolchain.toml` pinning stable.
5. `make rust-*` targets that operate on the (empty-for-now) workspace.

The first crate is **not** added by this ADR. Adding the first crate is what
resolves the fork — and that decision is gated on which workloads the runtime must
serve, not on developer convenience.

## Rationale

- **Decouple "can we write Rust" from "should we write Rust *now*."** With
  the scaffold in place, the moment the fork resolves we can populate
  `crates/repercep-cache` (or whichever component goes first) without spending
  a sprint on `pyproject.toml` ↔ `Cargo.toml` build wiring under deadline.
- **Cheap insurance against the robotics scenario.** Per the plan:
  Python-core means a 6–9 month Rust rewrite if closed-loop robotics
  becomes an early workload. The scaffold doesn't pre-pay any of that, but it
  removes the "build system bring-up" tax from the rewrite path entirely.
- **Forcing function for `kernels/` as a separate codebase.** The plan is
  explicit: "Mixing clean-architecture and perf code produces clean-but-slow
  or fast-but-unmaintainable. Treat the kernel layer as a separate codebase
  with different review standards." Putting `kernels/` at the repo root,
  with no expectation that mypy or ruff covers it, locks that culturally.
- **The Cargo + CMake + uv triad, not Bazel.** The plan explicitly defers Bazel
  until there's a "genuine polyglot monorepo coordination problem." We are
  far from that.

## Consequences

- **CI will need a Rust step soon.** None of `rust-fmt-check`, `rust-clippy`,
  `rust-test` does anything meaningful on an empty workspace, but the targets
  exist so the moment we land a crate, CI just needs to call them — no new
  pipeline design under pressure.
- **`maturin` is the planned Python↔Rust binding tool** (not pinned yet).
  Added to `pyproject.toml` `[project.optional-dependencies] dev` when the
  first PyO3 crate lands, not before.
- **`Cargo.lock` will be committed** once the first crate exists (workspace
  produces a Python extension module via PyO3 — reproducible builds are the
  right default). Until then there is no lock to commit.
- **No change to the running MI300X path.** This ADR is repo-shape only; the
  Cosmos-Predict-7B path, the 2.47× headline, the publish-ready writeup, the
  35-file Python codebase, and the 36-test suite are all untouched.

## Revisit when

The early workload mix is known and we know whether the closed-loop
robotics branch is live. At that point either:

- **Closed-loop robotics in:** populate `crates/repercep-cache` immediately (paged
  latent-cache manager has the fewest callers, cleanest PyO3 boundary).
  Then `crates/repercep-scheduler`, then `crates/repercep-router`.
- **All datacenter:** stay Python-core. Re-evaluate later with the operational scars we've actually accumulated.

The trigger is *workload mix*, not engineering preference.
