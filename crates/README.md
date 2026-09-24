# crates/ — Repercep Rust workspace

The Rust core of the Repercep Runtime. **Python is the surface; Rust is the
core.** Anything that customers touch (model loading, denoise loop, attention
op, serving HTTP/gRPC handler, CLI) stays in `src/repercep/`. Anything that
benefits from no GIL, deterministic latency, and one codebase shared between
the datacenter API and a future closed-loop edge path lives here.

The fork that put this directory on the map is recorded in
[`docs/adr/0005-rust-core-fork-resolved.md`](../docs/adr/0005-rust-core-fork-resolved.md).

## Members

| Crate | Scope | Status |
|---|---|---|
| `repercep-cache` | Paged latent-cache manager. Port of `src/repercep/runtime/latent_cache.py`. | Stage 3 |
| `repercep-scheduler` | Request scheduler — priority queue, FIFO within priority, preemption seam. | Stage 3 (greenfield) |
| `repercep-router` | Per-request state machine, frame ordering across batched requests. | Stage 3 (greenfield) |

## Shared conventions (every crate follows these)

**Library name.** Each crate's Cargo lib name is **unique within the
workspace** — `repercep_cache_native`, `repercep_router_native`,
`repercep_scheduler_native` — so the three never collide on the shared
`target/release/lib<name>.so` output. The Python import path is preserved
at `repercep_<name>._native` via `[tool.maturin] module-name =
"repercep_<name>._native"` in each crate's `pyproject.toml`, which renames
the .so as it's packaged into the wheel. Both `cdylib` (the Python
extension) and `rlib` (so other Rust crates can depend on this one)
crate-types are emitted. The Python wrapper at
`src/repercep/runtime/<name>.py` imports from `repercep_<name>._native`
(e.g. `from repercep_cache._native import …`) regardless of the Cargo lib
name — only the maturin module-name and the `#[pymodule] fn _native`
function matter for the import path.

**Versions.** All shared dep versions live in the root `Cargo.toml`'s
`[workspace.dependencies]`. Use `{ workspace = true }` to inherit. Never pin
a dep inside an individual crate unless it is the only consumer.

**Errors.**
- Library code uses `thiserror`-derived enums. The boundary types map to
  Python exceptions in the PyO3 `#[pyclass]` impl.
- `unwrap()`, `expect()`, `panic!()` are forbidden in library code (clippy
  enforced). Tests may use them.
- `anyhow` is forbidden in library crates. It belongs only in binaries or
  test code.

**Observability.** Use `tracing` spans/events. Never `println!` in library
code. Subscribers (`tracing-subscriber`) are wired only by tests or the
Python integration layer.

**Async.** `tokio` (multi-thread runtime). Crates that don't need async
(today: only `repercep-cache`) don't depend on it. Don't pull `async-std` or
`smol` — single runtime across the workspace.

**Locks.** `parking_lot::Mutex` / `RwLock` over std equivalents on hot paths.
Std locks are fine for cold paths; parking_lot is non-poisoning, faster, and
gives a smaller surface to reason about.

**Naming.**
- Crate names: `repercep-<role>` (kebab).
- Public Rust types: `PascalCase`, never prefixed with `Repercep` — the crate
  name is the prefix.
- PyO3 `#[pyclass(name = "…")]` names match the Python wrapper's public API
  so the wrapper is a thin shim, not a translator.

**Lints / format.**
- `cargo fmt --all` must be clean.
- `cargo clippy --workspace --all-targets -- -D warnings` must be clean.
- Crate-level lints (in `lib.rs`): `#![deny(unsafe_op_in_unsafe_fn)]` and
  `#![warn(missing_docs)]` for the public surface.

**Tests.**
- Rust unit tests inside `crates/<name>/src/lib.rs` `#[cfg(test)] mod tests`.
- Cross-language tests stay in Python pytest under `tests/test_<name>.py` —
  they're the integration contract.

## Python wrapper pattern

```python
# src/repercep/runtime/latent_cache.py
from __future__ import annotations

from repercep_cache._native import (
    PagedLatentCache as _Cache,
    LatentCacheError,
    CacheStats,
)

# Re-export so the public API is unchanged from the prior pure-Python module.
__all__ = ["PagedLatentCache", "LatentCacheError", "CacheStats"]

PagedLatentCache = _Cache  # or wrap if the surface needs Python-side polish
```

The wrapper must preserve the prior pure-Python module's public API verbatim
where one existed (cache is a port; scheduler and router are greenfield, so
the wrapper IS the API definition).

## Building

`make rust-install` runs `maturin build --release` for every crate, then
`uv pip install --reinstall` for all built wheels. We deliberately do NOT
use `maturin develop`: that command requires `pip` inside the venv, but
this project's venv is `uv`-managed (no pip), and `--uv` flag in maturin
1.13 doesn't reliably handle workspace-member crates with separate
per-crate `pyproject.toml`. Build-then-install is two steps but more
robust and CI-friendly. Wheels go to `target/wheels/`.

The user-facing dev loop is:

```bash
make rust-install      # build every crate's wheel, install into .venv
make check-all         # ruff + mypy + pytest + cargo fmt-check + clippy + cargo test
```

## What does NOT live here

- Triton / CUDA / HIP kernels — those live in `../kernels/`, with different
  rules (no clippy lints, no mypy expectations, perf-first).
- Anything Python — `../src/repercep/`. The wrapper file is the only Python
  the Rust crate cares about.
- The MLIR dialect (Phase 4+, later scope) — separate codebase when that
  arrives.
