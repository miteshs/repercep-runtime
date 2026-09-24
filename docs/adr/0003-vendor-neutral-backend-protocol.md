# ADR-0003 — Vendor-neutral backend Protocol

- **Status:** Accepted
- **Date:** 2026-05-22
- **Related:** ADR-0001 (MI300X-first)

## Context

Repercep leads on MI300X (ADR-0001) but the implementation plan's first milestone
includes a second hardware target, and the long-term thesis is multi-vendor
codegen. We must lead on one vendor *without* making the second a rewrite.

The plan (§3.4) is explicit: "Define interfaces between Runtime, Kernel, Studio
components before implementations ... comprehensive Protocol classes for
backends."

## Decision

A single seam — `repercep.backend.protocol.Backend`, a `typing.Protocol` — is the
only place Repercep touches a GPU vendor. It owns:

- capability declaration (`capabilities()` → `BackendCapabilities`)
- device discovery (`devices()` → `DeviceSpec` tuple)
- device handles (`torch_device()`)
- dtype policy (`default_dtype()`)
- kernel selection (`attention_op(shape, dtype)` → `AttentionOp`)

Everything above the seam — runtime, models, serving — imports `Backend` and
never a concrete backend. `ROCmBackend` is the first and currently only
implementation. `select_backend()` returns the first available one.

## Rationale

- **Adding NVIDIA = adding one class.** A `CUDABackend` satisfying the Protocol,
  appended to the registry's `_ALL_BACKENDS`. No existing file changes. This is
  what makes ADR-0001 (lead on AMD) safe.
- **Structural typing, not inheritance.** `Protocol` + `@runtime_checkable`
  means a backend conforms by shape, not by importing a base class. Tested
  directly (`test_rocm_backend_satisfies_protocol`).
- **The backend is policy, not a torch re-abstraction.** PyTorch already hides
  CUDA-vs-HIP behind `torch.cuda`. Re-wrapping that would be pure overhead. The
  backend instead carries the decisions torch does *not* make: which attention
  kernel, which dtype, what the hardware can do.
- **`mypy --strict` makes the Protocol a real contract.** A backend that drifts
  from the Protocol fails type-checking, not production.

## Consequences

- A small, deliberate indirection cost: the runtime asks the backend for an
  attention op rather than calling a kernel directly. This is the intended
  design, and the cost is one method call per layer construction, not per
  forward pass.
- The Protocol surface must stay minimal. Every method added is a method every
  future backend must implement. New capability flags go in
  `BackendCapabilities` (data), not as new Protocol methods, wherever possible.
- The kernel layer, when it lands, sits *below* this seam and is explicitly
  exempt from the clean-architecture rules here — see the plan §3.4.
