"""Native extension package for the Repercep per-request router.

The Python-facing API lives in ``repercep.runtime.router``; this package only
exposes the compiled `_native` submodule produced by maturin from the
``repercep-router`` crate. See crates/README.md and ADR-0005.
"""
