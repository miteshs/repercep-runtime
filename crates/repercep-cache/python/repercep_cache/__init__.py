"""repercep_cache — Rust-backed paged latent cache.

The public Python API for the cache lives at `repercep.runtime.latent_cache`;
this package exposes the raw `_native` extension module that wrapper imports
from. Application code should not import from this package directly.
"""

from repercep_cache._native import (
    CacheStats,
    LatentCacheError,
    LatentPage,
    PagedLatentCache,
)

__all__ = ["CacheStats", "LatentCacheError", "LatentPage", "PagedLatentCache"]
