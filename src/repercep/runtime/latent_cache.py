"""Paged latent cache.

The production path is the Rust/PyO3 ``repercep_cache._native`` implementation.
Developer machines can also have no native wheel, or worse, a stale placeholder
package on ``PYTHONPATH``.  In that case this module falls back to the original
pure-Python cache so runtime tests and CPU-only development remain useful.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

PageId = int
RequestId = str


class _PythonLatentCacheError(RuntimeError):
    """Raised when the cache cannot satisfy an allocation."""


@dataclass(slots=True)
class _PythonLatentPage:
    """One fixed-size latent tile's worth of cache bookkeeping."""

    page_id: PageId
    request_id: RequestId
    frame_index: int
    last_used_step: int
    in_use: bool = True


@dataclass(slots=True)
class _PythonCacheStats:
    """A snapshot of cache occupancy."""

    capacity: int
    allocated: int
    free: int
    pinned: int


@runtime_checkable
class EvictionPolicy(Protocol):
    """Chooses which evictable page to reclaim when the cache is full."""

    def victim(self, pages: list[_PythonLatentPage], current_frame: int) -> PageId:
        """Return the page id to evict, or raise ``LatentCacheError``."""
        ...


class FrameAwareEviction:
    """Evict the page whose frame is furthest behind the temporal window."""

    def victim(self, pages: list[_PythonLatentPage], current_frame: int) -> PageId:
        evictable = [p for p in pages if not p.in_use]
        if not evictable:
            raise _PythonLatentCacheError("no evictable pages: cache is fully pinned")
        victim = min(
            evictable,
            key=lambda p: (p.frame_index - current_frame, p.last_used_step),
        )
        return victim.page_id


class _PythonPagedLatentCache:
    """A fixed pool of latent pages with a frame-aware eviction policy."""

    def __init__(self, num_pages: int, policy: EvictionPolicy | None = None) -> None:
        if num_pages < 1:
            raise ValueError("num_pages must be >= 1")
        self._capacity = num_pages
        self._policy: EvictionPolicy = policy or FrameAwareEviction()
        self._pages: dict[PageId, _PythonLatentPage] = {}
        self._next_id: PageId = 0

    def allocate(
        self, request_id: RequestId, frame_index: int, step: int, current_frame: int
    ) -> PageId:
        """Reserve a page, evicting a victim first if the pool is full."""
        if len(self._pages) >= self._capacity:
            victim = self._policy.victim(list(self._pages.values()), current_frame)
            del self._pages[victim]
        page_id = self._next_id
        self._next_id += 1
        self._pages[page_id] = _PythonLatentPage(
            page_id=page_id,
            request_id=request_id,
            frame_index=frame_index,
            last_used_step=step,
        )
        return page_id

    def touch(self, page_id: PageId, step: int) -> None:
        """Record that ``page_id`` was read at diffusion ``step``."""
        page = self._pages.get(page_id)
        if page is not None:
            page.last_used_step = step

    def unpin(self, page_id: PageId) -> None:
        """Mark a page evictable without freeing it."""
        page = self._pages.get(page_id)
        if page is not None:
            page.in_use = False

    def release_request(self, request_id: RequestId) -> int:
        """Free every page held by a request. Returns the number freed."""
        victims = [pid for pid, p in self._pages.items() if p.request_id == request_id]
        for pid in victims:
            del self._pages[pid]
        return len(victims)

    def stats(self) -> _PythonCacheStats:
        pinned = sum(1 for p in self._pages.values() if p.in_use)
        return _PythonCacheStats(
            capacity=self._capacity,
            allocated=len(self._pages),
            free=self._capacity - len(self._pages),
            pinned=pinned,
        )


def _native_cache_is_usable(cache_cls: type[Any]) -> bool:
    try:
        cache = cache_cls(num_pages=1)
        stats = cache.stats()
    except Exception:
        return False
    return all(isinstance(getattr(stats, field, None), int) for field in _STATS_FIELDS)


def _load_native_cache() -> tuple[Any, Any, Any, Any] | None:
    try:
        native = importlib.import_module("repercep_cache._native")
    except Exception:
        return None
    native_cache_stats = native.CacheStats
    native_latent_cache_error = native.LatentCacheError
    native_latent_page = native.LatentPage
    native_paged_latent_cache = native.PagedLatentCache
    if not _native_cache_is_usable(native_paged_latent_cache):
        return None
    return (
        native_cache_stats,
        native_latent_cache_error,
        native_latent_page,
        native_paged_latent_cache,
    )


_STATS_FIELDS = ("capacity", "allocated", "free", "pinned")
_native_cache = _load_native_cache()

CacheStats: Any
LatentCacheError: Any
LatentPage: Any
PagedLatentCache: Any

if _native_cache is None:
    CacheStats = _PythonCacheStats
    LatentCacheError = _PythonLatentCacheError
    LatentPage = _PythonLatentPage
    PagedLatentCache = _PythonPagedLatentCache
else:
    CacheStats, LatentCacheError, LatentPage, PagedLatentCache = _native_cache


__all__ = ["CacheStats", "LatentCacheError", "LatentPage", "PagedLatentCache"]
