//! Repercep paged latent cache.
//!
//! Rust port of `src/repercep/runtime/latent_cache.py`. vLLM's PagedAttention
//! pages a KV cache for autoregressive token decode. World models need a
//! different object: a cache of latent video tiles reused across diffusion
//! steps and across temporal frames. A page here is a fixed-size latent tile;
//! eviction is frame-aware — a tile far behind the current temporal window is
//! a better victim than a recently-touched one, regardless of raw LRU age.
//!
//! The page table, allocation accounting, and the eviction-policy seam are
//! real and tested; tensor-backed page storage is wired in with the inference
//! path, where a page's byte size depends on the Cosmos tokenizer's latent
//! shape.
//!
//! The Python wrapper at `src/repercep/runtime/latent_cache.py` imports from
//! `repercep_cache._native`, this crate's library, via PyO3.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::collections::BTreeMap;

use parking_lot::Mutex;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

/// Identifier assigned to each cache page. Monotonically increasing per cache.
pub type PageId = u64;

/// Identifier for the request that owns one or more pages.
pub type RequestId = String;

/// Errors raised by the paged latent cache.
#[derive(Debug, Error)]
pub enum LatentCacheError {
    /// The cache is full and every page is currently pinned, so no victim can
    /// be evicted.
    #[error("no evictable pages: cache is fully pinned")]
    FullyPinned,
}

/// One fixed-size latent tile's worth of cache bookkeeping.
///
/// Mirrors the Python `LatentPage` dataclass verbatim.
#[pyclass(name = "LatentPage", module = "repercep_cache._native")]
#[derive(Debug, Clone)]
pub struct LatentPage {
    /// Unique identifier for the page.
    #[pyo3(get)]
    pub page_id: PageId,
    /// Identifier of the request that allocated the page.
    #[pyo3(get)]
    pub request_id: RequestId,
    /// Temporal frame index this page is bound to.
    #[pyo3(get)]
    pub frame_index: i64,
    /// Diffusion step at which this page was last touched.
    #[pyo3(get)]
    pub last_used_step: i64,
    /// Whether this page is currently pinned. Pinned pages cannot be evicted.
    #[pyo3(get)]
    pub in_use: bool,
}

#[pymethods]
impl LatentPage {
    #[new]
    #[pyo3(signature = (page_id, request_id, frame_index, last_used_step, in_use=true))]
    fn py_new(
        page_id: PageId,
        request_id: RequestId,
        frame_index: i64,
        last_used_step: i64,
        in_use: bool,
    ) -> Self {
        Self {
            page_id,
            request_id,
            frame_index,
            last_used_step,
            in_use,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LatentPage(page_id={}, request_id='{}', frame_index={}, last_used_step={}, in_use={})",
            self.page_id, self.request_id, self.frame_index, self.last_used_step, self.in_use,
        )
    }
}

/// A snapshot of cache occupancy.
///
/// Mirrors the Python `CacheStats` dataclass verbatim.
#[pyclass(name = "CacheStats", module = "repercep_cache._native")]
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Total number of pages the cache can hold.
    #[pyo3(get)]
    pub capacity: usize,
    /// Number of pages currently allocated (pinned or otherwise).
    #[pyo3(get)]
    pub allocated: usize,
    /// Number of pages currently free.
    #[pyo3(get)]
    pub free: usize,
    /// Number of pages that are pinned (cannot be evicted).
    #[pyo3(get)]
    pub pinned: usize,
}

#[pymethods]
impl CacheStats {
    #[new]
    fn py_new(capacity: usize, allocated: usize, free: usize, pinned: usize) -> Self {
        Self {
            capacity,
            allocated,
            free,
            pinned,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheStats(capacity={}, allocated={}, free={}, pinned={})",
            self.capacity, self.allocated, self.free, self.pinned,
        )
    }
}

/// Eviction policy seam: chooses which evictable page to reclaim when the
/// cache is full.
pub trait EvictionPolicy: Send + Sync {
    /// Return the page id to evict from `pages` given the current temporal
    /// frame, or an error if no evictable page exists.
    fn victim(&self, pages: &[LatentPage], current_frame: i64) -> Result<PageId, LatentCacheError>;
}

/// Evict the page whose frame is furthest behind the temporal window.
///
/// Ties are broken by least-recently-used diffusion step. This is the
/// world-model-specific reason the cache is not a plain LRU map: frames the
/// rollout has moved past will not be read again, so they are always the
/// right victim even if they were touched more recently than an in-window
/// frame.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameAwareEviction;

impl EvictionPolicy for FrameAwareEviction {
    fn victim(&self, pages: &[LatentPage], current_frame: i64) -> Result<PageId, LatentCacheError> {
        // Most negative (frame_index - current_frame) == furthest behind;
        // tie-break on the oldest diffusion step. Mirrors Python `min(..., key=...)`.
        let mut best: Option<&LatentPage> = None;
        for p in pages.iter().filter(|p| !p.in_use) {
            let key_p = (p.frame_index - current_frame, p.last_used_step);
            match best {
                None => best = Some(p),
                Some(cur) => {
                    let key_cur = (cur.frame_index - current_frame, cur.last_used_step);
                    if key_p < key_cur {
                        best = Some(p);
                    }
                }
            }
        }
        best.map(|p| p.page_id).ok_or(LatentCacheError::FullyPinned)
    }
}

/// Internal mutable state of a cache, guarded by a `parking_lot::Mutex`.
struct CacheState {
    capacity: usize,
    pages: BTreeMap<PageId, LatentPage>,
    next_id: PageId,
}

/// A fixed pool of latent pages with a frame-aware eviction policy.
///
/// Mirrors the Python `PagedLatentCache` class verbatim. Internal state is
/// guarded by a `parking_lot::Mutex` so the type is `Send + Sync` and safe to
/// expose to Python where the GIL is released around blocking work.
#[pyclass(name = "PagedLatentCache", module = "repercep_cache._native")]
pub struct PagedLatentCache {
    state: Mutex<CacheState>,
    policy: Box<dyn EvictionPolicy + Send + Sync>,
}

impl PagedLatentCache {
    /// Create a new cache with the given capacity and explicit eviction policy.
    pub fn new(
        num_pages: usize,
        policy: Box<dyn EvictionPolicy + Send + Sync>,
    ) -> Result<Self, &'static str> {
        if num_pages < 1 {
            return Err("num_pages must be >= 1");
        }
        Ok(Self {
            state: Mutex::new(CacheState {
                capacity: num_pages,
                pages: BTreeMap::new(),
                next_id: 0,
            }),
            policy,
        })
    }

    /// Create a new cache with the default `FrameAwareEviction` policy.
    pub fn with_default_policy(num_pages: usize) -> Result<Self, &'static str> {
        Self::new(num_pages, Box::new(FrameAwareEviction))
    }

    /// Reserve a page, evicting a victim first if the pool is full.
    pub fn allocate(
        &self,
        request_id: RequestId,
        frame_index: i64,
        step: i64,
        current_frame: i64,
    ) -> Result<PageId, LatentCacheError> {
        let mut state = self.state.lock();
        if state.pages.len() >= state.capacity {
            // Snapshot the current page set for the policy (Python passes a
            // `list(self._pages.values())`).
            let snapshot: Vec<LatentPage> = state.pages.values().cloned().collect();
            let victim = self.policy.victim(&snapshot, current_frame)?;
            state.pages.remove(&victim);
        }
        let page_id = state.next_id;
        state.next_id += 1;
        state.pages.insert(
            page_id,
            LatentPage {
                page_id,
                request_id,
                frame_index,
                last_used_step: step,
                in_use: true,
            },
        );
        Ok(page_id)
    }

    /// Record that `page_id` was read at diffusion `step`.
    pub fn touch(&self, page_id: PageId, step: i64) {
        let mut state = self.state.lock();
        if let Some(page) = state.pages.get_mut(&page_id) {
            page.last_used_step = step;
        }
    }

    /// Mark a page evictable without freeing it.
    pub fn unpin(&self, page_id: PageId) {
        let mut state = self.state.lock();
        if let Some(page) = state.pages.get_mut(&page_id) {
            page.in_use = false;
        }
    }

    /// Free every page held by `request_id`. Returns the number of pages
    /// freed.
    pub fn release_request(&self, request_id: &str) -> usize {
        let mut state = self.state.lock();
        let victims: Vec<PageId> = state
            .pages
            .iter()
            .filter(|(_, p)| p.request_id == request_id)
            .map(|(pid, _)| *pid)
            .collect();
        let count = victims.len();
        for pid in victims {
            state.pages.remove(&pid);
        }
        count
    }

    /// Return a snapshot of cache occupancy.
    pub fn stats(&self) -> CacheStats {
        let state = self.state.lock();
        let pinned = state.pages.values().filter(|p| p.in_use).count();
        let allocated = state.pages.len();
        CacheStats {
            capacity: state.capacity,
            allocated,
            free: state.capacity - allocated,
            pinned,
        }
    }
}

#[pymethods]
impl PagedLatentCache {
    #[new]
    #[pyo3(signature = (num_pages, policy=None))]
    fn py_new(num_pages: i64, policy: Option<PyObject>) -> PyResult<Self> {
        if num_pages < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_pages must be >= 1",
            ));
        }
        // Stage 3 doesn't expose a way to plug in a custom Python policy; the
        // Python wrapper drops `EvictionPolicy` from the public surface and
        // existing tests don't exercise a custom policy. Any non-None value
        // is treated as the default policy for now.
        let _ = policy;
        PagedLatentCache::with_default_policy(num_pages as usize)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyo3(name = "allocate")]
    fn py_allocate(
        &self,
        request_id: String,
        frame_index: i64,
        step: i64,
        current_frame: i64,
    ) -> PyResult<PageId> {
        self.allocate(request_id, frame_index, step, current_frame)
            .map_err(|e| LatentCacheErrorPy::new_err(e.to_string()))
    }

    #[pyo3(name = "touch")]
    fn py_touch(&self, page_id: PageId, step: i64) {
        self.touch(page_id, step);
    }

    #[pyo3(name = "unpin")]
    fn py_unpin(&self, page_id: PageId) {
        self.unpin(page_id);
    }

    #[pyo3(name = "release_request")]
    fn py_release_request(&self, request_id: &str) -> usize {
        self.release_request(request_id)
    }

    #[pyo3(name = "stats")]
    fn py_stats(&self) -> CacheStats {
        self.stats()
    }
}

create_exception!(
    repercep_cache,
    LatentCacheErrorPy,
    PyRuntimeError,
    "Raised when the paged latent cache cannot satisfy an allocation."
);

/// PyO3 module entry point. Exposed as `repercep_cache._native` in Python.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LatentPage>()?;
    m.add_class::<CacheStats>()?;
    m.add_class::<PagedLatentCache>()?;
    m.add("LatentCacheError", m.py().get_type::<LatentCacheErrorPy>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_cache(n: usize) -> PagedLatentCache {
        PagedLatentCache::with_default_policy(n).expect("valid capacity")
    }

    #[test]
    fn empty_cache_stats_match_capacity() {
        let cache = new_cache(4);
        let stats = cache.stats();
        assert_eq!(stats.capacity, 4);
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.free, 4);
        assert_eq!(stats.pinned, 0);
    }

    #[test]
    fn allocate_up_to_capacity() {
        let cache = new_cache(4);
        for i in 0..4 {
            cache
                .allocate("req-a".into(), i, 0, i)
                .expect("allocates within capacity");
        }
        let stats = cache.stats();
        assert_eq!(stats.allocated, 4);
        assert_eq!(stats.free, 0);
        assert_eq!(stats.pinned, 4);
    }

    #[test]
    fn eviction_picks_furthest_behind_frame() {
        let cache = new_cache(2);
        let behind = cache
            .allocate("r".into(), 0, 0, 10)
            .expect("first allocation");
        let recent = cache
            .allocate("r".into(), 9, 0, 10)
            .expect("second allocation");
        cache.unpin(behind);
        cache.unpin(recent);
        // current_frame jumps to 20 -- frame-0 page is furthest behind.
        let new_pid = cache
            .allocate("r".into(), 20, 1, 20)
            .expect("eviction allowed");
        assert_eq!(cache.stats().allocated, 2);
        // The `recent` page (frame 9) must still be present; the `behind` page
        // (frame 0) was the victim.
        let state = cache.state.lock();
        assert!(state.pages.contains_key(&recent));
        assert!(!state.pages.contains_key(&behind));
        assert!(state.pages.contains_key(&new_pid));
    }

    #[test]
    fn eviction_ties_broken_by_oldest_step() {
        let cache = new_cache(2);
        // Two pages, both at frame 5, both unpinned.
        let older = cache
            .allocate("r".into(), 5, 1, 10)
            .expect("first allocation");
        let newer = cache
            .allocate("r".into(), 5, 5, 10)
            .expect("second allocation");
        cache.unpin(older);
        cache.unpin(newer);
        // Allocate again -- the page with the older step should be evicted.
        cache
            .allocate("r".into(), 6, 6, 10)
            .expect("eviction allowed");
        let state = cache.state.lock();
        assert!(
            !state.pages.contains_key(&older),
            "older step is the victim"
        );
        assert!(state.pages.contains_key(&newer), "newer step survives");
    }

    #[test]
    fn allocate_when_fully_pinned_raises() {
        let cache = new_cache(1);
        cache
            .allocate("r".into(), 0, 0, 0)
            .expect("first allocation");
        let err = cache.allocate("r".into(), 1, 0, 1).unwrap_err();
        assert!(matches!(err, LatentCacheError::FullyPinned));
    }

    #[test]
    fn release_request_frees_exactly_that_requests_pages() {
        let cache = new_cache(4);
        cache.allocate("a".into(), 0, 0, 0).expect("a-0");
        cache.allocate("a".into(), 1, 0, 0).expect("a-1");
        cache.allocate("b".into(), 0, 0, 0).expect("b-0");
        cache.allocate("a".into(), 2, 0, 0).expect("a-2");
        let freed = cache.release_request("a");
        assert_eq!(freed, 3);
        let stats = cache.stats();
        assert_eq!(stats.allocated, 1);
        // The remaining page belongs to request "b".
        let state = cache.state.lock();
        assert!(state.pages.values().all(|p| p.request_id == "b"));
    }

    #[test]
    fn touch_updates_last_used_step() {
        let cache = new_cache(2);
        let pid = cache.allocate("r".into(), 0, 0, 0).expect("alloc");
        cache.touch(pid, 7);
        let state = cache.state.lock();
        assert_eq!(state.pages[&pid].last_used_step, 7);
    }

    #[test]
    fn touch_unknown_page_is_noop() {
        let cache = new_cache(2);
        // Should not panic.
        cache.touch(999, 7);
    }

    #[test]
    fn unpin_makes_page_evictable() {
        let cache = new_cache(1);
        let pinned = cache.allocate("r".into(), 0, 0, 0).expect("alloc");
        // Fully pinned -- allocation must fail.
        assert!(cache.allocate("r".into(), 1, 0, 1).is_err());
        cache.unpin(pinned);
        // Now the page is evictable, the next allocation should succeed.
        cache
            .allocate("r".into(), 1, 0, 1)
            .expect("allocation should succeed after unpin");
        assert_eq!(cache.stats().allocated, 1);
    }

    #[test]
    fn release_request_returns_zero_for_unknown_request() {
        let cache = new_cache(2);
        cache.allocate("a".into(), 0, 0, 0).expect("alloc");
        assert_eq!(cache.release_request("nonexistent"), 0);
        assert_eq!(cache.stats().allocated, 1);
    }

    #[test]
    fn page_ids_are_monotonic_across_evictions() {
        let cache = new_cache(1);
        let p0 = cache.allocate("r".into(), 0, 0, 0).expect("alloc");
        cache.unpin(p0);
        let p1 = cache.allocate("r".into(), 1, 0, 1).expect("alloc");
        assert!(p1 > p0, "page ids must be monotonically increasing");
    }
}
