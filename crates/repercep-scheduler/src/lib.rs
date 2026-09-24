//! Repercep request scheduler — Stage 3 greenfield seam.
//!
//! Three priority buckets (High > Normal > Low), FIFO within bucket,
//! cancellation via a skip-set, capacity backpressure at submit time, and a
//! shutdown drain. Heuristics — preemption, deadline-aware scheduling,
//! multi-engine fan-out, GPU memory accounting — are deliberately deferred;
//! this v0 is the SEAM through which the router and a future engine driver
//! coordinate, not the brains.
//!
//! ## Design choices recorded here so they survive a context reset
//!
//! - **RequestId is a `String`, not `Uuid`.** Callers (router, tests, future
//!   admission control) already mint stable ids — sometimes UUIDs, sometimes
//!   ulids, sometimes `req-<n>` synthetic ids in tests. Forcing UUID at this
//!   seam would either reject those upstream ids or silently rewrite them.
//!   `uuid` stays out of the dep tree until something genuinely needs it.
//! - **`parking_lot::Mutex` for the priority buckets and cancel set.** These
//!   are hot-path locks held for microseconds (push, pop, contains). They are
//!   NEVER held across `.await`. Anything that does need to await — currently
//!   only the wake-on-submit path — uses `tokio::sync::Notify`, which has its
//!   own internal sync.
//! - **Sync `next()` returning a future is overkill for v0.** PyO3 exposes
//!   `next_blocking(timeout_ms)` instead of a Python coroutine, sidestepping
//!   the `pyo3-async-runtimes` dependency and the per-call runtime acquisition
//!   it would impose. The Rust-native `next()` IS an async fn (returns a real
//!   `Future`), so a future async surface is a thin wrapper not a rewrite.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value as JsonValue;
use thiserror::Error;
use tokio::runtime::Runtime;
use tokio::sync::Notify;
use tracing::{debug, info_span};

/// Request priority. `next()` drains High before Normal before Low; within a
/// bucket the order is strict FIFO of submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Priority {
    /// Lowest priority bucket. Drained only when High and Normal are empty.
    Low,
    /// Default priority bucket.
    Normal,
    /// Highest priority bucket. Drained first.
    High,
}

impl Priority {
    /// Parse from the Python-facing string representation. Accepts
    /// `"low"`, `"normal"`, `"high"` case-insensitively.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Priority::Low),
            "normal" => Some(Priority::Normal),
            "high" => Some(Priority::High),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Priority::Low => "low",
            Priority::Normal => "normal",
            Priority::High => "high",
        }
    }
}

/// Opaque request identifier. A `String` so callers can carry forward
/// whatever id scheme they already use (uuid, ulid, `req-<n>`...). See the
/// top-of-file comment for the rationale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);

impl RequestId {
    /// Construct from anything that can be borrowed as `&str`.
    pub fn new(s: impl Into<String>) -> Self {
        RequestId(s.into())
    }
}

/// A single scheduled work item. The `payload` is opaque to the scheduler in
/// v0 — the engine driver downstream decodes it.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Caller-provided id; used by `cancel()`.
    pub id: RequestId,
    /// Which bucket this lands in.
    pub priority: Priority,
    /// Submission timestamp, set by `submit()`. Useful later for
    /// scheduling-latency metrics; carried but not consulted in v0.
    pub submitted_at: Instant,
    /// Opaque payload handed back to the engine driver via `next()`.
    pub payload: JsonValue,
}

/// All scheduler-boundary errors. Each variant maps 1:1 to a Python
/// `SchedulerError` raise site at the FFI boundary.
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// Submit rejected because the scheduler is at capacity. v0 makes
    /// backpressure the caller's problem.
    #[error("queue is full: {capacity}")]
    QueueFull {
        /// The configured capacity for context.
        capacity: usize,
    },
    /// `cancel()` was called with an id not currently in the queue and not
    /// already cancelled.
    #[error("request not found: {0:?}")]
    NotFound(RequestId),
    /// Submit or cancel called after `shutdown()`.
    #[error("scheduler is shutting down")]
    ShuttingDown,
}

// Internal state guarded by a single parking_lot mutex. Keeping all three
// buckets + cancel set + known-id set under ONE lock keeps the invariants
// (submit/cancel/next interactions) trivially atomic. Lock hold time is
// always O(1) ops on small collections, never held across `.await`.
struct Buckets {
    high: VecDeque<ScheduledRequest>,
    normal: VecDeque<ScheduledRequest>,
    low: VecDeque<ScheduledRequest>,
    // Marks ids the caller has cancelled. `next()` skips matching items
    // when popping. We don't try to splice out of the middle of a VecDeque —
    // the skip-set is O(1) and the queues never grow unboundedly (capacity
    // is enforced at submit). Entries are removed from `cancelled` when the
    // matching id is popped-and-skipped by `next()`.
    cancelled: HashSet<RequestId>,
    // Every id currently in the queue OR cancelled-but-not-yet-popped. Lets
    // cancel() distinguish "id I have queued" from "id I never saw" cheaply.
    known: HashSet<RequestId>,
}

impl Buckets {
    fn len(&self) -> usize {
        // The number of items the caller can still expect to receive from
        // `next()`: queued slots minus those marked as cancelled but still
        // physically sitting in a VecDeque.
        let physical = self.high.len() + self.normal.len() + self.low.len();
        physical.saturating_sub(self.cancelled.len())
    }

    fn pop_next(&mut self) -> Option<ScheduledRequest> {
        // Drain High → Normal → Low, skipping cancelled ids and clearing them
        // from the skip-set as we go (so the set stays bounded).
        for q in [&mut self.high, &mut self.normal, &mut self.low] {
            while let Some(item) = q.pop_front() {
                if self.cancelled.remove(&item.id) {
                    self.known.remove(&item.id);
                    continue;
                }
                self.known.remove(&item.id);
                return Some(item);
            }
        }
        None
    }
}

/// The scheduler itself. Cheaply clonable via `Arc` if a caller wants to
/// share submit/cancel handles across tasks; here we keep it simple and
/// expose the public methods on `&self`.
pub struct Scheduler {
    capacity: usize,
    inner: Arc<Inner>,
}

struct Inner {
    buckets: Mutex<Buckets>,
    // Wakes a `next()` caller when an item arrives or shutdown is requested.
    // tokio::sync::Notify is the cheapest "one consumer waiting on a binary
    // signal" primitive in the workspace.
    notify: Notify,
    // Toggled once by `shutdown()`. SeqCst is overkill but the path is cold —
    // shutdown happens once per process.
    shutting_down: AtomicBool,
}

impl Scheduler {
    /// Create a new scheduler with a fixed capacity. Capacity counts live
    /// (not-yet-popped, not-yet-cancelled) requests.
    pub fn new(capacity: usize) -> Self {
        Scheduler {
            capacity,
            inner: Arc::new(Inner {
                buckets: Mutex::new(Buckets {
                    high: VecDeque::new(),
                    normal: VecDeque::new(),
                    low: VecDeque::new(),
                    cancelled: HashSet::new(),
                    known: HashSet::new(),
                }),
                notify: Notify::new(),
                shutting_down: AtomicBool::new(false),
            }),
        }
    }

    /// Submit a request. Returns `QueueFull` if at capacity, `ShuttingDown`
    /// after `shutdown()` has been called.
    pub fn submit(&self, req: ScheduledRequest) -> Result<(), SchedulerError> {
        let span = info_span!("scheduler.submit", id = %req.id.0, priority = req.priority.as_str());
        let _enter = span.enter();

        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return Err(SchedulerError::ShuttingDown);
        }

        let mut b = self.inner.buckets.lock();
        if b.len() >= self.capacity {
            return Err(SchedulerError::QueueFull {
                capacity: self.capacity,
            });
        }

        b.known.insert(req.id.clone());
        match req.priority {
            Priority::High => b.high.push_back(req),
            Priority::Normal => b.normal.push_back(req),
            Priority::Low => b.low.push_back(req),
        }
        // Drop the lock BEFORE notifying so a woken waiter doesn't immediately
        // contend on the same mutex.
        drop(b);
        self.inner.notify.notify_one();
        debug!("submit accepted");
        Ok(())
    }

    /// Pop the next request in priority order, awaiting until one arrives or
    /// the scheduler shuts down and drains. Returns `None` only when shutdown
    /// has been requested AND the queue is empty.
    pub async fn next(&self) -> Option<ScheduledRequest> {
        loop {
            // Take a Notified BEFORE checking state to close the race where
            // submit() notifies between our peek and our await.
            let notified = self.inner.notify.notified();

            {
                let mut b = self.inner.buckets.lock();
                if let Some(item) = b.pop_next() {
                    let span = info_span!(
                        "scheduler.next",
                        id = %item.id.0,
                        priority = item.priority.as_str(),
                    );
                    let _e = span.enter();
                    debug!("next returning item");
                    return Some(item);
                }
                if self.inner.shutting_down.load(Ordering::SeqCst) {
                    debug!("scheduler.next: drained and shutting down");
                    return None;
                }
            }

            notified.await;
        }
    }

    /// Mark an id as cancelled. The item is not physically removed from the
    /// queue; `next()` will skip it when it surfaces. Returns `NotFound` if
    /// the id is neither queued nor already pending cancellation.
    pub fn cancel(&self, id: &RequestId) -> Result<(), SchedulerError> {
        let span = info_span!("scheduler.cancel", id = %id.0);
        let _enter = span.enter();

        if self.inner.shutting_down.load(Ordering::SeqCst) {
            return Err(SchedulerError::ShuttingDown);
        }

        let mut b = self.inner.buckets.lock();
        if !b.known.contains(id) {
            return Err(SchedulerError::NotFound(id.clone()));
        }
        b.cancelled.insert(id.clone());
        debug!("cancel marked");
        Ok(())
    }

    /// Number of live (queued, not cancelled) requests.
    pub fn len(&self) -> usize {
        self.inner.buckets.lock().len()
    }

    /// `true` iff no live requests are queued.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The fixed capacity supplied to `new`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Request shutdown. Subsequent `submit` / `cancel` return
    /// `ShuttingDown`; pending `next()` calls resolve to `None` once the
    /// queue drains.
    pub fn shutdown(&self) {
        let span = info_span!("scheduler.shutdown");
        let _enter = span.enter();
        self.inner.shutting_down.store(true, Ordering::SeqCst);
        // Wake EVERY waiter — they need to recheck and return None when the
        // queue is drained.
        self.inner.notify.notify_waiters();
        debug!("shutdown flag set");
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Scheduler::new(1024)
    }
}

// ============================================================================
// PyO3 surface
// ============================================================================

// `SchedulerError` is the Rust *enum*; `SchedulerExc` is its Python-facing
// exception twin. Naming them differently avoids the macro colliding with the
// enum identifier while keeping the Python class name set explicitly below.
create_exception!(
    _native,
    SchedulerExc,
    PyRuntimeError,
    "Python-side exception type for all scheduler boundary errors. \
     Wraps the Rust `SchedulerError` variants by message; exposed to Python \
     under the name `SchedulerError` in the `_native` module."
);

fn map_err(e: SchedulerError) -> PyErr {
    SchedulerExc::new_err(e.to_string())
}

// Convert a Python dict (or JSON-coercible object) into serde_json::Value via
// the json module — keeps the FFI boundary type-narrow without pulling
// pythonize/serde-pyobject as a dep. The payload crosses the boundary as JSON
// in both directions in v0; the engine driver downstream is the only consumer.
fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    let json_mod = py.import("json")?;
    let s: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyRuntimeError::new_err(format!("payload not JSON: {e}")))
}

fn json_to_py<'py>(py: Python<'py>, val: &JsonValue) -> PyResult<Bound<'py, PyAny>> {
    let json_mod = py.import("json")?;
    let s = serde_json::to_string(val)
        .map_err(|e| PyRuntimeError::new_err(format!("payload not serializable: {e}")))?;
    json_mod.call_method1("loads", (s,))
}

/// Python-facing wrapper. One-to-one with `Scheduler`; the wrapper Python
/// module is a thin re-export.
#[pyclass(name = "Scheduler", module = "repercep_scheduler._native")]
pub struct PyScheduler {
    inner: Arc<Scheduler>,
    // One Tokio runtime per scheduler, owned for the lifetime of the object,
    // so `next_blocking` doesn't pay runtime-spin-up per call. Multi-thread
    // because the workspace standard is multi-thread; a single-threaded
    // runtime would be cheaper but inconsistent with the rest of the stack.
    runtime: Runtime,
}

#[pymethods]
impl PyScheduler {
    #[new]
    fn py_new(capacity: usize) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
        Ok(PyScheduler {
            inner: Arc::new(Scheduler::new(capacity)),
            runtime,
        })
    }

    /// Submit a request. `priority` must be `"low"`, `"normal"`, or `"high"`.
    fn submit(
        &self,
        py: Python<'_>,
        request_id: String,
        priority: String,
        payload: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let prio = Priority::from_str(&priority)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown priority: {priority:?}")))?;
        let payload_json = py_to_json(py, &payload)?;
        let req = ScheduledRequest {
            id: RequestId(request_id),
            priority: prio,
            submitted_at: Instant::now(),
            payload: payload_json,
        };
        self.inner.submit(req).map_err(map_err)
    }

    /// Block until the next request is available or `timeout_ms` elapses
    /// (None means block forever). Returns `None` on timeout OR on shutdown
    /// drain. Callers can disambiguate via the scheduler's `len()` and the
    /// last-known shutdown state.
    #[pyo3(signature = (timeout_ms = None))]
    fn next_blocking<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Release the GIL while we wait so other Python threads aren't blocked
        // on a scheduler-side await. The Rust future itself never touches Py.
        let inner = Arc::clone(&self.inner);
        let result = py.allow_threads(|| {
            self.runtime.block_on(async move {
                match timeout_ms {
                    Some(ms) => tokio::time::timeout(Duration::from_millis(ms), inner.next())
                        .await
                        .ok()
                        .flatten(),
                    None => inner.next().await,
                }
            })
        });
        match result {
            Some(req) => {
                let dict = PyDict::new(py);
                dict.set_item("id", req.id.0)?;
                dict.set_item("priority", req.priority.as_str())?;
                dict.set_item("payload", json_to_py(py, &req.payload)?)?;
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Mark an id as cancelled. Raises `SchedulerError` if the id isn't known.
    fn cancel(&self, request_id: String) -> PyResult<()> {
        self.inner.cancel(&RequestId(request_id)).map_err(map_err)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The fixed capacity supplied at construction.
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Flip the shutdown flag. Pending `next_blocking` calls return `None`.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

/// Native PyO3 module entry point. Imported by the Python wrapper as
/// `repercep_scheduler._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScheduler>()?;
    m.add("SchedulerError", m.py().get_type::<SchedulerExc>())?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mk(id: &str, p: Priority) -> ScheduledRequest {
        ScheduledRequest {
            id: RequestId::new(id),
            priority: p,
            submitted_at: Instant::now(),
            payload: json!({}),
        }
    }

    #[tokio::test]
    async fn submit_then_next_roundtrip() {
        let s = Scheduler::new(4);
        s.submit(mk("a", Priority::Normal)).unwrap();
        let got = s.next().await.unwrap();
        assert_eq!(got.id, RequestId::new("a"));
        assert!(s.is_empty());
    }

    #[tokio::test]
    async fn priority_ordering_high_normal_low() {
        let s = Scheduler::new(16);
        s.submit(mk("l1", Priority::Low)).unwrap();
        s.submit(mk("n1", Priority::Normal)).unwrap();
        s.submit(mk("h1", Priority::High)).unwrap();
        s.submit(mk("n2", Priority::Normal)).unwrap();
        s.submit(mk("h2", Priority::High)).unwrap();
        s.submit(mk("l2", Priority::Low)).unwrap();

        // Drain — High FIFO, then Normal FIFO, then Low FIFO.
        let order: Vec<String> = futures_drain(&s, 6)
            .await
            .into_iter()
            .map(|r| r.id.0)
            .collect();
        assert_eq!(order, vec!["h1", "h2", "n1", "n2", "l1", "l2"]);
    }

    #[tokio::test]
    async fn fifo_within_priority() {
        let s = Scheduler::new(8);
        for i in 0..5 {
            s.submit(mk(&format!("r{i}"), Priority::Normal)).unwrap();
        }
        let order: Vec<String> = futures_drain(&s, 5)
            .await
            .into_iter()
            .map(|r| r.id.0)
            .collect();
        assert_eq!(order, vec!["r0", "r1", "r2", "r3", "r4"]);
    }

    #[tokio::test]
    async fn cancel_skips_at_next() {
        let s = Scheduler::new(4);
        s.submit(mk("a", Priority::Normal)).unwrap();
        s.submit(mk("b", Priority::Normal)).unwrap();
        s.submit(mk("c", Priority::Normal)).unwrap();
        s.cancel(&RequestId::new("b")).unwrap();
        // len() reflects the live count immediately, before next() runs.
        assert_eq!(s.len(), 2);
        let order: Vec<String> = futures_drain(&s, 2)
            .await
            .into_iter()
            .map(|r| r.id.0)
            .collect();
        assert_eq!(order, vec!["a", "c"]);
        assert!(s.is_empty());
    }

    #[test]
    fn cancel_unknown_id_returns_not_found() {
        let s = Scheduler::new(4);
        let err = s.cancel(&RequestId::new("nope")).unwrap_err();
        assert!(matches!(err, SchedulerError::NotFound(_)));
    }

    #[test]
    fn capacity_enforced_at_submit() {
        let s = Scheduler::new(2);
        s.submit(mk("a", Priority::Normal)).unwrap();
        s.submit(mk("b", Priority::High)).unwrap();
        let err = s.submit(mk("c", Priority::Low)).unwrap_err();
        assert!(matches!(err, SchedulerError::QueueFull { capacity: 2 }));
        assert_eq!(s.len(), 2);
        assert_eq!(s.capacity(), 2);
    }

    #[tokio::test]
    async fn capacity_recovers_after_next() {
        // Cancellation also frees capacity — a cancelled item still occupies
        // a slot until next() pops it. This documents the v0 contract.
        let s = Scheduler::new(2);
        s.submit(mk("a", Priority::Normal)).unwrap();
        s.submit(mk("b", Priority::Normal)).unwrap();
        s.next().await.unwrap();
        // Now there's a free slot.
        s.submit(mk("c", Priority::Normal)).unwrap();
        assert_eq!(s.len(), 2);
    }

    #[tokio::test]
    async fn shutdown_drains_remaining_then_returns_none() {
        let s = Scheduler::new(4);
        s.submit(mk("a", Priority::Normal)).unwrap();
        s.submit(mk("b", Priority::High)).unwrap();
        s.shutdown();
        // Submit after shutdown is rejected.
        let err = s.submit(mk("c", Priority::Low)).unwrap_err();
        assert!(matches!(err, SchedulerError::ShuttingDown));
        // But existing items drain.
        let r1 = s.next().await.unwrap();
        assert_eq!(r1.id.0, "b");
        let r2 = s.next().await.unwrap();
        assert_eq!(r2.id.0, "a");
        // And once empty, next() returns None promptly.
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn shutdown_wakes_pending_next() {
        let s = Arc::new(Scheduler::new(4));
        let s2 = Arc::clone(&s);
        let h = tokio::spawn(async move { s2.next().await });
        // Give the spawned task a moment to park on notify.
        tokio::task::yield_now().await;
        s.shutdown();
        let got = tokio::time::timeout(Duration::from_millis(500), h)
            .await
            .expect("next did not wake on shutdown")
            .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn cancel_after_shutdown_returns_shutting_down() {
        let s = Scheduler::new(2);
        s.submit(mk("a", Priority::Normal)).unwrap();
        s.shutdown();
        let err = s.cancel(&RequestId::new("a")).unwrap_err();
        assert!(matches!(err, SchedulerError::ShuttingDown));
    }

    #[test]
    fn len_and_capacity_reflect_state() {
        let s = Scheduler::new(3);
        assert_eq!(s.len(), 0);
        assert_eq!(s.capacity(), 3);
        s.submit(mk("a", Priority::Normal)).unwrap();
        assert_eq!(s.len(), 1);
        s.submit(mk("b", Priority::High)).unwrap();
        assert_eq!(s.len(), 2);
        s.cancel(&RequestId::new("a")).unwrap();
        assert_eq!(s.len(), 1);
    }

    #[tokio::test]
    async fn next_wakes_on_submit() {
        let s = Arc::new(Scheduler::new(4));
        let s2 = Arc::clone(&s);
        let h = tokio::spawn(async move { s2.next().await });
        tokio::task::yield_now().await;
        s.submit(mk("a", Priority::High)).unwrap();
        let got = tokio::time::timeout(Duration::from_millis(500), h)
            .await
            .expect("next did not wake on submit")
            .unwrap()
            .unwrap();
        assert_eq!(got.id.0, "a");
    }

    #[tokio::test]
    async fn cancel_then_resubmit_same_id_works() {
        // After a cancelled item is popped (and discarded by next()), its id
        // becomes available again. This is the contract callers downstream
        // can rely on for retry-after-cancel.
        let s = Scheduler::new(4);
        s.submit(mk("x", Priority::Normal)).unwrap();
        s.cancel(&RequestId::new("x")).unwrap();
        // Force the pop so the skip-set entry is cleared.
        // After cancel, len()==0 and the bucket still has the item; force a
        // drain by submitting another item and pulling both.
        s.submit(mk("y", Priority::Normal)).unwrap();
        let r = s.next().await.unwrap();
        assert_eq!(r.id.0, "y");
        // Now "x" is gone from known-set; resubmit succeeds.
        s.submit(mk("x", Priority::Normal)).unwrap();
        let r2 = s.next().await.unwrap();
        assert_eq!(r2.id.0, "x");
    }

    // Tiny helper — drain n items, panicking on early shutdown. Lives here
    // (not as a free function) to keep the test module self-contained.
    async fn futures_drain(s: &Scheduler, n: usize) -> Vec<ScheduledRequest> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(s.next().await.expect("unexpected shutdown during drain"));
        }
        out
    }
}
