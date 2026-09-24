//! Repercep per-request router.
//!
//! Greenfield component introduced in ADR-0005 / Stage 3. Owns:
//!
//! - the per-request state machine
//!   (`Received → Scheduled → Generating → Streaming → Complete`, with
//!   `Cancelled` / `Failed` as terminal off-ramps),
//! - bounded per-request frame channels (`tokio::sync::mpsc`), and
//! - the integration seam to a scheduler via the [`SchedulerHandle`] trait
//!   (which is **not** a Cargo dep on `repercep-scheduler` — the trait is
//!   implemented on the Python side via duck typing, see [`PySchedulerHandle`]).
//!
//! ## State machine
//!
//! The legal transitions are:
//!
//! ```text
//! Received ──accept()──▶ Scheduled
//! Scheduled ──first push_frame──▶ Generating
//! Generating ──same step──▶ Streaming         (collapsed in v0; see below)
//! Streaming ──frame.is_final──▶ Complete
//! * (non-terminal) ──cancel()──▶ Cancelled
//! * (non-terminal) ──internal──▶ Failed       (reserved, no caller in v0)
//! ```
//!
//! ### Why both `Generating` and `Streaming` exist
//!
//! In v0 they are entered in the same step (the first `push_frame` for a
//! request transitions `Scheduled → Generating → Streaming` atomically) and
//! could collapse to one variant. We keep them split because the eventual
//! batched-execution mode wants to express "the engine is producing tokens
//! for this request, but the network hasn't started draining frames yet" as
//! distinct from "the network has begun consuming the frame channel." The
//! cost of the extra variant is one extra match arm; the cost of collapsing
//! and re-introducing later is a wire-compatibility hassle.
//!
//! ## Async at the FFI boundary
//!
//! `push_frame` and `FrameStream.__anext__` are exposed as true Python
//! `async def` via [`pyo3_async_runtimes::tokio`]. Synchronous methods
//! (`accept`, `cancel`, `state`, `shutdown`, `subscribe`) take the GIL,
//! call into the router's internal state (guarded by `parking_lot::Mutex`),
//! and return. The tokio multi-thread runtime is owned by
//! `pyo3_async_runtimes::tokio::get_runtime()`; we do not spawn our own.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{info, info_span, warn};

// ---------------------------------------------------------------------------
// Public Rust surface
// ---------------------------------------------------------------------------

/// Per-request lifecycle state.
///
/// See the module-level docs for the legal transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    /// Created but not yet handed to a scheduler.
    Received,
    /// Handed to the scheduler. Awaiting first frame.
    Scheduled,
    /// Engine has begun producing frames for this request.
    Generating,
    /// At least one frame has been pushed; the per-request channel is live.
    Streaming,
    /// Final frame was pushed; the channel is closed.
    Complete,
    /// Caller cancelled; channel dropped.
    Cancelled,
    /// Reserved for future. No caller in v0.
    Failed,
}

impl RequestState {
    /// Terminal states do not transition further.
    fn is_terminal(self) -> bool {
        matches!(
            self,
            RequestState::Complete | RequestState::Cancelled | RequestState::Failed
        )
    }

    /// Stringly representation used at the Python boundary.
    fn as_str(self) -> &'static str {
        match self {
            RequestState::Received => "Received",
            RequestState::Scheduled => "Scheduled",
            RequestState::Generating => "Generating",
            RequestState::Streaming => "Streaming",
            RequestState::Complete => "Complete",
            RequestState::Cancelled => "Cancelled",
            RequestState::Failed => "Failed",
        }
    }
}

/// Request priority. Matches the scheduler's three-bucket convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    /// Background / best-effort.
    Low,
    /// Default.
    Normal,
    /// Latency-sensitive.
    High,
}

impl Priority {
    fn parse(s: &str) -> Result<Self, RouterError> {
        match s {
            "Low" | "low" => Ok(Priority::Low),
            "Normal" | "normal" => Ok(Priority::Normal),
            "High" | "high" => Ok(Priority::High),
            other => Err(RouterError::InvalidPriority(other.to_string())),
        }
    }
}

/// Unique identifier for a request. Matches the scheduler's id type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub String);

/// A request as it enters the router.
///
/// Internal bookkeeping (received_at, current state, last frame index)
/// lives on the [`Router`], not on this struct — this struct is the
/// inbound DTO.
#[derive(Debug, Clone)]
pub struct RouterRequest {
    /// Caller-supplied id.
    pub id: RequestId,
    /// Priority bucket.
    pub priority: Priority,
    /// Opaque body the engine driver will consume.
    pub payload: serde_json::Value,
}

/// A single frame the engine driver hands to the router.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Which request this frame belongs to.
    pub request_id: RequestId,
    /// Monotonic per-request index. The router rejects out-of-order pushes.
    pub frame_index: u64,
    /// Opaque frame body (e.g. NDJSON line bytes, gRPC protobuf bytes).
    pub bytes: Vec<u8>,
    /// True for the final frame of a request — triggers `Streaming → Complete`.
    pub is_final: bool,
}

/// Errors produced by the router.
#[derive(Debug, Error)]
pub enum RouterError {
    /// The id is not known (or was reaped because the request reached a
    /// terminal state).
    #[error("unknown request: {0:?}")]
    UnknownRequest(RequestId),
    /// `shutdown()` has been called; new ops are rejected.
    #[error("router is shutting down")]
    ShuttingDown,
    /// The bounded per-request channel is full. Caller (engine driver) must
    /// back off — there is no in-router buffering policy beyond the channel.
    #[error("backpressure: queue depth {depth} for request {id:?}")]
    Backpressure {
        /// Which request hit backpressure.
        id: RequestId,
        /// The configured channel depth (constant — a hint to the caller).
        depth: usize,
    },
    /// An illegal state-machine transition was attempted (e.g. cancelling a
    /// `Complete` request, or pushing a frame after the final frame).
    #[error("invalid state transition: {from:?} -> {to:?}")]
    InvalidTransition {
        /// The current state.
        from: RequestState,
        /// The proposed target state.
        to: RequestState,
    },
    /// `push_frame` received a frame index that is not strictly greater than
    /// the last one observed for this request.
    #[error("out-of-order frame for {id:?}: got index {got}, expected > {last}")]
    OutOfOrderFrame {
        /// Which request.
        id: RequestId,
        /// The (rejected) index that was pushed.
        got: u64,
        /// The last accepted index for this request.
        last: u64,
    },
    /// An id was accepted twice without an intervening terminal transition.
    #[error("duplicate request id: {0:?}")]
    DuplicateRequest(RequestId),
    /// The Python wrapper passed a priority string the router doesn't know.
    #[error("invalid priority: {0}")]
    InvalidPriority(String),
    /// The scheduler handle (likely a Python object) failed.
    #[error("scheduler error: {0}")]
    Scheduler(String),
}

/// Trait the router uses to talk to a request scheduler.
///
/// The real impl lives in `repercep-scheduler`; in this crate we only define
/// the trait and use it generically. There is **no** Cargo dep on
/// `repercep-scheduler` — the trait is the integration seam, and the Python
/// surface satisfies it via [`PySchedulerHandle`] using duck typing.
pub trait SchedulerHandle: Send + Sync + 'static {
    /// Submit a request id at the given priority to the scheduler.
    fn submit(&self, id: &RequestId, priority: Priority) -> Result<(), RouterError>;
    /// Notify the scheduler that a request was cancelled.
    fn cancel(&self, id: &RequestId) -> Result<(), RouterError>;
}

// ---------------------------------------------------------------------------
// Router implementation
// ---------------------------------------------------------------------------

/// Per-request entry the router tracks internally.
struct Entry {
    state: RequestState,
    /// Sender half of the per-request frame channel. `Some` while the channel
    /// is live; consumed by `subscribe` so the receiver belongs to one caller
    /// at a time, and dropped on terminal transitions to close the channel.
    sender: Option<mpsc::Sender<Frame>>,
    /// Receiver, parked here until the first `subscribe()` call takes it.
    receiver: Option<mpsc::Receiver<Frame>>,
    /// Last frame index accepted by `push_frame`; `None` before the first frame.
    last_frame_index: Option<u64>,
    priority: Priority,
}

/// Inner shared state. Wrapped in `Arc` so [`FrameStream`] can hold a handle.
struct Inner {
    /// Per-id entries. `parking_lot::Mutex` for cheap short critical sections.
    requests: Mutex<HashMap<RequestId, Entry>>,
    /// Frozen on `shutdown()`. New ops fail with [`RouterError::ShuttingDown`].
    shutting_down: Mutex<bool>,
    /// Constant bound on per-request channel depth.
    queue_depth: usize,
}

impl Inner {
    fn ensure_open(&self) -> Result<(), RouterError> {
        if *self.shutting_down.lock() {
            Err(RouterError::ShuttingDown)
        } else {
            Ok(())
        }
    }
}

/// The router.
///
/// Generic over a [`SchedulerHandle`] so tests can supply a stub without
/// pulling in the real `repercep-scheduler` crate. The Python entry point
/// uses `Router<PySchedulerHandle>` and goes through the duck-typed Python
/// scheduler object.
pub struct Router<S: SchedulerHandle> {
    inner: Arc<Inner>,
    scheduler: Arc<S>,
}

impl<S: SchedulerHandle> Router<S> {
    /// Build a new router.
    ///
    /// `per_request_queue_depth` is the bound on each per-request
    /// `tokio::sync::mpsc::channel`. When full, `push_frame` returns
    /// [`RouterError::Backpressure`] — the caller (engine driver) must back
    /// off; the router does not buffer further.
    pub fn new(scheduler: S, per_request_queue_depth: usize) -> Self {
        let depth = per_request_queue_depth.max(1);
        Self {
            inner: Arc::new(Inner {
                requests: Mutex::new(HashMap::new()),
                shutting_down: Mutex::new(false),
                queue_depth: depth,
            }),
            scheduler: Arc::new(scheduler),
        }
    }

    /// Accept a new request: allocate its frame channel, register state as
    /// `Scheduled`, and call into the scheduler. If the scheduler rejects the
    /// submission the request is rolled back and no entry remains.
    pub fn accept(&self, req: RouterRequest) -> Result<(), RouterError> {
        self.inner.ensure_open()?;
        let span = info_span!("request", id = %req.id.0, action = "accept");
        let _enter = span.enter();

        let (tx, rx) = mpsc::channel::<Frame>(self.inner.queue_depth);
        {
            let mut map = self.inner.requests.lock();
            if map.contains_key(&req.id) {
                return Err(RouterError::DuplicateRequest(req.id.clone()));
            }
            map.insert(
                req.id.clone(),
                Entry {
                    state: RequestState::Scheduled,
                    sender: Some(tx),
                    receiver: Some(rx),
                    last_frame_index: None,
                    priority: req.priority,
                },
            );
        }

        // Best-effort submit; if it fails we roll back the registration so
        // the caller can retry or fail upward cleanly.
        if let Err(e) = self.scheduler.submit(&req.id, req.priority) {
            warn!(error = %e, "scheduler.submit failed; rolling back");
            self.inner.requests.lock().remove(&req.id);
            return Err(e);
        }
        info!(state = "Scheduled", priority = ?req.priority, "request accepted");
        Ok(())
    }

    /// Push a frame into the router. The frame is forwarded, in order, to
    /// the per-request channel.
    ///
    /// Returns:
    ///
    /// - [`RouterError::UnknownRequest`] if the id is unknown (the request
    ///   may have been cancelled and reaped),
    /// - [`RouterError::OutOfOrderFrame`] if `frame.frame_index` is not
    ///   strictly greater than the last accepted index for this id,
    /// - [`RouterError::Backpressure`] if the bounded channel is full,
    /// - [`RouterError::InvalidTransition`] if the request is already in a
    ///   terminal state.
    pub async fn push_frame(&self, frame: Frame) -> Result<(), RouterError> {
        self.inner.ensure_open()?;
        let span = info_span!("request", id = %frame.request_id.0, action = "push_frame");
        let _enter = span.enter();

        // Phase 1: validate, take a clone of the sender, advance state.
        let sender = {
            let mut map = self.inner.requests.lock();
            let entry = map
                .get_mut(&frame.request_id)
                .ok_or_else(|| RouterError::UnknownRequest(frame.request_id.clone()))?;

            if entry.state.is_terminal() {
                return Err(RouterError::InvalidTransition {
                    from: entry.state,
                    to: if frame.is_final {
                        RequestState::Complete
                    } else {
                        RequestState::Streaming
                    },
                });
            }

            if let Some(last) = entry.last_frame_index
                && frame.frame_index <= last
            {
                return Err(RouterError::OutOfOrderFrame {
                    id: frame.request_id.clone(),
                    got: frame.frame_index,
                    last,
                });
            }

            // First-frame transition: Scheduled → Generating → Streaming.
            if matches!(entry.state, RequestState::Scheduled) {
                entry.state = RequestState::Generating;
                info!(state = "Generating", "first frame: Scheduled -> Generating");
                entry.state = RequestState::Streaming;
                info!(state = "Streaming", "first frame: Generating -> Streaming");
            }

            entry.last_frame_index = Some(frame.frame_index);
            entry
                .sender
                .as_ref()
                .ok_or_else(|| RouterError::UnknownRequest(frame.request_id.clone()))?
                .clone()
        };

        // Phase 2: try to send without holding the lock. `try_send` so we can
        // surface backpressure synchronously rather than wait forever.
        match sender.try_send(frame.clone()) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(RouterError::Backpressure {
                    id: frame.request_id.clone(),
                    depth: self.inner.queue_depth,
                });
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(RouterError::UnknownRequest(frame.request_id.clone()));
            }
        }

        // Phase 3: if this was the final frame, transition Streaming → Complete
        // and drop the sender so the receiver sees end-of-stream.
        if frame.is_final {
            let mut map = self.inner.requests.lock();
            if let Some(entry) = map.get_mut(&frame.request_id) {
                entry.state = RequestState::Complete;
                entry.sender = None;
                info!(state = "Complete", "final frame: Streaming -> Complete");
            }
        }
        Ok(())
    }

    /// Take the receiver for a request's frame channel.
    ///
    /// Each request's receiver can be taken once. After that, further
    /// `subscribe` calls return [`RouterError::UnknownRequest`] (the contract
    /// is "one consumer per request").
    pub fn subscribe(&self, id: &RequestId) -> Result<mpsc::Receiver<Frame>, RouterError> {
        self.inner.ensure_open()?;
        let span = info_span!("request", id = %id.0, action = "subscribe");
        let _enter = span.enter();

        let mut map = self.inner.requests.lock();
        let entry = map
            .get_mut(id)
            .ok_or_else(|| RouterError::UnknownRequest(id.clone()))?;
        entry
            .receiver
            .take()
            .ok_or_else(|| RouterError::UnknownRequest(id.clone()))
    }

    /// Cancel a request. Any non-terminal state transitions to `Cancelled`;
    /// the channel is dropped so any active subscriber sees end-of-stream.
    pub fn cancel(&self, id: &RequestId) -> Result<(), RouterError> {
        self.inner.ensure_open()?;
        let span = info_span!("request", id = %id.0, action = "cancel");
        let _enter = span.enter();

        let priority_opt = {
            let mut map = self.inner.requests.lock();
            let entry = map
                .get_mut(id)
                .ok_or_else(|| RouterError::UnknownRequest(id.clone()))?;
            if entry.state.is_terminal() {
                return Err(RouterError::InvalidTransition {
                    from: entry.state,
                    to: RequestState::Cancelled,
                });
            }
            entry.state = RequestState::Cancelled;
            // Drop the sender so the receiver sees the channel close.
            entry.sender = None;
            entry.receiver = None;
            Some(entry.priority)
        };
        info!(state = "Cancelled", "request cancelled");

        // Notify the scheduler. Best-effort: a scheduler-side error is logged
        // but doesn't undo the cancellation — the router has already moved on.
        if priority_opt.is_some()
            && let Err(e) = self.scheduler.cancel(id)
        {
            warn!(error = %e, "scheduler.cancel reported an error; router state already Cancelled");
        }
        Ok(())
    }

    /// Look up the current state. Returns `None` if the id is unknown.
    pub fn state(&self, id: &RequestId) -> Option<RequestState> {
        let map = self.inner.requests.lock();
        map.get(id).map(|e| e.state)
    }

    /// Mark the router as shutting down. New ops will fail with
    /// [`RouterError::ShuttingDown`]. Existing receivers continue to drain.
    pub fn shutdown(&self) {
        info!("router shutdown requested");
        *self.inner.shutting_down.lock() = true;
    }
}

// ---------------------------------------------------------------------------
// Python-side SchedulerHandle (duck-typed)
// ---------------------------------------------------------------------------

/// A [`SchedulerHandle`] backed by a Python object that duck-types
/// `submit(id, priority)` and `cancel(id)`.
///
/// This is the bridge that lets the router run with the real
/// `repercep_scheduler.Scheduler` (sister Agent B's crate) without this crate
/// ever depending on it at the Cargo level. The Python object can be the
/// real scheduler, a stub for testing, or any other class with the right
/// duck-typed methods.
pub struct PySchedulerHandle {
    obj: Py<PyAny>,
}

impl PySchedulerHandle {
    /// Build from a Python object reference.
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }

    fn call(&self, method: &str, args: (String, Option<&'static str>)) -> Result<(), RouterError> {
        Python::with_gil(|py| {
            let bound = self.obj.bind(py);
            // submit takes (id, priority); cancel takes (id,). We always
            // pass the optional priority and let Python error if the method
            // doesn't accept it — but we keep that arm out of cancel by
            // calling with a single-arg tuple when priority is None.
            let res = match args.1 {
                Some(p) => bound.call_method1(method, (args.0, p)),
                None => bound.call_method1(method, (args.0,)),
            };
            res.map_err(|e| RouterError::Scheduler(format!("{method}: {e}")))?;
            Ok(())
        })
    }
}

impl SchedulerHandle for PySchedulerHandle {
    fn submit(&self, id: &RequestId, priority: Priority) -> Result<(), RouterError> {
        let p = match priority {
            Priority::Low => "Low",
            Priority::Normal => "Normal",
            Priority::High => "High",
        };
        self.call("submit", (id.0.clone(), Some(p)))
    }

    fn cancel(&self, id: &RequestId) -> Result<(), RouterError> {
        self.call("cancel", (id.0.clone(), None))
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

// Python exception type for router errors. Wraps `RouterError` as a
// `RuntimeError` subclass so callers can `except RouterError` cleanly.
// The Python-visible class is named `RouterError`; we use `PyRouterError`
// as the Rust handle to avoid shadowing the `RouterError` enum.
#[allow(missing_docs)]
mod py_exc {
    use pyo3::exceptions::PyRuntimeError;
    pyo3::create_exception!(_native, PyRouterError, PyRuntimeError);
}
use py_exc::PyRouterError;

fn map_err(e: RouterError) -> PyErr {
    PyRouterError::new_err(e.to_string())
}

/// Convert a [`Frame`] to a Python dict for delivery to async iterators.
fn frame_to_pydict<'py>(py: Python<'py>, frame: &Frame) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("request_id", &frame.request_id.0)?;
    dict.set_item("frame_index", frame.frame_index)?;
    dict.set_item("payload", PyBytes::new(py, &frame.bytes))?;
    dict.set_item("is_final", frame.is_final)?;
    Ok(dict)
}

/// Python-visible router.
#[pyclass(name = "Router", module = "repercep_router._native")]
pub struct PyRouter {
    inner: Arc<Router<PySchedulerHandle>>,
}

#[pymethods]
impl PyRouter {
    #[new]
    #[pyo3(signature = (scheduler, per_request_queue_depth = 64))]
    fn new(scheduler: Py<PyAny>, per_request_queue_depth: usize) -> Self {
        let router = Router::new(PySchedulerHandle::new(scheduler), per_request_queue_depth);
        Self {
            inner: Arc::new(router),
        }
    }

    /// Accept a new request. Transitions state to `Scheduled` and calls
    /// `scheduler.submit(id, priority)`.
    fn accept(&self, request_id: String, priority: &str, payload: Py<PyAny>) -> PyResult<()> {
        let prio = Priority::parse(priority).map_err(map_err)?;
        // Serialize the opaque payload via Python's json module so the Rust
        // side holds a `serde_json::Value`. Failure here surfaces as a
        // RouterError rather than a TypeError to keep the surface uniform.
        let payload_str: String = Python::with_gil(|py| -> PyResult<String> {
            let json = py.import("json")?;
            let s: String = json.call_method1("dumps", (payload,))?.extract()?;
            Ok(s)
        })?;
        let payload_value: serde_json::Value = serde_json::from_str(&payload_str).map_err(|e| {
            PyRouterError::new_err(format!("payload is not JSON-serializable: {e}"))
        })?;
        let req = RouterRequest {
            id: RequestId(request_id),
            priority: prio,
            payload: payload_value,
        };
        self.inner.accept(req).map_err(map_err)
    }

    /// Push a frame. Exposed as Python `async def` via `pyo3_async_runtimes`.
    fn push_frame<'py>(
        &self,
        py: Python<'py>,
        request_id: String,
        frame_index: u64,
        payload: Vec<u8>,
        is_final: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let router = Arc::clone(&self.inner);
        let frame = Frame {
            request_id: RequestId(request_id),
            frame_index,
            bytes: payload,
            is_final,
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            router.push_frame(frame).await.map_err(map_err)
        })
    }

    /// Subscribe to a request's frame stream. Returns a [`PyFrameStream`].
    fn subscribe(&self, request_id: String) -> PyResult<PyFrameStream> {
        let rx = self
            .inner
            .subscribe(&RequestId(request_id))
            .map_err(map_err)?;
        Ok(PyFrameStream {
            rx: Arc::new(tokio::sync::Mutex::new(Some(rx))),
        })
    }

    /// Cancel a request.
    fn cancel(&self, request_id: String) -> PyResult<()> {
        self.inner.cancel(&RequestId(request_id)).map_err(map_err)
    }

    /// Look up the current state. Returns the variant name as a string
    /// (`"Received"`, `"Scheduled"`, ...) or `None` if the id is unknown.
    fn state(&self, request_id: String) -> Option<&'static str> {
        self.inner
            .state(&RequestId(request_id))
            .map(RequestState::as_str)
    }

    /// Mark the router as shutting down.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

/// Async iterator over a request's frames. Each `__anext__` yields a dict
/// `{"request_id", "frame_index", "payload", "is_final"}` or raises
/// `StopAsyncIteration` when the channel closes (final frame or cancel).
#[pyclass(name = "FrameStream", module = "repercep_router._native")]
pub struct PyFrameStream {
    /// `Arc<Mutex<Option<…>>>` so the receiver can be `take()`n inside an
    /// async block and replaced. Tokio mutex so we can `.await` over it.
    rx: Arc<tokio::sync::Mutex<Option<mpsc::Receiver<Frame>>>>,
}

#[pymethods]
impl PyFrameStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = rx.lock().await;
            let frame_opt = match guard.as_mut() {
                Some(receiver) => receiver.recv().await,
                None => None,
            };
            match frame_opt {
                Some(frame) => Python::with_gil(|py| {
                    let dict = frame_to_pydict(py, &frame)?;
                    Ok::<Py<PyAny>, PyErr>(dict.into())
                }),
                None => {
                    // Channel closed; drop the receiver so further calls
                    // also raise StopAsyncIteration immediately.
                    *guard = None;
                    Err(PyStopAsyncIteration::new_err(()))
                }
            }
        })
    }
}

/// PyO3 module registration: `repercep_router._native`.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRouter>()?;
    m.add_class::<PyFrameStream>()?;
    m.add("RouterError", py.get_type::<PyRouterError>())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Stub scheduler that records every call. Used as the `SchedulerHandle`
    /// impl in every Rust unit test — no dep on `repercep-scheduler` needed.
    #[derive(Default)]
    struct StubScheduler {
        submits: AtomicUsize,
        cancels: AtomicUsize,
        fail_submit: bool,
    }

    impl StubScheduler {
        fn new() -> Self {
            Self::default()
        }
        fn failing() -> Self {
            Self {
                fail_submit: true,
                ..Self::default()
            }
        }
    }

    impl SchedulerHandle for StubScheduler {
        fn submit(&self, _id: &RequestId, _priority: Priority) -> Result<(), RouterError> {
            self.submits.fetch_add(1, Ordering::SeqCst);
            if self.fail_submit {
                Err(RouterError::Scheduler("stub forced failure".into()))
            } else {
                Ok(())
            }
        }
        fn cancel(&self, _id: &RequestId) -> Result<(), RouterError> {
            self.cancels.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn req(id: &str, prio: Priority) -> RouterRequest {
        RouterRequest {
            id: RequestId(id.into()),
            priority: prio,
            payload: serde_json::json!({}),
        }
    }

    fn frame(id: &str, idx: u64, fin: bool) -> Frame {
        Frame {
            request_id: RequestId(id.into()),
            frame_index: idx,
            bytes: vec![idx as u8],
            is_final: fin,
        }
    }

    #[test]
    fn accept_sets_state_to_scheduled() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        assert_eq!(
            r.state(&RequestId("a".into())),
            Some(RequestState::Scheduled)
        );
    }

    #[test]
    fn duplicate_accept_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("dup", Priority::Normal)).unwrap();
        let err = r.accept(req("dup", Priority::Normal)).unwrap_err();
        assert!(matches!(err, RouterError::DuplicateRequest(_)));
    }

    #[test]
    fn scheduler_failure_rolls_back_accept() {
        let r = Router::new(StubScheduler::failing(), 4);
        let err = r.accept(req("x", Priority::Normal)).unwrap_err();
        assert!(matches!(err, RouterError::Scheduler(_)));
        assert_eq!(r.state(&RequestId("x".into())), None);
    }

    #[tokio::test]
    async fn push_frames_in_order_then_subscribe_drains() {
        let r = Router::new(StubScheduler::new(), 8);
        r.accept(req("a", Priority::Normal)).unwrap();
        let mut rx = r.subscribe(&RequestId("a".into())).unwrap();
        r.push_frame(frame("a", 0, false)).await.unwrap();
        r.push_frame(frame("a", 1, false)).await.unwrap();
        r.push_frame(frame("a", 2, true)).await.unwrap();

        let f0 = rx.recv().await.unwrap();
        assert_eq!(f0.frame_index, 0);
        let f1 = rx.recv().await.unwrap();
        assert_eq!(f1.frame_index, 1);
        let f2 = rx.recv().await.unwrap();
        assert_eq!(f2.frame_index, 2);
        assert!(f2.is_final);
        assert!(
            rx.recv().await.is_none(),
            "channel closes after final frame"
        );

        assert_eq!(
            r.state(&RequestId("a".into())),
            Some(RequestState::Complete)
        );
    }

    #[tokio::test]
    async fn push_first_frame_transitions_to_streaming() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        r.push_frame(frame("a", 0, false)).await.unwrap();
        assert_eq!(
            r.state(&RequestId("a".into())),
            Some(RequestState::Streaming)
        );
    }

    #[tokio::test]
    async fn out_of_order_push_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        r.push_frame(frame("a", 0, false)).await.unwrap();
        r.push_frame(frame("a", 5, false)).await.unwrap();
        let err = r.push_frame(frame("a", 3, false)).await.unwrap_err();
        assert!(matches!(err, RouterError::OutOfOrderFrame { .. }));
        // Repeating the same index is also out of order.
        let err = r.push_frame(frame("a", 5, false)).await.unwrap_err();
        assert!(matches!(err, RouterError::OutOfOrderFrame { .. }));
    }

    #[tokio::test]
    async fn push_for_unknown_request_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        let err = r.push_frame(frame("ghost", 0, false)).await.unwrap_err();
        assert!(matches!(err, RouterError::UnknownRequest(_)));
    }

    #[tokio::test]
    async fn push_after_terminal_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        r.push_frame(frame("a", 0, true)).await.unwrap();
        // Now in Complete.
        let err = r.push_frame(frame("a", 1, false)).await.unwrap_err();
        assert!(matches!(err, RouterError::InvalidTransition { .. }));
    }

    #[tokio::test]
    async fn backpressure_when_channel_full() {
        let r = Router::new(StubScheduler::new(), 2);
        r.accept(req("a", Priority::Normal)).unwrap();
        // Don't subscribe → no one drains → channel fills.
        r.push_frame(frame("a", 0, false)).await.unwrap();
        r.push_frame(frame("a", 1, false)).await.unwrap();
        let err = r.push_frame(frame("a", 2, false)).await.unwrap_err();
        match err {
            RouterError::Backpressure { depth, .. } => assert_eq!(depth, 2),
            other => panic!("expected Backpressure, got {other:?}"),
        }
    }

    #[test]
    fn cancel_unknown_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        let err = r.cancel(&RequestId("nope".into())).unwrap_err();
        assert!(matches!(err, RouterError::UnknownRequest(_)));
    }

    #[tokio::test]
    async fn cancel_transitions_state_and_drops_channel() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        let mut rx = r.subscribe(&RequestId("a".into())).unwrap();
        r.push_frame(frame("a", 0, false)).await.unwrap();
        let _f = rx.recv().await.unwrap();
        r.cancel(&RequestId("a".into())).unwrap();
        assert_eq!(
            r.state(&RequestId("a".into())),
            Some(RequestState::Cancelled)
        );
        // Receiver should now see end-of-stream.
        assert!(rx.recv().await.is_none());
        // Subsequent push on a cancelled request: channel was dropped on
        // cancel → the send phase reports Closed which we surface as
        // UnknownRequest (the contract on cancel) — but the state is still
        // Cancelled in the map, so the validation arm fires first.
        let err = r.push_frame(frame("a", 1, false)).await.unwrap_err();
        assert!(matches!(err, RouterError::InvalidTransition { .. }));
    }

    #[test]
    fn cancel_after_terminal_errors() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        r.cancel(&RequestId("a".into())).unwrap();
        let err = r.cancel(&RequestId("a".into())).unwrap_err();
        assert!(matches!(err, RouterError::InvalidTransition { .. }));
    }

    #[test]
    fn shutdown_blocks_new_ops() {
        let r = Router::new(StubScheduler::new(), 4);
        r.shutdown();
        let err = r.accept(req("a", Priority::Normal)).unwrap_err();
        assert!(matches!(err, RouterError::ShuttingDown));
    }

    #[test]
    fn state_returns_none_for_unknown() {
        let r = Router::new(StubScheduler::new(), 4);
        assert!(r.state(&RequestId("ghost".into())).is_none());
    }

    #[tokio::test]
    async fn subscribe_can_be_taken_only_once() {
        let r = Router::new(StubScheduler::new(), 4);
        r.accept(req("a", Priority::Normal)).unwrap();
        let _rx = r.subscribe(&RequestId("a".into())).unwrap();
        let err = r.subscribe(&RequestId("a".into())).unwrap_err();
        assert!(matches!(err, RouterError::UnknownRequest(_)));
    }

    #[test]
    fn priority_parse_accepts_canonical_and_lowercase() {
        assert_eq!(Priority::parse("Low").unwrap(), Priority::Low);
        assert_eq!(Priority::parse("normal").unwrap(), Priority::Normal);
        assert_eq!(Priority::parse("High").unwrap(), Priority::High);
        assert!(matches!(
            Priority::parse("Critical").unwrap_err(),
            RouterError::InvalidPriority(_)
        ));
    }

    #[test]
    fn state_variant_strings_match_python_enum_names() {
        assert_eq!(RequestState::Received.as_str(), "Received");
        assert_eq!(RequestState::Scheduled.as_str(), "Scheduled");
        assert_eq!(RequestState::Generating.as_str(), "Generating");
        assert_eq!(RequestState::Streaming.as_str(), "Streaming");
        assert_eq!(RequestState::Complete.as_str(), "Complete");
        assert_eq!(RequestState::Cancelled.as_str(), "Cancelled");
        assert_eq!(RequestState::Failed.as_str(), "Failed");
    }
}
