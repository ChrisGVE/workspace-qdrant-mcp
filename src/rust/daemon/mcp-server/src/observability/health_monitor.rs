//! Health monitor — mirrors `utils/health-monitor.ts`.
//!
//! Tracks daemon and Qdrant availability in the background, classifying the
//! combined state as `healthy` (both up) or `uncertain` (one or both down).
//! State is written on a 30-second timer (reusing [`HEARTBEAT_INTERVAL_MS`])
//! and via an explicit [`force_check`].
//!
//! # Parity notes
//! - Interval: `checkIntervalMs ?? 30000` (health-monitor.ts:67) — same value
//!   as [`HEARTBEAT_INTERVAL_MS`] (server-types.ts:11).
//! - Initial check fires before the interval timer (health-monitor.ts:95-100).
//! - Daemon `HEALTHY` **or** `DEGRADED` counts as available (health-monitor.ts:149-153).
//! - Classification: health-monitor.ts:174-208 → [`compute_state`].
//! - `augmentSearchResults` → [`augment_search_results`]: adds `health` key
//!   **only** when uncertain (health-monitor.ts:237-248).
//! - State transitions on the timer / `force_check` only — never edge-triggered
//!   by tool failures.
//!
//! # Probe injection
//! [`DaemonProbe`] and [`QdrantProbe`] traits allow hermetic testing without
//! live services.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::task::AbortHandle;
use tracing::{debug, warn};

use crate::server_types::HEARTBEAT_INTERVAL_MS;

// ─────────────────────────────────────────────────────────────────────────────
// Public types — mirror of health-monitor.ts interfaces
// ─────────────────────────────────────────────────────────────────────────────

/// Overall system health (health-monitor.ts:17).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Uncertain,
}

/// Reason for uncertain state (health-monitor.ts:18).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertainReason {
    DaemonUnavailable,
    QdrantUnavailable,
    BothUnavailable,
}

/// Current health snapshot (health-monitor.ts:20-27).
#[derive(Debug, Clone)]
pub struct HealthState {
    pub status: HealthStatus,
    pub daemon_available: bool,
    pub qdrant_available: bool,
    pub reason: Option<UncertainReason>,
    pub message: Option<String>,
}

impl HealthState {
    /// Optimistic initial state — mirrors health-monitor.ts:78-83.
    pub fn initial() -> Self {
        Self {
            status: HealthStatus::Healthy,
            daemon_available: true,
            qdrant_available: true,
            reason: None,
            message: None,
        }
    }
}

/// Metadata shape added to uncertain search responses (health-monitor.ts:29-33).
///
/// JSON shape:
/// ```json
/// {
///   "status": "uncertain",
///   "reason": "daemon_unavailable" | "qdrant_unavailable" | "both_unavailable",
///   "message": "<human-readable string>"
/// }
/// ```
/// `reason` and `message` are omitted when `None`
/// (`skip_serializing_if = "Option::is_none"`), matching TS
/// `exactOptionalPropertyTypes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetadata {
    pub status: HealthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UncertainReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Thread-safe handle to the live [`HealthState`].
pub type SharedHealthState = Arc<RwLock<HealthState>>;

// ─────────────────────────────────────────────────────────────────────────────
// Injectable probe traits
// ─────────────────────────────────────────────────────────────────────────────

/// Checks daemon availability.
///
/// Returns `true` when the daemon is reachable AND its status is
/// `HEALTHY` or `DEGRADED` (health-monitor.ts:149-153).
pub trait DaemonProbe: Send + Sync + 'static {
    fn check(&self) -> impl std::future::Future<Output = bool> + Send;
}

/// Checks Qdrant availability.
///
/// Returns `true` when Qdrant is reachable (mirrors `getCollections()` call in
/// health-monitor.ts:163-168).
pub trait QdrantProbe: Send + Sync + 'static {
    fn check(&self) -> impl std::future::Future<Output = bool> + Send;
}

// ─────────────────────────────────────────────────────────────────────────────
// HealthMonitorBuilder + StartedHealthMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for a health monitor.  Mirrors the TS `new HealthMonitor(config,
/// daemonClient)` followed by `monitor.start()`.
///
/// ```text
/// let monitor = HealthMonitorBuilder::new(daemon_probe, qdrant_probe).build();
/// // monitor.state() gives the Arc<RwLock<HealthState>>
/// // Drop monitor to stop background task.
/// ```
pub struct HealthMonitorBuilder<D: DaemonProbe, Q: QdrantProbe> {
    daemon_probe: D,
    qdrant_probe: Q,
    interval: Duration,
}

impl<D: DaemonProbe, Q: QdrantProbe> HealthMonitorBuilder<D, Q> {
    /// Create a builder with the default 30-second interval.
    pub fn new(daemon_probe: D, qdrant_probe: Q) -> Self {
        Self {
            daemon_probe,
            qdrant_probe,
            interval: Duration::from_millis(HEARTBEAT_INTERVAL_MS),
        }
    }

    /// Override the check interval (primarily for tests).
    #[cfg(test)]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Spawn the background task and return a [`StartedHealthMonitor`].
    pub fn build(self) -> StartedHealthMonitor {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let abort = spawn_health_loop(
            Arc::clone(&state),
            self.daemon_probe,
            self.qdrant_probe,
            self.interval,
        );
        StartedHealthMonitor {
            state,
            abort_handle: abort,
        }
    }
}

/// A running health monitor.  Stopping (or dropping) it cancels the background
/// task.
pub struct StartedHealthMonitor {
    state: SharedHealthState,
    abort_handle: AbortHandle,
}

impl StartedHealthMonitor {
    /// Return a cheap `Arc` clone of the shared state.
    pub fn state(&self) -> SharedHealthState {
        Arc::clone(&self.state)
    }

    /// Cancel the background task.
    pub fn stop(&self) {
        self.abort_handle.abort();
    }
}

impl Drop for StartedHealthMonitor {
    fn drop(&mut self) {
        self.abort_handle.abort();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// force_check — public helper for explicit immediate check
// ─────────────────────────────────────────────────────────────────────────────

/// Run an immediate health check and write the result to `state`.
///
/// Mirrors `forceCheck() / performHealthCheck()` in health-monitor.ts:253-255.
/// Useful for the first check on session start or for manual probes in tests.
pub async fn force_check<D: DaemonProbe, Q: QdrantProbe>(
    state: &SharedHealthState,
    daemon_probe: &D,
    qdrant_probe: &Q,
) -> HealthState {
    run_check(state, daemon_probe, qdrant_probe).await;
    state
        .read()
        .map(|g| g.clone())
        .unwrap_or_else(|_| HealthState::initial())
}

// ─────────────────────────────────────────────────────────────────────────────
// Background loop internals
// ─────────────────────────────────────────────────────────────────────────────

fn spawn_health_loop<D: DaemonProbe, Q: QdrantProbe>(
    state: SharedHealthState,
    daemon_probe: D,
    qdrant_probe: Q,
    interval: Duration,
) -> AbortHandle {
    let handle = tokio::spawn(health_loop(state, daemon_probe, qdrant_probe, interval));
    handle.abort_handle()
}

/// Background task: initial check, then repeat every `interval`.
pub(crate) async fn health_loop<D: DaemonProbe, Q: QdrantProbe>(
    state: SharedHealthState,
    daemon_probe: D,
    qdrant_probe: Q,
    interval: Duration,
) {
    // Immediate first check before the interval fires (health-monitor.ts:95).
    run_check(&state, &daemon_probe, &qdrant_probe).await;

    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await; // consume zero-delay first tick
    loop {
        ticker.tick().await;
        run_check(&state, &daemon_probe, &qdrant_probe).await;
    }
}

/// Execute one check and update shared state.
async fn run_check<D: DaemonProbe, Q: QdrantProbe>(
    state: &SharedHealthState,
    daemon_probe: &D,
    qdrant_probe: &Q,
) {
    let (daemon_ok, qdrant_ok) = tokio::join!(daemon_probe.check(), qdrant_probe.check());
    let new_state = compute_state(daemon_ok, qdrant_ok);
    debug!(
        daemon_available = daemon_ok,
        qdrant_available = qdrant_ok,
        status = ?new_state.status,
        "health check complete"
    );
    match state.write() {
        Ok(mut guard) => *guard = new_state,
        Err(_) => warn!("health monitor: RwLock poisoned, skipping state update"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// State classification — mirrors health-monitor.ts:174-208
// ─────────────────────────────────────────────────────────────────────────────

/// Classify the system state from individual component availability.
///
/// | daemon | qdrant | status    | reason               |
/// |--------|--------|-----------|----------------------|
/// | true   | true   | healthy   | —                    |
/// | false  | false  | uncertain | `both_unavailable`   |
/// | false  | true   | uncertain | `daemon_unavailable` |
/// | true   | false  | uncertain | `qdrant_unavailable` |
pub fn compute_state(daemon_available: bool, qdrant_available: bool) -> HealthState {
    if daemon_available && qdrant_available {
        return HealthState {
            status: HealthStatus::Healthy,
            daemon_available: true,
            qdrant_available: true,
            reason: None,
            message: None,
        };
    }

    let (reason, msg) = if !daemon_available && !qdrant_available {
        (
            UncertainReason::BothUnavailable,
            "Both daemon and Qdrant are unavailable. \
             Search results may be incomplete or unavailable.",
        )
    } else if !daemon_available {
        (
            UncertainReason::DaemonUnavailable,
            "Daemon is unavailable. \
             Search results may use cached data and new content cannot be indexed.",
        )
    } else {
        (
            UncertainReason::QdrantUnavailable,
            "Qdrant is unavailable. Search functionality is limited.",
        )
    };

    HealthState {
        status: HealthStatus::Uncertain,
        daemon_available,
        qdrant_available,
        reason: Some(reason),
        message: Some(msg.to_string()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search result augmentation — mirrors health-monitor.ts:213-248
// ─────────────────────────────────────────────────────────────────────────────

/// Extract health metadata from a state snapshot.
///
/// Returns `None` when healthy — no `health` key added (health-monitor.ts:213-216).
/// Returns `Some` when uncertain (health-monitor.ts:219-228).
pub fn get_health_metadata(state: &HealthState) -> Option<HealthMetadata> {
    if state.status == HealthStatus::Healthy {
        return None;
    }
    Some(HealthMetadata {
        status: state.status,
        reason: state.reason,
        message: state.message.clone(),
    })
}

/// Augment a search result JSON value with a `health` key when uncertain.
///
/// Mirrors `augmentSearchResults<T>` in health-monitor.ts:237-248.
///
/// - **Healthy**: returns `value` unchanged — no `health` key, byte-identical.
/// - **Uncertain**: inserts `"health": { "status": "uncertain", "reason": "…",
///   "message": "…" }` into the top-level object.
/// - Non-object `value`: returned unchanged (defensive, should not occur).
pub fn augment_search_results(
    state: &HealthState,
    mut value: serde_json::Value,
) -> serde_json::Value {
    if let Some(metadata) = get_health_metadata(state) {
        if let Some(obj) = value.as_object_mut() {
            if let Ok(meta_val) = serde_json::to_value(&metadata) {
                obj.insert("health".to_string(), meta_val);
            }
        }
    }
    value
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (sibling file)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "health_monitor_tests.rs"]
mod tests;
