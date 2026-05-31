//! Heartbeat loop for MCP sessions.
//!
//! Mirrors `startHeartbeat` / `sendHeartbeat` in
//! `src/typescript/mcp-server/src/session-lifecycle.ts` (lines 218-259).
//!
//! # Design
//! `start_heartbeat` spawns a tokio task that fires immediately and then every
//! [`HEARTBEAT_INTERVAL_MS`] milliseconds.  The task captures an
//! `Arc<tokio::sync::Mutex<SessionState>>` (so it can flip `daemon_connected`)
//! and a `HeartbeatFn` closure that performs the actual RPC call.
//!
//! The closure approach avoids holding a mutex across an `await` point and
//! keeps the heartbeat module independent of the full [`DaemonOps`] trait —
//! tests can inject a simple async closure.
//!
//! Returns a [`tokio::task::AbortHandle`]; the lifecycle module stores this
//! and calls `abort()` during [`super::lifecycle::cleanup_session`].
//!
//! # Fire-and-forget semantics
//! Heartbeat failures MUST NOT propagate (matching TS `sendHeartbeat`).
//! Any error simply sets `daemon_connected = false` on the shared state.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::AbortHandle;
use tracing::{debug, error};

use crate::server_types::{SessionState, HEARTBEAT_INTERVAL_MS};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Start the heartbeat loop.
///
/// Spawns a tokio task that:
/// 1. Calls `heartbeat_fn` immediately (matches TS `sendHeartbeatFn()` before
///    `setInterval`).
/// 2. Repeats every [`HEARTBEAT_INTERVAL_MS`] ms.
///
/// On each tick the current `project_id` and `daemon_connected` are read from
/// `state`; if disconnected or no project, the tick is a no-op.
///
/// Returns an [`AbortHandle`] the caller uses to stop the loop on cleanup.
///
/// # Type parameters
/// - `F`: factory closure `FnMut(String) -> Fut` — called with `project_id`.
/// - `Fut`: the future returned by `F`; resolves to `Result<bool, String>`
///   where `Ok(acknowledged)` is success and `Err(msg)` is a failure.
pub fn start_heartbeat<F, Fut>(state: Arc<Mutex<SessionState>>, heartbeat_fn: F) -> AbortHandle
where
    F: FnMut(String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<bool, String>> + Send + 'static,
{
    let interval = Duration::from_millis(HEARTBEAT_INTERVAL_MS);
    debug!(
        interval_secs = HEARTBEAT_INTERVAL_MS / 1000,
        "Heartbeat started"
    );

    let handle = tokio::spawn(heartbeat_loop(state, heartbeat_fn, interval));
    handle.abort_handle()
}

/// Inner loop — separated for testability with injectable interval.
pub(crate) async fn heartbeat_loop<F, Fut>(
    state: Arc<Mutex<SessionState>>,
    mut heartbeat_fn: F,
    interval: Duration,
) where
    F: FnMut(String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<bool, String>> + Send + 'static,
{
    // Immediate first beat (mirrors TS `sendHeartbeatFn()` before `setInterval`).
    tick_heartbeat(&state, &mut heartbeat_fn).await;

    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await; // consume the zero-delay first tick
    loop {
        ticker.tick().await;
        tick_heartbeat(&state, &mut heartbeat_fn).await;
    }
}

/// Execute one heartbeat tick.
///
/// Reads project_id + daemon_connected from state without holding the lock
/// across the await.  Writes `daemon_connected = false` on failure.
async fn tick_heartbeat<F, Fut>(state: &Arc<Mutex<SessionState>>, heartbeat_fn: &mut F)
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<bool, String>>,
{
    let (project_id, connected) = {
        let s = state.lock().await;
        (s.project_id.clone(), s.daemon_connected)
    };

    let project_id = match project_id {
        Some(id) if connected => id,
        _ => return,
    };

    match heartbeat_fn(project_id.clone()).await {
        Ok(acknowledged) => {
            debug!(project_id = %project_id, acknowledged, "heartbeat ok");
        }
        Err(e) => {
            error!(project_id = %project_id, error = %e, "Heartbeat failed");
            state.lock().await.daemon_connected = false;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "heartbeat_tests.rs"]
mod tests;
