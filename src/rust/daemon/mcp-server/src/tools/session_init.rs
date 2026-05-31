//! Session-initialization glue for the rmcp `initialize` handler.
//!
//! Extracted from `ToolsHandler::initialize` so it can be unit-tested without
//! constructing an rmcp `RequestContext` (which is impractical). The rmcp
//! handler is a thin wrapper that stashes peer info and calls
//! [`run_session_initialize`].

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::AbortHandle;

use crate::server_types::SessionState;
use crate::session::{default_detect_fn, initialize_session, start_heartbeat, DaemonOps};
use crate::sqlite::SharedStateManager;

/// Run the session lifecycle for an `initialize` request.
///
/// 1. Detect the project from `cwd` under a short SQLite lock that is dropped
///    before the daemon `.await` (so no std-mutex guard is held across await).
/// 2. `initialize_session`: daemon health → register_project → start heartbeat.
///    The heartbeat `AbortHandle` is stored in `hb_slot` for cleanup.
///
/// Generic over the daemon type `D` so tests can inject a mock; the rmcp handler
/// passes the concrete `DaemonClient`.
///
/// Lock order is daemon-then-session, which MUST match
/// [`crate::tools::ToolsHandler::call_tool`] to avoid an ABBA deadlock.
pub(crate) async fn run_session_initialize<D>(
    state: &Arc<SharedStateManager>,
    session: &Arc<Mutex<SessionState>>,
    daemon: &Arc<Mutex<D>>,
    hb_slot: &Arc<std::sync::Mutex<Option<AbortHandle>>>,
    cwd: &Path,
) where
    D: DaemonOps + Send + 'static,
{
    // Project detection: short synchronous SQLite lock, dropped here.
    let detected = {
        let guard = state.lock();
        default_detect_fn(cwd, &guard)
    };

    let mut daemon_guard = daemon.lock().await;
    let mut session_guard = session.lock().await;

    let hb_session = Arc::clone(session);
    let hb_daemon = Arc::clone(daemon);
    let hb_slot = Arc::clone(hb_slot);
    initialize_session(
        &mut session_guard,
        &mut *daemon_guard,
        cwd,
        detected,
        move || {
            let handle = start_heartbeat(hb_session, move |pid| {
                let d = Arc::clone(&hb_daemon);
                async move { DaemonOps::heartbeat(&mut *d.lock().await, &pid).await }
            });
            *hb_slot.lock().unwrap() = Some(handle);
        },
    )
    .await;
}

#[cfg(test)]
#[path = "session_init_tests.rs"]
mod tests;
