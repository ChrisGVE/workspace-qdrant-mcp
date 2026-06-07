//! Session-initialization glue for the rmcp `initialize` handler.
//!
//! Extracted from `ToolsHandler::initialize` so it can be unit-tested without
//! constructing an rmcp `RequestContext` (which is impractical). The rmcp
//! handler is a thin wrapper that stashes peer info and calls
//! [`run_session_initialize`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::AbortHandle;
use tracing::{debug, info};

use crate::server_types::SessionState;
use crate::session::{
    default_detect_fn, initialize_session, register_project, start_heartbeat, DaemonOps,
};
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

/// Re-bind the session's project to a client-reported workspace root (#97).
///
/// In stdio mode the server's process cwd is the *client launch* cwd, which
/// need not match the conversation's working directory — initial detection in
/// [`run_session_initialize`] can therefore miss or resolve the wrong project.
/// Once the client's `roots/list` answer arrives, this re-runs detection
/// anchored at the reported root and, when it finds a different project,
/// applies it to the session and registers it with the daemon.
///
/// Conservative: an already-resolved identical project is left untouched, and
/// a root that resolves to NO project never clears an existing binding.
///
/// Lock order is daemon-then-session, matching `call_tool` (ABBA guard).
pub(crate) async fn rebind_session_project<D>(
    state: &Arc<SharedStateManager>,
    session: &Arc<Mutex<SessionState>>,
    daemon: &Arc<Mutex<D>>,
    client_root: PathBuf,
) where
    D: DaemonOps + Send + 'static,
{
    // Project detection: short synchronous SQLite lock, dropped here.
    let detected = {
        let guard = state.lock();
        default_detect_fn(&client_root, &guard)
    };

    let mut daemon_guard = daemon.lock().await;
    let mut session_guard = session.lock().await;

    session_guard.client_cwd = Some(client_root.clone());

    // Only a root that resolves to a REGISTERED project (project_id present)
    // justifies a rebind; a bare path-only detection must not displace or
    // re-register an existing binding.
    let Some(info) = detected.filter(|i| i.project_id.is_some()) else {
        debug!(
            root = %client_root.display(),
            "Client root resolves to no registered project; keeping existing session binding"
        );
        return;
    };

    if session_guard.project_id == info.project_id {
        debug!(root = %client_root.display(), "Client root matches session project; no rebind");
        return;
    }

    info!(
        root = %client_root.display(),
        project_path = %info.project_path.display(),
        "Rebinding session project to client root (#97)"
    );
    session_guard.project_path = Some(info.project_path.clone());
    if let Some(pid) = info.project_id.clone() {
        session_guard.project_id = Some(pid);
    }
    session_guard.current_branch = Some(info.branch.clone());

    if session_guard.daemon_connected {
        register_project(&mut session_guard, &mut *daemon_guard).await;
    }
}

/// Convert a `file://` URI (as sent in MCP `roots/list` results) to a path.
///
/// Handles the `file://` scheme, an optional empty authority, and percent-
/// encoded bytes (e.g. `%20` for spaces). Returns `None` for non-file URIs.
pub(crate) fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // Strip a non-empty authority (e.g. "file://localhost/x") down to the path.
    let path = match rest.find('/') {
        Some(0) => rest,           // "file:///abs/path" — empty authority
        Some(idx) => &rest[idx..], // "file://host/abs/path"
        None => return None,       // no path component
    };
    Some(PathBuf::from(percent_decode(path)))
}

/// Minimal percent-decoder for file-URI paths (no external dependency).
/// Invalid escapes are passed through verbatim.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let (Some(h), Some(l)) = (
                bytes.get(i + 1).and_then(|b| (*b as char).to_digit(16)),
                bytes.get(i + 2).and_then(|b| (*b as char).to_digit(16)),
            ) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
#[path = "session_init_tests.rs"]
mod tests;
