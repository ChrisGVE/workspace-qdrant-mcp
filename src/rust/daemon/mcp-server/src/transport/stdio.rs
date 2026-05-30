//! Stdio (stdin/stdout) MCP transport.
//!
//! # Protocol
//!
//! JSON-RPC 2.0 frames are exchanged over `stdin` (incoming) and `stdout`
//! (outgoing) per the MCP specification for stdio servers.  `stderr` is
//! exclusively used for logs (see [`crate::observability::logging`]).
//!
//! # Serve loop
//!
//! [`serve_stdio`] uses `rmcp::serve_server` with the built-in
//! `rmcp::transport::io::stdio()` transport, which returns a
//! `(tokio::io::Stdin, tokio::io::Stdout)` pair.  The rmcp library adapts
//! the pair to its line-delimited JSON framing automatically via the
//! `transport-io` feature (enabled in `Cargo.toml`).
//!
//! # Shutdown
//!
//! After the `RunningService` loop exits (transport closed or cancelled),
//! [`cleanup_session`] is called to:
//! 1. Abort the background heartbeat task (if one was started).
//! 2. Deprioritize the project with the daemon.
//! 3. Mark the session as cleaned.

use tokio::task::AbortHandle;
use tracing::{debug, error, info, warn};

use rmcp::{serve_server, transport::io::stdio};
use secrecy::SecretString;

use crate::grpc::client::DaemonClient;
use crate::observability::health_monitor::{HealthState, SharedHealthState};
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::session::lifecycle::cleanup_session;
use crate::sqlite::StateManager;
use crate::tools::ToolsHandler;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Serve the MCP server over stdio until the client closes the connection.
///
/// # Arguments
///
/// * `daemon`       — gRPC client to the memexd daemon (may be disconnected).
/// * `qdrant`       — read-only Qdrant client.
/// * `state`        — SQLite state manager (may be in degraded mode).
/// * `session`      — pre-populated session state (from `initialize_session`).
/// * `hb_handle`    — optional heartbeat `AbortHandle` to cancel on shutdown.
/// * `health_state` — optional shared health state from a running
///   [`HealthMonitorBuilder`](crate::observability::health_monitor::HealthMonitorBuilder).
///   When `None` a default optimistic (healthy) state is used.
///
/// # Steps
///
/// 1. Build a [`ToolsHandler`] from the supplied dependencies.
/// 2. Start the rmcp server over `(stdin, stdout)`.
/// 3. Await the `RunningService` loop (blocks until connection is closed).
/// 4. Call [`cleanup_session`] — stop heartbeat, deprioritize, mark cleaned.
///
/// # Errors
///
/// Returns `Err` if the rmcp initialization handshake fails (e.g. the client
/// sent a malformed `initialize` request).  Transport errors after
/// initialization are logged and result in a clean exit.
pub async fn serve_stdio(
    daemon: DaemonClient,
    qdrant: QdrantReadClient,
    state: StateManager,
    session: SessionState,
    hb_handle: Option<AbortHandle>,
    health_state: Option<SharedHealthState>,
) -> anyhow::Result<()> {
    let health_state = health_state.unwrap_or_else(|| {
        use std::sync::RwLock;
        std::sync::Arc::new(RwLock::new(HealthState::initial()))
    });
    let handler = ToolsHandler::new(daemon, qdrant, state, session, health_state);

    // Grab Arc handles for post-serve cleanup.
    let daemon_arc = handler.daemon();
    let session_arc = handler.session();

    info!("MCP stdio transport: listening on stdin/stdout");

    // Serve over (stdin, stdout).  rmcp's `transport-io` feature adapts the
    // tokio AsyncRead/AsyncWrite pair to the line-delimited JSON framing.
    let running = serve_server(handler, stdio()).await.map_err(|e| {
        error!(error = %e, "MCP server initialization failed");
        anyhow::anyhow!("MCP server initialization failed: {e}")
    })?;

    debug!("MCP handshake complete — entering serve loop");

    // Block until the transport is closed or the server is cancelled.
    match running.waiting().await {
        Ok(reason) => {
            info!(reason = ?reason, "MCP serve loop exited");
        }
        Err(e) => {
            warn!(error = %e, "MCP serve loop task join error");
        }
    }

    // Graceful cleanup: stop heartbeat, deprioritize project with daemon.
    let mut session_guard = session_arc.lock().await;
    let mut daemon_guard = daemon_arc.lock().await;
    cleanup_session(&mut session_guard, &mut *daemon_guard, hb_handle).await;

    info!("MCP server shutdown complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Qdrant client builder
// ---------------------------------------------------------------------------

/// Build a `QdrantReadClient` from environment variables or defaults.
///
/// Reads `QDRANT_URL` (default: `http://localhost:6333`) and `QDRANT_API_KEY`.
/// Called from `main` before passing the client to `serve_stdio`.
pub fn build_qdrant_client() -> QdrantReadClient {
    let url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string());
    let api_key = std::env::var("QDRANT_API_KEY")
        .ok()
        .map(|s| SecretString::new(s.into_boxed_str()));
    QdrantReadClient::new(url, api_key)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `build_qdrant_client` uses the default URL when the
    /// environment variable is absent (exercising the fallback path).
    #[test]
    fn build_qdrant_client_uses_default_url() {
        // Remove env var to ensure default path is exercised.
        // We cannot assert the URL directly (it's hidden inside the inner Arc)
        // but the call must not panic.
        let _client = {
            let prev = std::env::var("QDRANT_URL").ok();
            // SAFETY: tests run in-process; unsetting env is not thread-safe in
            // general, but this module test runs serially and only reads env.
            unsafe { std::env::remove_var("QDRANT_URL") };
            let c = build_qdrant_client();
            if let Some(v) = prev {
                unsafe { std::env::set_var("QDRANT_URL", v) };
            }
            c
        };
    }
}
