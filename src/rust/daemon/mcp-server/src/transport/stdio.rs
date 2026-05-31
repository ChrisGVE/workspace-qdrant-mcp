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
/// * `daemon`              — gRPC client to the memexd daemon (may be disconnected).
/// * `qdrant`              — read-only Qdrant client.
/// * `state`               — SQLite state manager (may be in degraded mode).
/// * `session`             — fresh session state; the lifecycle runs on the
///   client's `initialize` request (see [`ToolsHandler::initialize`]).
/// * `health_state`        — optional shared health state from a running
///   [`HealthMonitorBuilder`](crate::observability::health_monitor::HealthMonitorBuilder).
///   When `None` a default optimistic (healthy) state is used.
/// * `rules_dup_threshold` — optional duplication threshold from loaded config,
///   mirrors `config.rules?.duplicationThreshold` from TS `server-factory.ts:52`.
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
    health_state: Option<SharedHealthState>,
    rules_dup_threshold: Option<f64>,
) -> anyhow::Result<()> {
    let health_state = health_state.unwrap_or_else(|| {
        use std::sync::RwLock;
        std::sync::Arc::new(RwLock::new(HealthState::initial()))
    });
    let handler = ToolsHandler::new_with_config(
        daemon,
        qdrant,
        state,
        session,
        health_state,
        rules_dup_threshold,
    );

    // Grab Arc handles for post-serve cleanup.
    let daemon_arc = handler.daemon();
    let session_arc = handler.session();
    let hb_arc = handler.hb_handle();

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
    // The heartbeat `AbortHandle` (if any) was stored by `ToolsHandler::initialize`.
    let hb_handle = hb_arc.lock().expect("hb_handle mutex poisoned").take();
    let mut session_guard = session_arc.lock().await;
    let mut daemon_guard = daemon_arc.lock().await;
    cleanup_session(&mut session_guard, &mut *daemon_guard, hb_handle).await;

    info!("MCP server shutdown complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Qdrant client builder
// ---------------------------------------------------------------------------

/// Build a `QdrantReadClient` from a loaded `ServerConfig`.
///
/// Uses `config.qdrant.url` and `config.qdrant.api_key`, which already
/// incorporate env-override precedence (QDRANT_URL / QDRANT_API_KEY).
/// Mirrors TS `src/index.ts:106`: `new QdrantClient({ url, apiKey })`.
pub fn build_qdrant_client_from_config(config: &crate::config::ServerConfig) -> QdrantReadClient {
    let api_key = config
        .qdrant
        .api_key
        .as_deref()
        .map(|s| SecretString::new(s.to_string().into_boxed_str()));
    QdrantReadClient::new(config.qdrant.url.clone(), api_key)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `build_qdrant_client_from_config` constructs a client
    /// without panicking, using the URL and optional API key from a config.
    #[test]
    fn build_qdrant_client_from_config_uses_config_url() {
        let mut config = crate::config::ServerConfig::default();
        config.qdrant.url = "http://localhost:6333".to_string();
        config.qdrant.api_key = None;
        // Must not panic.
        let _client = build_qdrant_client_from_config(&config);
    }

    /// Verify that `build_qdrant_client_from_config` passes through an API key.
    #[test]
    fn build_qdrant_client_from_config_with_api_key() {
        let mut config = crate::config::ServerConfig::default();
        config.qdrant.url = "http://remote-qdrant:6333".to_string();
        config.qdrant.api_key = Some("test-api-key".to_string());
        // Must not panic.
        let _client = build_qdrant_client_from_config(&config);
    }
}
