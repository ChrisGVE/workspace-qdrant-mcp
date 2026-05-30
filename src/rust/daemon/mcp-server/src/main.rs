//! workspace-qdrant MCP server entry point.
//!
//! # Mode resolution
//!
//! The operating mode is selected by the `MCP_SERVER_MODE` environment
//! variable:
//!
//! | `MCP_SERVER_MODE`     | Transport                                   |
//! |-----------------------|---------------------------------------------|
//! | `"stdio"` (default)   | JSON-RPC over stdin/stdout                  |
//! | `"http"`              | Streamable HTTP (task 32, not yet wired)    |
//!
//! Unrecognised values log a warning and fall back to `stdio`.
//!
//! # Stdout purity
//!
//! In stdio mode, `stdout` carries **only** JSON-RPC frames.  Logging is
//! always directed to `stderr` (see [`mcp_server::observability::logging`]).
//! The binary is tested for stdout purity by `tests/stdout_purity.rs`.
//!
//! # Graceful shutdown
//!
//! tokio SIGINT / SIGTERM handlers trigger graceful shutdown; the transport
//! layer calls `cleanup_session` after the serve loop exits.

use mcp_server::grpc::client::DaemonClient;
use mcp_server::observability::logging::init_logging;
use mcp_server::observability::metrics_http::serve_metrics;
use mcp_server::server_types::{ServerMode, BUILD_NUMBER, SERVER_NAME, SERVER_VERSION_BASE};
use mcp_server::sqlite::StateManager;
use mcp_server::transport::http::serve_http;
use mcp_server::transport::stdio::{build_qdrant_client, serve_stdio};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Resolve operating mode from env (default: stdio).
    let mode = resolve_mode();

    // 2. Init logging — MUST come before any tracing calls, and MUST use stderr
    //    in stdio mode so stdout stays reserved for JSON-RPC frames.
    init_logging(mode);

    info!(
        server = SERVER_NAME,
        version = SERVER_VERSION_BASE,
        build = BUILD_NUMBER,
        "workspace-qdrant MCP server starting"
    );

    match mode {
        ServerMode::Stdio => run_stdio().await,
        ServerMode::Http => run_http().await,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stdio runner
// ─────────────────────────────────────────────────────────────────────────────

async fn run_stdio() -> anyhow::Result<()> {
    // Open SQLite (degraded-mode safe — read-only, no daemon required).
    let state = StateManager::open();

    // Build Qdrant read client from environment (QDRANT_URL / QDRANT_API_KEY).
    let qdrant = build_qdrant_client();

    // Connect daemon gRPC channel (lazy — no TCP handshake at construction).
    let daemon = DaemonClient::connect_default().unwrap_or_else(|e| {
        warn!(error = %e, "Failed to create DaemonClient; continuing in degraded mode");
        // Panic is safe here: connect_default only fails on a malformed
        // URI constant — which would be a programming error.
        panic!("Unexpected DaemonClient construction error: {e}")
    });

    // Fresh session — project detection and daemon registration happen inside
    // initialize_session, which is called by serve_stdio after the MCP
    // handshake (session init is deferred to transport).
    let session = mcp_server::server_types::SessionState::new();

    // Install OS signal handlers for graceful shutdown.
    //
    // tokio::signal::ctrl_c() covers SIGINT (Ctrl-C) on all platforms.
    // On Unix we also handle SIGTERM (e.g. `kill`, systemd stop, Docker stop).
    // When either signal fires we let the serve loop drain naturally by closing
    // the transport — the `serve_stdio` future resolves and calls cleanup.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown_tx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx)));

    let shutdown_tx_clone = std::sync::Arc::clone(&shutdown_tx);
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            warn!(error = %e, "ctrl_c signal handler error");
            return;
        }
        info!("Received SIGINT — initiating graceful shutdown");
        if let Some(tx) = shutdown_tx_clone.lock().await.take() {
            let _ = tx.send(());
        }
    });

    #[cfg(unix)]
    {
        let shutdown_tx_sigterm = std::sync::Arc::clone(&shutdown_tx);
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "SIGTERM signal handler setup failed");
                    return;
                }
            };
            sigterm.recv().await;
            info!("Received SIGTERM — initiating graceful shutdown");
            if let Some(tx) = shutdown_tx_sigterm.lock().await.take() {
                let _ = tx.send(());
            }
        });
    }

    // Run the serve loop; no heartbeat handle at this stage (heartbeat is
    // started post-handshake inside the session lifecycle).
    let serve_result = tokio::select! {
        result = serve_stdio(daemon, qdrant, state, session, None, None) => result,
        _ = async {
            let _ = shutdown_rx.await;
        } => {
            info!("Shutdown signal received — exiting");
            Ok(())
        }
    };

    serve_result
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP runner
// ─────────────────────────────────────────────────────────────────────────────

async fn run_http() -> anyhow::Result<()> {
    // Open SQLite (read-only, degraded-mode safe).
    let state = StateManager::open();

    // Build Qdrant read client.
    let qdrant = build_qdrant_client();

    // Connect daemon gRPC channel (lazy).
    let daemon = DaemonClient::connect_default().unwrap_or_else(|e| {
        warn!(error = %e, "Failed to create DaemonClient; continuing in degraded mode");
        panic!("Unexpected DaemonClient construction error: {e}")
    });

    let session = mcp_server::server_types::SessionState::new();

    // Build a CancellationToken that fires on SIGINT / SIGTERM.
    let shutdown_token = CancellationToken::new();

    {
        let ct = shutdown_token.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!(error = %e, "ctrl_c signal handler error");
                return;
            }
            info!("Received SIGINT — initiating graceful shutdown (HTTP)");
            ct.cancel();
        });
    }

    #[cfg(unix)]
    {
        let ct = shutdown_token.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "SIGTERM signal handler setup failed");
                    return;
                }
            };
            sigterm.recv().await;
            info!("Received SIGTERM — initiating graceful shutdown (HTTP)");
            ct.cancel();
        });
    }

    // Spawn the metrics HTTP server alongside the MCP server.
    // Non-fatal on bind failure (the warning is already logged inside serve_metrics).
    // Mirrors the TS `startMetricsServer()` call in `src/index.ts` (HTTP mode only).
    if let Err(e) = serve_metrics(shutdown_token.child_token()).await {
        warn!(error = %e, "Metrics server failed to start; continuing without metrics endpoint");
    }

    serve_http(daemon, qdrant, state, session, shutdown_token).await
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the server mode from the `MCP_SERVER_MODE` environment variable.
///
/// `"http"` → [`ServerMode::Http`]; anything else (including unset) → `Stdio`.
fn resolve_mode() -> ServerMode {
    match std::env::var("MCP_SERVER_MODE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "http" => ServerMode::Http,
        _ => ServerMode::Stdio,
    }
}
