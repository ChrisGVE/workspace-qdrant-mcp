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

use mcp_server::config::{load_config, ServerConfig};
use mcp_server::grpc::client::DaemonClient;
use mcp_server::observability::logging::init_logging;
use mcp_server::observability::metrics_http::serve_metrics;
use mcp_server::server_types::{ServerMode, BUILD_NUMBER, SERVER_NAME, SERVER_VERSION_BASE};
use mcp_server::sqlite::StateManager;
use mcp_server::transport::http::serve_http;
use mcp_server::transport::stdio::{build_qdrant_client_from_config, serve_stdio};
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

/// Construct the daemon gRPC client from config, degrading gracefully.
///
/// A malformed configured endpoint must NOT crash the server — TS catches the
/// connect failure and continues disconnected (session-lifecycle.ts:63-74).
/// On an invalid config endpoint we log and fall back to the default localhost
/// client (lazy-connect, so an unreachable daemon is handled by degraded mode).
fn build_daemon_client(config: &ServerConfig) -> anyhow::Result<DaemonClient> {
    match DaemonClient::from_config(config) {
        Ok(c) => Ok(c),
        Err(e) => {
            warn!(error = %e, "Invalid daemon endpoint in config; falling back to default endpoint");
            DaemonClient::connect_default()
                .map_err(|e2| anyhow::anyhow!("daemon client construction failed: {e2}"))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stdio runner
// ─────────────────────────────────────────────────────────────────────────────

async fn run_stdio() -> anyhow::Result<()> {
    // Load server config (file → env overrides → tilde expand).
    // Mirrors TS `loadConfig()` in `src/index.ts:106`.
    let config = load_config().unwrap_or_else(|e| {
        warn!(error = %e, "Config load failed; falling back to defaults");
        mcp_server::config::ServerConfig::default()
    });

    // Extract rules duplication threshold from config before config is moved.
    // Mirrors TS `config.rules?.duplicationThreshold` in `server-factory.ts:52`.
    let rules_dup_threshold = config.rules.as_ref().and_then(|r| r.duplication_threshold);

    // Open SQLite at the config-resolved path (degraded-mode safe — read-only).
    let state = StateManager::open_at(&config.database.path);

    // Build Qdrant read client from config (URL + API key already resolved).
    let qdrant = build_qdrant_client_from_config(&config);

    // Connect daemon gRPC channel using config host/port (lazy — no TCP handshake
    // at construction).  Mirrors TS `DaemonClient::from_config` pattern.
    let daemon = build_daemon_client(&config)?;

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
        result = serve_stdio(daemon, qdrant, state, session, None, None, rules_dup_threshold) => result,
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
    // Load server config (file → env overrides → tilde expand).
    // Mirrors TS `loadConfig()` in `src/index.ts:106`.
    let config = load_config().unwrap_or_else(|e| {
        warn!(error = %e, "Config load failed; falling back to defaults");
        mcp_server::config::ServerConfig::default()
    });

    // Extract rules duplication threshold from config.
    // Mirrors TS `config.rules?.duplicationThreshold` in `server-factory.ts:52`.
    let rules_dup_threshold = config.rules.as_ref().and_then(|r| r.duplication_threshold);

    // Open SQLite at the config-resolved path (degraded-mode safe — read-only).
    let state = StateManager::open_at(&config.database.path);

    // Build Qdrant read client from config (URL + API key already resolved).
    let qdrant = build_qdrant_client_from_config(&config);

    // Connect daemon gRPC channel using config host/port (lazy).
    let daemon = build_daemon_client(&config)?;

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

    serve_http(
        daemon,
        qdrant,
        state,
        session,
        shutdown_token,
        rules_dup_threshold,
    )
    .await
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
