//! CLI-side connection helpers over the shared [`wqm_client::DaemonClient`].
//!
//! The shared client connects **lazily** (no TCP handshake at construction) and
//! adds retry/timeout — the right default for the MCP server's degraded mode.
//! The CLI, however, wants the previous *eager* semantics: surface
//! "daemon not running" up front rather than on the first RPC. We get that by
//! constructing the lazy client and then issuing a single `health()` probe.
//!
//! Address resolution (`WQM_DAEMON_ADDR` > active cli-config profile > default)
//! is owned by `wqm_client` (`DaemonClient::connect_default`), which uses the
//! same `wqm_common::cli_profiles` chain the CLI previously resolved inline.

use anyhow::{Context, Result};

pub use wqm_client::DaemonClient;

/// Connect to the daemon at the resolved default address and verify liveness.
///
/// Constructs the lazy [`DaemonClient`] (resolving `WQM_DAEMON_ADDR` > profile >
/// `http://127.0.0.1:50051`) then probes `health()` so an unreachable daemon
/// fails here instead of on the first business RPC — preserving the previous
/// eager-`connect()` behaviour.
///
/// # Errors
/// Returns an error if the address is invalid or the daemon is unreachable.
pub async fn connect_default() -> Result<DaemonClient> {
    let mut client = DaemonClient::connect_default().context("Invalid daemon address")?;
    client
        .health()
        .await
        .context("Failed to connect to memexd daemon. Is it running?")?;
    Ok(client)
}

/// Check that the daemon is running and return a connected client.
///
/// Returns a clear, actionable error message if the daemon is not available.
/// Use this before any operation that requires the daemon.
///
/// # Errors
/// Returns an error if the daemon is not running or not reachable.
pub async fn ensure_daemon_available() -> Result<DaemonClient> {
    connect_default().await.map_err(|e| {
        anyhow::anyhow!(
            "Daemon not running. Start memexd to execute this command.\n\
             Hint: wqm service start\n\
             Cause: {}",
            e
        )
    })
}
