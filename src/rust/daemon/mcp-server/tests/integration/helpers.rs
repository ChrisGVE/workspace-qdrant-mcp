// Shared helpers for the integration test suite.
//
// Prerequisites:
//   - Running memexd daemon  (default gRPC: http://127.0.0.1:50051)
//   - Running Qdrant         (default HTTP: http://localhost:6333)
//   - Existing state.db      (WQM_DATABASE_PATH or XDG default)
//
// Runtime skip contract: helpers return None/false when services are absent.
// Tests do:  let Some(x) = helpers::probe_daemon().await else { return };
// This avoids #[ignore] entirely; tests are compiled out by default.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

/// gRPC endpoint for the daemon (env GRPC_HOST / GRPC_PORT).
pub fn daemon_endpoint() -> String {
    let host = std::env::var("GRPC_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("GRPC_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(50051);
    format!("http://{host}:{port}")
}

/// Qdrant URL (env QDRANT_URL).
pub fn qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string())
}

/// Optional Qdrant API key (env QDRANT_API_KEY).
pub fn qdrant_api_key() -> Option<String> {
    std::env::var("QDRANT_API_KEY").ok()
}

/// Resolve state.db path: WQM_DATABASE_PATH env var or XDG default.
pub fn state_db_path() -> Option<std::path::PathBuf> {
    if let Ok(v) = std::env::var("WQM_DATABASE_PATH") {
        let p = std::path::PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
        eprintln!("SKIP: WQM_DATABASE_PATH={} does not exist", p.display());
        return None;
    }
    match wqm_common::paths::get_database_path() {
        Ok(p) if p.exists() => Some(p),
        Ok(p) => {
            eprintln!("SKIP: default state.db at {} does not exist", p.display());
            None
        }
        Err(e) => {
            eprintln!("SKIP: cannot resolve database path: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Probe helpers
// ---------------------------------------------------------------------------

/// Health-probe the daemon.  Returns a live DaemonClient or None.
pub async fn probe_daemon() -> Option<mcp_server::grpc::client::DaemonClient> {
    let endpoint = daemon_endpoint();
    let mut client = match mcp_server::grpc::client::DaemonClient::new(&endpoint) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: cannot build DaemonClient for {endpoint}: {e}");
            return None;
        }
    };
    match tokio::time::timeout(Duration::from_secs(3), client.health()).await {
        Ok(Ok(_)) => Some(client),
        Ok(Err(e)) => {
            eprintln!("SKIP: daemon health failed at {endpoint}: {e}");
            None
        }
        Err(_) => {
            eprintln!("SKIP: daemon health timed out at {endpoint}");
            None
        }
    }
}

/// TCP-probe Qdrant.  Returns true when the port accepts connections.
pub async fn probe_qdrant() -> bool {
    let url = qdrant_url();
    let addr = url
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    match tokio::time::timeout(Duration::from_secs(3), tokio::net::TcpStream::connect(addr)).await {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            eprintln!("SKIP: Qdrant TCP probe to {addr} failed: {e}");
            false
        }
        Err(_) => {
            eprintln!("SKIP: Qdrant TCP probe to {addr} timed out");
            false
        }
    }
}

/// Open a read-only StateManager against the live state.db.
pub fn open_state_manager() -> Option<mcp_server::sqlite::StateManager> {
    let path = state_db_path()?;
    let mgr = mcp_server::sqlite::StateManager::open_at(&path);
    if mgr.is_connected() {
        Some(mgr)
    } else {
        eprintln!("SKIP: StateManager could not open {}", path.display());
        None
    }
}
