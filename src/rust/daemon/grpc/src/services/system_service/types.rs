//! Types for the SystemService gRPC implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::proto::ServerState;

/// Tracks the status of a server component (MCP server, CLI, etc.)
#[derive(Debug, Clone)]
pub struct ServerStatusEntry {
    /// Current state (UP or DOWN)
    pub state: ServerState,
    /// Project name (if project-scoped)
    pub project_name: Option<String>,
    /// Project root path
    pub project_root: Option<String>,
    /// Timestamp of last status update
    pub updated_at: SystemTime,
}

/// Thread-safe store for server status entries, keyed by component identifier
pub type ServerStatusStore = Arc<RwLock<HashMap<String, ServerStatusEntry>>>;
