//! Priority Manager Module
//!
//! Manages project activation state based on MCP server lifecycle events.
//! Priority ordering is computed at dequeue time by JOINing `watch_folders.is_active`
//! and collection type — the stored `priority` column in `unified_queue` is NOT used
//! for dequeue ordering (see `queue_operations::dequeue_unified`).
//!
//! This module manages only `watch_folders.is_active` and `last_activity_at`:
//! - `register_session`: Sets is_active=1, updates last_activity_at
//! - `heartbeat`: Updates last_activity_at timestamp for active projects
//! - `unregister_session`: Sets is_active=0
//! - `set_priority`: Maps "high"/"normal" to is_active=1/0
//! - `cleanup_orphaned_sessions`: Detects stale active projects (>60s without heartbeat)
//!
//! ## Schema Compliance (WORKSPACE_QDRANT_MCP.md v1.6.7)
//!
//! This module uses only the `watch_folders` table for activity state.
//! Queue ordering is computed at dequeue time, not stored.

mod manager;
mod session_monitor;

pub use manager::PriorityManager;
pub use session_monitor::SessionMonitor;

use chrono::{DateTime, Utc};
use thiserror::Error;

/// Priority levels for the queue system — re-exported from wqm_common
pub use wqm_common::constants::priority;

/// Session monitoring configuration
#[derive(Debug, Clone)]
pub struct SessionMonitorConfig {
    /// Heartbeat timeout in seconds (default: 60)
    pub heartbeat_timeout_secs: u64,
    /// Check interval in seconds (default: 30)
    pub check_interval_secs: u64,
}

impl Default for SessionMonitorConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout_secs: 60,
            check_interval_secs: 30,
        }
    }
}

/// Priority management errors
#[derive(Error, Debug)]
pub enum PriorityError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Invalid priority value: {0}")]
    InvalidPriority(i32),

    #[error("Empty tenant_id or branch")]
    EmptyParameter,

    #[error("Project not found: {0}")]
    ProjectNotFound(String),

    #[error("Session monitor already running")]
    MonitorAlreadyRunning,

    #[error("Session monitor not running")]
    MonitorNotRunning,
}

/// Result type for priority operations
pub type PriorityResult<T> = Result<T, PriorityError>;

impl From<crate::lifecycle::WatchFolderLifecycleError> for PriorityError {
    fn from(err: crate::lifecycle::WatchFolderLifecycleError) -> Self {
        match err {
            crate::lifecycle::WatchFolderLifecycleError::Database(e) => Self::Database(e),
            crate::lifecycle::WatchFolderLifecycleError::NotFound(msg) => Self::ProjectNotFound(msg),
        }
    }
}

/// Session information for tracking active MCP server connections
///
/// Uses `watch_folders.is_active` for activity state per spec.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Watch ID (tenant identifier)
    pub watch_id: String,
    /// Tenant ID (project_id for projects)
    pub tenant_id: String,
    /// Whether this project is currently active (has active sessions)
    pub is_active: bool,
    /// Last heartbeat timestamp
    pub last_activity_at: Option<DateTime<Utc>>,
    /// Current priority level (derived from is_active)
    pub priority: String,
}

/// Result of orphaned session cleanup
#[derive(Debug, Clone)]
pub struct OrphanedSessionCleanup {
    /// Number of projects with orphaned sessions detected
    pub projects_affected: usize,
    /// Total sessions cleaned up (same as projects_affected with boolean model)
    pub sessions_cleaned: i32,
    /// Tenant IDs that were demoted
    pub demoted_projects: Vec<String>,
}

#[cfg(test)]
mod tests;
