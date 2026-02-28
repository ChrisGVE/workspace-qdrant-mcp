//! Branch lifecycle management for monitoring branch changes in repositories.

mod detector;
#[cfg(test)]
mod tests;

pub use detector::BranchLifecycleDetector;

use serde::{Deserialize, Serialize};

use super::types::GitResult;

/// Branch lifecycle event types
///
/// These events are emitted when branches are created, deleted, renamed,
/// or when the default branch changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchEvent {
    /// A new branch was created
    Created {
        /// Name of the created branch
        branch: String,
        /// Commit hash the branch points to
        commit_hash: Option<String>,
    },
    /// A branch was deleted
    Deleted {
        /// Name of the deleted branch
        branch: String,
    },
    /// A branch was renamed
    Renamed {
        /// Old branch name
        old_name: String,
        /// New branch name
        new_name: String,
    },
    /// The default branch changed
    DefaultChanged {
        /// Previous default branch
        old_default: String,
        /// New default branch
        new_default: String,
    },
    /// Branch was switched to (HEAD changed)
    Switched {
        /// Previous branch
        from_branch: Option<String>,
        /// New branch
        to_branch: String,
    },
}

impl BranchEvent {
    /// Get the primary branch name involved in this event
    pub fn branch_name(&self) -> &str {
        match self {
            BranchEvent::Created { branch, .. } => branch,
            BranchEvent::Deleted { branch } => branch,
            BranchEvent::Renamed { new_name, .. } => new_name,
            BranchEvent::DefaultChanged { new_default, .. } => new_default,
            BranchEvent::Switched { to_branch, .. } => to_branch,
        }
    }

    /// Check if this event affects a specific branch
    pub fn affects_branch(&self, branch: &str) -> bool {
        match self {
            BranchEvent::Created { branch: b, .. } => b == branch,
            BranchEvent::Deleted { branch: b } => b == branch,
            BranchEvent::Renamed { old_name, new_name } => old_name == branch || new_name == branch,
            BranchEvent::DefaultChanged { old_default, new_default } => {
                old_default == branch || new_default == branch
            }
            BranchEvent::Switched { from_branch, to_branch } => {
                from_branch.as_deref() == Some(branch) || to_branch == branch
            }
        }
    }
}

/// Configuration for branch lifecycle detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchLifecycleConfig {
    /// Enable branch lifecycle tracking
    pub enabled: bool,
    /// Auto-delete branch documents when branch is deleted
    pub auto_delete_on_branch_delete: bool,
    /// Scan interval for detecting branch changes (seconds)
    pub scan_interval_seconds: u64,
    /// Rename correlation timeout (milliseconds)
    pub rename_correlation_timeout_ms: u64,
}

impl Default for BranchLifecycleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_delete_on_branch_delete: true,
            scan_interval_seconds: 5,
            rename_correlation_timeout_ms: 500,
        }
    }
}

/// Statistics about branch lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchLifecycleStats {
    /// Number of tracked branches
    pub tracked_branches: usize,
    /// Number of pending delete events (waiting for rename correlation)
    pub pending_deletes: usize,
    /// Current default branch
    pub default_branch: Option<String>,
}

/// Handler for branch lifecycle events that integrates with Qdrant
#[async_trait::async_trait]
pub trait BranchEventHandler: Send + Sync {
    /// Handle a branch creation event
    async fn handle_branch_created(
        &self,
        project_id: &str,
        branch: &str,
        commit_hash: Option<&str>,
    ) -> GitResult<()>;

    /// Handle a branch deletion event
    async fn handle_branch_deleted(&self, project_id: &str, branch: &str) -> GitResult<()>;

    /// Handle a branch rename event
    async fn handle_branch_renamed(
        &self,
        project_id: &str,
        old_branch: &str,
        new_branch: &str,
    ) -> GitResult<()>;

    /// Handle a default branch change event
    async fn handle_default_changed(
        &self,
        project_id: &str,
        old_default: &str,
        new_default: &str,
    ) -> GitResult<()>;
}

/// SQL schemas for branch lifecycle tracking
pub mod branch_schema {
    /// Add default_branch column to watch_folders
    pub const ALTER_ADD_DEFAULT_BRANCH: &str = r#"
        ALTER TABLE watch_folders ADD COLUMN default_branch TEXT
    "#;
}
