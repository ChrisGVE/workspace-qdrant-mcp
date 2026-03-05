//! Types for move detection: errors, actions, configuration, and internal state.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error;

/// Errors that can occur during move detection
#[derive(Error, Debug)]
pub enum MoveDetectorError {
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Correlation timeout: {0}")]
    CorrelationTimeout(String),

    #[error("Database error: {0}")]
    Database(String),
}

/// Result of a rename operation analysis
#[derive(Debug, Clone, PartialEq)]
pub enum RenameAction {
    /// Move within the same filesystem (both old and new paths known)
    IntraFilesystemMove {
        old_path: PathBuf,
        new_path: PathBuf,
        is_directory: bool,
    },

    /// Move across filesystems (detected as delete - MOVED_FROM without MOVED_TO)
    CrossFilesystemMove {
        deleted_path: PathBuf,
        is_directory: bool,
    },

    /// Simple rename (file/folder renamed in place)
    SimpleRename {
        old_path: PathBuf,
        new_path: PathBuf,
        is_directory: bool,
    },

    /// Pending - waiting for MOVED_TO event
    Pending,
}

/// Tracks a pending move operation (MOVED_FROM without MOVED_TO yet)
#[derive(Debug, Clone)]
pub(super) struct PendingMove {
    /// Original path of the file/folder
    pub old_path: PathBuf,

    /// Whether this is a directory
    pub is_directory: bool,

    /// When the MOVED_FROM was received
    pub timestamp: Instant,
}

/// Configuration for the move correlator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveCorrelatorConfig {
    /// Timeout for correlating MOVED_FROM with MOVED_TO (seconds)
    pub correlation_timeout_secs: u64,

    /// Maximum number of pending moves to track
    pub max_pending_moves: usize,

    /// Whether to enable file ID correlation (platform-specific)
    pub enable_file_id_correlation: bool,
}

impl Default for MoveCorrelatorConfig {
    fn default() -> Self {
        Self {
            correlation_timeout_secs: 30,
            max_pending_moves: 10000,
            enable_file_id_correlation: true,
        }
    }
}

/// Statistics for the move correlator
#[derive(Debug, Clone, Default)]
pub struct MoveCorrelatorStats {
    pub pending_by_id: usize,
    pub pending_by_path: usize,
    pub max_capacity: usize,
}
