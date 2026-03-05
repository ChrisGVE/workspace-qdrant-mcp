//! Enhanced file watcher using notify-debouncer-full.
//!
//! This module provides a file watcher that uses notify-debouncer-full to:
//! - Correlate rename events using FileIdMap
//! - Debounce events for efficient processing
//! - Handle cross-platform differences in rename event delivery

mod handle;
#[cfg(test)]
mod tests;
mod watcher;

pub use handle::WatcherHandle;
pub use watcher::EnhancedFileWatcher;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

use super::move_detector::MoveCorrelatorConfig;
use super::path_validator::PathValidatorConfig;

/// Errors that can occur with the enhanced file watcher
#[derive(Error, Debug)]
pub enum EnhancedWatcherError {
    #[error("Watcher error: {0}")]
    Watcher(#[from] notify::Error),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Path error: {0}")]
    Path(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration for the enhanced file watcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedWatcherConfig {
    /// Debounce delay for events
    pub debounce_delay_ms: u64,

    /// Whether to use file ID cache for rename correlation
    pub enable_file_id_cache: bool,

    /// Move correlator configuration
    pub move_correlator: MoveCorrelatorConfig,

    /// Path validator configuration
    pub path_validator: PathValidatorConfig,

    /// Tick interval for periodic tasks (ms)
    pub tick_interval_ms: u64,
}

impl Default for EnhancedWatcherConfig {
    fn default() -> Self {
        Self {
            debounce_delay_ms: 1000,
            enable_file_id_cache: true,
            move_correlator: MoveCorrelatorConfig::default(),
            path_validator: PathValidatorConfig::default(),
            tick_interval_ms: 5000,
        }
    }
}

/// Events emitted by the enhanced file watcher
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// A file was created
    Created { path: PathBuf, is_directory: bool },

    /// A file was modified
    Modified { path: PathBuf, is_directory: bool },

    /// A file was deleted
    Deleted { path: PathBuf, is_directory: bool },

    /// A file was renamed/moved within the same filesystem
    Renamed {
        old_path: PathBuf,
        new_path: PathBuf,
        is_directory: bool,
    },

    /// A file/folder was moved across filesystems (detected as delete)
    CrossFilesystemMove {
        deleted_path: PathBuf,
        is_directory: bool,
    },

    /// Root watch folder was renamed/moved
    RootRenamed {
        old_path: PathBuf,
        new_path: PathBuf,
        tenant_id: String,
    },

    /// Watch error occurred
    Error {
        path: Option<PathBuf>,
        message: String,
    },
}

/// Tracks watched paths and their associated tenant IDs
#[derive(Clone)]
pub(super) struct WatchEntry {
    pub(super) tenant_id: String,
    pub(super) recursive: bool,
}

/// Statistics for the enhanced watcher
#[derive(Debug, Clone, Default)]
pub struct EnhancedWatcherStats {
    pub watched_paths: usize,
    pub move_correlator_pending: usize,
    pub path_validator_pending_orphans: usize,
}
