//! Move detection and rename correlation for file watching.
//!
//! This module provides functionality to correlate rename events and detect
//! folder moves within the filesystem. It handles:
//! - Intra-filesystem moves (MOVED_FROM + MOVED_TO correlation)
//! - Cross-filesystem moves (detected as delete when MOVED_TO times out)
//! - Root folder moves (MOVE_SELF or RENAME events)

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
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
struct PendingMove {
    /// Original path of the file/folder
    old_path: PathBuf,

    /// Whether this is a directory
    is_directory: bool,

    /// When the MOVED_FROM was received
    timestamp: Instant,

    /// File ID if available (for correlation)
    file_id: Option<u64>,
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

/// Correlates rename events to determine move type
pub struct MoveCorrelator {
    /// Pending MOVED_FROM events waiting for MOVED_TO
    pending_moves: HashMap<u64, PendingMove>,

    /// Path-based pending moves (when file ID not available)
    pending_by_path: HashMap<PathBuf, PendingMove>,

    /// Configuration
    config: MoveCorrelatorConfig,

    /// Counter for generating synthetic IDs when file ID not available
    synthetic_id_counter: u64,
}

impl MoveCorrelator {
    /// Create a new move correlator with default configuration
    pub fn new() -> Self {
        Self::with_config(MoveCorrelatorConfig::default())
    }

    /// Create a new move correlator with custom configuration
    pub fn with_config(config: MoveCorrelatorConfig) -> Self {
        Self {
            pending_moves: HashMap::with_capacity(1000),
            pending_by_path: HashMap::with_capacity(1000),
            config,
            synthetic_id_counter: 0,
        }
    }

    /// Handle a MOVED_FROM (rename from) event
    ///
    /// Returns Pending to indicate we're waiting for the MOVED_TO
    pub fn handle_moved_from(
        &mut self,
        path: PathBuf,
        is_directory: bool,
        file_id: Option<u64>,
    ) -> RenameAction {
        // Clean up expired pending moves first
        self.cleanup_expired();

        // Enforce capacity limit
        if self.pending_moves.len() >= self.config.max_pending_moves {
            tracing::warn!(
                "Pending moves at capacity ({}), dropping oldest",
                self.config.max_pending_moves
            );
            self.drop_oldest_pending();
        }

        let pending = PendingMove {
            old_path: path.clone(),
            is_directory,
            timestamp: Instant::now(),
            file_id,
        };

        // Store by file ID if available, otherwise by path
        if let Some(id) = file_id {
            self.pending_moves.insert(id, pending);
        } else {
            self.pending_by_path.insert(path, pending);
        }

        RenameAction::Pending
    }

    /// Handle a MOVED_TO (rename to) event
    ///
    /// Attempts to correlate with a pending MOVED_FROM
    pub fn handle_moved_to(
        &mut self,
        new_path: PathBuf,
        is_directory: bool,
        file_id: Option<u64>,
    ) -> RenameAction {
        let timeout = Duration::from_secs(self.config.correlation_timeout_secs);

        // Try to find matching MOVED_FROM by file ID first
        if let Some(id) = file_id {
            if let Some(pending) = self.pending_moves.remove(&id) {
                if pending.timestamp.elapsed() < timeout {
                    // Check if it's a simple rename (same parent) or a move
                    let old_parent = pending.old_path.parent();
                    let new_parent = new_path.parent();

                    if old_parent == new_parent {
                        return RenameAction::SimpleRename {
                            old_path: pending.old_path,
                            new_path,
                            is_directory,
                        };
                    } else {
                        return RenameAction::IntraFilesystemMove {
                            old_path: pending.old_path,
                            new_path,
                            is_directory,
                        };
                    }
                }
            }
        }

        // If no file ID match, try parent directory heuristic for directories
        // (a directory moved_to often arrives shortly after the parent's moved_from)
        if is_directory {
            // Look for any pending move in the same parent or where the new path
            // could be a destination of a pending move
            let mut matched_path = None;
            for (path, pending) in &self.pending_by_path {
                if pending.timestamp.elapsed() < timeout && pending.is_directory {
                    // Simple heuristic: same filename in different location
                    if path.file_name() == new_path.file_name() {
                        matched_path = Some(path.clone());
                        break;
                    }
                }
            }

            if let Some(path) = matched_path {
                if let Some(pending) = self.pending_by_path.remove(&path) {
                    let old_parent = pending.old_path.parent();
                    let new_parent = new_path.parent();

                    if old_parent == new_parent {
                        return RenameAction::SimpleRename {
                            old_path: pending.old_path,
                            new_path,
                            is_directory,
                        };
                    } else {
                        return RenameAction::IntraFilesystemMove {
                            old_path: pending.old_path,
                            new_path,
                            is_directory,
                        };
                    }
                }
            }
        }

        // No correlation found - this might be a create event misidentified
        // or a cross-filesystem move where we only see the destination
        tracing::debug!(
            "MOVED_TO without matching MOVED_FROM for {:?}",
            new_path
        );

        // Treat as a new file/directory creation
        RenameAction::IntraFilesystemMove {
            old_path: PathBuf::new(), // Unknown old path
            new_path,
            is_directory,
        }
    }

    /// Handle a rename event (combined from/to)
    ///
    /// Used when the filesystem provides both paths in a single event
    pub fn handle_rename_event(
        &mut self,
        old_path: PathBuf,
        new_path: PathBuf,
        is_directory: bool,
    ) -> RenameAction {
        let old_parent = old_path.parent();
        let new_parent = new_path.parent();

        if old_parent == new_parent {
            RenameAction::SimpleRename {
                old_path,
                new_path,
                is_directory,
            }
        } else {
            RenameAction::IntraFilesystemMove {
                old_path,
                new_path,
                is_directory,
            }
        }
    }

    /// Get all expired pending moves (cross-filesystem moves)
    ///
    /// Returns paths that have timed out waiting for MOVED_TO
    pub fn get_expired_moves(&mut self) -> Vec<RenameAction> {
        let timeout = Duration::from_secs(self.config.correlation_timeout_secs);
        let mut expired = Vec::new();

        // Check file ID-based pending moves
        let expired_ids: Vec<u64> = self.pending_moves
            .iter()
            .filter(|(_, pending)| pending.timestamp.elapsed() >= timeout)
            .map(|(id, _)| *id)
            .collect();

        for id in expired_ids {
            if let Some(pending) = self.pending_moves.remove(&id) {
                expired.push(RenameAction::CrossFilesystemMove {
                    deleted_path: pending.old_path,
                    is_directory: pending.is_directory,
                });
            }
        }

        // Check path-based pending moves
        let expired_paths: Vec<PathBuf> = self.pending_by_path
            .iter()
            .filter(|(_, pending)| pending.timestamp.elapsed() >= timeout)
            .map(|(path, _)| path.clone())
            .collect();

        for path in expired_paths {
            if let Some(pending) = self.pending_by_path.remove(&path) {
                expired.push(RenameAction::CrossFilesystemMove {
                    deleted_path: pending.old_path,
                    is_directory: pending.is_directory,
                });
            }
        }

        expired
    }

    /// Clean up expired pending moves
    fn cleanup_expired(&mut self) {
        let timeout = Duration::from_secs(self.config.correlation_timeout_secs);

        self.pending_moves.retain(|_, pending| {
            pending.timestamp.elapsed() < timeout
        });

        self.pending_by_path.retain(|_, pending| {
            pending.timestamp.elapsed() < timeout
        });
    }

    /// Drop the oldest pending move to make room for new ones
    fn drop_oldest_pending(&mut self) {
        // Find and remove the oldest entry from pending_moves
        if let Some(oldest_id) = self.pending_moves
            .iter()
            .min_by_key(|(_, pending)| pending.timestamp)
            .map(|(id, _)| *id)
        {
            self.pending_moves.remove(&oldest_id);
            return;
        }

        // If no pending_moves, try pending_by_path
        if let Some(oldest_path) = self.pending_by_path
            .iter()
            .min_by_key(|(_, pending)| pending.timestamp)
            .map(|(path, _)| path.clone())
        {
            self.pending_by_path.remove(&oldest_path);
        }
    }

    /// Get statistics about pending moves
    pub fn stats(&self) -> MoveCorrelatorStats {
        MoveCorrelatorStats {
            pending_by_id: self.pending_moves.len(),
            pending_by_path: self.pending_by_path.len(),
            max_capacity: self.config.max_pending_moves,
        }
    }

    /// Clear all pending moves
    pub fn clear(&mut self) {
        self.pending_moves.clear();
        self.pending_by_path.clear();
    }
}

impl Default for MoveCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the move correlator
#[derive(Debug, Clone, Default)]
pub struct MoveCorrelatorStats {
    pub pending_by_id: usize,
    pub pending_by_path: usize,
    pub max_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_simple_rename_same_directory() {
        let mut correlator = MoveCorrelator::new();

        let old_path = PathBuf::from("/project/old_name.txt");
        let new_path = PathBuf::from("/project/new_name.txt");

        // Simulate MOVED_FROM
        let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
        assert_eq!(result, RenameAction::Pending);

        // Simulate MOVED_TO with same file ID
        let result = correlator.handle_moved_to(new_path.clone(), false, Some(12345));

        assert!(matches!(result, RenameAction::SimpleRename { .. }));
        if let RenameAction::SimpleRename { old_path: op, new_path: np, is_directory } = result {
            assert_eq!(op, old_path);
            assert_eq!(np, new_path);
            assert!(!is_directory);
        }
    }

    #[test]
    fn test_intra_filesystem_move() {
        let mut correlator = MoveCorrelator::new();

        let old_path = PathBuf::from("/project/src/file.txt");
        let new_path = PathBuf::from("/project/dest/file.txt");

        // Simulate MOVED_FROM
        let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
        assert_eq!(result, RenameAction::Pending);

        // Simulate MOVED_TO with same file ID
        let result = correlator.handle_moved_to(new_path.clone(), false, Some(12345));

        assert!(matches!(result, RenameAction::IntraFilesystemMove { .. }));
        if let RenameAction::IntraFilesystemMove { old_path: op, new_path: np, is_directory } = result {
            assert_eq!(op, old_path);
            assert_eq!(np, new_path);
            assert!(!is_directory);
        }
    }

    #[test]
    fn test_directory_rename() {
        let mut correlator = MoveCorrelator::new();

        let old_path = PathBuf::from("/project/old_folder");
        let new_path = PathBuf::from("/project/new_folder");

        // Simulate MOVED_FROM for directory
        let result = correlator.handle_moved_from(old_path.clone(), true, Some(12345));
        assert_eq!(result, RenameAction::Pending);

        // Simulate MOVED_TO with same file ID
        let result = correlator.handle_moved_to(new_path.clone(), true, Some(12345));

        assert!(matches!(result, RenameAction::SimpleRename { .. }));
        if let RenameAction::SimpleRename { old_path: op, new_path: np, is_directory } = result {
            assert_eq!(op, old_path);
            assert_eq!(np, new_path);
            assert!(is_directory);
        }
    }

    #[test]
    fn test_combined_rename_event() {
        let mut correlator = MoveCorrelator::new();

        let old_path = PathBuf::from("/project/old.txt");
        let new_path = PathBuf::from("/project/new.txt");

        let result = correlator.handle_rename_event(old_path.clone(), new_path.clone(), false);

        assert!(matches!(result, RenameAction::SimpleRename { .. }));
    }

    #[test]
    fn test_expired_moves_become_cross_filesystem() {
        let config = MoveCorrelatorConfig {
            correlation_timeout_secs: 0, // Immediate timeout for testing
            ..Default::default()
        };
        let mut correlator = MoveCorrelator::with_config(config);

        let old_path = PathBuf::from("/project/moved_file.txt");

        // Simulate MOVED_FROM
        let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
        assert_eq!(result, RenameAction::Pending);

        // Wait for timeout
        sleep(Duration::from_millis(10));

        // Get expired moves
        let expired = correlator.get_expired_moves();

        assert_eq!(expired.len(), 1);
        assert!(matches!(expired[0], RenameAction::CrossFilesystemMove { .. }));
        if let RenameAction::CrossFilesystemMove { deleted_path, is_directory } = &expired[0] {
            assert_eq!(deleted_path, &old_path);
            assert!(!is_directory);
        }
    }

    #[test]
    fn test_stats() {
        let mut correlator = MoveCorrelator::new();

        correlator.handle_moved_from(PathBuf::from("/a"), false, Some(1));
        correlator.handle_moved_from(PathBuf::from("/b"), false, Some(2));
        correlator.handle_moved_from(PathBuf::from("/c"), false, None);

        let stats = correlator.stats();
        assert_eq!(stats.pending_by_id, 2);
        assert_eq!(stats.pending_by_path, 1);
    }

    #[test]
    fn test_clear() {
        let mut correlator = MoveCorrelator::new();

        correlator.handle_moved_from(PathBuf::from("/a"), false, Some(1));
        correlator.handle_moved_from(PathBuf::from("/b"), false, None);

        correlator.clear();

        let stats = correlator.stats();
        assert_eq!(stats.pending_by_id, 0);
        assert_eq!(stats.pending_by_path, 0);
    }
}
