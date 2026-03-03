//! MoveCorrelator: correlates rename events to determine move type.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use super::types::{MoveCorrelatorConfig, MoveCorrelatorStats, PendingMove, RenameAction};

/// Correlates rename events to determine move type
pub struct MoveCorrelator {
    /// Pending MOVED_FROM events waiting for MOVED_TO
    pending_moves: HashMap<u64, PendingMove>,

    /// Path-based pending moves (when file ID not available)
    pending_by_path: HashMap<PathBuf, PendingMove>,

    /// Configuration
    config: MoveCorrelatorConfig,
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
            timestamp: std::time::Instant::now(),
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
