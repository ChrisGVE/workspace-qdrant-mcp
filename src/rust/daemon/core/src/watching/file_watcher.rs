//! Enhanced file watcher using notify-debouncer-full.
//!
//! This module provides a file watcher that uses notify-debouncer-full to:
//! - Correlate rename events using FileIdMap
//! - Debounce events for efficient processing
//! - Handle cross-platform differences in rename event delivery

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use notify::{EventKind, RecursiveMode, RecommendedWatcher};
use notify::event::{CreateKind, ModifyKind, RemoveKind, RenameMode};
use notify_debouncer_full::{
    new_debouncer,
    DebounceEventResult,
    DebouncedEvent,
    Debouncer,
    RecommendedCache,
};
use tokio::sync::{mpsc, RwLock, Mutex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::move_detector::{MoveCorrelator, MoveCorrelatorConfig, RenameAction};
use super::path_validator::{PathValidator, PathValidatorConfig};

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
    Created {
        path: PathBuf,
        is_directory: bool,
    },

    /// A file was modified
    Modified {
        path: PathBuf,
        is_directory: bool,
    },

    /// A file was deleted
    Deleted {
        path: PathBuf,
        is_directory: bool,
    },

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
struct WatchEntry {
    tenant_id: String,
    recursive: bool,
}

/// Enhanced file watcher with rename correlation
pub struct EnhancedFileWatcher {
    /// Configuration
    config: EnhancedWatcherConfig,

    /// Move correlator for rename tracking
    move_correlator: Arc<Mutex<MoveCorrelator>>,

    /// Path validator for orphan detection
    path_validator: Arc<PathValidator>,

    /// Channel for sending watch events
    event_sender: mpsc::Sender<WatchEvent>,

    /// Watched paths and their tenant IDs
    watch_entries: Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,

    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl EnhancedFileWatcher {
    /// Create a new enhanced file watcher
    pub fn new(
        config: EnhancedWatcherConfig,
        event_sender: mpsc::Sender<WatchEvent>,
    ) -> Self {
        let move_correlator = MoveCorrelator::with_config(config.move_correlator.clone());
        let path_validator = PathValidator::with_config(config.path_validator.clone());

        Self {
            config,
            move_correlator: Arc::new(Mutex::new(move_correlator)),
            path_validator: Arc::new(path_validator),
            event_sender,
            watch_entries: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the file watcher
    ///
    /// This spawns the watcher thread and returns a handle for control
    pub async fn start(
        self: Arc<Self>,
    ) -> Result<WatcherHandle, EnhancedWatcherError> {
        // Create the notify-debouncer-full debouncer
        let debounce_delay = Duration::from_millis(self.config.debounce_delay_ms);

        // Clone Arc references for the callback
        let event_sender = self.event_sender.clone();
        let move_correlator = self.move_correlator.clone();
        let watch_entries = self.watch_entries.clone();
        let path_validator = self.path_validator.clone();

        // Create debouncer with FileIdMap for rename correlation
        let (tx, mut rx) = mpsc::channel::<Vec<DebouncedEvent>>(1000);

        let debouncer = new_debouncer(
            debounce_delay,
            None, // Use default tick rate
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) => {
                        // Send events to processing channel (blocking)
                        let _ = tx.blocking_send(events);
                    }
                    Err(errors) => {
                        for error in errors {
                            tracing::error!("Debouncer error: {:?}", error);
                        }
                    }
                }
            },
        )?;

        // Mark as running
        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let running_flag = self.running.clone();
        let tick_interval = Duration::from_millis(self.config.tick_interval_ms);

        // Spawn event processing task
        let process_task = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(tick_interval);

            loop {
                tokio::select! {
                    // Process incoming debounced events
                    Some(events) = rx.recv() => {
                        Self::process_events(
                            events,
                            &event_sender,
                            &move_correlator,
                            &watch_entries,
                            &path_validator,
                        ).await;
                    }

                    // Periodic tick for maintenance tasks
                    _ = ticker.tick() => {
                        // Check for expired moves (cross-filesystem)
                        let mut correlator = move_correlator.lock().await;
                        let expired = correlator.get_expired_moves();
                        drop(correlator);

                        for action in expired {
                            if let RenameAction::CrossFilesystemMove { deleted_path, is_directory } = action {
                                let _ = event_sender.send(WatchEvent::CrossFilesystemMove {
                                    deleted_path,
                                    is_directory,
                                }).await;
                            }
                        }

                        // Check if we should run path validation
                        if path_validator.is_validation_due().await {
                            tracing::debug!("Path validation due, will be triggered by daemon");
                            // Note: Actual validation is triggered by the daemon
                            // which has access to the database
                        }
                    }

                    // Check if we should stop
                    else => {
                        let running = running_flag.read().await;
                        if !*running {
                            break;
                        }
                    }
                }
            }

            tracing::info!("Enhanced file watcher stopped");
        });

        Ok(WatcherHandle {
            debouncer: Arc::new(Mutex::new(debouncer)),
            running: self.running.clone(),
            watch_entries: self.watch_entries.clone(),
            _process_task: process_task,
        })
    }

    /// Process debounced events
    async fn process_events(
        events: Vec<DebouncedEvent>,
        event_sender: &mpsc::Sender<WatchEvent>,
        move_correlator: &Arc<Mutex<MoveCorrelator>>,
        watch_entries: &Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        path_validator: &Arc<PathValidator>,
    ) {
        for debounced in events {
            let event = debounced.event;
            let paths = &event.paths;

            match event.kind {
                EventKind::Create(kind) => {
                    let is_directory = matches!(kind, CreateKind::Folder);
                    for path in paths {
                        let _ = event_sender.send(WatchEvent::Created {
                            path: path.clone(),
                            is_directory,
                        }).await;
                    }
                }

                EventKind::Modify(kind) => {
                    // Ignore metadata-only changes
                    if matches!(kind, ModifyKind::Metadata(_)) {
                        continue;
                    }

                    let is_directory = paths.first()
                        .map(|p| p.is_dir())
                        .unwrap_or(false);

                    // Check for rename events
                    if let ModifyKind::Name(rename_mode) = kind {
                        Self::handle_rename_event(
                            rename_mode,
                            paths,
                            is_directory,
                            event_sender,
                            move_correlator,
                            watch_entries,
                            path_validator,
                        ).await;
                    } else {
                        for path in paths {
                            let _ = event_sender.send(WatchEvent::Modified {
                                path: path.clone(),
                                is_directory,
                            }).await;
                        }
                    }
                }

                EventKind::Remove(kind) => {
                    let is_directory = matches!(kind, RemoveKind::Folder);
                    for path in paths {
                        let _ = event_sender.send(WatchEvent::Deleted {
                            path: path.clone(),
                            is_directory,
                        }).await;
                    }
                }

                EventKind::Access(_) => {
                    // Ignore access events
                }

                EventKind::Other | EventKind::Any => {
                    // Log but ignore
                    tracing::trace!("Unhandled event kind: {:?}", event.kind);
                }
            }
        }
    }

    /// Handle rename events
    async fn handle_rename_event(
        rename_mode: RenameMode,
        paths: &[PathBuf],
        is_directory: bool,
        event_sender: &mpsc::Sender<WatchEvent>,
        move_correlator: &Arc<Mutex<MoveCorrelator>>,
        watch_entries: &Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        path_validator: &Arc<PathValidator>,
    ) {
        match rename_mode {
            RenameMode::From => {
                // MOVED_FROM - store for correlation
                if let Some(path) = paths.first() {
                    let mut correlator = move_correlator.lock().await;
                    // Note: file ID would come from debouncer if available
                    correlator.handle_moved_from(path.clone(), is_directory, None);

                    // Reset path validator timer on folder operations
                    if is_directory {
                        path_validator.reset_timer().await;
                    }
                }
            }

            RenameMode::To => {
                // MOVED_TO - try to correlate with pending MOVED_FROM
                if let Some(path) = paths.first() {
                    let mut correlator = move_correlator.lock().await;
                    let action = correlator.handle_moved_to(path.clone(), is_directory, None);
                    drop(correlator);

                    // Emit appropriate event based on correlation result
                    match action {
                        RenameAction::SimpleRename { old_path, new_path, is_directory } |
                        RenameAction::IntraFilesystemMove { old_path, new_path, is_directory } => {
                            // Check if this is a root folder rename
                            let entries = watch_entries.read().await;
                            if let Some(entry) = entries.get(&old_path) {
                                let _ = event_sender.send(WatchEvent::RootRenamed {
                                    old_path: old_path.clone(),
                                    new_path: new_path.clone(),
                                    tenant_id: entry.tenant_id.clone(),
                                }).await;
                            } else {
                                let _ = event_sender.send(WatchEvent::Renamed {
                                    old_path,
                                    new_path,
                                    is_directory,
                                }).await;
                            }

                            // Reset path validator timer on folder operations
                            if is_directory {
                                path_validator.reset_timer().await;
                            }
                        }

                        RenameAction::CrossFilesystemMove { deleted_path, is_directory } => {
                            let _ = event_sender.send(WatchEvent::CrossFilesystemMove {
                                deleted_path,
                                is_directory,
                            }).await;
                        }

                        RenameAction::Pending => {
                            // Waiting for more events
                        }
                    }
                }
            }

            RenameMode::Both => {
                // Both paths provided in one event
                if paths.len() >= 2 {
                    let old_path = &paths[0];
                    let new_path = &paths[1];

                    let mut correlator = move_correlator.lock().await;
                    let action = correlator.handle_rename_event(
                        old_path.clone(),
                        new_path.clone(),
                        is_directory,
                    );
                    drop(correlator);

                    match action {
                        RenameAction::SimpleRename { old_path, new_path, is_directory } => {
                            // Check if this is a root folder rename
                            let entries = watch_entries.read().await;
                            if let Some(entry) = entries.get(&old_path) {
                                let _ = event_sender.send(WatchEvent::RootRenamed {
                                    old_path: old_path.clone(),
                                    new_path: new_path.clone(),
                                    tenant_id: entry.tenant_id.clone(),
                                }).await;
                            } else {
                                let _ = event_sender.send(WatchEvent::Renamed {
                                    old_path,
                                    new_path,
                                    is_directory,
                                }).await;
                            }
                        }

                        RenameAction::IntraFilesystemMove { old_path, new_path, is_directory } => {
                            // Check if this is a root folder rename
                            let entries = watch_entries.read().await;
                            if let Some(entry) = entries.get(&old_path) {
                                let _ = event_sender.send(WatchEvent::RootRenamed {
                                    old_path: old_path.clone(),
                                    new_path: new_path.clone(),
                                    tenant_id: entry.tenant_id.clone(),
                                }).await;
                            } else {
                                let _ = event_sender.send(WatchEvent::Renamed {
                                    old_path,
                                    new_path,
                                    is_directory,
                                }).await;
                            }
                        }

                        _ => {}
                    }

                    // Reset path validator timer on folder operations
                    if is_directory {
                        path_validator.reset_timer().await;
                    }
                }
            }

            RenameMode::Any | RenameMode::Other => {
                // Generic rename - treat as modify
                for path in paths {
                    let _ = event_sender.send(WatchEvent::Modified {
                        path: path.clone(),
                        is_directory,
                    }).await;
                }
            }
        }
    }
}

/// Handle for controlling the file watcher
pub struct WatcherHandle {
    /// The underlying debouncer
    debouncer: Arc<Mutex<Debouncer<RecommendedWatcher, RecommendedCache>>>,

    /// Running flag
    running: Arc<RwLock<bool>>,

    /// Watch entries
    watch_entries: Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,

    /// Processing task handle
    _process_task: tokio::task::JoinHandle<()>,
}

impl WatcherHandle {
    /// Add a path to watch
    pub async fn watch(
        &self,
        path: &Path,
        tenant_id: String,
        recursive: bool,
    ) -> Result<(), EnhancedWatcherError> {
        let canonical = path.canonicalize()
            .map_err(|e| EnhancedWatcherError::Path(format!(
                "Failed to canonicalize path {}: {}", path.display(), e
            )))?;

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        // Add to debouncer (v0.5+ API provides watch/unwatch directly on Debouncer)
        {
            let mut debouncer = self.debouncer.lock().await;
            debouncer.watch(&canonical, mode)?;
        }

        // Record the watch entry
        {
            let mut entries = self.watch_entries.write().await;
            entries.insert(canonical, WatchEntry {
                tenant_id,
                recursive,
            });
        }

        Ok(())
    }

    /// Remove a path from watching
    pub async fn unwatch(&self, path: &Path) -> Result<(), EnhancedWatcherError> {
        let canonical = path.canonicalize()
            .map_err(|e| EnhancedWatcherError::Path(format!(
                "Failed to canonicalize path {}: {}", path.display(), e
            )))?;

        // Remove from debouncer (v0.5+ API provides watch/unwatch directly on Debouncer)
        {
            let mut debouncer = self.debouncer.lock().await;
            debouncer.unwatch(&canonical)?;
        }

        // Remove from entries
        {
            let mut entries = self.watch_entries.write().await;
            entries.remove(&canonical);
        }

        Ok(())
    }

    /// Update watch path after a rename
    pub async fn update_watch_path(
        &self,
        old_path: &Path,
        new_path: &Path,
    ) -> Result<(), EnhancedWatcherError> {
        // Get the entry for the old path
        let entry = {
            let entries = self.watch_entries.read().await;
            entries.get(old_path).cloned()
        };

        if let Some(entry) = entry {
            // Unwatch old path
            self.unwatch(old_path).await.ok(); // Ignore error if already unwatched

            // Watch new path with same settings
            self.watch(new_path, entry.tenant_id, entry.recursive).await?;
        }

        Ok(())
    }

    /// Stop the watcher
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Get the list of watched paths
    pub async fn watched_paths(&self) -> Vec<PathBuf> {
        let entries = self.watch_entries.read().await;
        entries.keys().cloned().collect()
    }

    /// Get tenant ID for a path
    pub async fn get_tenant_id(&self, path: &Path) -> Option<String> {
        let entries = self.watch_entries.read().await;

        // Try exact match first
        if let Some(entry) = entries.get(path) {
            return Some(entry.tenant_id.clone());
        }

        // Try to find parent watch that covers this path
        for (watched_path, entry) in entries.iter() {
            if entry.recursive && path.starts_with(watched_path) {
                return Some(entry.tenant_id.clone());
            }
        }

        None
    }
}

/// Statistics for the enhanced watcher
#[derive(Debug, Clone, Default)]
pub struct EnhancedWatcherStats {
    pub watched_paths: usize,
    pub move_correlator_pending: usize,
    pub path_validator_pending_orphans: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_default() {
        let config = EnhancedWatcherConfig::default();
        assert_eq!(config.debounce_delay_ms, 1000);
        assert!(config.enable_file_id_cache);
    }

    #[tokio::test]
    async fn test_watch_event_variants() {
        // Test that all event variants can be created
        let events = vec![
            WatchEvent::Created {
                path: PathBuf::from("/test"),
                is_directory: false,
            },
            WatchEvent::Modified {
                path: PathBuf::from("/test"),
                is_directory: false,
            },
            WatchEvent::Deleted {
                path: PathBuf::from("/test"),
                is_directory: false,
            },
            WatchEvent::Renamed {
                old_path: PathBuf::from("/old"),
                new_path: PathBuf::from("/new"),
                is_directory: false,
            },
            WatchEvent::CrossFilesystemMove {
                deleted_path: PathBuf::from("/deleted"),
                is_directory: true,
            },
            WatchEvent::RootRenamed {
                old_path: PathBuf::from("/old"),
                new_path: PathBuf::from("/new"),
                tenant_id: "tenant-123".to_string(),
            },
            WatchEvent::Error {
                path: Some(PathBuf::from("/error")),
                message: "test error".to_string(),
            },
        ];

        assert_eq!(events.len(), 7);
    }
}
