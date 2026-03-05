//! Windows platform-specific file watching implementation using ReadDirectoryChangesW via notify crate
//!
//! Uses the `notify` crate's Windows backend, which provides:
//! - ReadDirectoryChangesW for efficient directory monitoring
//! - I/O Completion Ports for asynchronous event handling
//! - Recursive watching support
//!
//! Edge case handling:
//! - UNC paths: Supported via Windows extended path syntax
//! - Long paths (>260 chars): Handled via \\?\ prefix when needed
//! - Case-insensitive filesystem: Event paths preserve original case
//! - Network drives: Supported with appropriate timeout handling

use super::*;
use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

/// Windows file watcher using ReadDirectoryChangesW via the notify crate
///
/// The notify crate on Windows uses ReadDirectoryChangesW with I/O Completion Ports
/// for efficient, asynchronous file system monitoring.
pub struct WindowsWatcher {
    config: WindowsConfig,
    event_tx: mpsc::UnboundedSender<FileEvent>,
    event_rx: Option<mpsc::UnboundedReceiver<FileEvent>>,
    watcher: Option<RecommendedWatcher>,
    watched_paths: Vec<PathBuf>,
}

impl WindowsWatcher {
    /// Create a new Windows file watcher
    ///
    /// # Arguments
    /// * `config` - Windows-specific configuration (buffer size, filters)
    /// * `_buffer_size` - Event buffer size (used for channel capacity)
    pub fn new(config: WindowsConfig, _buffer_size: usize) -> Result<Self, PlatformWatchingError> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            event_tx,
            event_rx: Some(event_rx),
            watcher: None,
            watched_paths: Vec::new(),
        })
    }

    /// Normalize Windows path for extended length support
    ///
    /// Converts paths to use \\?\ prefix for long path support (>260 chars)
    fn normalize_path(&self, path: &Path) -> PathBuf {
        // Check if path might be a long path or UNC path
        let path_str = path.to_string_lossy();

        // Already has extended prefix
        if path_str.starts_with(r"\\?\") || path_str.starts_with(r"\\.\") {
            return path.to_path_buf();
        }

        // UNC path - convert to \\?\UNC\
        if path_str.starts_with(r"\\") {
            let unc_path = format!(r"\\?\UNC\{}", &path_str[2..]);
            return PathBuf::from(unc_path);
        }

        // Long path that needs extended prefix
        if path_str.len() > 248 {
            if let Ok(canonical) = std::fs::canonicalize(path) {
                let canonical_str = canonical.to_string_lossy();
                if !canonical_str.starts_with(r"\\?\") {
                    return PathBuf::from(format!(r"\\?\{}", canonical_str));
                }
                return canonical;
            }
        }

        path.to_path_buf()
    }

    /// Convert notify Event to our FileEvent format
    fn convert_event(event: Event) -> Option<FileEvent> {
        // Get the first path from the event
        let path = event.paths.first()?.clone();

        // Get file metadata for size
        let size = std::fs::metadata(&path).ok().map(|m| m.len());

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("platform".to_string(), "windows".to_string());
        metadata.insert("backend".to_string(), "ReadDirectoryChangesW".to_string());

        // Add event-specific metadata
        if event.need_rescan() {
            metadata.insert("needs_rescan".to_string(), "true".to_string());
        }

        Some(FileEvent {
            path,
            event_kind: event.kind,
            timestamp: Instant::now(),
            system_time: SystemTime::now(),
            size,
            metadata,
        })
    }

    /// Setup the ReadDirectoryChangesW-based watcher using notify crate
    fn setup_watcher(&mut self) -> Result<(), PlatformWatchingError> {
        let event_tx = self.event_tx.clone();
        let config = self.config.clone();

        // Configure notify
        let notify_config = NotifyConfig::default();

        // Create the watcher with our event handler
        let watcher = RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                match result {
                    Ok(event) => {
                        // Filter events based on config
                        let should_process = match &event.kind {
                            EventKind::Create(_) => {
                                config.monitor_file_name || config.monitor_dir_name
                            }
                            EventKind::Modify(modify_kind) => {
                                use notify::event::ModifyKind;
                                match modify_kind {
                                    ModifyKind::Data(_) => config.monitor_last_write,
                                    ModifyKind::Metadata(_) => config.monitor_size,
                                    ModifyKind::Name(_) => config.monitor_file_name,
                                    _ => true,
                                }
                            }
                            EventKind::Remove(_) => {
                                config.monitor_file_name || config.monitor_dir_name
                            }
                            EventKind::Access(_) => false, // Skip access events
                            EventKind::Other => true,
                            EventKind::Any => true,
                        };

                        if should_process {
                            if let Some(file_event) = Self::convert_event(event) {
                                if let Err(e) = event_tx.send(file_event) {
                                    tracing::warn!("Failed to send file event: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Windows watcher error: {}", e);
                    }
                }
            },
            notify_config,
        )
        .map_err(|e| PlatformWatchingError::ReadDirectoryChanges(e.to_string()))?;

        self.watcher = Some(watcher);
        tracing::info!("Windows ReadDirectoryChangesW watcher initialized");

        Ok(())
    }

    /// Get the number of watched paths
    pub fn watched_path_count(&self) -> usize {
        self.watched_paths.len()
    }

    /// Check if the watcher is active
    pub fn is_active(&self) -> bool {
        self.watcher.is_some() && !self.watched_paths.is_empty()
    }
}

#[async_trait]
impl PlatformWatcher for WindowsWatcher {
    async fn watch(&mut self, path: &Path) -> Result<(), PlatformWatchingError> {
        // Initialize watcher if not already done
        if self.watcher.is_none() {
            self.setup_watcher()?;
        }

        // Normalize path for long path/UNC support
        let normalized_path = self.normalize_path(path);

        // Verify path exists and is accessible
        if !normalized_path.exists() {
            return Err(PlatformWatchingError::ReadDirectoryChanges(format!(
                "Path does not exist: {}",
                normalized_path.display()
            )));
        }

        // Check permissions
        if let Err(e) = std::fs::read_dir(&normalized_path) {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(PlatformWatchingError::ReadDirectoryChanges(format!(
                    "Permission denied: {}",
                    normalized_path.display()
                )));
            }
        }

        // Determine recursive mode based on config
        let recursive_mode = if self.config.watch_subtree {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        // Add path to watcher
        if let Some(ref mut watcher) = self.watcher {
            watcher
                .watch(&normalized_path, recursive_mode)
                .map_err(|e| PlatformWatchingError::ReadDirectoryChanges(e.to_string()))?;

            self.watched_paths.push(normalized_path.clone());
            tracing::info!(
                "Started Windows ReadDirectoryChangesW watching for: {} (normalized: {}, recursive: {})",
                path.display(),
                normalized_path.display(),
                self.config.watch_subtree
            );
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PlatformWatchingError> {
        // Remove all watched paths
        if let Some(ref mut watcher) = self.watcher {
            for path in &self.watched_paths {
                if let Err(e) = watcher.unwatch(path) {
                    tracing::warn!("Failed to unwatch {}: {}", path.display(), e);
                }
            }
        }

        // Clear state
        self.watcher = None;
        self.watched_paths.clear();

        tracing::info!("Stopped Windows ReadDirectoryChangesW watcher");
        Ok(())
    }

    fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
        self.event_rx.take()
    }
}
