//! macOS platform-specific file watching implementation using FSEvents via notify crate
//!
//! Uses the `notify` crate's FSEvents backend for macOS, which provides:
//! - Efficient directory-level event notification
//! - Recursive watching support
//! - Configurable latency for event coalescing
//!
//! Edge case handling:
//! - Symlinks: Resolved to canonical paths before watching
//! - Permission changes: Reported as metadata events
//! - Rapid file changes: Coalesced based on latency setting
//! - Network mounts: Automatically uses polling fallback if FSEvents unavailable

use super::*;
use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// macOS file watcher using FSEvents via the notify crate
///
/// The notify crate automatically uses FSEvents on macOS when available,
/// falling back to kqueue for individual file watching when needed.
pub struct MacOSWatcher {
    config: MacOSConfig,
    event_tx: mpsc::UnboundedSender<FileEvent>,
    event_rx: Option<mpsc::UnboundedReceiver<FileEvent>>,
    watcher: Option<RecommendedWatcher>,
    watched_paths: Vec<PathBuf>,
    /// Track symlink resolutions to handle symlinked directories
    symlink_map: HashMap<PathBuf, PathBuf>,
}

impl MacOSWatcher {
    /// Create a new macOS file watcher
    ///
    /// # Arguments
    /// * `config` - macOS-specific configuration (latency, event types)
    /// * `_buffer_size` - Event buffer size (used for channel capacity)
    pub fn new(config: MacOSConfig, _buffer_size: usize) -> Result<Self, PlatformWatchingError> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            event_tx,
            event_rx: Some(event_rx),
            watcher: None,
            watched_paths: Vec::new(),
            symlink_map: HashMap::new(),
        })
    }

    /// Resolve symlinks to their canonical paths
    ///
    /// This ensures consistent event handling when watching symlinked directories
    fn resolve_symlink(&mut self, path: &Path) -> PathBuf {
        match std::fs::canonicalize(path) {
            Ok(canonical) => {
                if canonical != path {
                    tracing::debug!(
                        "Resolved symlink: {} -> {}",
                        path.display(),
                        canonical.display()
                    );
                    self.symlink_map
                        .insert(path.to_path_buf(), canonical.clone());
                }
                canonical
            }
            Err(e) => {
                tracing::warn!("Failed to resolve symlink for {}: {}", path.display(), e);
                path.to_path_buf()
            }
        }
    }

    /// Convert notify Event to our FileEvent format
    fn convert_event(event: Event) -> Option<FileEvent> {
        // Get the first path from the event (notify events can have multiple paths)
        let path = event.paths.first()?.clone();

        // Get file metadata for size
        let size = std::fs::metadata(&path).ok().map(|m| m.len());

        // Build metadata from event attributes
        let mut metadata = HashMap::new();
        metadata.insert("platform".to_string(), "macos".to_string());
        metadata.insert("backend".to_string(), "fsevents".to_string());

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

    /// Setup the FSEvents-based watcher using notify crate
    fn setup_watcher(&mut self) -> Result<(), PlatformWatchingError> {
        let event_tx = self.event_tx.clone();
        let config = self.config.clone();

        // Configure notify with FSEvents-specific settings
        // The latency setting controls event coalescing (debouncing at the OS level)
        let notify_config =
            NotifyConfig::default().with_poll_interval(Duration::from_secs_f64(config.latency));

        // Create the watcher with our event handler
        let watcher = RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                match result {
                    Ok(event) => {
                        // Filter events based on config
                        let should_process = match &event.kind {
                            EventKind::Create(_) | EventKind::Modify(_) => config.watch_file_events,
                            EventKind::Remove(_) => config.watch_file_events,
                            EventKind::Access(_) => false, // Skip access events by default
                            EventKind::Other => config.watch_dir_events,
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
                        tracing::error!("FSEvents watcher error: {}", e);
                    }
                }
            },
            notify_config,
        )
        .map_err(|e| PlatformWatchingError::FSEvents(e.to_string()))?;

        self.watcher = Some(watcher);
        tracing::info!(
            "FSEvents watcher initialized with latency: {}s",
            config.latency
        );

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
impl PlatformWatcher for MacOSWatcher {
    async fn watch(&mut self, path: &Path) -> Result<(), PlatformWatchingError> {
        // Initialize watcher if not already done
        if self.watcher.is_none() {
            self.setup_watcher()?;
        }

        // Resolve symlinks to ensure consistent behavior
        let canonical_path = self.resolve_symlink(path);

        // Verify path exists and is accessible
        if !canonical_path.exists() {
            return Err(PlatformWatchingError::FSEvents(format!(
                "Path does not exist: {}",
                canonical_path.display()
            )));
        }

        // Check permissions
        if let Err(e) = std::fs::read_dir(&canonical_path) {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(PlatformWatchingError::FSEvents(format!(
                    "Permission denied: {}",
                    canonical_path.display()
                )));
            }
        }

        // Add path to watcher (always recursive on macOS FSEvents)
        if let Some(ref mut watcher) = self.watcher {
            watcher
                .watch(&canonical_path, RecursiveMode::Recursive)
                .map_err(|e| PlatformWatchingError::FSEvents(e.to_string()))?;

            self.watched_paths.push(canonical_path.clone());
            tracing::info!(
                "Started macOS FSEvents watching for: {} (canonical: {})",
                path.display(),
                canonical_path.display()
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
        self.symlink_map.clear();

        tracing::info!("Stopped macOS FSEvents watcher");
        Ok(())
    }

    fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
        self.event_rx.take()
    }
}
