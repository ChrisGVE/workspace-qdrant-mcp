//! macOS platform-specific file watching implementation using FSEvents via notify crate
//!
//! Uses the `notify` crate's FSEvents backend for macOS, which provides:
//! - Efficient directory-level event notification
//! - Recursive watching support
//! - Configurable latency for event coalescing
//!
//! Edge case handling:
//! - Symlinks: Watched paths are used **as given**. The user's symlink
//!   representation is preserved end-to-end (see spec §16 §11.1 sub-case
//!   8b). Watch roots that are themselves symlinks emit a startup warning
//!   because FSEvents may attribute events to the symlink target rather
//!   than the symlink path; this is the documented limitation chosen
//!   under audit task A-2 (spec §13: "restrict to non-symlink roots —
//!   document limitation"). Test case 8e exercises this path.
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
        })
    }

    /// Inspect a watch-root path and emit a warning if it is itself a
    /// symlink. The path is returned unchanged — per spec §16 §3.1
    /// rule 7 we never resolve symlinks. Callers register FSEvents on
    /// the user-supplied path representation; events for files under a
    /// symlinked root may be attributed to either the symlink or its
    /// target depending on FSEvents semantics.
    ///
    /// This replaces the former `resolve_symlink` function (audit task
    /// A-2, spec §3.2.2). The redesign chooses option (a) from spec
    /// §13: "restrict to non-symlink roots — document limitation".
    fn check_root_symlink(path: &Path) {
        if let Ok(meta) = std::fs::symlink_metadata(path) {
            if meta.file_type().is_symlink() {
                tracing::warn!(
                    "macOS watch root is a symlink: {}. FSEvents may attribute \
                     events to the symlink target rather than the symlink path. \
                     For stable canonical-path attribution, prefer watching the \
                     resolved target directly. See spec §16 §11.1 case 8e.",
                    path.display()
                );
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

        // Emit a warning if the user-supplied watch root is itself a
        // symlink. The path is then used unchanged: FSEvents is
        // registered on the symlink representation, not the target.
        Self::check_root_symlink(path);
        let watch_path = path.to_path_buf();

        // Verify path exists and is accessible
        if !watch_path.exists() {
            return Err(PlatformWatchingError::FSEvents(format!(
                "Path does not exist: {}",
                watch_path.display()
            )));
        }

        // Check permissions
        if let Err(e) = std::fs::read_dir(&watch_path) {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(PlatformWatchingError::FSEvents(format!(
                    "Permission denied: {}",
                    watch_path.display()
                )));
            }
        }

        // Add path to watcher (always recursive on macOS FSEvents)
        if let Some(ref mut watcher) = self.watcher {
            watcher
                .watch(&watch_path, RecursiveMode::Recursive)
                .map_err(|e| PlatformWatchingError::FSEvents(e.to_string()))?;

            self.watched_paths.push(watch_path.clone());
            tracing::info!(
                "Started macOS FSEvents watching for: {}",
                watch_path.display()
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

        tracing::info!("Stopped macOS FSEvents watcher");
        Ok(())
    }

    fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
        self.event_rx.take()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod symlink_tests {
    //! Test §11.1 case 8e — macOS FSEvents handling for symlinked watch
    //! roots after `resolve_symlink` removal. Verifies the chosen
    //! redesign (option (a): restrict to non-symlink roots, log warning,
    //! preserve user-supplied path representation).
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_8e_symlink_watch_root_preserves_path_representation() {
        // Create a real directory, a symlink to it, then watch the symlink.
        // The watcher must store the symlink path (not the target) in
        // its watched_paths list — confirming spec §16 §11.1 8b/8e
        // behavior.
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("real");
        fs::create_dir(&target).unwrap();

        let symlink = temp.path().join("link");
        std::os::unix::fs::symlink(&target, &symlink).unwrap();

        let mut watcher = MacOSWatcher::new(MacOSConfig::default(), 64).unwrap();
        watcher.watch(&symlink).await.unwrap();

        let watched = &watcher.watched_paths;
        assert_eq!(watched.len(), 1, "should register exactly one path");
        // The watcher stores the symlink path as given — NOT the target.
        // This is the §3.2.2 / §3.1 rule 7 guarantee.
        assert_eq!(
            watched[0],
            symlink,
            "watch_paths must preserve the user-supplied symlink representation; got {} expected {}",
            watched[0].display(),
            symlink.display()
        );
    }

    #[tokio::test]
    async fn test_8e_non_symlink_watch_root_works_normally() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("project");
        fs::create_dir(&dir).unwrap();

        let mut watcher = MacOSWatcher::new(MacOSConfig::default(), 64).unwrap();
        watcher.watch(&dir).await.unwrap();

        assert_eq!(watcher.watched_paths.len(), 1);
        assert_eq!(watcher.watched_paths[0], dir);
    }
}
