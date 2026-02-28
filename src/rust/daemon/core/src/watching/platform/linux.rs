//! Linux platform-specific file watching implementation using inotify
//!
//! Uses the `inotify` crate directly with `tokio::task::spawn_blocking` for the
//! blocking read loop. This is simpler and more reliable than AsyncFd since
//! `Inotify` from the inotify crate is `!Sync` in some configurations.
//!
//! Limitations:
//! - inotify is NOT recursive (unlike FSEvents) - each subdirectory needs its own watch
//! - Watch limit: default 8192 per user (configurable via /proc/sys/fs/inotify/max_user_watches)
//! - Initial implementation: single-level watching per added path; recursive support is a follow-up
//!
//! Edge case handling:
//! - Symlinks: Resolved to canonical paths before watching
//! - Rapid file changes: Handled by event deduplication at the FileWatcher level
//! - Watch limit exhaustion: Returns error with clear message

use super::*;
use inotify::{Inotify, WatchMask, WatchDescriptor};
use notify::event::{CreateKind, ModifyKind, RemoveKind, RenameMode, DataChange};
use notify::EventKind;
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

pub struct LinuxWatcher {
    config: LinuxConfig,
    event_tx: mpsc::UnboundedSender<FileEvent>,
    event_rx: Option<mpsc::UnboundedReceiver<FileEvent>>,
    inotify: Option<Inotify>,
    watched_paths: Vec<PathBuf>,
    /// Maps inotify watch descriptors back to their base directories
    watch_descriptors: HashMap<WatchDescriptor, PathBuf>,
    /// Signal to stop the event processing loop
    stop_flag: Arc<AtomicBool>,
    /// Handle to the background event processing task
    event_task: Option<tokio::task::JoinHandle<()>>,
}

impl LinuxWatcher {
    /// Create a new Linux file watcher
    ///
    /// # Arguments
    /// * `config` - Linux-specific configuration (buffer size, event types)
    /// * `_buffer_size` - Event buffer size (used for channel capacity)
    pub fn new(
        config: LinuxConfig,
        _buffer_size: usize,
    ) -> Result<Self, PlatformWatchingError> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            event_tx,
            event_rx: Some(event_rx),
            inotify: None,
            watched_paths: Vec::new(),
            watch_descriptors: HashMap::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            event_task: None,
        })
    }

    /// Build the inotify watch mask from config
    fn build_watch_mask(&self) -> WatchMask {
        let mut mask = WatchMask::empty();

        if self.config.monitor_create {
            mask |= WatchMask::CREATE;
        }
        if self.config.monitor_modify {
            mask |= WatchMask::MODIFY | WatchMask::CLOSE_WRITE;
        }
        if self.config.monitor_delete {
            mask |= WatchMask::DELETE;
        }
        if self.config.track_moves {
            mask |= WatchMask::MOVED_FROM | WatchMask::MOVED_TO;
        }

        mask
    }

    /// Set up inotify and add a watch for the given path
    fn setup_inotify(&mut self, path: &Path) -> Result<(), PlatformWatchingError> {
        // Initialize inotify if not already done
        if self.inotify.is_none() {
            let inotify = Inotify::init()
                .map_err(|e| PlatformWatchingError::Inotify(e.to_string()))?;
            self.inotify = Some(inotify);
        }

        let mask = self.build_watch_mask();

        let wd = self
            .inotify
            .as_mut()
            .unwrap()
            .watches()
            .add(path, mask)
            .map_err(|e| {
                if e.kind() == ErrorKind::Other {
                    PlatformWatchingError::Inotify(format!(
                        "Failed to add watch for {}: {} (check /proc/sys/fs/inotify/max_user_watches)",
                        path.display(),
                        e
                    ))
                } else {
                    PlatformWatchingError::Inotify(format!(
                        "Failed to add watch for {}: {}",
                        path.display(),
                        e
                    ))
                }
            })?;

        self.watch_descriptors.insert(wd, path.to_path_buf());
        self.watched_paths.push(path.to_path_buf());

        tracing::info!(
            "Set up inotify for path: {} with mask: {:?}",
            path.display(),
            mask
        );
        Ok(())
    }

    /// Convert inotify event mask to notify::EventKind
    fn mask_to_event_kind(mask: inotify::EventMask) -> EventKind {
        if mask.contains(inotify::EventMask::CREATE) {
            if mask.contains(inotify::EventMask::ISDIR) {
                EventKind::Create(CreateKind::Folder)
            } else {
                EventKind::Create(CreateKind::File)
            }
        } else if mask.contains(inotify::EventMask::MODIFY)
            || mask.contains(inotify::EventMask::CLOSE_WRITE)
        {
            EventKind::Modify(ModifyKind::Data(DataChange::Any))
        } else if mask.contains(inotify::EventMask::DELETE) {
            if mask.contains(inotify::EventMask::ISDIR) {
                EventKind::Remove(RemoveKind::Folder)
            } else {
                EventKind::Remove(RemoveKind::File)
            }
        } else if mask.contains(inotify::EventMask::MOVED_FROM) {
            EventKind::Modify(ModifyKind::Name(RenameMode::From))
        } else if mask.contains(inotify::EventMask::MOVED_TO) {
            EventKind::Modify(ModifyKind::Name(RenameMode::To))
        } else {
            EventKind::Other
        }
    }

    /// Start the background event processing loop
    ///
    /// Spawns a blocking task that reads from inotify in a loop and converts
    /// events to FileEvent messages sent via the channel.
    fn start_event_processing(&mut self) -> Result<(), PlatformWatchingError> {
        let inotify = self
            .inotify
            .take()
            .ok_or_else(|| PlatformWatchingError::Inotify("inotify not initialized".into()))?;

        let event_tx = self.event_tx.clone();
        let stop_flag = Arc::clone(&self.stop_flag);
        let buffer_size = self.config.buffer_size;
        let watch_descriptors = self.watch_descriptors.clone();

        let task = tokio::task::spawn_blocking(move || {
            let mut buffer = vec![0u8; buffer_size];
            let mut inotify = inotify;

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    tracing::info!("inotify event loop stopping (stop flag set)");
                    break;
                }

                match inotify.read_events(&mut buffer) {
                    Ok(events) => {
                        for event in events {
                            // Resolve the base path from the watch descriptor
                            let base_path = match watch_descriptors.get(&event.wd) {
                                Some(path) => path.clone(),
                                None => {
                                    tracing::warn!(
                                        "Received event for unknown watch descriptor"
                                    );
                                    continue;
                                }
                            };

                            // Build full path: base_dir + event.name
                            let full_path = match event.name {
                                Some(name) => base_path.join(name),
                                None => base_path.clone(),
                            };

                            let event_kind = Self::mask_to_event_kind(event.mask);

                            // Get file metadata for size (may fail for deleted files)
                            let size = std::fs::metadata(&full_path).ok().map(|m| m.len());

                            let mut metadata = HashMap::new();
                            metadata
                                .insert("platform".to_string(), "linux".to_string());
                            metadata
                                .insert("backend".to_string(), "inotify".to_string());

                            if event.mask.contains(inotify::EventMask::ISDIR) {
                                metadata
                                    .insert("is_dir".to_string(), "true".to_string());
                            }

                            let file_event = FileEvent {
                                path: full_path,
                                event_kind,
                                timestamp: Instant::now(),
                                system_time: SystemTime::now(),
                                size,
                                metadata,
                            };

                            if let Err(e) = event_tx.send(file_event) {
                                tracing::warn!("Failed to send inotify event: {}", e);
                                // Channel closed, stop the loop
                                break;
                            }
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => {
                        // EINTR - just retry
                        continue;
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        // No events available, sleep briefly and retry
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    }
                    Err(e) => {
                        tracing::error!("inotify read error: {}", e);
                        break;
                    }
                }
            }

            tracing::info!("inotify event loop exited");
        });

        self.event_task = Some(task);
        tracing::info!("Started Linux inotify event processing");
        Ok(())
    }

    /// Get the number of watched paths
    pub fn watched_path_count(&self) -> usize {
        self.watched_paths.len()
    }

    /// Check if the watcher is active
    pub fn is_active(&self) -> bool {
        self.event_task.is_some() && !self.watched_paths.is_empty()
    }
}

#[async_trait]
impl PlatformWatcher for LinuxWatcher {
    async fn watch(&mut self, path: &Path) -> Result<(), PlatformWatchingError> {
        // Verify path exists and is accessible
        if !path.exists() {
            return Err(PlatformWatchingError::Inotify(format!(
                "Path does not exist: {}",
                path.display()
            )));
        }

        // Check permissions
        if let Err(e) = std::fs::read_dir(path) {
            if e.kind() == ErrorKind::PermissionDenied {
                return Err(PlatformWatchingError::Inotify(format!(
                    "Permission denied: {}",
                    path.display()
                )));
            }
        }

        // Set up inotify watch
        self.setup_inotify(path)?;

        // Start event processing loop (only once, for the first watch)
        if self.event_task.is_none() {
            self.start_event_processing()?;
        }

        tracing::info!("Started Linux inotify watching for: {}", path.display());
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PlatformWatchingError> {
        // Signal the event loop to stop
        self.stop_flag.store(true, Ordering::SeqCst);

        // Wait for the event processing task to finish
        if let Some(task) = self.event_task.take() {
            // The blocking task will exit when it sees the stop flag
            // Give it a reasonable timeout
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => {
                    tracing::info!("inotify event task stopped cleanly");
                }
                Ok(Err(e)) => {
                    tracing::warn!("inotify event task panicked: {}", e);
                }
                Err(_) => {
                    tracing::warn!("inotify event task did not stop within 5 seconds");
                }
            }
        }

        // Clean up state
        self.inotify = None;
        self.watched_paths.clear();
        self.watch_descriptors.clear();
        // Reset stop flag for potential reuse
        self.stop_flag.store(false, Ordering::SeqCst);

        tracing::info!("Stopped Linux inotify watcher");
        Ok(())
    }

    fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
        self.event_rx.take()
    }
}
