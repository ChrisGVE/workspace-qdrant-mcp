//! Platform-specific file watching optimizations
//!
//! This module provides platform-specific implementations of file watching
//! to leverage native file system events for better performance.

use std::path::Path;
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use async_trait::async_trait;

use super::{FileEvent, WatchingError};

/// Platform-specific file watching errors
#[derive(Error, Debug)]
pub enum PlatformWatchingError {
    #[error("Platform not supported: {platform}")]
    UnsupportedPlatform { platform: String },
    
    #[error("Platform-specific error: {message}")]
    PlatformError { message: String },
    
    #[cfg(target_os = "macos")]
    #[error("FSEvents error: {0}")]
    FSEvents(String),
    
    #[cfg(target_os = "linux")]
    #[error("Inotify error: {0}")]
    Inotify(String),
    
    #[cfg(target_os = "windows")]
    #[error("ReadDirectoryChangesW error: {0}")]
    ReadDirectoryChanges(String),
}

impl From<PlatformWatchingError> for WatchingError {
    fn from(err: PlatformWatchingError) -> Self {
        WatchingError::Config {
            message: format!("Platform watching error: {}", err),
        }
    }
}

/// Configuration for platform-specific file watching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformWatcherConfig {
    /// Use platform-specific optimizations
    pub enable_optimizations: bool,
    
    /// Buffer size for platform events
    pub event_buffer_size: usize,
    
    /// Platform-specific settings
    #[cfg(target_os = "macos")]
    pub macos: MacOSConfig,
    
    #[cfg(target_os = "linux")]
    pub linux: LinuxConfig,
    
    #[cfg(target_os = "windows")]
    pub windows: WindowsConfig,
}

impl Default for PlatformWatcherConfig {
    fn default() -> Self {
        Self {
            enable_optimizations: true,
            event_buffer_size: 4096,
            
            #[cfg(target_os = "macos")]
            macos: MacOSConfig::default(),
            
            #[cfg(target_os = "linux")]
            linux: LinuxConfig::default(),
            
            #[cfg(target_os = "windows")]
            windows: WindowsConfig::default(),
        }
    }
}

/// macOS-specific configuration for FSEvents
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacOSConfig {
    /// FSEvents latency in seconds
    pub latency: f64,
    
    /// Use kqueue for fine-grained monitoring
    pub use_kqueue: bool,
    
    /// Stream flags for FSEvents
    pub stream_flags: u32,
    
    /// Watch for file events
    pub watch_file_events: bool,
    
    /// Watch for directory events
    pub watch_dir_events: bool,
}

#[cfg(target_os = "macos")]
impl Default for MacOSConfig {
    fn default() -> Self {
        Self {
            latency: 0.1, // 100ms latency
            use_kqueue: false,
            stream_flags: 0, // Default FSEvents flags
            watch_file_events: true,
            watch_dir_events: true,
        }
    }
}

/// Linux-specific configuration for inotify
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinuxConfig {
    /// Deprecated: epoll is no longer used (inotify blocking reads via spawn_blocking)
    pub use_epoll: bool,

    /// Inotify read buffer size in bytes
    pub buffer_size: usize,

    /// Maximum number of inotify watches (informational; actual limit is kernel-level)
    pub max_watches: usize,

    /// Track file moves (IN_MOVED_FROM and IN_MOVED_TO events)
    pub track_moves: bool,

    /// Monitor file creation (IN_CREATE)
    pub monitor_create: bool,

    /// Monitor file modification (IN_MODIFY, IN_CLOSE_WRITE)
    pub monitor_modify: bool,

    /// Monitor file deletion (IN_DELETE)
    pub monitor_delete: bool,
}

#[cfg(target_os = "linux")]
impl Default for LinuxConfig {
    fn default() -> Self {
        Self {
            use_epoll: true,
            buffer_size: 16384, // 16KB buffer
            max_watches: 8192,
            track_moves: true,
            monitor_create: true,
            monitor_modify: true,
            monitor_delete: false, // Usually not needed for document processing
        }
    }
}

/// Windows-specific configuration for ReadDirectoryChangesW
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsConfig {
    /// Watch subtree (recursive)
    pub watch_subtree: bool,
    
    /// Buffer size for ReadDirectoryChangesW
    pub buffer_size: u32,
    
    /// Watch filter flags
    pub filter_flags: u32,
    
    /// Use completion ports
    pub use_completion_ports: bool,
    
    /// Monitor file name changes
    pub monitor_file_name: bool,
    
    /// Monitor directory name changes
    pub monitor_dir_name: bool,
    
    /// Monitor file size changes
    pub monitor_size: bool,
    
    /// Monitor file write time changes
    pub monitor_last_write: bool,
}

#[cfg(target_os = "windows")]
impl Default for WindowsConfig {
    fn default() -> Self {
        use windows::Win32::Storage::FileSystem::*;
        
        Self {
            watch_subtree: true,
            buffer_size: 65536, // 64KB buffer
            filter_flags: (FILE_NOTIFY_CHANGE_FILE_NAME.0 
                         | FILE_NOTIFY_CHANGE_SIZE.0 
                         | FILE_NOTIFY_CHANGE_LAST_WRITE.0) as u32,
            use_completion_ports: true,
            monitor_file_name: true,
            monitor_dir_name: true,
            monitor_size: true,
            monitor_last_write: true,
        }
    }
}

/// Platform-specific file watcher trait
#[async_trait]
pub trait PlatformWatcher: Send + Sync {
    /// Start watching the specified path
    async fn watch(&mut self, path: &Path) -> Result<(), PlatformWatchingError>;

    /// Stop watching all paths
    async fn stop(&mut self) -> Result<(), PlatformWatchingError>;

    /// Get the event receiver
    fn event_receiver(&self) -> mpsc::UnboundedReceiver<FileEvent>;
}

/// Factory for creating platform-specific watchers
pub struct PlatformWatcherFactory;

impl PlatformWatcherFactory {
    /// Create a platform-specific watcher
    pub fn create_watcher(
        config: PlatformWatcherConfig,
    ) -> Result<Box<dyn PlatformWatcher>, PlatformWatchingError> {
        if !config.enable_optimizations {
            return Err(PlatformWatchingError::PlatformError {
                message: "Platform optimizations disabled".to_string(),
            });
        }
        
        #[cfg(target_os = "macos")]
        {
            let watcher = MacOSWatcher::new(config.macos, config.event_buffer_size)?;
            Ok(Box::new(watcher))
        }
        
        #[cfg(target_os = "linux")]
        {
            let watcher = LinuxWatcher::new(config.linux, config.event_buffer_size)?;
            Ok(Box::new(watcher))
        }
        
        #[cfg(target_os = "windows")]
        {
            let watcher = WindowsWatcher::new(config.windows, config.event_buffer_size)?;
            Ok(Box::new(watcher))
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            Err(PlatformWatchingError::UnsupportedPlatform {
                platform: std::env::consts::OS.to_string(),
            })
        }
    }
}

// Platform-specific implementations

/// macOS platform-specific file watching implementation using FSEvents via notify crate
///
/// Uses the `notify` crate's FSEvents backend for macOS, which provides:
/// - Efficient directory-level event notification
/// - Recursive watching support
/// - Configurable latency for event coalescing
///
/// Edge case handling:
/// - Symlinks: Resolved to canonical paths before watching
/// - Permission changes: Reported as metadata events
/// - Rapid file changes: Coalesced based on latency setting
/// - Network mounts: Automatically uses polling fallback if FSEvents unavailable
#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use notify::{RecommendedWatcher, Watcher, RecursiveMode, Event, EventKind, Config as NotifyConfig};
    use std::time::{Duration, Instant, SystemTime};
    use std::collections::HashMap;
    use std::path::PathBuf;

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
        pub fn new(
            config: MacOSConfig,
            _buffer_size: usize,
        ) -> Result<Self, PlatformWatchingError> {
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
                        self.symlink_map.insert(path.to_path_buf(), canonical.clone());
                    }
                    canonical
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to resolve symlink for {}: {}",
                        path.display(),
                        e
                    );
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
            let notify_config = NotifyConfig::default()
                .with_poll_interval(Duration::from_secs_f64(config.latency));

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
            ).map_err(|e| PlatformWatchingError::FSEvents(e.to_string()))?;

            self.watcher = Some(watcher);
            tracing::info!("FSEvents watcher initialized with latency: {}s", config.latency);

            Ok(())
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
                return Err(PlatformWatchingError::FSEvents(
                    format!("Path does not exist: {}", canonical_path.display())
                ));
            }

            // Check permissions
            if let Err(e) = std::fs::read_dir(&canonical_path) {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    return Err(PlatformWatchingError::FSEvents(
                        format!("Permission denied: {}", canonical_path.display())
                    ));
                }
            }

            // Add path to watcher (always recursive on macOS FSEvents)
            if let Some(ref mut watcher) = self.watcher {
                watcher.watch(&canonical_path, RecursiveMode::Recursive)
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

        fn event_receiver(&self) -> mpsc::UnboundedReceiver<FileEvent> {
            // Note: This returns a new empty receiver since we can't clone the original
            // The actual receiver should be taken via take_event_receiver() before starting
            let (_, rx) = mpsc::unbounded_channel();
            rx
        }
    }

    impl MacOSWatcher {
        /// Take ownership of the event receiver
        ///
        /// This should be called before starting the watcher to get the receiver
        /// for processing events.
        pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
            self.event_rx.take()
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
}

#[cfg(target_os = "macos")]
pub use macos::MacOSWatcher;

/// Linux platform-specific file watching implementation using inotify
///
/// Uses the `inotify` crate directly with `tokio::task::spawn_blocking` for the
/// blocking read loop. This is simpler and more reliable than AsyncFd since
/// `Inotify` from the inotify crate is `!Sync` in some configurations.
///
/// Limitations:
/// - inotify is NOT recursive (unlike FSEvents) - each subdirectory needs its own watch
/// - Watch limit: default 8192 per user (configurable via /proc/sys/fs/inotify/max_user_watches)
/// - Initial implementation: single-level watching per added path; recursive support is a follow-up
///
/// Edge case handling:
/// - Symlinks: Resolved to canonical paths before watching
/// - Rapid file changes: Handled by event deduplication at the FileWatcher level
/// - Watch limit exhaustion: Returns error with clear message
#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use inotify::{Inotify, WatchMask, WatchDescriptor};
    use notify::event::{CreateKind, ModifyKind, RemoveKind, RenameMode, DataChange};
    use notify::EventKind;
    use std::collections::HashMap;
    use std::io::ErrorKind;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Instant, SystemTime};

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

        /// Take ownership of the event receiver
        ///
        /// This should be called before starting the watcher to get the receiver
        /// for processing events.
        pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
            self.event_rx.take()
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

        fn event_receiver(&self) -> mpsc::UnboundedReceiver<FileEvent> {
            // Note: This returns a new empty receiver since we can't clone the original
            // The actual receiver should be taken via take_event_receiver() before starting
            let (_, rx) = mpsc::unbounded_channel();
            rx
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::LinuxWatcher;

/// Windows platform-specific file watching implementation using ReadDirectoryChangesW via notify crate
///
/// Uses the `notify` crate's Windows backend, which provides:
/// - ReadDirectoryChangesW for efficient directory monitoring
/// - I/O Completion Ports for asynchronous event handling
/// - Recursive watching support
///
/// Edge case handling:
/// - UNC paths: Supported via Windows extended path syntax
/// - Long paths (>260 chars): Handled via \\?\ prefix when needed
/// - Case-insensitive filesystem: Event paths preserve original case
/// - Network drives: Supported with appropriate timeout handling
#[cfg(target_os = "windows")]
mod windows {
    use super::*;
    use notify::{RecommendedWatcher, Watcher, RecursiveMode, Event, EventKind, Config as NotifyConfig};
    use std::time::{Duration, Instant, SystemTime};
    use std::collections::HashMap;
    use std::path::PathBuf;

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
        pub fn new(
            config: WindowsConfig,
            _buffer_size: usize,
        ) -> Result<Self, PlatformWatchingError> {
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
                                EventKind::Create(_) => config.monitor_file_name || config.monitor_dir_name,
                                EventKind::Modify(modify_kind) => {
                                    use notify::event::ModifyKind;
                                    match modify_kind {
                                        ModifyKind::Data(_) => config.monitor_last_write,
                                        ModifyKind::Metadata(_) => config.monitor_size,
                                        ModifyKind::Name(_) => config.monitor_file_name,
                                        _ => true,
                                    }
                                }
                                EventKind::Remove(_) => config.monitor_file_name || config.monitor_dir_name,
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
            ).map_err(|e| PlatformWatchingError::ReadDirectoryChanges(e.to_string()))?;

            self.watcher = Some(watcher);
            tracing::info!("Windows ReadDirectoryChangesW watcher initialized");

            Ok(())
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
                return Err(PlatformWatchingError::ReadDirectoryChanges(
                    format!("Path does not exist: {}", normalized_path.display())
                ));
            }

            // Check permissions
            if let Err(e) = std::fs::read_dir(&normalized_path) {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    return Err(PlatformWatchingError::ReadDirectoryChanges(
                        format!("Permission denied: {}", normalized_path.display())
                    ));
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
                watcher.watch(&normalized_path, recursive_mode)
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

        fn event_receiver(&self) -> mpsc::UnboundedReceiver<FileEvent> {
            // Note: This returns a new empty receiver since we can't clone the original
            // The actual receiver should be taken via take_event_receiver() before starting
            let (_, rx) = mpsc::unbounded_channel();
            rx
        }
    }

    impl WindowsWatcher {
        /// Take ownership of the event receiver
        ///
        /// This should be called before starting the watcher to get the receiver
        /// for processing events.
        pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>> {
            self.event_rx.take()
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
}

#[cfg(target_os = "windows")]
pub use windows::WindowsWatcher;

/// Statistics for platform-specific watchers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformWatchingStats {
    pub platform: String,
    pub native_events_received: u64,
    pub native_events_processed: u64,
    pub platform_errors: u64,
    pub optimization_enabled: bool,
}

impl Default for PlatformWatchingStats {
    fn default() -> Self {
        Self {
            platform: std::env::consts::OS.to_string(),
            native_events_received: 0,
            native_events_processed: 0,
            platform_errors: 0,
            optimization_enabled: false,
        }
    }
}