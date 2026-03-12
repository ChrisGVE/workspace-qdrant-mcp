//! Platform-specific file watching optimizations
//!
//! This module provides platform-specific implementations of file watching
//! to leverage native file system events for better performance.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio::sync::mpsc;

use super::{FileEvent, WatchingError};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
pub use linux::LinuxWatcher;
#[cfg(target_os = "macos")]
pub use macos::MacOSWatcher;
#[cfg(target_os = "windows")]
pub use windows::WindowsWatcher;

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
        use winapi::um::winnt::{
            FILE_NOTIFY_CHANGE_FILE_NAME, FILE_NOTIFY_CHANGE_LAST_WRITE, FILE_NOTIFY_CHANGE_SIZE,
        };

        Self {
            watch_subtree: true,
            buffer_size: 65536, // 64KB buffer
            filter_flags: FILE_NOTIFY_CHANGE_FILE_NAME
                | FILE_NOTIFY_CHANGE_SIZE
                | FILE_NOTIFY_CHANGE_LAST_WRITE,
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

    /// Take ownership of the event receiver.
    ///
    /// Must be called before starting the watcher. Returns `Some` on the first
    /// call and `None` on subsequent calls (the receiver can only be taken once).
    fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<FileEvent>>;
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
