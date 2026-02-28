//! File watching system
//!
//! Cross-platform file watching with event debouncing, pattern matching, and priority-based processing.
//! Integrates with the task pipeline system for responsive file processing.

use glob::PatternError;
use thiserror::Error;

pub mod config;
mod compiled_patterns;
mod debouncer;
pub mod events;
pub mod telemetry;
pub mod watcher;
mod watcher_processing;

pub mod platform;
pub mod move_detector;
pub mod path_validator;
pub mod file_watcher;

// Re-export primary types
pub use config::{WatcherConfig, TelemetryConfig, BatchConfig};
pub use events::{FileEvent, PausedEventBuffer};
pub use telemetry::{TelemetrySnapshot, WatchingStats};
pub use watcher::FileWatcher;

pub use platform::{PlatformWatcherConfig, PlatformWatcherFactory, PlatformWatchingStats};
pub use move_detector::{
    MoveCorrelator, MoveCorrelatorConfig, MoveCorrelatorStats,
    MoveDetectorError, RenameAction,
};
pub use path_validator::{
    PathValidator, PathValidatorConfig, PathValidatorStats, PathValidatorError,
    OrphanedProject, RegisteredProject, OrphanCleanupActions,
};
pub use file_watcher::{
    EnhancedFileWatcher, EnhancedWatcherConfig, EnhancedWatcherError,
    EnhancedWatcherStats, WatcherHandle, WatchEvent,
};

/// Errors that can occur during file watching
#[derive(Error, Debug)]
pub enum WatchingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Pattern compilation error: {0}")]
    Pattern(#[from] PatternError),

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Task submission error: {0}")]
    TaskSubmission(String),
}

#[cfg(test)]
mod tests;
