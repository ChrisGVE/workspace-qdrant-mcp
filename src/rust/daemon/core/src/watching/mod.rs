//! File watching system
//!
//! Cross-platform file watching with event debouncing, pattern matching, and priority-based processing.
//! Integrates with the task pipeline system for responsive file processing.

use glob::PatternError;
use thiserror::Error;

mod compiled_patterns;
pub mod config;
mod debouncer;
pub mod events;
pub mod telemetry;
pub mod watcher;
mod watcher_processing;

pub mod file_watcher;
pub mod move_detector;
pub mod path_validator;
pub mod platform;

// Re-export primary types
pub use config::{BatchConfig, TelemetryConfig, WatcherConfig};
pub use events::{FileEvent, PausedEventBuffer};
pub use telemetry::{TelemetrySnapshot, WatchingStats};
pub use watcher::FileWatcher;

pub use file_watcher::{
    EnhancedFileWatcher, EnhancedWatcherConfig, EnhancedWatcherError, EnhancedWatcherStats,
    WatchEvent, WatcherHandle,
};
pub use move_detector::{
    MoveCorrelator, MoveCorrelatorConfig, MoveCorrelatorStats, MoveDetectorError, RenameAction,
};
pub use path_validator::{
    OrphanCleanupActions, OrphanedProject, PathValidator, PathValidatorConfig, PathValidatorError,
    PathValidatorStats, RegisteredProject,
};
pub use platform::{PlatformWatcherConfig, PlatformWatcherFactory, PlatformWatchingStats};

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
