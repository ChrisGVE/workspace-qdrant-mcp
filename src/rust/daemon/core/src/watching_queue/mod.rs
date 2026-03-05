//! File Watching with SQLite Queue Integration
//!
//! This module provides Rust-based file watching that writes directly to the
//! unified_queue SQLite table per docs/specs/04-write-path.md specification.

mod coordinator;
mod error_state;
mod error_types;
mod file_watcher;
mod file_watcher_ops;
mod manager;
mod manager_ops;
mod throttle;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items at the same path they had before
pub use coordinator::*;
pub use error_state::*;
pub use error_types::*;
pub use file_watcher::*;
pub use manager::*;
pub use throttle::*;
pub use types::*;
