//! File Watching with SQLite Queue Integration
//!
//! This module provides Rust-based file watching that writes directly to the
//! unified_queue SQLite table per WORKSPACE_QDRANT_MCP.md specification.

mod types;
mod file_watcher;
mod file_watcher_ops;
mod error_types;
mod error_state;
mod throttle;
mod coordinator;
mod manager;
mod manager_ops;

#[cfg(test)]
mod tests;

// Re-export all public items at the same path they had before
pub use types::*;
pub use file_watcher::*;
pub use error_types::*;
pub use error_state::*;
pub use throttle::*;
pub use coordinator::*;
pub use manager::*;
