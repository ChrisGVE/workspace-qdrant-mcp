//! SQLite read-only access to the daemon's `state.db`.
//!
//! This module provides:
//! - [`manager::StateManager`] — connection lifecycle and degraded-mode
//! - [`queue_stats`] — unified_queue statistics
//! - [`rules_mirror`] — rules_mirror reads
//! - [`scratchpad_mirror`] — scratchpad_mirror reads
//! - [`project_queries`] — watch_folders / instance queries
//! - [`tag_queries`] — tags / keyword_baskets / canonical_tags queries
//! - [`tracked_files`] — tracked_files / submodules / project_components queries

pub mod manager;
pub mod project_queries;
pub mod queue_stats;
pub mod rules_mirror;
pub mod scratchpad_mirror;
pub mod tag_queries;
pub mod tracked_files;

// Convenience re-exports of the most commonly used types.
pub use manager::{DegradedReason, QueryResult, QueryStatus, StateManager};
