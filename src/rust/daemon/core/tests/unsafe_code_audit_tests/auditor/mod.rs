//! Core audit engine for unsafe code validation
//!
//! Submodules:
//! - `engine`   -- `UnsafeCodeAuditor` and all audit logic
//! - `trackers` -- `MemoryTracker` and `ConcurrencyTracker` helper types

mod engine;
pub(crate) mod trackers;

pub use engine::UnsafeCodeAuditor;
// Re-export tracker types so sibling modules can use `super::auditor::X` paths.
pub(crate) use trackers::{AccessType, ConcurrencyTracker, MemoryTracker};
