//! Canonical data queries — single source of truth for CLI metrics.
//!
//! Grouped by entity domain. Every metric displayed by multiple commands
//! must be queried here to ensure consistency.

mod documents;
mod libraries;
mod projects;
mod queue;

pub use documents::*;
pub use libraries::*;
pub use projects::*;
pub use queue::*;
