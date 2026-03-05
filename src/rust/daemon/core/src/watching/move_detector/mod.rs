//! Move detection and rename correlation for file watching.
//!
//! This module provides functionality to correlate rename events and detect
//! folder moves within the filesystem. It handles:
//! - Intra-filesystem moves (MOVED_FROM + MOVED_TO correlation)
//! - Cross-filesystem moves (detected as delete when MOVED_TO times out)
//! - Root folder moves (MOVE_SELF or RENAME events)

mod correlator;
mod types;

#[cfg(test)]
mod tests;

pub use correlator::MoveCorrelator;
pub use types::{MoveCorrelatorConfig, MoveCorrelatorStats, MoveDetectorError, RenameAction};
