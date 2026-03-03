//! Idle/Active State Transition History
//!
//! Tracks state transitions in `~/.workspace-qdrant/idle_history.jsonl` for
//! adaptive flip-flop detection. When the system detects frequent flip-flopping
//! (e.g., >10 transitions/hour during genuinely idle periods), it can
//! recommend increasing the cooling-off period.

pub mod history;
pub mod tracker;
pub mod types;

mod tests;

// Re-export all public items to preserve the original module API.
pub use history::IdleHistory;
pub use tracker::ModeTracker;
pub use types::{FlipFlopAnalysis, StateTransition};
