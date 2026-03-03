//! Types for idle/active state transition history.

use serde::{Deserialize, Serialize};

/// A single state transition record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// ISO 8601 timestamp of the transition
    pub timestamp: String,
    /// Mode we transitioned FROM
    pub from_mode: String,
    /// Mode we transitioned TO
    pub to_mode: String,
    /// Seconds since last user input at time of transition
    pub idle_seconds: f64,
    /// How long we spent in the previous mode (seconds)
    pub duration_in_previous_secs: f64,
}

/// Flip-flop analysis result for a time window.
#[derive(Debug, Clone)]
pub struct FlipFlopAnalysis {
    /// Number of transitions in the analysis window
    pub transition_count: usize,
    /// Transitions per hour
    pub transitions_per_hour: f64,
    /// Average duration in each mode before flipping (seconds)
    pub avg_mode_duration_secs: f64,
    /// Number of "short" transitions (< 30 seconds in a mode)
    pub short_transitions: usize,
    /// Whether flip-flop rate exceeds threshold
    pub is_flip_flopping: bool,
    /// Recommended cooloff_polls increase (0 if not needed)
    pub recommended_cooloff_increase: u32,
}
