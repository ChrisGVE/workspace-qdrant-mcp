//! ModeTracker — integrates IdleHistory into the adaptive_resources polling loop.

use tracing::info;

use crate::adaptive_resources::ResourceLevel;

use super::history::IdleHistory;
use super::types::FlipFlopAnalysis;

/// Track mode changes and record transitions.
///
/// Call this from the adaptive resource polling loop whenever mode changes.
pub struct ModeTracker {
    pub(super) history: Option<IdleHistory>,
    pub(super) last_level: ResourceLevel,
    pub(super) last_change: std::time::Instant,
}

impl ModeTracker {
    /// Create a new mode tracker.
    pub fn new() -> Self {
        let history = IdleHistory::new();
        if history.is_some() {
            info!("Idle history tracking enabled");
        }
        Self {
            history,
            last_level: ResourceLevel::Normal,
            last_change: std::time::Instant::now(),
        }
    }

    /// Record a level change if the level actually changed.
    /// Returns true if a transition was recorded.
    ///
    /// Tracks `ResourceLevel` directly (not `ResourceMode`) to avoid
    /// ambiguity between Active Processing and Active ramp-down level.
    pub fn on_mode_change(&mut self, new_level: ResourceLevel, idle_seconds: f64) -> bool {
        if new_level == self.last_level {
            return false;
        }

        let duration_in_previous = self.last_change.elapsed();

        if let Some(ref history) = self.history {
            history.record_transition(
                self.last_level,
                new_level,
                idle_seconds,
                duration_in_previous,
            );
        }

        self.last_level = new_level;
        self.last_change = std::time::Instant::now();
        true
    }

    /// Rotate old history entries. Call periodically (e.g., once per hour).
    pub fn rotate(&self) {
        if let Some(ref history) = self.history {
            history.rotate();
        }
    }

    /// Get flip-flop analysis for the last N hours.
    pub fn analyze(&self, hours: f64) -> Option<FlipFlopAnalysis> {
        self.history.as_ref().map(|h| h.analyze_flip_flops(hours))
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> Option<String> {
        self.history.as_ref().map(|h| h.summary())
    }
}
