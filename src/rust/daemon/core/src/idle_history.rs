//! Idle/Active State Transition History
//!
//! Tracks state transitions in `~/.workspace-qdrant/idle_history.jsonl` for
//! adaptive flip-flop detection. When the system detects frequent flip-flopping
//! (e.g., >10 transitions/hour during genuinely idle periods), it can
//! recommend increasing the cooling-off period.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, warn};
use wqm_common::paths;

use crate::adaptive_resources::ResourceMode;

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

/// Manages idle state transition history.
pub struct IdleHistory {
    /// Path to the JSONL history file
    path: PathBuf,
    /// Maximum age of entries to keep (rotation)
    max_age: Duration,
    /// Flip-flop detection threshold (transitions per hour)
    flip_flop_threshold: f64,
}

impl IdleHistory {
    /// Create a new IdleHistory with default settings.
    ///
    /// File: `~/.workspace-qdrant/idle_history.jsonl`
    /// Rotation: 7 days
    /// Flip-flop threshold: 10 transitions/hour
    pub fn new() -> Option<Self> {
        let config_dir = paths::get_config_dir().ok()?;
        Some(Self {
            path: config_dir.join("idle_history.jsonl"),
            max_age: Duration::from_secs(7 * 24 * 3600), // 7 days
            flip_flop_threshold: 10.0,
        })
    }

    /// Create with a custom path (for testing).
    #[cfg(test)]
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            path,
            max_age: Duration::from_secs(7 * 24 * 3600),
            flip_flop_threshold: 10.0,
        }
    }

    /// Record a state transition.
    pub fn record_transition(
        &self,
        from_mode: ResourceMode,
        to_mode: ResourceMode,
        idle_seconds: f64,
        duration_in_previous: Duration,
    ) {
        let entry = StateTransition {
            timestamp: wqm_common::timestamps::now_utc(),
            from_mode: from_mode.as_str().to_string(),
            to_mode: to_mode.as_str().to_string(),
            idle_seconds,
            duration_in_previous_secs: duration_in_previous.as_secs_f64(),
        };

        if let Err(e) = self.append_entry(&entry) {
            warn!("Failed to write idle history entry: {}", e);
        }
    }

    /// Append a single entry to the JSONL file.
    fn append_entry(&self, entry: &StateTransition) -> std::io::Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let json = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// Read recent transitions (last `window` duration).
    pub fn read_recent(&self, window: Duration) -> Vec<StateTransition> {
        let cutoff = Utc::now() - chrono::Duration::from_std(window).unwrap_or_default();
        self.read_since(&cutoff)
    }

    /// Read transitions since a specific time.
    fn read_since(&self, since: &DateTime<Utc>) -> Vec<StateTransition> {
        let file = match fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let reader = BufReader::new(file);
        let since_str = wqm_common::timestamps::format_utc(since);

        reader
            .lines()
            .filter_map(|line| line.ok())
            .filter_map(|line| serde_json::from_str::<StateTransition>(&line).ok())
            .filter(|entry| entry.timestamp >= since_str)
            .collect()
    }

    /// Analyze flip-flop patterns in the last N hours.
    pub fn analyze_flip_flops(&self, hours: f64) -> FlipFlopAnalysis {
        let window = Duration::from_secs_f64(hours * 3600.0);
        let entries = self.read_recent(window);

        let transition_count = entries.len();
        let transitions_per_hour = if hours > 0.0 {
            transition_count as f64 / hours
        } else {
            0.0
        };

        let avg_mode_duration_secs = if transition_count > 0 {
            entries.iter().map(|e| e.duration_in_previous_secs).sum::<f64>()
                / transition_count as f64
        } else {
            0.0
        };

        let short_transitions = entries
            .iter()
            .filter(|e| e.duration_in_previous_secs < 30.0)
            .count();

        let is_flip_flopping = transitions_per_hour > self.flip_flop_threshold;

        // If flip-flopping, recommend increasing cooloff by the ratio
        // of actual rate to threshold. E.g., 20 transitions/hr with threshold 10
        // → recommend +2 polls
        let recommended_cooloff_increase = if is_flip_flopping {
            ((transitions_per_hour / self.flip_flop_threshold).ceil() as u32).saturating_sub(1)
        } else {
            0
        };

        FlipFlopAnalysis {
            transition_count,
            transitions_per_hour,
            avg_mode_duration_secs,
            short_transitions,
            is_flip_flopping,
            recommended_cooloff_increase,
        }
    }

    /// Rotate the history file, removing entries older than max_age.
    pub fn rotate(&self) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(self.max_age).unwrap_or_default();
        let entries = self.read_since(&cutoff);

        if entries.is_empty() {
            // Remove the file entirely if no recent entries
            let _ = fs::remove_file(&self.path);
            return;
        }

        // Rewrite with only recent entries
        match fs::File::create(&self.path) {
            Ok(mut file) => {
                for entry in &entries {
                    if let Ok(json) = serde_json::to_string(entry) {
                        let _ = writeln!(file, "{}", json);
                    }
                }
                debug!("Rotated idle history: kept {} entries", entries.len());
            }
            Err(e) => {
                warn!("Failed to rotate idle history: {}", e);
            }
        }
    }

    /// Get a human-readable summary of recent idle patterns.
    pub fn summary(&self) -> String {
        let analysis_1h = self.analyze_flip_flops(1.0);
        let analysis_24h = self.analyze_flip_flops(24.0);

        let mut lines = Vec::new();
        lines.push(format!("Idle History Summary"));
        lines.push(format!("  Last 1 hour:"));
        lines.push(format!(
            "    Transitions: {} ({:.1}/hr)",
            analysis_1h.transition_count, analysis_1h.transitions_per_hour
        ));
        lines.push(format!(
            "    Avg mode duration: {:.1}s",
            analysis_1h.avg_mode_duration_secs
        ));
        lines.push(format!(
            "    Short transitions (<30s): {}",
            analysis_1h.short_transitions
        ));
        if analysis_1h.is_flip_flopping {
            lines.push(format!(
                "    WARNING: flip-flopping detected (recommend +{} cooloff polls)",
                analysis_1h.recommended_cooloff_increase
            ));
        }

        lines.push(format!("  Last 24 hours:"));
        lines.push(format!(
            "    Transitions: {} ({:.1}/hr)",
            analysis_24h.transition_count, analysis_24h.transitions_per_hour
        ));
        lines.push(format!(
            "    Avg mode duration: {:.1}s",
            analysis_24h.avg_mode_duration_secs
        ));
        lines.push(format!(
            "    Short transitions (<30s): {}",
            analysis_24h.short_transitions
        ));
        if analysis_24h.is_flip_flopping {
            lines.push(format!(
                "    WARNING: flip-flopping detected (recommend +{} cooloff polls)",
                analysis_24h.recommended_cooloff_increase
            ));
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Integration with adaptive_resources polling loop
// ---------------------------------------------------------------------------

/// Track mode changes and record transitions.
///
/// Call this from the adaptive resource polling loop whenever mode changes.
pub struct ModeTracker {
    history: Option<IdleHistory>,
    last_mode: ResourceMode,
    last_change: std::time::Instant,
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
            last_mode: ResourceMode::Normal,
            last_change: std::time::Instant::now(),
        }
    }

    /// Record a mode change if the mode actually changed.
    /// Returns true if a transition was recorded.
    pub fn on_mode_change(&mut self, new_mode: ResourceMode, idle_seconds: f64) -> bool {
        if new_mode == self.last_mode {
            return false;
        }

        let duration_in_previous = self.last_change.elapsed();

        if let Some(ref history) = self.history {
            history.record_transition(
                self.last_mode,
                new_mode,
                idle_seconds,
                duration_in_previous,
            );
        }

        self.last_mode = new_mode;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_test_history() -> (IdleHistory, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idle_history.jsonl");
        let history = IdleHistory::with_path(path);
        (history, dir)
    }

    #[test]
    fn test_record_and_read() {
        let (history, _dir) = make_test_history();

        history.record_transition(
            ResourceMode::Normal,
            ResourceMode::RampingUp(1),
            125.0,
            Duration::from_secs(300),
        );

        history.record_transition(
            ResourceMode::RampingUp(1),
            ResourceMode::Burst,
            200.0,
            Duration::from_secs(60),
        );

        let entries = history.read_recent(Duration::from_secs(3600));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].from_mode, "normal");
        assert_eq!(entries[0].to_mode, "elevated");
        assert!((entries[0].idle_seconds - 125.0).abs() < 0.01);
        assert!((entries[0].duration_in_previous_secs - 300.0).abs() < 0.01);
        assert_eq!(entries[1].from_mode, "elevated");
        assert_eq!(entries[1].to_mode, "burst");
    }

    #[test]
    fn test_flip_flop_detection() {
        let (history, _dir) = make_test_history();

        // Simulate rapid flip-flopping: 15 transitions in quick succession
        for i in 0..15 {
            let from = if i % 2 == 0 {
                ResourceMode::Burst
            } else {
                ResourceMode::Normal
            };
            let to = if i % 2 == 0 {
                ResourceMode::Normal
            } else {
                ResourceMode::Burst
            };
            history.record_transition(from, to, 5.0, Duration::from_secs(10));
        }

        let analysis = history.analyze_flip_flops(1.0);
        assert_eq!(analysis.transition_count, 15);
        assert!(analysis.transitions_per_hour > 10.0);
        assert!(analysis.is_flip_flopping);
        assert!(analysis.recommended_cooloff_increase > 0);
        assert_eq!(analysis.short_transitions, 15); // all < 30s
    }

    #[test]
    fn test_no_flip_flop_normal_usage() {
        let (history, _dir) = make_test_history();

        // Normal usage: 2 transitions (idle → burst → back to normal)
        history.record_transition(
            ResourceMode::Normal,
            ResourceMode::Burst,
            200.0,
            Duration::from_secs(7200), // 2 hours in normal
        );
        history.record_transition(
            ResourceMode::Burst,
            ResourceMode::Normal,
            5.0,
            Duration::from_secs(3600), // 1 hour in burst
        );

        let analysis = history.analyze_flip_flops(4.0);
        assert_eq!(analysis.transition_count, 2);
        assert!(!analysis.is_flip_flopping);
        assert_eq!(analysis.recommended_cooloff_increase, 0);
        assert_eq!(analysis.short_transitions, 0); // both > 30s
    }

    #[test]
    fn test_rotate_removes_old_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idle_history.jsonl");
        let history = IdleHistory {
            path: path.clone(),
            max_age: Duration::from_secs(3600), // 1 hour for test
            flip_flop_threshold: 10.0,
        };

        // Write an "old" entry by manually writing JSON with old timestamp
        let old_entry = StateTransition {
            timestamp: "2020-01-01T00:00:00.000Z".to_string(),
            from_mode: "normal".to_string(),
            to_mode: "burst".to_string(),
            idle_seconds: 200.0,
            duration_in_previous_secs: 3600.0,
        };
        history.append_entry(&old_entry).unwrap();

        // Write a recent entry
        history.record_transition(
            ResourceMode::Normal,
            ResourceMode::Burst,
            200.0,
            Duration::from_secs(60),
        );

        // Before rotation, 2 entries
        let all = history.read_since(
            &DateTime::parse_from_rfc3339("2019-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert_eq!(all.len(), 2);

        // After rotation, only the recent entry remains
        history.rotate();
        let all = history.read_since(
            &DateTime::parse_from_rfc3339("2019-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert_eq!(all.len(), 1);
        assert_ne!(all[0].timestamp, "2020-01-01T00:00:00.000Z");
    }

    #[test]
    fn test_mode_tracker() {
        let mut tracker = ModeTracker {
            history: None, // No file I/O for this test
            last_mode: ResourceMode::Normal,
            last_change: std::time::Instant::now(),
        };

        // Same mode → no transition
        assert!(!tracker.on_mode_change(ResourceMode::Normal, 5.0));

        // Different mode → transition
        assert!(tracker.on_mode_change(ResourceMode::Burst, 200.0));
        assert_eq!(tracker.last_mode, ResourceMode::Burst);

        // Same mode again → no transition
        assert!(!tracker.on_mode_change(ResourceMode::Burst, 300.0));
    }

    #[test]
    fn test_empty_history_analysis() {
        let (history, _dir) = make_test_history();
        let analysis = history.analyze_flip_flops(1.0);
        assert_eq!(analysis.transition_count, 0);
        assert!(!analysis.is_flip_flopping);
        assert_eq!(analysis.recommended_cooloff_increase, 0);
    }

    #[test]
    fn test_summary_format() {
        let (history, _dir) = make_test_history();
        let summary = history.summary();
        assert!(summary.contains("Idle History Summary"));
        assert!(summary.contains("Last 1 hour"));
        assert!(summary.contains("Last 24 hours"));
    }
}
