//! IdleHistory — JSONL file management for state transition records.

use chrono::{DateTime, Utc};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, warn};
use wqm_common::paths;

use crate::adaptive_resources::ResourceLevel;

use super::types::{FlipFlopAnalysis, StateTransition};

/// Manages idle state transition history.
pub struct IdleHistory {
    /// Path to the JSONL history file
    pub(super) path: PathBuf,
    /// Maximum age of entries to keep (rotation)
    pub(super) max_age: Duration,
    /// Flip-flop detection threshold (transitions per hour)
    pub(super) flip_flop_threshold: f64,
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
        from_level: ResourceLevel,
        to_level: ResourceLevel,
        idle_seconds: f64,
        duration_in_previous: Duration,
    ) {
        let entry = StateTransition {
            timestamp: wqm_common::timestamps::now_utc(),
            from_mode: from_level.as_str().to_string(),
            to_mode: to_level.as_str().to_string(),
            idle_seconds,
            duration_in_previous_secs: duration_in_previous.as_secs_f64(),
        };

        if let Err(e) = self.append_entry(&entry) {
            warn!("Failed to write idle history entry: {}", e);
        }
    }

    /// Append a single entry to the JSONL file.
    pub(super) fn append_entry(&self, entry: &StateTransition) -> std::io::Result<()> {
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
    pub(super) fn read_since(&self, since: &DateTime<Utc>) -> Vec<StateTransition> {
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
            entries
                .iter()
                .map(|e| e.duration_in_previous_secs)
                .sum::<f64>()
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
        let cutoff = Utc::now() - chrono::Duration::from_std(self.max_age).unwrap_or_default();
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
