//! Tests for idle history tracking.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::Utc;
    use tempfile::tempdir;

    use crate::adaptive_resources::ResourceLevel;

    use crate::idle_history::{history::IdleHistory, tracker::ModeTracker, types::StateTransition};

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
            ResourceLevel::Normal,
            ResourceLevel::Elevated,
            125.0,
            Duration::from_secs(300),
        );

        history.record_transition(
            ResourceLevel::Elevated,
            ResourceLevel::Burst,
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
                ResourceLevel::Burst
            } else {
                ResourceLevel::Normal
            };
            let to = if i % 2 == 0 {
                ResourceLevel::Normal
            } else {
                ResourceLevel::Burst
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
            ResourceLevel::Normal,
            ResourceLevel::Burst,
            200.0,
            Duration::from_secs(7200), // 2 hours in normal
        );
        history.record_transition(
            ResourceLevel::Burst,
            ResourceLevel::Normal,
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
            ResourceLevel::Normal,
            ResourceLevel::Burst,
            200.0,
            Duration::from_secs(60),
        );

        // Before rotation, 2 entries
        let all = history.read_since(
            &chrono::DateTime::parse_from_rfc3339("2019-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert_eq!(all.len(), 2);

        // After rotation, only the recent entry remains
        history.rotate();
        let all = history.read_since(
            &chrono::DateTime::parse_from_rfc3339("2019-01-01T00:00:00Z")
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
            last_level: ResourceLevel::Normal,
            last_change: std::time::Instant::now(),
        };

        // Same level → no transition
        assert!(!tracker.on_mode_change(ResourceLevel::Normal, 5.0));

        // Different level → transition
        assert!(tracker.on_mode_change(ResourceLevel::Burst, 200.0));
        assert_eq!(tracker.last_level, ResourceLevel::Burst);

        // Same level again → no transition
        assert!(!tracker.on_mode_change(ResourceLevel::Burst, 300.0));
    }

    #[test]
    fn test_mode_tracker_full_descent_no_skip() {
        // Reproduce the exact bug scenario:
        // last_level starts at Active (from Active Processing),
        // then system ramps up to Burst, then descends.
        // All intermediate transitions must be recorded.
        let mut tracker = ModeTracker {
            history: None,
            last_level: ResourceLevel::Active,
            last_change: std::time::Instant::now(),
        };

        // Ramp up: Active → Burst
        assert!(tracker.on_mode_change(ResourceLevel::Burst, 300.0));
        assert_eq!(tracker.last_level, ResourceLevel::Burst);

        // Ramp down: Burst → Elevated
        assert!(tracker.on_mode_change(ResourceLevel::Elevated, 5.0));
        assert_eq!(tracker.last_level, ResourceLevel::Elevated);

        // Ramp down: Elevated → Active (this was previously skipped
        // because ResourceMode::Active == ResourceMode::Active)
        assert!(tracker.on_mode_change(ResourceLevel::Active, 5.0));
        assert_eq!(tracker.last_level, ResourceLevel::Active);

        // Ramp down: Active → Normal
        assert!(tracker.on_mode_change(ResourceLevel::Normal, 5.0));
        assert_eq!(tracker.last_level, ResourceLevel::Normal);
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
