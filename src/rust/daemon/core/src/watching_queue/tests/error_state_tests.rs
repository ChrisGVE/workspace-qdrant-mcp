//! Tests for WatchErrorState and WatchErrorTracker (Task 461).

use super::super::*;

#[test]
fn test_watch_error_state_new() {
    let state = WatchErrorState::new();
    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.total_errors, 0);
    assert_eq!(state.backoff_level, 0);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.last_error_time.is_none());
    assert!(state.can_process());
}

#[test]
fn test_watch_error_state_record_error() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // First error - should remain healthy
    let delay = state.record_error("test error 1", &config);
    assert_eq!(state.consecutive_errors, 1);
    assert_eq!(state.total_errors, 1);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert_eq!(delay, 0); // No backoff yet

    // Third error - should become degraded
    state.record_error("test error 2", &config);
    let delay = state.record_error("test error 3", &config);
    assert_eq!(state.consecutive_errors, 3);
    assert_eq!(state.health_status, WatchHealthStatus::Degraded);
    assert_eq!(delay, 0); // Degraded but no backoff yet
}

#[test]
fn test_watch_error_state_backoff_threshold() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record errors up to backoff threshold
    for i in 1..=5 {
        let delay = state.record_error(&format!("error {}", i), &config);
        if i >= 5 {
            // Should be in backoff now
            assert_eq!(state.health_status, WatchHealthStatus::Backoff);
            assert!(delay > 0, "Should have backoff delay");
        }
    }
}

#[test]
fn test_watch_error_state_disable_threshold() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record errors up to disable threshold (Task 461.15: threshold is now 20)
    for _ in 0..20 {
        state.record_error("repeated error", &config);
    }

    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert!(state.should_disable());
    assert!(!state.can_process());
}

#[test]
fn test_watch_error_state_record_success() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Get into degraded state
    for _ in 0..3 {
        state.record_error("error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Degraded);

    // Record successes to recover
    for _ in 0..3 {
        state.record_success(&config);
    }

    // Should be fully reset after success_reset_count successes
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.backoff_level, 0);
}

#[test]
fn test_watch_error_state_reset() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record some errors
    for _ in 0..5 {
        state.record_error("error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Backoff);

    // Reset
    state.reset();

    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.backoff_level, 0);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    // Total errors should still be tracked
    assert_eq!(state.total_errors, 5);
}

#[test]
fn test_backoff_delay_calculation() {
    let config = BackoffConfig {
        base_delay_ms: 1000,
        max_delay_ms: 60_000,
        ..BackoffConfig::default()
    };
    let mut state = WatchErrorState::new();

    // Level 0 - no delay
    assert_eq!(state.calculate_backoff_delay(&config), 0);

    // Level 1 - base delay (~1000ms with jitter)
    state.backoff_level = 1;
    let delay1 = state.calculate_backoff_delay(&config);
    assert!(
        delay1 >= 900 && delay1 <= 1100,
        "Level 1 delay should be ~1000ms, got {}",
        delay1
    );

    // Level 2 - 2x base delay (~2000ms with jitter)
    state.backoff_level = 2;
    let delay2 = state.calculate_backoff_delay(&config);
    assert!(
        delay2 >= 1800 && delay2 <= 2200,
        "Level 2 delay should be ~2000ms, got {}",
        delay2
    );

    // Level 3 - 4x base delay (~4000ms with jitter)
    state.backoff_level = 3;
    let delay3 = state.calculate_backoff_delay(&config);
    assert!(
        delay3 >= 3600 && delay3 <= 4400,
        "Level 3 delay should be ~4000ms, got {}",
        delay3
    );
}

#[test]
fn test_backoff_delay_max_cap() {
    let config = BackoffConfig {
        base_delay_ms: 1000,
        max_delay_ms: 5000,
        ..BackoffConfig::default()
    };
    let mut state = WatchErrorState::new();

    // Very high level should be capped
    state.backoff_level = 20;
    let delay = state.calculate_backoff_delay(&config);
    assert!(
        delay <= 5500,
        "Delay should be capped at max_delay + jitter, got {}",
        delay
    );
}

#[test]
fn test_watch_health_status_as_str() {
    assert_eq!(WatchHealthStatus::Healthy.as_str(), "healthy");
    assert_eq!(WatchHealthStatus::Degraded.as_str(), "degraded");
    assert_eq!(WatchHealthStatus::Backoff.as_str(), "backoff");
    assert_eq!(WatchHealthStatus::Disabled.as_str(), "disabled");
}

#[test]
fn test_backoff_config_default() {
    let config = BackoffConfig::default();
    assert_eq!(config.base_delay_ms, 1000);
    assert_eq!(config.max_delay_ms, 300_000);
    assert_eq!(config.degraded_threshold, 3);
    assert_eq!(config.backoff_threshold, 5);
    assert_eq!(config.disable_threshold, 20); // Task 461.15: updated threshold
    assert_eq!(config.success_reset_count, 3);
    // Circuit breaker settings (Task 461.15)
    assert_eq!(config.window_error_threshold, 50);
    assert_eq!(config.window_duration_secs, 3600);
    assert_eq!(config.cooldown_secs, 3600);
    assert_eq!(config.half_open_success_threshold, 3);
}

#[tokio::test]
async fn test_watch_error_tracker_basic() {
    let tracker = WatchErrorTracker::new();

    // Record error
    let delay = tracker.record_error("watch-1", "test error").await;
    assert_eq!(delay, 0); // First error, no backoff

    // Check status
    let status = tracker.get_health_status("watch-1").await;
    assert_eq!(status, WatchHealthStatus::Healthy);

    // Record success
    tracker.record_success("watch-1").await;

    // Should still be able to process
    assert!(tracker.can_process("watch-1").await);
}

#[tokio::test]
async fn test_watch_error_tracker_multiple_watches() {
    let tracker = WatchErrorTracker::new();

    // Record errors for multiple watches
    for _ in 0..5 {
        tracker.record_error("watch-bad", "error").await;
    }
    tracker.record_error("watch-good", "single error").await;

    // Check different states
    let bad_status = tracker.get_health_status("watch-bad").await;
    let good_status = tracker.get_health_status("watch-good").await;

    assert_eq!(bad_status, WatchHealthStatus::Backoff);
    assert_eq!(good_status, WatchHealthStatus::Healthy);
}

#[tokio::test]
async fn test_watch_error_tracker_get_error_summary() {
    let tracker = WatchErrorTracker::new();

    tracker.record_error("watch-1", "error 1").await;
    tracker.record_error("watch-2", "error 2").await;
    tracker.record_error("watch-2", "error 3").await;

    let summary = tracker.get_error_summary().await;
    assert_eq!(summary.len(), 2);

    let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1").unwrap();
    assert_eq!(watch1_summary.consecutive_errors, 1);

    let watch2_summary = summary.iter().find(|s| s.watch_id == "watch-2").unwrap();
    assert_eq!(watch2_summary.consecutive_errors, 2);
}

#[tokio::test]
async fn test_watch_error_tracker_reset_watch() {
    let tracker = WatchErrorTracker::new();

    // Get into bad state (Task 461.15: threshold is now 20)
    for _ in 0..20 {
        tracker.record_error("watch-1", "error").await;
    }
    assert_eq!(
        tracker.get_health_status("watch-1").await,
        WatchHealthStatus::Disabled
    );

    // Reset
    tracker.reset_watch("watch-1").await;

    // Should be healthy again
    assert_eq!(
        tracker.get_health_status("watch-1").await,
        WatchHealthStatus::Healthy
    );
    assert!(tracker.can_process("watch-1").await);
}

#[tokio::test]
async fn test_watch_error_tracker_remove_watch() {
    let tracker = WatchErrorTracker::new();

    tracker.record_error("watch-1", "error").await;
    assert_eq!(tracker.get_error_summary().await.len(), 1);

    tracker.remove_watch("watch-1").await;
    assert_eq!(tracker.get_error_summary().await.len(), 0);
}
