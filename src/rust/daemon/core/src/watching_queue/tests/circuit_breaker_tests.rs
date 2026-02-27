//! Circuit breaker state transition tests (Task 461.15).

use super::super::*;

#[test]
fn test_circuit_breaker_config_defaults() {
    let config = BackoffConfig::default();
    assert_eq!(config.disable_threshold, 20);
    assert_eq!(config.window_error_threshold, 50);
    assert_eq!(config.window_duration_secs, 3600);
    assert_eq!(config.cooldown_secs, 3600);
    assert_eq!(config.half_open_success_threshold, 3);
}

#[test]
fn test_circuit_breaker_opens_on_consecutive_errors() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record 19 errors - should not open circuit yet
    for _ in 0..19 {
        state.record_error("test error", &config);
    }
    assert_ne!(state.health_status, WatchHealthStatus::Disabled);

    // 20th error should open circuit
    state.record_error("test error", &config);
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert!(state.circuit_opened_at.is_some());
}

#[test]
fn test_circuit_breaker_state_info() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Initially closed
    let circuit_state = state.get_circuit_state();
    assert!(!circuit_state.is_open);
    assert!(!circuit_state.is_half_open);

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }

    let circuit_state = state.get_circuit_state();
    assert!(circuit_state.is_open);
    assert!(!circuit_state.is_half_open);
    assert!(circuit_state.opened_at.is_some());
    assert_eq!(circuit_state.errors_in_window, 20);
}

#[test]
fn test_half_open_state_transition() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0; // Immediate cooldown for testing
    let mut state = WatchErrorState::new();

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);

    // Should transition to half-open after cooldown
    assert!(state.should_attempt_half_open(&config));
    state.transition_to_half_open();
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

    let circuit_state = state.get_circuit_state();
    assert!(!circuit_state.is_open);
    assert!(circuit_state.is_half_open);
}

#[test]
fn test_half_open_error_reopens_circuit() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0;
    let mut state = WatchErrorState::new();

    // Open and transition to half-open
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    state.transition_to_half_open();
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

    // Error in half-open should reopen circuit
    state.record_error("retry failed", &config);
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert_eq!(state.half_open_attempts, 1);
}

#[test]
fn test_half_open_success_closes_circuit() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0;
    config.half_open_success_threshold = 2;
    let mut state = WatchErrorState::new();

    // Open and transition to half-open
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    state.transition_to_half_open();

    // First success - still half-open
    let changed = state.record_success(&config);
    assert!(!changed);
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);
    assert_eq!(state.half_open_successes, 1);

    // Second success - closes circuit
    let changed = state.record_success(&config);
    assert!(changed);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.circuit_opened_at.is_none());
}

#[test]
fn test_manual_circuit_reset() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);

    // Manual reset
    state.manual_circuit_reset();
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.circuit_opened_at.is_none());
    assert_eq!(state.consecutive_errors, 0);
    assert!(state.errors_in_window.is_empty());
}

#[test]
fn test_errors_in_window_tracking() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record some errors
    for _ in 0..10 {
        state.record_error("test error", &config);
    }

    let circuit_state = state.get_circuit_state();
    assert_eq!(circuit_state.errors_in_window, 10);
    assert_eq!(state.errors_in_window.len(), 10);
}

#[test]
fn test_watch_health_status_half_open_as_str() {
    assert_eq!(WatchHealthStatus::HalfOpen.as_str(), "half_open");
}
