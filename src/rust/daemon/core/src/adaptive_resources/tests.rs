use std::time::Duration;
use tokio_util::sync::CancellationToken;

use super::idle_detection::is_cpu_under_pressure;
use super::*;
use crate::config::ResourceLimitsConfig;

fn test_limits() -> ResourceLimitsConfig {
    ResourceLimitsConfig {
        max_concurrent_embeddings: 2,
        inter_item_delay_ms: 50,
        idle_threshold_secs: 120,
        burst_concurrency_multiplier: 2.0,
        burst_inter_item_delay_ms: 0,
        cpu_pressure_threshold: 0.6,
        idle_poll_interval_secs: 5,
        ..Default::default()
    }
}

// --- ResourceProfile tests ---

#[test]
fn test_resource_profile_equality() {
    let a = ResourceProfile {
        max_concurrent_embeddings: 2,
        inter_item_delay_ms: 50,
    };
    let b = ResourceProfile {
        max_concurrent_embeddings: 2,
        inter_item_delay_ms: 50,
    };
    let c = ResourceProfile {
        max_concurrent_embeddings: 4,
        inter_item_delay_ms: 0,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// --- ResourceLevel tests ---

#[test]
fn test_level_ordering() {
    assert!(ResourceLevel::Normal < ResourceLevel::Active);
    assert!(ResourceLevel::Active < ResourceLevel::Elevated);
    assert!(ResourceLevel::Elevated < ResourceLevel::Burst);
}

#[test]
fn test_level_up() {
    assert_eq!(ResourceLevel::Normal.up(), ResourceLevel::Active);
    assert_eq!(ResourceLevel::Active.up(), ResourceLevel::Elevated);
    assert_eq!(ResourceLevel::Elevated.up(), ResourceLevel::Burst);
    assert_eq!(ResourceLevel::Burst.up(), ResourceLevel::Burst); // capped
}

#[test]
fn test_level_down() {
    assert_eq!(ResourceLevel::Burst.down(), ResourceLevel::Elevated);
    assert_eq!(ResourceLevel::Elevated.down(), ResourceLevel::Active);
    assert_eq!(ResourceLevel::Active.down(), ResourceLevel::Normal);
    assert_eq!(ResourceLevel::Normal.down(), ResourceLevel::Normal); // floored
}

#[test]
fn test_level_as_u8() {
    assert_eq!(ResourceLevel::Normal.as_u8(), 0);
    assert_eq!(ResourceLevel::Active.as_u8(), 1);
    assert_eq!(ResourceLevel::Elevated.as_u8(), 2);
    assert_eq!(ResourceLevel::Burst.as_u8(), 3);
}

// --- ResourceMode tests ---

#[test]
fn test_resource_mode_as_str() {
    assert_eq!(ResourceMode::Normal.as_str(), "normal");
    assert_eq!(ResourceMode::Active.as_str(), "active");
    assert_eq!(ResourceMode::RampingUp(2).as_str(), "elevated");
    assert_eq!(ResourceMode::Burst.as_str(), "burst");
}

#[test]
fn test_resource_mode_from_level() {
    assert_eq!(
        ResourceMode::from(ResourceLevel::Normal),
        ResourceMode::Normal
    );
    assert_eq!(
        ResourceMode::from(ResourceLevel::Active),
        ResourceMode::Active
    );
    assert_eq!(
        ResourceMode::from(ResourceLevel::Elevated),
        ResourceMode::RampingUp(2)
    );
    assert_eq!(
        ResourceMode::from(ResourceLevel::Burst),
        ResourceMode::Burst
    );
}

// --- AdaptiveResourceConfig tests ---

#[test]
fn test_adaptive_config_from_resource_limits() {
    let limits = test_limits();
    let config = AdaptiveResourceConfig::from_resource_limits(&limits);
    assert_eq!(config.normal_max_concurrent_embeddings, 2);
    assert_eq!(config.burst_max_concurrent_embeddings, 4); // 2 * 2.0
    assert_eq!(config.normal_inter_item_delay_ms, 50);
    assert_eq!(config.burst_inter_item_delay_ms, 0);
    assert_eq!(config.idle_threshold_secs, 120);
    assert_eq!(config.idle_confirmation_secs, 300);
    assert_eq!(config.ramp_up_step_secs, 120);
    assert_eq!(config.ramp_down_step_secs, 300);
    assert_eq!(config.burst_hold_secs, 600);
}

#[test]
fn test_adaptive_config_burst_minimum() {
    let mut limits = test_limits();
    limits.max_concurrent_embeddings = 1;
    limits.burst_concurrency_multiplier = 1.0;
    let config = AdaptiveResourceConfig::from_resource_limits(&limits);
    // max(1+1, 1*1.0=1) = max(2, 1) = 2
    assert_eq!(config.burst_max_concurrent_embeddings, 2);
}

// --- SystemState tests ---

#[test]
fn test_system_state_new() {
    let state = SystemState::new();
    assert_eq!(state.level, ResourceLevel::Normal);
    assert!(state.idle_detected_at.is_none());
    assert!(state.activity_detected_at.is_none());
}

#[test]
fn test_system_state_transition() {
    let mut state = SystemState::new();
    let before = state.level_entered_at;
    std::thread::sleep(Duration::from_millis(10));
    state.transition_to(ResourceLevel::Active);
    assert_eq!(state.level, ResourceLevel::Active);
    assert!(state.level_entered_at > before);
}

// --- AdaptiveResourceState (atomic shared state) tests ---

#[test]
fn test_adaptive_resource_state() {
    let state = AdaptiveResourceState::new();
    assert_eq!(state.mode(), ResourceMode::Normal);
    assert!((state.idle_seconds() - 0.0).abs() < f64::EPSILON);

    state.set_mode(ResourceMode::RampingUp(2));
    assert_eq!(state.mode(), ResourceMode::RampingUp(2));

    state.set_mode(ResourceMode::Burst);
    assert_eq!(state.mode(), ResourceMode::Burst);

    state.set_idle_seconds(125.5);
    assert!((state.idle_seconds() - 125.5).abs() < 0.02);
}

#[test]
fn test_mode_encoding_all_variants() {
    let state = AdaptiveResourceState::new();

    state.set_mode(ResourceMode::Normal);
    assert_eq!(state.mode(), ResourceMode::Normal);

    state.set_mode(ResourceMode::Active);
    assert_eq!(state.mode(), ResourceMode::Active);

    state.set_mode(ResourceMode::RampingUp(2));
    assert_eq!(state.mode(), ResourceMode::RampingUp(2));

    state.set_mode(ResourceMode::Burst);
    assert_eq!(state.mode(), ResourceMode::Burst);
}

// --- Profile for level tests ---

#[test]
fn test_profile_for_level() {
    let normal = ResourceProfile {
        max_concurrent_embeddings: 2,
        inter_item_delay_ms: 50,
    };
    let active = ResourceProfile {
        max_concurrent_embeddings: 3,
        inter_item_delay_ms: 25,
    };
    let elevated = ResourceProfile {
        max_concurrent_embeddings: 3,
        inter_item_delay_ms: 12,
    };
    let burst = ResourceProfile {
        max_concurrent_embeddings: 4,
        inter_item_delay_ms: 0,
    };

    assert_eq!(
        profile_for_level(ResourceLevel::Normal, &normal, &active, &elevated, &burst),
        normal
    );
    assert_eq!(
        profile_for_level(ResourceLevel::Active, &normal, &active, &elevated, &burst),
        active
    );
    assert_eq!(
        profile_for_level(ResourceLevel::Elevated, &normal, &active, &elevated, &burst),
        elevated
    );
    assert_eq!(
        profile_for_level(ResourceLevel::Burst, &normal, &active, &elevated, &burst),
        burst
    );
}

// --- Manager lifecycle tests ---

#[tokio::test]
async fn test_adaptive_manager_starts_with_normal_profile() {
    let limits = test_limits();
    let config = AdaptiveResourceConfig::from_resource_limits(&limits);
    let token = CancellationToken::new();
    let manager = AdaptiveResourceManager::start(config, token.clone(), None);

    let profile = manager.current_profile();
    assert_eq!(profile.max_concurrent_embeddings, 2);
    assert_eq!(profile.inter_item_delay_ms, 50);
    assert_eq!(manager.state().mode(), ResourceMode::Normal);

    token.cancel();
}

#[tokio::test]
async fn test_adaptive_manager_subscribe() {
    let limits = test_limits();
    let config = AdaptiveResourceConfig::from_resource_limits(&limits);
    let token = CancellationToken::new();
    let manager = AdaptiveResourceManager::start(config, token.clone(), None);

    let rx = manager.subscribe();
    let profile = *rx.borrow();
    assert_eq!(profile.max_concurrent_embeddings, 2);

    token.cancel();
}

// --- Config defaults tests ---

#[test]
fn test_active_profile_values() {
    let limits = test_limits();
    let config = AdaptiveResourceConfig::from_resource_limits(&limits);

    let active_embeddings = std::cmp::max(
        config.normal_max_concurrent_embeddings + 1,
        (config.normal_max_concurrent_embeddings as f64 * config.active_concurrency_multiplier)
            .round() as usize,
    );
    assert_eq!(active_embeddings, 3, "2 * 1.5 = 3 embeddings");
    assert_eq!(config.active_inter_item_delay_ms, 25);
}

#[test]
fn test_active_config_defaults() {
    let limits = ResourceLimitsConfig::default();
    assert!((limits.active_concurrency_multiplier - 1.5).abs() < f64::EPSILON);
    assert_eq!(limits.active_inter_item_delay_ms, 25);
}

#[test]
fn test_new_config_defaults() {
    let limits = ResourceLimitsConfig::default();
    assert_eq!(limits.idle_confirmation_secs, 300);
    assert_eq!(limits.ramp_up_step_secs, 120);
    assert_eq!(limits.ramp_down_step_secs, 300);
    assert_eq!(limits.burst_hold_secs, 600);
}

// --- Timing tests ---

#[test]
fn test_heartbeat_interval_calculation() {
    let poll_secs: u64 = 5;
    let interval = 60 / poll_secs.max(1);
    assert_eq!(interval, 12);

    let poll_secs: u64 = 10;
    let interval = 60 / poll_secs.max(1);
    assert_eq!(interval, 6);

    let poll_secs: u64 = 0;
    let interval = 60 / poll_secs.max(1);
    assert_eq!(interval, 60);
}

#[test]
fn test_cpu_pressure_check() {
    // Use an absurdly high threshold that real load can never reach,
    // so the assertion is stable regardless of system load during test runs.
    assert!(!is_cpu_under_pressure(10_000.0, 1));
    // Threshold of 0.0 means any non-zero load triggers pressure.
    // On a running system, 1-min load average is always > 0.
    assert!(is_cpu_under_pressure(0.0, 1));
}

#[cfg(target_os = "macos")]
#[test]
fn test_macos_idle_detection_returns_value() {
    let secs = idle_detection::seconds_since_last_input();
    assert!(secs.is_some(), "macOS should always return idle time");
    assert!(secs.unwrap() >= 0.0, "Idle time should be non-negative");
}
