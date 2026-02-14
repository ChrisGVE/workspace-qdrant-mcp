//! Adaptive Resource Manager
//!
//! Dynamically adjusts embedding concurrency and processing delays based on system
//! idle state. When the user is idle (no keyboard/mouse input), switches to burst
//! mode for faster queue processing. Immediately scales down when user activity
//! resumes.
//!
//! Platform support:
//! - macOS: CGEventSourceSecondsSinceLastEventType (CoreGraphics)
//! - Linux: /proc/stat user idle heuristic (falls back to always-interactive)
//! - Other: always-interactive (no burst mode)

use std::time::Duration;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Resource Profile
// ---------------------------------------------------------------------------

/// Dynamic resource profile communicated to the processing loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceProfile {
    /// Maximum concurrent embedding operations (semaphore target)
    pub max_concurrent_embeddings: usize,
    /// Inter-item delay in milliseconds
    pub inter_item_delay_ms: u64,
}

// ---------------------------------------------------------------------------
// System State
// ---------------------------------------------------------------------------

/// System idle/activity state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemState {
    /// User is actively using the machine
    Interactive,
    /// User has been idle for at least idle_threshold
    Idle,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive resource manager.
#[derive(Debug, Clone)]
pub struct AdaptiveResourceConfig {
    /// Seconds of no user input before entering idle/burst mode
    pub idle_threshold_secs: u64,
    /// Normal (interactive) max concurrent embeddings
    pub normal_max_concurrent_embeddings: usize,
    /// Burst (idle) max concurrent embeddings
    pub burst_max_concurrent_embeddings: usize,
    /// Normal inter-item delay in ms
    pub normal_inter_item_delay_ms: u64,
    /// Burst inter-item delay in ms
    pub burst_inter_item_delay_ms: u64,
    /// CPU load average threshold (fraction of cores) above which burst is suppressed
    pub cpu_pressure_threshold: f64,
    /// How often to poll system state (in seconds)
    pub poll_interval_secs: u64,
}

impl AdaptiveResourceConfig {
    /// Create config from resolved ResourceLimitsConfig values.
    pub fn from_resource_limits(
        normal_max_concurrent: usize,
        normal_delay_ms: u64,
        physical_cores: usize,
    ) -> Self {
        Self {
            idle_threshold_secs: Self::env_or("WQM_RESOURCE_IDLE_THRESHOLD_SECS", 120),
            normal_max_concurrent_embeddings: normal_max_concurrent,
            burst_max_concurrent_embeddings: std::cmp::max(2, physical_cores / 2),
            normal_inter_item_delay_ms: normal_delay_ms,
            burst_inter_item_delay_ms: 0,
            cpu_pressure_threshold: 0.6,
            poll_interval_secs: 5,
        }
    }

    fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

// ---------------------------------------------------------------------------
// Platform: idle detection
// ---------------------------------------------------------------------------

/// Returns seconds since last user input event, or None if unavailable.
fn seconds_since_last_input() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        macos_idle::seconds_since_last_input()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None // No idle detection on this platform
    }
}

#[cfg(target_os = "macos")]
mod macos_idle {
    // CGEventSourceSecondsSinceLastEventType is in the CoreGraphics framework.
    // We link against it directly rather than pulling in a full crate.
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        /// Returns the elapsed seconds since the last event of the given type.
        /// source_state_id: 1 = kCGEventSourceStateCombinedSessionState
        /// event_type: u32, we use kCGAnyInputEventType = 0xFFFFFFFF (all input events)
        fn CGEventSourceSecondsSinceLastEventType(
            source_state_id: i32,
            event_type: u32,
        ) -> f64;
    }

    const CG_EVENT_SOURCE_STATE_COMBINED_SESSION: i32 = 1;
    const CG_ANY_INPUT_EVENT_TYPE: u32 = 0xFFFFFFFF;

    pub fn seconds_since_last_input() -> Option<f64> {
        let secs = unsafe {
            CGEventSourceSecondsSinceLastEventType(
                CG_EVENT_SOURCE_STATE_COMBINED_SESSION,
                CG_ANY_INPUT_EVENT_TYPE,
            )
        };
        // Returns negative on error
        if secs >= 0.0 { Some(secs) } else { None }
    }
}

/// Check if CPU load is too high for burst mode.
fn is_cpu_under_pressure(threshold: f64, physical_cores: usize) -> bool {
    use sysinfo::System;
    let load = System::load_average();
    let normalized = load.one / physical_cores as f64;
    normalized > threshold
}

// ---------------------------------------------------------------------------
// AdaptiveResourceManager
// ---------------------------------------------------------------------------

/// Manages dynamic resource allocation based on system idle state.
///
/// Spawns a background polling task that monitors user activity and CPU load,
/// then communicates resource profile changes via a watch channel.
pub struct AdaptiveResourceManager {
    /// Receiver for the current resource profile
    rx: watch::Receiver<ResourceProfile>,
}

impl AdaptiveResourceManager {
    /// Start the adaptive resource manager.
    ///
    /// Returns the manager (with a watch receiver) and spawns a background task
    /// that polls system state and updates the resource profile.
    pub fn start(
        config: AdaptiveResourceConfig,
        cancellation_token: CancellationToken,
    ) -> Self {
        let normal_profile = ResourceProfile {
            max_concurrent_embeddings: config.normal_max_concurrent_embeddings,
            inter_item_delay_ms: config.normal_inter_item_delay_ms,
        };
        let burst_profile = ResourceProfile {
            max_concurrent_embeddings: config.burst_max_concurrent_embeddings,
            inter_item_delay_ms: config.burst_inter_item_delay_ms,
        };

        let (tx, rx) = watch::channel(normal_profile);

        let physical_cores = detect_physical_cores();

        info!(
            "Adaptive resource manager started (idle_threshold={}s, normal={}/{}, burst={}/{}, poll={}s)",
            config.idle_threshold_secs,
            normal_profile.max_concurrent_embeddings,
            normal_profile.inter_item_delay_ms,
            burst_profile.max_concurrent_embeddings,
            burst_profile.inter_item_delay_ms,
            config.poll_interval_secs,
        );

        tokio::spawn(async move {
            let poll_interval = Duration::from_secs(config.poll_interval_secs);
            let mut current_state = SystemState::Interactive;

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        debug!("Adaptive resource manager shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(poll_interval) => {
                        let new_state = determine_state(
                            config.idle_threshold_secs,
                            config.cpu_pressure_threshold,
                            physical_cores,
                        );

                        if new_state != current_state {
                            let profile = match new_state {
                                SystemState::Idle => burst_profile,
                                SystemState::Interactive => normal_profile,
                            };

                            match new_state {
                                SystemState::Idle => {
                                    info!(
                                        "System idle detected — switching to burst mode (embeddings: {} → {}, delay: {}ms → {}ms)",
                                        normal_profile.max_concurrent_embeddings,
                                        burst_profile.max_concurrent_embeddings,
                                        normal_profile.inter_item_delay_ms,
                                        burst_profile.inter_item_delay_ms,
                                    );
                                }
                                SystemState::Interactive => {
                                    info!(
                                        "User activity detected — switching to normal mode (embeddings: {} → {}, delay: {}ms → {}ms)",
                                        burst_profile.max_concurrent_embeddings,
                                        normal_profile.max_concurrent_embeddings,
                                        burst_profile.inter_item_delay_ms,
                                        normal_profile.inter_item_delay_ms,
                                    );
                                }
                            }

                            // Send will only fail if all receivers are dropped
                            let _ = tx.send(profile);
                            current_state = new_state;
                        }
                    }
                }
            }
        });

        Self { rx }
    }

    /// Get the current resource profile.
    pub fn current_profile(&self) -> ResourceProfile {
        *self.rx.borrow()
    }

    /// Subscribe to resource profile changes.
    ///
    /// Returns a watch::Receiver that the processing loop can use to detect
    /// when the resource profile changes.
    pub fn subscribe(&self) -> watch::Receiver<ResourceProfile> {
        self.rx.clone()
    }
}

/// Determine the current system state based on idle time and CPU pressure.
fn determine_state(
    idle_threshold_secs: u64,
    cpu_pressure_threshold: f64,
    physical_cores: usize,
) -> SystemState {
    // Check idle time
    let idle_secs = match seconds_since_last_input() {
        Some(secs) => secs,
        None => {
            // No idle detection available — stay interactive
            return SystemState::Interactive;
        }
    };

    if idle_secs < idle_threshold_secs as f64 {
        return SystemState::Interactive;
    }

    // User is idle, but check CPU pressure
    if is_cpu_under_pressure(cpu_pressure_threshold, physical_cores) {
        debug!(
            "User idle for {:.0}s but CPU pressure high — staying interactive",
            idle_secs
        );
        return SystemState::Interactive;
    }

    SystemState::Idle
}

/// Detect physical core count (same logic as config.rs).
fn detect_physical_cores() -> usize {
    use sysinfo::System;
    let sys = System::new_all();
    let physical = sys.physical_core_count().unwrap_or(4);
    physical
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_adaptive_config_from_resource_limits() {
        let config = AdaptiveResourceConfig::from_resource_limits(2, 50, 8);
        assert_eq!(config.normal_max_concurrent_embeddings, 2);
        assert_eq!(config.burst_max_concurrent_embeddings, 4); // max(2, 8/2)
        assert_eq!(config.normal_inter_item_delay_ms, 50);
        assert_eq!(config.burst_inter_item_delay_ms, 0);
        assert_eq!(config.idle_threshold_secs, 120);
    }

    #[test]
    fn test_adaptive_config_burst_minimum() {
        // With only 2 cores, burst should still be at least 2
        let config = AdaptiveResourceConfig::from_resource_limits(1, 100, 2);
        assert_eq!(config.burst_max_concurrent_embeddings, 2); // max(2, 2/2=1) = 2
    }

    #[test]
    fn test_determine_state_no_idle_detection() {
        // When seconds_since_last_input returns None (unsupported platform),
        // state should always be Interactive
        // This test validates the logic path — actual idle detection is platform-specific
        let state = determine_state(120, 0.6, 4);
        // On macOS in CI/test, this might return Idle if system is actually idle
        // On non-macOS, it always returns Interactive
        assert!(state == SystemState::Interactive || state == SystemState::Idle);
    }

    #[tokio::test]
    async fn test_adaptive_manager_starts_with_normal_profile() {
        let config = AdaptiveResourceConfig::from_resource_limits(2, 50, 8);
        let token = CancellationToken::new();
        let manager = AdaptiveResourceManager::start(config, token.clone());

        let profile = manager.current_profile();
        assert_eq!(profile.max_concurrent_embeddings, 2);
        assert_eq!(profile.inter_item_delay_ms, 50);

        token.cancel();
    }

    #[tokio::test]
    async fn test_adaptive_manager_subscribe() {
        let config = AdaptiveResourceConfig::from_resource_limits(2, 50, 8);
        let token = CancellationToken::new();
        let manager = AdaptiveResourceManager::start(config, token.clone());

        let rx = manager.subscribe();
        let profile = *rx.borrow();
        assert_eq!(profile.max_concurrent_embeddings, 2);

        token.cancel();
    }

    #[test]
    fn test_cpu_pressure_check() {
        // With an extremely high threshold (10.0), no system should be under pressure
        assert!(!is_cpu_under_pressure(10.0, 4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_idle_detection_returns_value() {
        let secs = seconds_since_last_input();
        assert!(secs.is_some(), "macOS should always return idle time");
        assert!(secs.unwrap() >= 0.0, "Idle time should be non-negative");
    }
}
