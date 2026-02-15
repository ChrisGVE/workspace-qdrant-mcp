//! Adaptive Resource Manager
//!
//! Dynamically adjusts embedding concurrency and processing delays based on
//! system idle state. When the user is idle (no keyboard/mouse input), the
//! daemon *gradually* ramps up processing throughput over a configurable
//! duration. When user activity resumes, it immediately snaps back to normal.
//!
//! Platform support:
//! - macOS: CGEventSourceSecondsSinceLastEventType (CoreGraphics),
//!          with IOKit HIDIdleTime fallback (works during screen lock)
//! - Linux: falls back to always-interactive (no burst mode)
//! - Other: always-interactive (no burst mode)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::config::ResourceLimitsConfig;

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
// Resource Mode (exposed for status reporting)
// ---------------------------------------------------------------------------

/// Human-readable resource mode, exposed via gRPC status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceMode {
    /// Normal interactive mode
    Normal,
    /// Gradually ramping up (includes step index, 0-based)
    RampingUp(u32),
    /// Full burst mode
    Burst,
}

impl ResourceMode {
    /// Label for display / gRPC.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceMode::Normal => "normal",
            ResourceMode::RampingUp(_) => "ramping",
            ResourceMode::Burst => "burst",
        }
    }
}

// ---------------------------------------------------------------------------
// System State (internal)
// ---------------------------------------------------------------------------

/// Internal system state including ramp-up tracking.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SystemState {
    /// User is actively using the machine
    Interactive,
    /// Ramping up: `entered_idle_at` is the instant when idle was first detected.
    RampingUp { entered_idle_at: Instant },
    /// Fully ramped to burst mode
    FullBurst,
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
    /// Total duration to ramp from normal to burst (seconds)
    pub ramp_duration_secs: u64,
    /// Number of discrete steps during ramp-up
    pub ramp_steps: u32,
}

impl AdaptiveResourceConfig {
    /// Create config from resolved ResourceLimitsConfig values.
    pub fn from_resource_limits(limits: &ResourceLimitsConfig) -> Self {
        let normal_max = limits.max_concurrent_embeddings;
        let burst_max = std::cmp::max(
            normal_max + 1,
            (normal_max as f64 * limits.burst_concurrency_multiplier).round() as usize,
        );

        Self {
            idle_threshold_secs: limits.idle_threshold_secs,
            normal_max_concurrent_embeddings: normal_max,
            burst_max_concurrent_embeddings: burst_max,
            normal_inter_item_delay_ms: limits.inter_item_delay_ms,
            burst_inter_item_delay_ms: limits.burst_inter_item_delay_ms,
            cpu_pressure_threshold: limits.cpu_pressure_threshold,
            poll_interval_secs: limits.idle_poll_interval_secs,
            ramp_duration_secs: limits.idle_ramp_duration_secs,
            ramp_steps: std::cmp::max(1, limits.idle_ramp_steps),
        }
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
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(
            source_state_id: i32,
            event_type: u32,
        ) -> f64;
    }

    // IOKit framework for HIDIdleTime (works during screen lock).
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceGetMatchingService(
            main_port: u32,
            matching: *const std::ffi::c_void,
        ) -> u32;
        fn IOServiceMatching(name: *const std::ffi::c_char) -> *const std::ffi::c_void;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: *const std::ffi::c_void,
            allocator: *const std::ffi::c_void,
            options: u32,
        ) -> *const std::ffi::c_void;
        fn IOObjectRelease(object: u32) -> u32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const std::ffi::c_void,
            c_str: *const std::ffi::c_char,
            encoding: u32,
        ) -> *const std::ffi::c_void;
        fn CFNumberGetValue(
            number: *const std::ffi::c_void,
            the_type: i64,
            value_ptr: *mut std::ffi::c_void,
        ) -> bool;
        fn CFRelease(cf: *const std::ffi::c_void);
    }

    const CG_EVENT_SOURCE_STATE_COMBINED_SESSION: i32 = 1;
    const CG_ANY_INPUT_EVENT_TYPE: u32 = 0xFFFFFFFF;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
    const K_CF_NUMBER_SINT64_TYPE: i64 = 4;
    const K_IO_MAIN_PORT_DEFAULT: u32 = 0;

    /// Primary idle detection via CoreGraphics.
    fn cg_idle_seconds() -> Option<f64> {
        let secs = unsafe {
            CGEventSourceSecondsSinceLastEventType(
                CG_EVENT_SOURCE_STATE_COMBINED_SESSION,
                CG_ANY_INPUT_EVENT_TYPE,
            )
        };
        if secs >= 0.0 { Some(secs) } else { None }
    }

    /// Fallback idle detection via IOKit HIDIdleTime.
    /// Works during screen lock when CGEventSource may not update.
    fn iokit_idle_seconds() -> Option<f64> {
        unsafe {
            let name = b"IOHIDSystem\0".as_ptr() as *const std::ffi::c_char;
            let matching = IOServiceMatching(name);
            if matching.is_null() {
                return None;
            }

            let service = IOServiceGetMatchingService(K_IO_MAIN_PORT_DEFAULT, matching);
            if service == 0 {
                return None;
            }

            let key_str = b"HIDIdleTime\0".as_ptr() as *const std::ffi::c_char;
            let cf_key = CFStringCreateWithCString(
                std::ptr::null(),
                key_str,
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf_key.is_null() {
                IOObjectRelease(service);
                return None;
            }

            let cf_number = IORegistryEntryCreateCFProperty(
                service,
                cf_key,
                std::ptr::null(),
                0,
            );
            CFRelease(cf_key);
            IOObjectRelease(service);

            if cf_number.is_null() {
                return None;
            }

            let mut idle_ns: i64 = 0;
            let ok = CFNumberGetValue(
                cf_number,
                K_CF_NUMBER_SINT64_TYPE,
                &mut idle_ns as *mut i64 as *mut std::ffi::c_void,
            );
            CFRelease(cf_number);

            if ok && idle_ns >= 0 {
                Some(idle_ns as f64 / 1_000_000_000.0)
            } else {
                None
            }
        }
    }

    /// Returns seconds since last user input.
    /// Tries CoreGraphics first (lighter), falls back to IOKit (works during screen lock).
    pub fn seconds_since_last_input() -> Option<f64> {
        cg_idle_seconds().or_else(iokit_idle_seconds)
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
// Interpolation
// ---------------------------------------------------------------------------

/// Compute an intermediate resource profile at a given ramp step.
///
/// `step` ranges from 0 (just entered idle, slight increase) to
/// `total_steps` (full burst). Linear interpolation between normal and burst.
fn interpolate_profile(
    normal: &ResourceProfile,
    burst: &ResourceProfile,
    step: u32,
    total_steps: u32,
) -> ResourceProfile {
    if step == 0 || total_steps == 0 {
        return *normal;
    }
    if step >= total_steps {
        return *burst;
    }

    let fraction = step as f64 / total_steps as f64;

    let embeddings = normal.max_concurrent_embeddings as f64
        + fraction * (burst.max_concurrent_embeddings as f64 - normal.max_concurrent_embeddings as f64);
    let delay = normal.inter_item_delay_ms as f64
        + fraction * (burst.inter_item_delay_ms as f64 - normal.inter_item_delay_ms as f64);

    ResourceProfile {
        max_concurrent_embeddings: std::cmp::max(1, embeddings.round() as usize),
        inter_item_delay_ms: delay.round().max(0.0) as u64,
    }
}

// ---------------------------------------------------------------------------
// Shared state for status reporting
// ---------------------------------------------------------------------------

/// Shared state exposed to gRPC status reporting.
///
/// Uses atomic fields for lock-free reads from the gRPC status handler.
pub struct AdaptiveResourceState {
    /// Current resource mode (encoded as: 0=Normal, 1=Burst, 100+step=RampingUp)
    mode: AtomicU64,
    /// Seconds since last user input (×100 for 2-decimal precision)
    idle_centiseconds: AtomicU64,
    /// Current max concurrent embeddings
    max_concurrent_embeddings: AtomicU64,
    /// Current inter-item delay in ms
    inter_item_delay_ms: AtomicU64,
}

impl AdaptiveResourceState {
    fn new() -> Self {
        Self {
            mode: AtomicU64::new(0),
            idle_centiseconds: AtomicU64::new(0),
            max_concurrent_embeddings: AtomicU64::new(0),
            inter_item_delay_ms: AtomicU64::new(0),
        }
    }

    fn set_mode(&self, mode: ResourceMode) {
        let encoded = match mode {
            ResourceMode::Normal => 0,
            ResourceMode::Burst => 1,
            ResourceMode::RampingUp(step) => 100 + step as u64,
        };
        self.mode.store(encoded, Ordering::Relaxed);
    }

    fn set_idle_seconds(&self, secs: f64) {
        self.idle_centiseconds.store((secs * 100.0) as u64, Ordering::Relaxed);
    }

    fn set_profile(&self, profile: &ResourceProfile) {
        self.max_concurrent_embeddings.store(profile.max_concurrent_embeddings as u64, Ordering::Relaxed);
        self.inter_item_delay_ms.store(profile.inter_item_delay_ms, Ordering::Relaxed);
    }

    /// Get the current resource mode.
    pub fn mode(&self) -> ResourceMode {
        match self.mode.load(Ordering::Relaxed) {
            0 => ResourceMode::Normal,
            1 => ResourceMode::Burst,
            n if n >= 100 => ResourceMode::RampingUp((n - 100) as u32),
            _ => ResourceMode::Normal,
        }
    }

    /// Get seconds since last user input.
    pub fn idle_seconds(&self) -> f64 {
        self.idle_centiseconds.load(Ordering::Relaxed) as f64 / 100.0
    }

    /// Get the current max concurrent embeddings.
    pub fn max_concurrent_embeddings(&self) -> usize {
        self.max_concurrent_embeddings.load(Ordering::Relaxed) as usize
    }

    /// Get the current inter-item delay in ms.
    pub fn inter_item_delay_ms(&self) -> u64 {
        self.inter_item_delay_ms.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for AdaptiveResourceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveResourceState")
            .field("mode", &self.mode())
            .field("idle_seconds", &self.idle_seconds())
            .field("max_concurrent_embeddings", &self.max_concurrent_embeddings())
            .field("inter_item_delay_ms", &self.inter_item_delay_ms())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AdaptiveResourceManager
// ---------------------------------------------------------------------------

/// Manages dynamic resource allocation based on system idle state.
///
/// Spawns a background polling task that monitors user activity and CPU load,
/// then communicates resource profile changes via a watch channel. Gradually
/// ramps up throughput during idle periods rather than switching instantly.
pub struct AdaptiveResourceManager {
    /// Receiver for the current resource profile
    rx: watch::Receiver<ResourceProfile>,
    /// Shared state for status reporting
    state: Arc<AdaptiveResourceState>,
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
        let state = Arc::new(AdaptiveResourceState::new());
        state.set_profile(&normal_profile);
        let state_clone = Arc::clone(&state);

        let physical_cores = detect_physical_cores();

        info!(
            "Adaptive resource manager started (idle_threshold={}s, ramp={}s/{} steps, normal={}/{}, burst={}/{}, poll={}s)",
            config.idle_threshold_secs,
            config.ramp_duration_secs,
            config.ramp_steps,
            normal_profile.max_concurrent_embeddings,
            normal_profile.inter_item_delay_ms,
            burst_profile.max_concurrent_embeddings,
            burst_profile.inter_item_delay_ms,
            config.poll_interval_secs,
        );

        tokio::spawn(async move {
            let poll_interval = Duration::from_secs(config.poll_interval_secs);
            let ramp_duration = Duration::from_secs(config.ramp_duration_secs);
            let ramp_steps = config.ramp_steps;

            let mut current_state = SystemState::Interactive;
            let mut current_profile = normal_profile;

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        debug!("Adaptive resource manager shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(poll_interval) => {
                        // Read idle time
                        let idle_secs = seconds_since_last_input().unwrap_or(0.0);
                        state_clone.set_idle_seconds(idle_secs);

                        let user_is_idle = idle_secs >= config.idle_threshold_secs as f64;
                        let cpu_ok = !is_cpu_under_pressure(
                            config.cpu_pressure_threshold,
                            physical_cores,
                        );

                        let (new_state, new_profile, new_mode) = if !user_is_idle || !cpu_ok {
                            // User active or CPU too busy → back to normal
                            if current_state != SystemState::Interactive {
                                info!(
                                    "User activity detected — switching to normal mode (embeddings: {} → {}, delay: {}ms → {}ms)",
                                    current_profile.max_concurrent_embeddings,
                                    normal_profile.max_concurrent_embeddings,
                                    current_profile.inter_item_delay_ms,
                                    normal_profile.inter_item_delay_ms,
                                );
                            }
                            (SystemState::Interactive, normal_profile, ResourceMode::Normal)
                        } else {
                            // User is idle and CPU is ok
                            match current_state {
                                SystemState::Interactive => {
                                    // Just entered idle → start ramping
                                    let now = Instant::now();
                                    let step1_profile = interpolate_profile(
                                        &normal_profile, &burst_profile, 1, ramp_steps,
                                    );
                                    info!(
                                        "System idle detected — starting ramp-up step 1/{} (embeddings: {} → {}, delay: {}ms → {}ms)",
                                        ramp_steps,
                                        normal_profile.max_concurrent_embeddings,
                                        step1_profile.max_concurrent_embeddings,
                                        normal_profile.inter_item_delay_ms,
                                        step1_profile.inter_item_delay_ms,
                                    );
                                    (
                                        SystemState::RampingUp { entered_idle_at: now },
                                        step1_profile,
                                        ResourceMode::RampingUp(1),
                                    )
                                }
                                SystemState::RampingUp { entered_idle_at } => {
                                    let elapsed = entered_idle_at.elapsed();
                                    if elapsed >= ramp_duration {
                                        // Ramp complete → full burst
                                        info!(
                                            "Ramp-up complete — full burst mode (embeddings: {} → {}, delay: {}ms → {}ms)",
                                            current_profile.max_concurrent_embeddings,
                                            burst_profile.max_concurrent_embeddings,
                                            current_profile.inter_item_delay_ms,
                                            burst_profile.inter_item_delay_ms,
                                        );
                                        (SystemState::FullBurst, burst_profile, ResourceMode::Burst)
                                    } else {
                                        // Compute current step
                                        let fraction = elapsed.as_secs_f64() / ramp_duration.as_secs_f64();
                                        let step = ((fraction * ramp_steps as f64).ceil() as u32)
                                            .min(ramp_steps);
                                        let profile = interpolate_profile(
                                            &normal_profile, &burst_profile, step, ramp_steps,
                                        );
                                        if profile != current_profile {
                                            debug!(
                                                "Ramp-up step {}/{} (embeddings: {}, delay: {}ms)",
                                                step, ramp_steps,
                                                profile.max_concurrent_embeddings,
                                                profile.inter_item_delay_ms,
                                            );
                                        }
                                        (
                                            SystemState::RampingUp { entered_idle_at },
                                            profile,
                                            ResourceMode::RampingUp(step),
                                        )
                                    }
                                }
                                SystemState::FullBurst => {
                                    // Already at full burst, nothing to change
                                    (SystemState::FullBurst, burst_profile, ResourceMode::Burst)
                                }
                            }
                        };

                        state_clone.set_mode(new_mode);
                        state_clone.set_profile(&new_profile);

                        if new_profile != current_profile {
                            let _ = tx.send(new_profile);
                            current_profile = new_profile;
                        }
                        current_state = new_state;
                    }
                }
            }
        });

        Self { rx, state }
    }

    /// Get the current resource profile.
    pub fn current_profile(&self) -> ResourceProfile {
        *self.rx.borrow()
    }

    /// Subscribe to resource profile changes.
    pub fn subscribe(&self) -> watch::Receiver<ResourceProfile> {
        self.rx.clone()
    }

    /// Get shared state handle for status reporting (gRPC/CLI).
    pub fn state(&self) -> Arc<AdaptiveResourceState> {
        Arc::clone(&self.state)
    }
}

/// Detect physical core count (same logic as config.rs).
fn detect_physical_cores() -> usize {
    use sysinfo::System;
    let sys = System::new_all();
    sys.physical_core_count().unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits() -> ResourceLimitsConfig {
        ResourceLimitsConfig {
            max_concurrent_embeddings: 2,
            inter_item_delay_ms: 50,
            idle_threshold_secs: 120,
            idle_ramp_duration_secs: 300,
            idle_ramp_steps: 5,
            burst_concurrency_multiplier: 2.0,
            burst_inter_item_delay_ms: 0,
            cpu_pressure_threshold: 0.6,
            idle_poll_interval_secs: 5,
            ..Default::default()
        }
    }

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
        let limits = test_limits();
        let config = AdaptiveResourceConfig::from_resource_limits(&limits);
        assert_eq!(config.normal_max_concurrent_embeddings, 2);
        assert_eq!(config.burst_max_concurrent_embeddings, 4); // 2 * 2.0
        assert_eq!(config.normal_inter_item_delay_ms, 50);
        assert_eq!(config.burst_inter_item_delay_ms, 0);
        assert_eq!(config.idle_threshold_secs, 120);
        assert_eq!(config.ramp_duration_secs, 300);
        assert_eq!(config.ramp_steps, 5);
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

    #[test]
    fn test_interpolate_profile_boundaries() {
        let normal = ResourceProfile { max_concurrent_embeddings: 2, inter_item_delay_ms: 50 };
        let burst = ResourceProfile { max_concurrent_embeddings: 6, inter_item_delay_ms: 0 };

        // Step 0 → normal
        let p0 = interpolate_profile(&normal, &burst, 0, 5);
        assert_eq!(p0, normal);

        // Step 5/5 → burst
        let p5 = interpolate_profile(&normal, &burst, 5, 5);
        assert_eq!(p5, burst);

        // Step >= total → burst
        let p6 = interpolate_profile(&normal, &burst, 6, 5);
        assert_eq!(p6, burst);
    }

    #[test]
    fn test_interpolate_profile_intermediate() {
        let normal = ResourceProfile { max_concurrent_embeddings: 2, inter_item_delay_ms: 50 };
        let burst = ResourceProfile { max_concurrent_embeddings: 6, inter_item_delay_ms: 0 };

        // Step 2/5 → 40% of the way
        let p = interpolate_profile(&normal, &burst, 2, 5);
        // embeddings: 2 + 0.4*(6-2) = 2 + 1.6 = 3.6 → round to 4
        assert_eq!(p.max_concurrent_embeddings, 4);
        // delay: 50 + 0.4*(0-50) = 50 - 20 = 30
        assert_eq!(p.inter_item_delay_ms, 30);

        // Step 3/5 → 60%
        let p = interpolate_profile(&normal, &burst, 3, 5);
        // embeddings: 2 + 0.6*4 = 4.4 → round to 4
        assert_eq!(p.max_concurrent_embeddings, 4);
        // delay: 50 - 30 = 20
        assert_eq!(p.inter_item_delay_ms, 20);
    }

    #[test]
    fn test_interpolate_profile_min_embeddings() {
        let normal = ResourceProfile { max_concurrent_embeddings: 1, inter_item_delay_ms: 100 };
        let burst = ResourceProfile { max_concurrent_embeddings: 1, inter_item_delay_ms: 0 };

        // Even at step 0, embeddings should never go below 1
        let p = interpolate_profile(&normal, &burst, 1, 5);
        assert!(p.max_concurrent_embeddings >= 1);
    }

    #[test]
    fn test_resource_mode_as_str() {
        assert_eq!(ResourceMode::Normal.as_str(), "normal");
        assert_eq!(ResourceMode::RampingUp(3).as_str(), "ramping");
        assert_eq!(ResourceMode::Burst.as_str(), "burst");
    }

    #[test]
    fn test_adaptive_resource_state() {
        let state = AdaptiveResourceState::new();
        assert_eq!(state.mode(), ResourceMode::Normal);
        assert!((state.idle_seconds() - 0.0).abs() < f64::EPSILON);

        state.set_mode(ResourceMode::RampingUp(3));
        assert_eq!(state.mode(), ResourceMode::RampingUp(3));

        state.set_mode(ResourceMode::Burst);
        assert_eq!(state.mode(), ResourceMode::Burst);

        state.set_idle_seconds(125.5);
        assert!((state.idle_seconds() - 125.5).abs() < 0.02);
    }

    #[tokio::test]
    async fn test_adaptive_manager_starts_with_normal_profile() {
        let limits = test_limits();
        let config = AdaptiveResourceConfig::from_resource_limits(&limits);
        let token = CancellationToken::new();
        let manager = AdaptiveResourceManager::start(config, token.clone());

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
