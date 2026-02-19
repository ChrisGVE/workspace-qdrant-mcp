//! Adaptive Resource Manager
//!
//! Dynamically adjusts embedding concurrency and processing delays based on
//! system idle state using a 4-level state machine with gradual transitions.
//!
//! ## Resource Levels (ordered)
//!
//! | Level | Name     | When                           | Resources           |
//! |-------|----------|--------------------------------|---------------------|
//! | 0     | Normal   | User active                    | Baseline            |
//! | 1     | Active   | Short idle period detected     | +50% concurrency    |
//! | 2     | Elevated | Idle confirmed, ramping up     | ~75% of burst       |
//! | 3     | Burst    | Sustained idle, full resources  | Maximum             |
//!
//! ## Transition Rules
//!
//! - **Max 1-level change per evaluation**: No jumping from Burst to Normal.
//! - **Ramp-up**: Requires `idle_confirmation_secs` (default 300s) of sustained
//!   idle before the first upward transition. Each subsequent level requires
//!   `ramp_up_step_secs` at the current level.
//! - **Ramp-down**: Each downward level transition requires `ramp_down_step_secs`
//!   (default 300s) of sustained user activity. If idle resumes during ramp-down,
//!   the direction reverses to ramp-up.
//! - **Burst hold**: Once at Burst, stays for at least `burst_hold_secs` (default
//!   600s) before allowing ramp-down ("generous time at max").
//!
//! ## Platform support
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
// Resource Level (4-level hierarchy)
// ---------------------------------------------------------------------------

/// Ordered resource level. Each level maps to a specific resource profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ResourceLevel {
    /// Level 0: Baseline — user active, no queue work
    Normal = 0,
    /// Level 1: Boosted — user active, queue has work (+50% resources)
    Active = 1,
    /// Level 2: High — idle confirmed, intermediate resources (~75% of burst)
    Elevated = 2,
    /// Level 3: Maximum — sustained idle, full burst resources
    Burst = 3,
}

impl ResourceLevel {
    /// Numeric level (0-3).
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Go up one level, capped at Burst.
    fn up(self) -> Self {
        match self {
            Self::Normal => Self::Active,
            Self::Active => Self::Elevated,
            Self::Elevated => Self::Burst,
            Self::Burst => Self::Burst,
        }
    }

    /// Go down one level, floored at Normal.
    fn down(self) -> Self {
        match self {
            Self::Normal => Self::Normal,
            Self::Active => Self::Normal,
            Self::Elevated => Self::Active,
            Self::Burst => Self::Elevated,
        }
    }

    /// String label for the level (used in idle_history logging).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Active => "active",
            Self::Elevated => "elevated",
            Self::Burst => "burst",
        }
    }
}

// ---------------------------------------------------------------------------
// Resource Mode (exposed for status reporting)
// ---------------------------------------------------------------------------

/// Human-readable resource mode, exposed via gRPC status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceMode {
    /// Normal interactive mode (no queue work or queue empty)
    Normal,
    /// Active processing: queue has work while user is present (+50% resources)
    Active,
    /// Elevated idle resources (step number for backwards compatibility)
    RampingUp(u32),
    /// Full burst mode
    Burst,
}

impl ResourceMode {
    /// Label for display / gRPC.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceMode::Normal => "normal",
            ResourceMode::Active => "active",
            ResourceMode::RampingUp(_) => "elevated",
            ResourceMode::Burst => "burst",
        }
    }
}

impl From<ResourceLevel> for ResourceMode {
    fn from(level: ResourceLevel) -> Self {
        match level {
            ResourceLevel::Normal => ResourceMode::Normal,
            ResourceLevel::Active => ResourceMode::Active,
            ResourceLevel::Elevated => ResourceMode::RampingUp(2),
            ResourceLevel::Burst => ResourceMode::Burst,
        }
    }
}

// ---------------------------------------------------------------------------
// System State (internal)
// ---------------------------------------------------------------------------

/// Internal state machine tracking level and transition timers.
#[derive(Debug, Clone, Copy)]
struct SystemState {
    /// Current resource level (0-3)
    level: ResourceLevel,
    /// When we entered the current level
    level_entered_at: Instant,
    /// When continuous idle was first detected (for idle confirmation).
    /// Reset whenever user activity is detected.
    idle_detected_at: Option<Instant>,
    /// When continuous user activity was first detected (for ramp-down).
    /// Reset whenever idle is detected.
    activity_detected_at: Option<Instant>,
}

impl SystemState {
    fn new() -> Self {
        Self {
            level: ResourceLevel::Normal,
            level_entered_at: Instant::now(),
            idle_detected_at: None,
            activity_detected_at: None,
        }
    }

    /// Transition to a new level, resetting the level timer.
    fn transition_to(&mut self, new_level: ResourceLevel) {
        self.level = new_level;
        self.level_entered_at = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive resource manager.
#[derive(Debug, Clone)]
pub struct AdaptiveResourceConfig {
    /// Seconds of no user input before considering idle
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
    /// Seconds of sustained idle required before the first upward level transition
    pub idle_confirmation_secs: u64,
    /// Seconds to spend at each level during ramp-up (after confirmation)
    pub ramp_up_step_secs: u64,
    /// Seconds of sustained user activity required before each downward level transition
    pub ramp_down_step_secs: u64,
    /// Minimum seconds to hold at Burst before allowing ramp-down
    pub burst_hold_secs: u64,
    /// Multiplier for active processing mode (user present, queue has work)
    pub active_concurrency_multiplier: f64,
    /// Inter-item delay in active processing mode (ms)
    pub active_inter_item_delay_ms: u64,
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
            idle_confirmation_secs: limits.idle_confirmation_secs,
            ramp_up_step_secs: limits.ramp_up_step_secs,
            ramp_down_step_secs: limits.ramp_down_step_secs,
            burst_hold_secs: limits.burst_hold_secs,
            active_concurrency_multiplier: limits.active_concurrency_multiplier,
            active_inter_item_delay_ms: limits.active_inter_item_delay_ms,
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
// Level profiles
// ---------------------------------------------------------------------------

/// Compute the resource profile for a given level.
fn profile_for_level(
    level: ResourceLevel,
    normal: &ResourceProfile,
    active: &ResourceProfile,
    elevated: &ResourceProfile,
    burst: &ResourceProfile,
) -> ResourceProfile {
    match level {
        ResourceLevel::Normal => *normal,
        ResourceLevel::Active => *active,
        ResourceLevel::Elevated => *elevated,
        ResourceLevel::Burst => *burst,
    }
}

// ---------------------------------------------------------------------------
// Shared state for status reporting
// ---------------------------------------------------------------------------

/// Shared state exposed to gRPC status reporting.
///
/// Uses atomic fields for lock-free reads from the gRPC status handler.
pub struct AdaptiveResourceState {
    /// Current resource mode (encoded as: 0=Normal, 1=Burst, 2=Active, 100+step=Elevated)
    mode: AtomicU64,
    /// Seconds since last user input (x100 for 2-decimal precision)
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
            ResourceMode::Active => 2,
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
            2 => ResourceMode::Active,
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
/// then communicates resource profile changes via a watch channel. Uses a
/// 4-level state machine (Normal < Active < Elevated < Burst) with gradual
/// transitions — max 1 level per evaluation, with configurable confirmation
/// delays for both ramp-up and ramp-down.
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
    ///
    /// `queue_depth` is an optional shared counter of pending queue items.
    /// When provided and > 0 while the user is active, the manager enters
    /// Active Processing mode with +50% resources.
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

        // Active processing profile (+50% concurrency, halved delay)
        let active_profile = {
            let active_embeddings = std::cmp::max(
                normal_profile.max_concurrent_embeddings + 1,
                (normal_profile.max_concurrent_embeddings as f64 * config.active_concurrency_multiplier)
                    .round() as usize,
            );
            ResourceProfile {
                max_concurrent_embeddings: std::cmp::min(
                    active_embeddings,
                    burst_profile.max_concurrent_embeddings,
                ),
                inter_item_delay_ms: config.active_inter_item_delay_ms,
            }
        };

        // Elevated profile: midpoint between active and burst
        let elevated_profile = {
            let mid_embeddings = (active_profile.max_concurrent_embeddings
                + burst_profile.max_concurrent_embeddings) / 2;
            let mid_delay = (active_profile.inter_item_delay_ms
                + burst_profile.inter_item_delay_ms) / 2;
            ResourceProfile {
                max_concurrent_embeddings: std::cmp::max(
                    active_profile.max_concurrent_embeddings,
                    mid_embeddings,
                ),
                inter_item_delay_ms: mid_delay,
            }
        };

        let (tx, rx) = watch::channel(normal_profile);
        let state = Arc::new(AdaptiveResourceState::new());
        state.set_profile(&normal_profile);
        let state_clone = Arc::clone(&state);

        let physical_cores = detect_physical_cores();

        info!(
            "Adaptive resource manager started \
             (idle_threshold={}s, confirm={}s, ramp_up_step={}s, ramp_down_step={}s, burst_hold={}s, \
             normal={}/{}, active={}/{}, elevated={}/{}, burst={}/{}, poll={}s)",
            config.idle_threshold_secs,
            config.idle_confirmation_secs,
            config.ramp_up_step_secs,
            config.ramp_down_step_secs,
            config.burst_hold_secs,
            normal_profile.max_concurrent_embeddings,
            normal_profile.inter_item_delay_ms,
            active_profile.max_concurrent_embeddings,
            active_profile.inter_item_delay_ms,
            elevated_profile.max_concurrent_embeddings,
            elevated_profile.inter_item_delay_ms,
            burst_profile.max_concurrent_embeddings,
            burst_profile.inter_item_delay_ms,
            config.poll_interval_secs,
        );

        tokio::spawn(async move {
            let poll_interval = Duration::from_secs(config.poll_interval_secs);
            let idle_confirmation = Duration::from_secs(config.idle_confirmation_secs);
            let ramp_up_step = Duration::from_secs(config.ramp_up_step_secs);
            let ramp_down_step = Duration::from_secs(config.ramp_down_step_secs);
            let burst_hold = Duration::from_secs(config.burst_hold_secs);

            let mut sys_state = SystemState::new();
            let mut current_profile = normal_profile;
            let mut heartbeat_counter: u64 = 0;
            let heartbeat_interval: u64 = 60 / config.poll_interval_secs.max(1);
            let mut mode_tracker = crate::idle_history::ModeTracker::new();
            let rotation_interval: u64 = 3600 / config.poll_interval_secs.max(1);

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        debug!("Adaptive resource manager shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(poll_interval) => {
                        let idle_secs = seconds_since_last_input().unwrap_or(0.0);
                        state_clone.set_idle_seconds(idle_secs);

                        let user_is_idle = idle_secs >= config.idle_threshold_secs as f64;
                        let cpu_ok = !is_cpu_under_pressure(
                            config.cpu_pressure_threshold,
                            physical_cores,
                        );

                        let old_level = sys_state.level;

                        if user_is_idle && cpu_ok {
                            // --- User is idle, CPU ok: ramp UP ---
                            sys_state.activity_detected_at = None; // reset descent timer

                            if sys_state.idle_detected_at.is_none() {
                                sys_state.idle_detected_at = Some(Instant::now());
                                debug!(
                                    "Idle detected at level {:?}, starting confirmation timer ({}s)",
                                    sys_state.level, config.idle_confirmation_secs,
                                );
                            }

                            let idle_duration = sys_state.idle_detected_at
                                .map(|t| t.elapsed())
                                .unwrap_or_default();

                            // Must confirm idle before any upward transition
                            if idle_duration >= idle_confirmation && sys_state.level < ResourceLevel::Burst {
                                let time_at_level = sys_state.level_entered_at.elapsed();

                                if time_at_level >= ramp_up_step {
                                    let new_level = sys_state.level.up();
                                    info!(
                                        "Ramp-up: {:?} -> {:?} (idle {:.0}s, at level {:.0}s)",
                                        sys_state.level, new_level,
                                        idle_secs, time_at_level.as_secs_f64(),
                                    );
                                    sys_state.transition_to(new_level);
                                } else {
                                    debug!(
                                        "Ramp-up: holding at {:?} ({:.0}s/{:.0}s before next level)",
                                        sys_state.level,
                                        time_at_level.as_secs_f64(),
                                        ramp_up_step.as_secs_f64(),
                                    );
                                }
                            } else if idle_duration < idle_confirmation {
                                debug!(
                                    "Idle confirmation: {:.0}s/{:.0}s",
                                    idle_duration.as_secs_f64(),
                                    idle_confirmation.as_secs_f64(),
                                );
                            }
                        } else {
                            // --- User active or CPU pressure: ramp DOWN ---
                            sys_state.idle_detected_at = None; // reset idle confirmation

                            // Target is always Normal when user is active — resource level
                            // is driven purely by idle time, not queue depth.
                            let target = ResourceLevel::Normal;

                            if sys_state.level <= target {
                                // Already at or below target
                                sys_state.activity_detected_at = None;
                            } else {
                                // Need to ramp down
                                if sys_state.activity_detected_at.is_none() {
                                    sys_state.activity_detected_at = Some(Instant::now());
                                    debug!(
                                        "Activity detected at level {:?}, starting ramp-down timer ({}s/level)",
                                        sys_state.level, config.ramp_down_step_secs,
                                    );
                                }

                                // Burst hold: enforce minimum time at burst
                                if sys_state.level == ResourceLevel::Burst {
                                    let burst_time = sys_state.level_entered_at.elapsed();
                                    if burst_time < burst_hold {
                                        debug!(
                                            "Burst hold: {:.0}s/{:.0}s before ramp-down allowed",
                                            burst_time.as_secs_f64(),
                                            burst_hold.as_secs_f64(),
                                        );
                                        // Don't start the ramp-down timer until burst hold expires
                                        sys_state.activity_detected_at = None;
                                        // Continue in next branch
                                    }
                                }

                                // Check if enough sustained activity for a level drop
                                if let Some(activity_start) = sys_state.activity_detected_at {
                                    let activity_duration = activity_start.elapsed();
                                    if activity_duration >= ramp_down_step {
                                        let new_level = sys_state.level.down();
                                        info!(
                                            "Ramp-down: {:?} -> {:?} (active {:.0}s)",
                                            sys_state.level, new_level,
                                            activity_duration.as_secs_f64(),
                                        );
                                        sys_state.transition_to(new_level);
                                        // Reset activity timer for next level's descent
                                        sys_state.activity_detected_at = Some(Instant::now());
                                    } else {
                                        debug!(
                                            "Ramp-down: holding at {:?} ({:.0}s/{:.0}s before drop)",
                                            sys_state.level,
                                            activity_duration.as_secs_f64(),
                                            ramp_down_step.as_secs_f64(),
                                        );
                                    }
                                }
                            }
                        }

                        // Compute profile and mode from current level
                        let new_profile = profile_for_level(
                            sys_state.level,
                            &normal_profile, &active_profile,
                            &elevated_profile, &burst_profile,
                        );
                        let new_mode = ResourceMode::from(sys_state.level);

                        state_clone.set_mode(new_mode);
                        state_clone.set_profile(&new_profile);

                        // Track level transitions for history (uses ResourceLevel
                        // directly to avoid ResourceMode ambiguity between Active
                        // Processing and Active ramp-down level)
                        mode_tracker.on_mode_change(sys_state.level, idle_secs);

                        if new_profile != current_profile {
                            if old_level != sys_state.level {
                                info!(
                                    "Profile changed: embeddings {} -> {}, delay {}ms -> {}ms",
                                    current_profile.max_concurrent_embeddings,
                                    new_profile.max_concurrent_embeddings,
                                    current_profile.inter_item_delay_ms,
                                    new_profile.inter_item_delay_ms,
                                );
                            }
                            let _ = tx.send(new_profile);
                            current_profile = new_profile;
                        }

                        // Periodic heartbeat
                        heartbeat_counter += 1;
                        if heartbeat_counter % heartbeat_interval == 0 {
                            info!(
                                "Adaptive resources heartbeat: level={:?}, mode={}, idle_secs={:.0}, cpu_pressure={}, embeddings={}, delay={}ms",
                                sys_state.level,
                                new_mode.as_str(),
                                idle_secs,
                                !cpu_ok,
                                new_profile.max_concurrent_embeddings,
                                new_profile.inter_item_delay_ms,
                            );
                        }

                        // Rotate idle history once per hour
                        if heartbeat_counter % rotation_interval == 0 {
                            mode_tracker.rotate();
                        }
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
        let a = ResourceProfile { max_concurrent_embeddings: 2, inter_item_delay_ms: 50 };
        let b = ResourceProfile { max_concurrent_embeddings: 2, inter_item_delay_ms: 50 };
        let c = ResourceProfile { max_concurrent_embeddings: 4, inter_item_delay_ms: 0 };
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
        assert_eq!(ResourceMode::from(ResourceLevel::Normal), ResourceMode::Normal);
        assert_eq!(ResourceMode::from(ResourceLevel::Active), ResourceMode::Active);
        assert_eq!(ResourceMode::from(ResourceLevel::Elevated), ResourceMode::RampingUp(2));
        assert_eq!(ResourceMode::from(ResourceLevel::Burst), ResourceMode::Burst);
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
        let normal = ResourceProfile { max_concurrent_embeddings: 2, inter_item_delay_ms: 50 };
        let active = ResourceProfile { max_concurrent_embeddings: 3, inter_item_delay_ms: 25 };
        let elevated = ResourceProfile { max_concurrent_embeddings: 3, inter_item_delay_ms: 12 };
        let burst = ResourceProfile { max_concurrent_embeddings: 4, inter_item_delay_ms: 0 };

        assert_eq!(profile_for_level(ResourceLevel::Normal, &normal, &active, &elevated, &burst), normal);
        assert_eq!(profile_for_level(ResourceLevel::Active, &normal, &active, &elevated, &burst), active);
        assert_eq!(profile_for_level(ResourceLevel::Elevated, &normal, &active, &elevated, &burst), elevated);
        assert_eq!(profile_for_level(ResourceLevel::Burst, &normal, &active, &elevated, &burst), burst);
    }

    // --- Manager lifecycle tests ---

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
