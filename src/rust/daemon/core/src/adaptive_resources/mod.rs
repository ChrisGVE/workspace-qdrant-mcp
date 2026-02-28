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
use std::time::Instant;

use tracing::info;

use crate::config::ResourceLimitsConfig;

pub(crate) mod idle_detection;
mod manager;

pub use manager::AdaptiveResourceManager;

#[cfg(test)]
mod tests;

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
    pub(super) fn up(self) -> Self {
        match self {
            Self::Normal => Self::Active,
            Self::Active => Self::Elevated,
            Self::Elevated => Self::Burst,
            Self::Burst => Self::Burst,
        }
    }

    /// Go down one level, floored at Normal.
    pub(super) fn down(self) -> Self {
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
// System State (internal, shared with manager submodule)
// ---------------------------------------------------------------------------

/// Internal state machine tracking level and transition timers.
#[derive(Debug, Clone, Copy)]
pub(super) struct SystemState {
    /// Current resource level (0-3)
    pub(super) level: ResourceLevel,
    /// When we entered the current level
    pub(super) level_entered_at: Instant,
    /// When continuous idle was first detected (for idle confirmation).
    /// Reset whenever user activity is detected.
    pub(super) idle_detected_at: Option<Instant>,
    /// When continuous user activity was first detected (for ramp-down).
    /// Reset whenever idle is detected.
    pub(super) activity_detected_at: Option<Instant>,
}

impl SystemState {
    pub(super) fn new() -> Self {
        Self {
            level: ResourceLevel::Normal,
            level_entered_at: Instant::now(),
            idle_detected_at: None,
            activity_detected_at: None,
        }
    }

    /// Transition to a new level, resetting the level timer.
    pub(super) fn transition_to(&mut self, new_level: ResourceLevel) {
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
// Profile helpers (shared with manager submodule)
// ---------------------------------------------------------------------------

/// All four resource profiles built from config.
pub(super) struct Profiles {
    pub(super) normal: ResourceProfile,
    pub(super) active: ResourceProfile,
    pub(super) elevated: ResourceProfile,
    pub(super) burst: ResourceProfile,
}

/// Build the four resource profiles from adaptive config.
pub(super) fn build_profiles(config: &AdaptiveResourceConfig) -> Profiles {
    let normal = ResourceProfile {
        max_concurrent_embeddings: config.normal_max_concurrent_embeddings,
        inter_item_delay_ms: config.normal_inter_item_delay_ms,
    };
    let burst = ResourceProfile {
        max_concurrent_embeddings: config.burst_max_concurrent_embeddings,
        inter_item_delay_ms: config.burst_inter_item_delay_ms,
    };
    let active_embeddings = std::cmp::min(
        burst.max_concurrent_embeddings,
        std::cmp::max(
            normal.max_concurrent_embeddings + 1,
            (normal.max_concurrent_embeddings as f64 * config.active_concurrency_multiplier)
                .round() as usize,
        ),
    );
    let active = ResourceProfile {
        max_concurrent_embeddings: active_embeddings,
        inter_item_delay_ms: config.active_inter_item_delay_ms,
    };
    let elevated = ResourceProfile {
        max_concurrent_embeddings: std::cmp::max(
            active.max_concurrent_embeddings,
            (active.max_concurrent_embeddings + burst.max_concurrent_embeddings) / 2,
        ),
        inter_item_delay_ms: (active.inter_item_delay_ms + burst.inter_item_delay_ms) / 2,
    };
    Profiles { normal, active, elevated, burst }
}

/// Compute the resource profile for a given level.
pub(super) fn profile_for_level(
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

/// Detect physical core count (same logic as config.rs).
pub(super) fn detect_physical_cores() -> usize {
    use sysinfo::System;
    let sys = System::new_all();
    sys.physical_core_count().unwrap_or(4)
}

/// Log the startup configuration.
pub(super) fn log_startup_config(config: &AdaptiveResourceConfig, p: &Profiles) {
    info!(
        "Adaptive resource manager started \
         (idle_threshold={}s, confirm={}s, ramp_up={}s, ramp_down={}s, burst_hold={}s, \
         normal={}/{}, active={}/{}, elevated={}/{}, burst={}/{}, poll={}s)",
        config.idle_threshold_secs, config.idle_confirmation_secs,
        config.ramp_up_step_secs, config.ramp_down_step_secs, config.burst_hold_secs,
        p.normal.max_concurrent_embeddings, p.normal.inter_item_delay_ms,
        p.active.max_concurrent_embeddings, p.active.inter_item_delay_ms,
        p.elevated.max_concurrent_embeddings, p.elevated.inter_item_delay_ms,
        p.burst.max_concurrent_embeddings, p.burst.inter_item_delay_ms,
        config.poll_interval_secs,
    );
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
    pub(super) fn new() -> Self {
        Self {
            mode: AtomicU64::new(0),
            idle_centiseconds: AtomicU64::new(0),
            max_concurrent_embeddings: AtomicU64::new(0),
            inter_item_delay_ms: AtomicU64::new(0),
        }
    }

    pub(super) fn set_mode(&self, mode: ResourceMode) {
        let encoded = match mode {
            ResourceMode::Normal => 0,
            ResourceMode::Burst => 1,
            ResourceMode::Active => 2,
            ResourceMode::RampingUp(step) => 100 + step as u64,
        };
        self.mode.store(encoded, Ordering::Relaxed);
    }

    pub(super) fn set_idle_seconds(&self, secs: f64) {
        self.idle_centiseconds.store((secs * 100.0) as u64, Ordering::Relaxed);
    }

    pub(super) fn set_profile(&self, profile: &ResourceProfile) {
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
