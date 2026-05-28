//! Resource limits configuration (CPU, memory, adaptive idle mode)
use serde::{Deserialize, Serialize};

mod env_overrides;
mod platform;
mod validation;

pub use platform::detect_physical_cores;

#[cfg(test)]
mod tests;

fn default_nice_level() -> i32 {
    10
}
fn default_max_concurrent_embeddings() -> usize {
    0
} // 0 = auto-detect
fn default_max_memory_percent() -> u8 {
    70
}
fn default_onnx_intra_threads() -> usize {
    0
} // 0 = auto-detect
fn default_idle_threshold_secs() -> u64 {
    120
}
fn default_idle_confirmation_secs() -> u64 {
    300
}
fn default_ramp_up_step_secs() -> u64 {
    120
}
fn default_ramp_down_step_secs() -> u64 {
    300
}
fn default_burst_hold_secs() -> u64 {
    600
}
fn default_burst_concurrency_multiplier() -> f64 {
    2.0
}
fn default_cpu_pressure_threshold() -> f64 {
    0.6
}
fn default_idle_poll_interval_secs() -> u64 {
    5
}
fn default_active_concurrency_multiplier() -> f64 {
    1.5
}
fn default_linux_idle_source() -> String {
    "none".to_string()
}
fn default_linux_idle_load_threshold() -> f64 {
    0.1
}

/// Resource limits configuration section
///
/// Controls how the daemon manages system resources to be a good neighbor.
/// Four levels: OS priority, processing pacing, embedding concurrency, memory pressure.
///
/// `max_concurrent_embeddings` and `onnx_intra_threads` support auto-detection:
/// set to `0` (the default) and the daemon computes optimal values at startup
/// based on the machine's CPU core count:
///   - `max_concurrent_embeddings` = max(1, physical_cores / 4)
///   - `onnx_intra_threads` = 2 (optimal for all-MiniLM-L6-v2 regardless of core count)
///
/// Total embedding CPU budget = max_concurrent_embeddings x onnx_intra_threads,
/// which is roughly half a quarter of the machine -- leaving headroom for file watching,
/// tree-sitter, gRPC, LSP, and the user's own work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimitsConfig {
    /// Unix nice level for the daemon process (-20 highest priority, 19 lowest)
    /// Default: 10 (low priority - daemon should yield to interactive processes)
    #[serde(default = "default_nice_level")]
    pub nice_level: i32,

    /// Maximum concurrent embedding operations (semaphore on ONNX ops).
    /// 0 = auto-detect from CPU core count (physical_cores / 4, minimum 1).
    #[serde(default = "default_max_concurrent_embeddings")]
    pub max_concurrent_embeddings: usize,

    /// Pause processing when available memory falls below (100 - this)%.
    /// e.g. 70 means pause when less than 30% of system memory is available.
    /// Default: 70
    #[serde(default = "default_max_memory_percent")]
    pub max_memory_percent: u8,

    /// Number of ONNX intra-op threads per embedding session.
    /// 0 = auto-detect (always resolves to 2; the all-MiniLM-L6-v2 model
    /// is too small to benefit from more intra-op parallelism).
    #[serde(default = "default_onnx_intra_threads")]
    pub onnx_intra_threads: usize,

    // --- Adaptive Idle Mode ---
    /// Seconds of no user input before entering idle mode.
    #[serde(default = "default_idle_threshold_secs")]
    pub idle_threshold_secs: u64,

    /// Seconds of sustained idle required before the first upward level transition.
    /// Default: 300 (5 minutes)
    #[serde(default = "default_idle_confirmation_secs")]
    pub idle_confirmation_secs: u64,

    /// Seconds to spend at each level during ramp-up (after confirmation).
    /// Default: 120 (2 minutes per level, so Active->Burst takes ~4 min)
    #[serde(default = "default_ramp_up_step_secs")]
    pub ramp_up_step_secs: u64,

    /// Seconds of sustained user activity required before each downward level transition.
    /// Default: 300 (5 minutes per level)
    #[serde(default = "default_ramp_down_step_secs")]
    pub ramp_down_step_secs: u64,

    /// Minimum seconds to hold at Burst before allowing ramp-down.
    /// Default: 600 (10 minutes -- "generous time at max")
    #[serde(default = "default_burst_hold_secs")]
    pub burst_hold_secs: u64,

    /// Multiplier for max_concurrent_embeddings in burst mode.
    #[serde(default = "default_burst_concurrency_multiplier")]
    pub burst_concurrency_multiplier: f64,

    /// CPU load fraction above which burst is suppressed.
    #[serde(default = "default_cpu_pressure_threshold")]
    pub cpu_pressure_threshold: f64,

    /// How often to poll idle state (seconds).
    #[serde(default = "default_idle_poll_interval_secs")]
    pub idle_poll_interval_secs: u64,

    // NOTE: idle_cooloff_polls removed -- replaced by ramp_down_step_secs
    // which provides proper per-level descent timing instead of a crude poll counter.

    // --- Active Processing Mode ---
    /// Multiplier for max_concurrent_embeddings when user is active but queue has work.
    /// Default: 1.5 (+50% over normal)
    #[serde(default = "default_active_concurrency_multiplier")]
    pub active_concurrency_multiplier: f64,

    // --- Linux Idle Detection ---
    /// Backend for Linux idle detection. Only consulted when running on Linux.
    /// - `"none"`: No idle detection (default). Adaptive manager stays at Normal/Active.
    /// - `"proc"`: `/proc/loadavg` heuristic. Treats host as idle when the 1-minute
    ///   load average normalized by physical cores falls below
    ///   `linux_idle_load_threshold`.
    ///
    /// Future values: `"systemd"` (logind `IdleHint` via DBus), `"manual"` (operator
    /// command), `"x11"` (Wayland/X11 session bind-mount).
    #[serde(default = "default_linux_idle_source")]
    pub linux_idle_source: String,

    /// Normalized load-average threshold for the `/proc` Linux idle heuristic.
    /// Below this ratio (1-minute load / physical cores), the host is considered idle.
    /// Default: 0.1
    #[serde(default = "default_linux_idle_load_threshold")]
    pub linux_idle_load_threshold: f64,
}

impl Default for ResourceLimitsConfig {
    fn default() -> Self {
        Self {
            nice_level: default_nice_level(),
            max_concurrent_embeddings: default_max_concurrent_embeddings(),
            max_memory_percent: default_max_memory_percent(),
            onnx_intra_threads: default_onnx_intra_threads(),
            idle_threshold_secs: default_idle_threshold_secs(),
            idle_confirmation_secs: default_idle_confirmation_secs(),
            ramp_up_step_secs: default_ramp_up_step_secs(),
            ramp_down_step_secs: default_ramp_down_step_secs(),
            burst_hold_secs: default_burst_hold_secs(),
            burst_concurrency_multiplier: default_burst_concurrency_multiplier(),
            cpu_pressure_threshold: default_cpu_pressure_threshold(),
            idle_poll_interval_secs: default_idle_poll_interval_secs(),
            active_concurrency_multiplier: default_active_concurrency_multiplier(),
            linux_idle_source: default_linux_idle_source(),
            linux_idle_load_threshold: default_linux_idle_load_threshold(),
        }
    }
}
