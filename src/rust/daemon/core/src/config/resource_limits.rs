//! Resource limits configuration (CPU, memory, adaptive idle mode)
use serde::{Deserialize, Serialize};

fn default_nice_level() -> i32 { 10 }
fn default_inter_item_delay_ms() -> u64 { 50 }
fn default_max_concurrent_embeddings() -> usize { 0 } // 0 = auto-detect
fn default_max_memory_percent() -> u8 { 70 }
fn default_onnx_intra_threads() -> usize { 0 } // 0 = auto-detect
fn default_idle_threshold_secs() -> u64 { 120 }
fn default_idle_confirmation_secs() -> u64 { 300 }
fn default_ramp_up_step_secs() -> u64 { 120 }
fn default_ramp_down_step_secs() -> u64 { 300 }
fn default_burst_hold_secs() -> u64 { 600 }
fn default_burst_concurrency_multiplier() -> f64 { 2.0 }
fn default_burst_inter_item_delay_ms() -> u64 { 0 }
fn default_cpu_pressure_threshold() -> f64 { 0.6 }
fn default_idle_poll_interval_secs() -> u64 { 5 }
fn default_active_concurrency_multiplier() -> f64 { 1.5 }
fn default_active_inter_item_delay_ms() -> u64 { 25 }

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

    /// Delay in milliseconds between processing items (breathing room)
    /// Default: 50ms
    #[serde(default = "default_inter_item_delay_ms")]
    pub inter_item_delay_ms: u64,

    /// Maximum concurrent embedding operations (semaphore on ONNX ops).
    /// 0 = auto-detect from CPU core count (physical_cores / 4, minimum 1).
    #[serde(default = "default_max_concurrent_embeddings")]
    pub max_concurrent_embeddings: usize,

    /// Pause processing when system memory usage exceeds this percentage
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

    /// Inter-item delay in full burst mode (ms).
    #[serde(default = "default_burst_inter_item_delay_ms")]
    pub burst_inter_item_delay_ms: u64,

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

    /// Inter-item delay in ms during active processing (user present, queue has work).
    /// Default: 25 (half of normal 50ms)
    #[serde(default = "default_active_inter_item_delay_ms")]
    pub active_inter_item_delay_ms: u64,
}

impl Default for ResourceLimitsConfig {
    fn default() -> Self {
        Self {
            nice_level: default_nice_level(),
            inter_item_delay_ms: default_inter_item_delay_ms(),
            max_concurrent_embeddings: default_max_concurrent_embeddings(),
            max_memory_percent: default_max_memory_percent(),
            onnx_intra_threads: default_onnx_intra_threads(),
            idle_threshold_secs: default_idle_threshold_secs(),
            idle_confirmation_secs: default_idle_confirmation_secs(),
            ramp_up_step_secs: default_ramp_up_step_secs(),
            ramp_down_step_secs: default_ramp_down_step_secs(),
            burst_hold_secs: default_burst_hold_secs(),
            burst_concurrency_multiplier: default_burst_concurrency_multiplier(),
            burst_inter_item_delay_ms: default_burst_inter_item_delay_ms(),
            cpu_pressure_threshold: default_cpu_pressure_threshold(),
            idle_poll_interval_secs: default_idle_poll_interval_secs(),
            active_concurrency_multiplier: default_active_concurrency_multiplier(),
            active_inter_item_delay_ms: default_active_inter_item_delay_ms(),
        }
    }
}

impl ResourceLimitsConfig {
    /// Resolve auto-detected values (0 = auto) based on hardware.
    ///
    /// Call this after loading config and applying env overrides, before validation.
    /// Values that are already non-zero (explicitly set) are left unchanged.
    ///
    /// Formula:
    ///   - `max_concurrent_embeddings`: max(1, physical_cores / 4)
    ///   - `onnx_intra_threads`: 2 (all-MiniLM-L6-v2 is too small to benefit from more)
    pub fn resolve_auto_values(&mut self) {
        let physical_cores = detect_physical_cores();

        if self.max_concurrent_embeddings == 0 {
            self.max_concurrent_embeddings = std::cmp::max(1, physical_cores / 4);
            tracing::info!(
                "Auto-detected max_concurrent_embeddings = {} (physical_cores={}, formula: cores/4)",
                self.max_concurrent_embeddings, physical_cores
            );
        }

        if self.onnx_intra_threads == 0 {
            // 2 is optimal for the small all-MiniLM-L6-v2 model regardless of core count.
            // The ONNX graph is narrow enough that additional threads yield diminishing returns.
            self.onnx_intra_threads = 2;
            tracing::info!(
                "Auto-detected onnx_intra_threads = 2 (optimal for all-MiniLM-L6-v2)"
            );
        }

        tracing::info!(
            "Embedding resource budget: {} workers x {} threads/worker = {} total threads (physical_cores={})",
            self.max_concurrent_embeddings, self.onnx_intra_threads,
            self.max_concurrent_embeddings * self.onnx_intra_threads, physical_cores
        );
    }

    /// Validate configuration settings.
    ///
    /// Must be called AFTER `resolve_auto_values()` -- 0 is not valid post-resolution.
    pub fn validate(&self) -> Result<(), String> {
        if self.nice_level < -20 || self.nice_level > 19 {
            return Err("nice_level must be between -20 and 19".to_string());
        }
        if self.inter_item_delay_ms > 5000 {
            return Err("inter_item_delay_ms should not exceed 5000".to_string());
        }
        if self.max_concurrent_embeddings == 0 || self.max_concurrent_embeddings > 8 {
            return Err("max_concurrent_embeddings must be between 1 and 8 (0 should have been auto-resolved)".to_string());
        }
        if self.max_memory_percent < 20 || self.max_memory_percent > 95 {
            return Err("max_memory_percent must be between 20 and 95".to_string());
        }
        if self.onnx_intra_threads == 0 || self.onnx_intra_threads > 16 {
            return Err("onnx_intra_threads must be between 1 and 16 (0 should have been auto-resolved)".to_string());
        }
        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        apply_env_i32("WQM_RESOURCE_NICE_LEVEL", &mut self.nice_level);
        apply_env_u64("WQM_RESOURCE_INTER_ITEM_DELAY_MS", &mut self.inter_item_delay_ms);
        apply_env_usize("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS", &mut self.max_concurrent_embeddings);
        apply_env_u8("WQM_RESOURCE_MAX_MEMORY_PERCENT", &mut self.max_memory_percent);
        apply_env_usize("WQM_RESOURCE_ONNX_INTRA_THREADS", &mut self.onnx_intra_threads);
        apply_env_u64("WQM_RESOURCE_IDLE_THRESHOLD_SECS", &mut self.idle_threshold_secs);
        apply_env_u64("WQM_RESOURCE_IDLE_CONFIRMATION_SECS", &mut self.idle_confirmation_secs);
        apply_env_u64("WQM_RESOURCE_RAMP_UP_STEP_SECS", &mut self.ramp_up_step_secs);
        apply_env_u64("WQM_RESOURCE_RAMP_DOWN_STEP_SECS", &mut self.ramp_down_step_secs);
        apply_env_u64("WQM_RESOURCE_BURST_HOLD_SECS", &mut self.burst_hold_secs);
        apply_env_f64("WQM_RESOURCE_BURST_CONCURRENCY_MULTIPLIER", &mut self.burst_concurrency_multiplier);
        apply_env_u64("WQM_RESOURCE_BURST_INTER_ITEM_DELAY_MS", &mut self.burst_inter_item_delay_ms);
        apply_env_f64("WQM_RESOURCE_CPU_PRESSURE_THRESHOLD", &mut self.cpu_pressure_threshold);
        apply_env_u64("WQM_RESOURCE_IDLE_POLL_INTERVAL_SECS", &mut self.idle_poll_interval_secs);
        apply_env_f64("WQM_RESOURCE_ACTIVE_CONCURRENCY_MULTIPLIER", &mut self.active_concurrency_multiplier);
        apply_env_u64("WQM_RESOURCE_ACTIVE_INTER_ITEM_DELAY_MS", &mut self.active_inter_item_delay_ms);
    }
}

/// Apply an environment variable override to an `i32` field.
fn apply_env_i32(var: &str, field: &mut i32) {
    if let Ok(val) = std::env::var(var) {
        if let Ok(parsed) = val.parse() {
            *field = parsed;
        }
    }
}

/// Apply an environment variable override to a `u8` field.
fn apply_env_u8(var: &str, field: &mut u8) {
    if let Ok(val) = std::env::var(var) {
        if let Ok(parsed) = val.parse() {
            *field = parsed;
        }
    }
}

/// Apply an environment variable override to a `u64` field.
fn apply_env_u64(var: &str, field: &mut u64) {
    if let Ok(val) = std::env::var(var) {
        if let Ok(parsed) = val.parse() {
            *field = parsed;
        }
    }
}

/// Apply an environment variable override to a `usize` field.
fn apply_env_usize(var: &str, field: &mut usize) {
    if let Ok(val) = std::env::var(var) {
        if let Ok(parsed) = val.parse() {
            *field = parsed;
        }
    }
}

/// Apply an environment variable override to an `f64` field.
fn apply_env_f64(var: &str, field: &mut f64) {
    if let Ok(val) = std::env::var(var) {
        if let Ok(parsed) = val.parse() {
            *field = parsed;
        }
    }
}

/// Detect the number of physical CPU cores on the current machine.
///
/// Uses `sysinfo::System::physical_core_count()` with a fallback to
/// `std::thread::available_parallelism()` (which returns logical cores).
/// Returns 4 as a safe fallback if both methods fail.
pub fn detect_physical_cores() -> usize {
    use sysinfo::System;

    let sys = System::new();
    if let Some(physical) = sys.physical_core_count() {
        return physical;
    }

    // Fallback: logical cores (includes hyperthreading)
    if let Ok(logical) = std::thread::available_parallelism() {
        return logical.get();
    }

    // Ultimate fallback
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_limits_config_defaults() {
        let config = ResourceLimitsConfig::default();
        assert_eq!(config.nice_level, 10);
        assert_eq!(config.inter_item_delay_ms, 50);
        assert_eq!(config.max_concurrent_embeddings, 0, "default should be 0 (auto-detect)");
        assert_eq!(config.max_memory_percent, 70);
        assert_eq!(config.onnx_intra_threads, 0, "default should be 0 (auto-detect)");
    }

    #[test]
    fn test_resource_limits_auto_detection() {
        let mut config = ResourceLimitsConfig::default();
        assert_eq!(config.max_concurrent_embeddings, 0);
        assert_eq!(config.onnx_intra_threads, 0);

        config.resolve_auto_values();

        // After resolution, values should be non-zero
        assert!(config.max_concurrent_embeddings >= 1, "auto-detected embeddings should be >= 1");
        assert!(config.max_concurrent_embeddings <= 8, "auto-detected embeddings should be <= 8");
        assert_eq!(config.onnx_intra_threads, 2, "onnx_intra_threads always resolves to 2");

        // Validation should pass after resolution
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_explicit_values_not_overridden() {
        let mut config = ResourceLimitsConfig::default();
        config.max_concurrent_embeddings = 3;
        config.onnx_intra_threads = 4;

        config.resolve_auto_values();

        // Explicitly set values should not be changed
        assert_eq!(config.max_concurrent_embeddings, 3);
        assert_eq!(config.onnx_intra_threads, 4);
    }

    #[test]
    fn test_resource_limits_config_validation() {
        let mut config = ResourceLimitsConfig::default();
        config.resolve_auto_values(); // Must resolve before validation

        // Valid settings after resolution
        assert!(config.validate().is_ok());

        // Invalid nice_level (too low)
        config.nice_level = -21;
        assert!(config.validate().is_err());
        // Invalid nice_level (too high)
        config.nice_level = 20;
        assert!(config.validate().is_err());
        config.nice_level = 10;

        // Invalid inter_item_delay_ms (too high)
        config.inter_item_delay_ms = 5001;
        assert!(config.validate().is_err());
        config.inter_item_delay_ms = 50;

        // Invalid max_concurrent_embeddings (zero = unresolved auto-detect)
        config.max_concurrent_embeddings = 0;
        assert!(config.validate().is_err());
        // Invalid max_concurrent_embeddings (too high)
        config.max_concurrent_embeddings = 9;
        assert!(config.validate().is_err());
        config.max_concurrent_embeddings = 2;

        // Invalid max_memory_percent (too low)
        config.max_memory_percent = 19;
        assert!(config.validate().is_err());
        // Invalid max_memory_percent (too high)
        config.max_memory_percent = 96;
        assert!(config.validate().is_err());
        config.max_memory_percent = 70;

        // Invalid onnx_intra_threads (zero = unresolved)
        config.onnx_intra_threads = 0;
        assert!(config.validate().is_err());
        // Invalid onnx_intra_threads (too high)
        config.onnx_intra_threads = 17;
        assert!(config.validate().is_err());
        config.onnx_intra_threads = 2;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_config_boundary_values() {
        let mut config = ResourceLimitsConfig::default();
        config.resolve_auto_values(); // Must resolve before validation

        config.nice_level = -20;
        assert!(config.validate().is_ok());
        config.nice_level = 19;
        assert!(config.validate().is_ok());

        config.inter_item_delay_ms = 0;
        assert!(config.validate().is_ok());
        config.inter_item_delay_ms = 5000;
        assert!(config.validate().is_ok());

        config.max_concurrent_embeddings = 1;
        assert!(config.validate().is_ok());
        config.max_concurrent_embeddings = 8;
        assert!(config.validate().is_ok());

        config.onnx_intra_threads = 1;
        assert!(config.validate().is_ok());
        config.onnx_intra_threads = 16;
        assert!(config.validate().is_ok());

        config.max_memory_percent = 20;
        assert!(config.validate().is_ok());
        config.max_memory_percent = 95;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_env_overrides() {
        let mut config = ResourceLimitsConfig::default();

        // Set environment variables
        std::env::set_var("WQM_RESOURCE_NICE_LEVEL", "5");
        std::env::set_var("WQM_RESOURCE_INTER_ITEM_DELAY_MS", "100");
        std::env::set_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS", "4");
        std::env::set_var("WQM_RESOURCE_MAX_MEMORY_PERCENT", "80");

        config.apply_env_overrides();

        assert_eq!(config.nice_level, 5);
        assert_eq!(config.inter_item_delay_ms, 100);
        assert_eq!(config.max_concurrent_embeddings, 4);
        assert_eq!(config.max_memory_percent, 80);

        // Clean up
        std::env::remove_var("WQM_RESOURCE_NICE_LEVEL");
        std::env::remove_var("WQM_RESOURCE_INTER_ITEM_DELAY_MS");
        std::env::remove_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS");
        std::env::remove_var("WQM_RESOURCE_MAX_MEMORY_PERCENT");
    }

    #[test]
    fn test_resource_limits_serialization() {
        let config = ResourceLimitsConfig {
            nice_level: 5,
            inter_item_delay_ms: 100,
            max_concurrent_embeddings: 4,
            max_memory_percent: 80,
            onnx_intra_threads: 2,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"nice_level\":5"));
        assert!(json.contains("\"max_memory_percent\":80"));

        let deserialized: ResourceLimitsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nice_level, 5);
        assert_eq!(deserialized.max_memory_percent, 80);
        // Idle config should have defaults
        assert_eq!(deserialized.idle_threshold_secs, 120);
        assert_eq!(deserialized.idle_confirmation_secs, 300);
        assert_eq!(deserialized.ramp_up_step_secs, 120);
        assert_eq!(deserialized.ramp_down_step_secs, 300);
        assert_eq!(deserialized.burst_hold_secs, 600);
        assert!((deserialized.burst_concurrency_multiplier - 2.0).abs() < f64::EPSILON);
    }
}
