//! Environment variable overrides for [`ResourceLimitsConfig`].
use super::ResourceLimitsConfig;

impl ResourceLimitsConfig {
    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        apply_env_i32("WQM_RESOURCE_NICE_LEVEL", &mut self.nice_level);
        apply_env_usize(
            "WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS",
            &mut self.max_concurrent_embeddings,
        );
        apply_env_u8(
            "WQM_RESOURCE_MAX_MEMORY_PERCENT",
            &mut self.max_memory_percent,
        );
        apply_env_usize(
            "WQM_RESOURCE_ONNX_INTRA_THREADS",
            &mut self.onnx_intra_threads,
        );
        apply_env_u64(
            "WQM_RESOURCE_IDLE_THRESHOLD_SECS",
            &mut self.idle_threshold_secs,
        );
        apply_env_u64(
            "WQM_RESOURCE_IDLE_CONFIRMATION_SECS",
            &mut self.idle_confirmation_secs,
        );
        apply_env_u64(
            "WQM_RESOURCE_RAMP_UP_STEP_SECS",
            &mut self.ramp_up_step_secs,
        );
        apply_env_u64(
            "WQM_RESOURCE_RAMP_DOWN_STEP_SECS",
            &mut self.ramp_down_step_secs,
        );
        apply_env_u64("WQM_RESOURCE_BURST_HOLD_SECS", &mut self.burst_hold_secs);
        apply_env_f64(
            "WQM_RESOURCE_BURST_CONCURRENCY_MULTIPLIER",
            &mut self.burst_concurrency_multiplier,
        );
        apply_env_f64(
            "WQM_RESOURCE_CPU_PRESSURE_THRESHOLD",
            &mut self.cpu_pressure_threshold,
        );
        apply_env_u64(
            "WQM_RESOURCE_IDLE_POLL_INTERVAL_SECS",
            &mut self.idle_poll_interval_secs,
        );
        apply_env_f64(
            "WQM_RESOURCE_ACTIVE_CONCURRENCY_MULTIPLIER",
            &mut self.active_concurrency_multiplier,
        );
        apply_env_string(
            "WQM_RESOURCE_LINUX_IDLE_SOURCE",
            &mut self.linux_idle_source,
        );
        apply_env_f64(
            "WQM_RESOURCE_LINUX_IDLE_LOAD_THRESHOLD",
            &mut self.linux_idle_load_threshold,
        );
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

/// Apply an environment variable override to a `String` field.
fn apply_env_string(var: &str, field: &mut String) {
    if let Ok(val) = std::env::var(var) {
        if !val.is_empty() {
            *field = val;
        }
    }
}
