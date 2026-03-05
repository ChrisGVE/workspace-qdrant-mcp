//! Observability configuration (logging, monitoring, metrics, telemetry)

use serde::{Deserialize, Serialize};

// --- Logging ---

/// Logging configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub info_includes_connection_events: bool,
    pub info_includes_transport_details: bool,
    pub info_includes_retry_attempts: bool,
    pub info_includes_fallback_behavior: bool,
    pub error_includes_stack_trace: bool,
    pub error_includes_connection_state: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            info_includes_connection_events: true,
            info_includes_transport_details: true,
            info_includes_retry_attempts: true,
            info_includes_fallback_behavior: true,
            error_includes_stack_trace: true,
            error_includes_connection_state: true,
        }
    }
}

// --- Monitoring ---

fn default_check_interval_hours() -> u64 {
    24
}
fn default_check_on_startup() -> bool {
    true
}
fn default_enable_monitoring() -> bool {
    true
}

/// Tool monitoring configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Check interval in hours
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u64,

    /// Check on daemon startup
    #[serde(default = "default_check_on_startup")]
    pub check_on_startup: bool,

    /// Enable tool availability monitoring
    #[serde(default = "default_enable_monitoring")]
    pub enable_monitoring: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            check_interval_hours: default_check_interval_hours(),
            check_on_startup: default_check_on_startup(),
            enable_monitoring: default_enable_monitoring(),
        }
    }
}

impl MonitoringConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.check_interval_hours == 0 {
            return Err("check_interval_hours must be greater than 0".to_string());
        }
        if self.check_interval_hours > 8760 {
            // 1 year
            return Err("check_interval_hours should not exceed 8760 (1 year)".to_string());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_MONITOR_CHECK_INTERVAL_HOURS") {
            if let Ok(parsed) = val.parse() {
                self.check_interval_hours = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_MONITOR_CHECK_ON_STARTUP") {
            self.check_on_startup = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("WQM_MONITOR_ENABLE") {
            self.enable_monitoring = val.to_lowercase() == "true" || val == "1";
        }
    }
}

// --- Observability (metrics + telemetry) ---

fn default_collection_interval() -> u64 {
    60
}
fn default_history_retention() -> usize {
    120
}
fn default_telemetry_enabled() -> bool {
    true
}

/// Observability configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Collection interval in seconds
    #[serde(default = "default_collection_interval")]
    pub collection_interval: u64,

    /// Basic metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Detailed telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsConfig {
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_history_retention")]
    pub history_retention: usize,

    #[serde(default = "default_telemetry_enabled")]
    pub cpu_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub memory_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub latency: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub queue_depth: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub throughput: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_retention: default_history_retention(),
            cpu_usage: default_telemetry_enabled(),
            memory_usage: default_telemetry_enabled(),
            latency: default_telemetry_enabled(),
            queue_depth: default_telemetry_enabled(),
            throughput: default_telemetry_enabled(),
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            collection_interval: default_collection_interval(),
            metrics: MetricsConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_config_defaults() {
        let config = MonitoringConfig::default();
        assert_eq!(config.check_interval_hours, 24);
        assert!(config.check_on_startup);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_monitoring_config_validation() {
        let mut config = MonitoringConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid check_interval_hours
        config.check_interval_hours = 0;
        assert!(config.validate().is_err());
        config.check_interval_hours = 8761;
        assert!(config.validate().is_err());
        config.check_interval_hours = 24;

        // Valid again
        assert!(config.validate().is_ok());
    }
}
