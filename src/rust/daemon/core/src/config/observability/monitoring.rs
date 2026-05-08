//! Tool monitoring configuration

use serde::{Deserialize, Serialize};

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
