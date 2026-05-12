//! Integration configuration (Git and updates)

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

// --- Git Config ---

fn default_enable_branch_detection() -> bool {
    true
}
fn default_cache_ttl_seconds() -> u64 {
    60
}

/// Git integration configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// Enable Git branch detection
    #[serde(default = "default_enable_branch_detection")]
    pub enable_branch_detection: bool,

    /// Cache TTL in seconds for branch info
    #[serde(default = "default_cache_ttl_seconds")]
    pub cache_ttl_seconds: u64,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            enable_branch_detection: default_enable_branch_detection(),
            cache_ttl_seconds: default_cache_ttl_seconds(),
        }
    }
}

impl GitConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.cache_ttl_seconds == 0 {
            return Err("cache_ttl_seconds must be greater than 0".to_string());
        }
        if self.cache_ttl_seconds > 3600 {
            return Err("cache_ttl_seconds should not exceed 3600 (1 hour)".to_string());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_GIT_ENABLE_BRANCH_DETECTION") {
            self.enable_branch_detection = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("WQM_GIT_CACHE_TTL_SECONDS") {
            if let Ok(parsed) = val.parse() {
                self.cache_ttl_seconds = parsed;
            }
        }
    }
}

// --- Updates Config ---

/// Update channel for daemon self-updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UpdateChannel {
    /// Stable releases only (default)
    #[default]
    Stable,
    /// Beta releases for testing
    Beta,
    /// Development builds (may be unstable)
    Dev,
}

fn default_update_check_interval() -> u32 {
    24
} // Daily

/// Daemon self-update configuration
///
/// Controls how the daemon checks for and applies updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatesConfig {
    /// Check for updates on daemon startup
    #[serde(default = "default_true")]
    pub auto_check: bool,

    /// Update channel to follow
    #[serde(default)]
    pub channel: UpdateChannel,

    /// Only notify about updates, don't auto-install
    /// When true, updates are announced but not automatically applied
    #[serde(default = "default_true")]
    pub notify_only: bool,

    /// Interval in hours to check for updates (when auto_check is true)
    /// Default: 24 (daily)
    #[serde(default = "default_update_check_interval")]
    pub check_interval_hours: u32,
}

impl Default for UpdatesConfig {
    fn default() -> Self {
        Self {
            auto_check: default_true(),
            channel: UpdateChannel::default(),
            notify_only: default_true(),
            check_interval_hours: default_update_check_interval(),
        }
    }
}

impl UpdatesConfig {
    /// Validate configuration settings.
    ///
    /// `check_interval_hours` must be in the range [1, 8760] (1 hour – 1 year).
    pub fn validate(&self) -> Result<(), String> {
        if self.check_interval_hours == 0 {
            return Err("check_interval_hours must be at least 1".to_string());
        }
        if self.check_interval_hours > 8760 {
            return Err("check_interval_hours must not exceed 8760 (1 year)".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_config_defaults() {
        let config = GitConfig::default();
        assert!(config.enable_branch_detection);
        assert_eq!(config.cache_ttl_seconds, 60);
    }

    #[test]
    fn test_git_config_validation() {
        let mut config = GitConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid cache_ttl_seconds
        config.cache_ttl_seconds = 0;
        assert!(config.validate().is_err());
        config.cache_ttl_seconds = 3601;
        assert!(config.validate().is_err());
        config.cache_ttl_seconds = 60;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_updates_config_validate_default_ok() {
        assert!(UpdatesConfig::default().validate().is_ok());
    }

    #[test]
    fn test_updates_config_validate_rejects_zero_interval() {
        let config = UpdatesConfig {
            check_interval_hours: 0,
            ..UpdatesConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_updates_config_validate_rejects_over_8760() {
        let config = UpdatesConfig {
            check_interval_hours: 8761,
            ..UpdatesConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_updates_config_validate_accepts_boundary_values() {
        let min = UpdatesConfig {
            check_interval_hours: 1,
            ..UpdatesConfig::default()
        };
        assert!(min.validate().is_ok());

        let max = UpdatesConfig {
            check_interval_hours: 8760,
            ..UpdatesConfig::default()
        };
        assert!(max.validate().is_ok());
    }
}
