//! Configuration management
//!
//! This module contains configuration management for the priority processing engine

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Processing engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: Option<usize>,
    /// Default timeout for tasks in milliseconds
    pub default_timeout_ms: Option<u64>,
    /// Enable task preemption
    pub enable_preemption: bool,
    /// Document processing chunk size
    pub chunk_size: usize,
    /// Enable LSP support
    pub enable_lsp: bool,
    /// Log level configuration
    pub log_level: String,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
}

impl Config {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self {
            max_concurrent_tasks: Some(4),
            default_timeout_ms: Some(30_000),
            enable_preemption: true,
            chunk_size: 1000,
            enable_lsp: true,
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 60,
        }
    }
    
    /// Create configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_concurrent_tasks: Some(8),
            default_timeout_ms: Some(10_000),
            enable_preemption: true,
            chunk_size: 2000,
            enable_lsp: false, // Disable LSP for better performance
            log_level: "warn".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 30,
        }
    }
    
    /// Create configuration optimized for responsiveness (MCP servers)
    pub fn responsive() -> Self {
        Self {
            max_concurrent_tasks: Some(2),
            default_timeout_ms: Some(5_000),
            enable_preemption: true,
            chunk_size: 500,
            enable_lsp: true,
            log_level: "debug".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 10,
        }
    }
    
    /// Create configuration for resource-constrained environments
    pub fn low_resource() -> Self {
        Self {
            max_concurrent_tasks: Some(1),
            default_timeout_ms: Some(60_000),
            enable_preemption: false,
            chunk_size: 500,
            enable_lsp: false,
            log_level: "error".to_string(),
            enable_metrics: false,
            metrics_interval_secs: 300,
        }
    }
    
    /// Get default timeout as Duration
    pub fn default_timeout(&self) -> Option<Duration> {
        self.default_timeout_ms.map(Duration::from_millis)
    }
    
    /// Get metrics interval as Duration
    pub fn metrics_interval(&self) -> Duration {
        Duration::from_secs(self.metrics_interval_secs)
    }
    
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if let Some(max_concurrent) = self.max_concurrent_tasks {
            if max_concurrent == 0 {
                return Err("max_concurrent_tasks must be greater than 0".to_string());
            }
            if max_concurrent > 100 {
                return Err("max_concurrent_tasks should not exceed 100".to_string());
            }
        }
        
        if let Some(timeout) = self.default_timeout_ms {
            if timeout == 0 {
                return Err("default_timeout_ms must be greater than 0".to_string());
            }
            if timeout > 300_000 {
                return Err("default_timeout_ms should not exceed 5 minutes".to_string());
            }
        }
        
        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0".to_string());
        }
        if self.chunk_size > 10_000 {
            return Err("chunk_size should not exceed 10,000".to_string());
        }
        
        if !matches!(self.log_level.as_str(), "trace" | "debug" | "info" | "warn" | "error") {
            return Err("log_level must be one of: trace, debug, info, warn, error".to_string());
        }
        
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy engine configuration (for backward compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub chunk_size: usize,
    pub enable_lsp: bool,
}

impl EngineConfig {
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            enable_lsp: true,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert legacy config to new config
impl From<EngineConfig> for Config {
    fn from(legacy: EngineConfig) -> Self {
        Self {
            chunk_size: legacy.chunk_size,
            enable_lsp: legacy.enable_lsp,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = Config::new();
        assert_eq!(config.max_concurrent_tasks, Some(4));
        assert_eq!(config.default_timeout_ms, Some(30_000));
        assert!(config.enable_preemption);
        assert_eq!(config.chunk_size, 1000);
        assert!(config.enable_lsp);
    }
    
    #[test]
    fn test_high_throughput_config() {
        let config = Config::high_throughput();
        assert_eq!(config.max_concurrent_tasks, Some(8));
        assert_eq!(config.default_timeout_ms, Some(10_000));
        assert_eq!(config.chunk_size, 2000);
        assert!(!config.enable_lsp); // Should be disabled for performance
    }
    
    #[test]
    fn test_responsive_config() {
        let config = Config::responsive();
        assert_eq!(config.max_concurrent_tasks, Some(2));
        assert_eq!(config.default_timeout_ms, Some(5_000));
        assert_eq!(config.chunk_size, 500);
        assert!(config.enable_lsp);
    }
    
    #[test]
    fn test_low_resource_config() {
        let config = Config::low_resource();
        assert_eq!(config.max_concurrent_tasks, Some(1));
        assert_eq!(config.default_timeout_ms, Some(60_000));
        assert!(!config.enable_preemption);
        assert!(!config.enable_metrics);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::new();
        assert!(config.validate().is_ok());
        
        // Test invalid max_concurrent_tasks
        config.max_concurrent_tasks = Some(0);
        assert!(config.validate().is_err());
        
        config.max_concurrent_tasks = Some(150);
        assert!(config.validate().is_err());
        
        // Reset and test invalid timeout
        config.max_concurrent_tasks = Some(4);
        config.default_timeout_ms = Some(0);
        assert!(config.validate().is_err());
        
        // Test invalid chunk size
        config.default_timeout_ms = Some(30_000);
        config.chunk_size = 0;
        assert!(config.validate().is_err());
        
        config.chunk_size = 20_000;
        assert!(config.validate().is_err());
        
        // Test invalid log level
        config.chunk_size = 1000;
        config.log_level = "invalid".to_string();
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_legacy_config_conversion() {
        let legacy = EngineConfig {
            chunk_size: 2000,
            enable_lsp: false,
        };
        
        let config: Config = legacy.into();
        assert_eq!(config.chunk_size, 2000);
        assert!(!config.enable_lsp);
        // Other fields should be defaults
        assert_eq!(config.max_concurrent_tasks, Some(4));
    }
    
    #[test]
    fn test_duration_helpers() {
        let config = Config {
            default_timeout_ms: Some(5000),
            metrics_interval_secs: 120,
            ..Default::default()
        };
        
        assert_eq!(config.default_timeout(), Some(Duration::from_millis(5000)));
        assert_eq!(config.metrics_interval(), Duration::from_secs(120));
        
        let config_no_timeout = Config {
            default_timeout_ms: None,
            ..Default::default()
        };
        assert_eq!(config_no_timeout.default_timeout(), None);
    }
}