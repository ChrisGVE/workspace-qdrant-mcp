//! Main LSP configuration and feature flags
//!
//! Contains the top-level `LspConfig` struct with global timeout settings,
//! resource limits, logging, and feature flags. Also handles configuration
//! file I/O (JSON and YAML) and validation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::lsp::Language;

use super::language_config::{self, LanguageConfig};
use super::server_config::{self, ServerConfig};

/// Main LSP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Path to SQLite database for unified daemon state management
    pub database_path: PathBuf,

    /// Global timeout settings
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub shutdown_timeout: Duration,

    /// Health check configuration
    pub health_check_interval: Duration,
    pub health_check_timeout: Duration,
    pub max_consecutive_failures: u32,

    /// Restart policy settings
    pub enable_auto_restart: bool,
    pub max_restart_attempts: u32,
    pub restart_base_delay: Duration,
    pub restart_max_delay: Duration,
    pub restart_backoff_multiplier: f64,

    /// Resource limits
    pub max_memory_mb: Option<u64>,
    pub max_cpu_percent: Option<f32>,
    pub max_concurrent_requests: u32,

    /// Logging settings
    pub log_communication: bool,
    pub log_level: String,
    pub max_log_retention_days: u32,

    /// Language-specific configurations
    pub language_configs: HashMap<Language, LanguageConfig>,

    /// Server-specific configurations
    pub server_configs: HashMap<String, ServerConfig>,

    /// Global server environment variables
    pub global_environment: HashMap<String, String>,

    /// Feature flags
    pub features: LspFeatures,
}

/// LSP feature flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspFeatures {
    /// Enable LSP integration
    pub enabled: bool,

    /// Enable automatic server detection
    pub auto_detection: bool,

    /// Enable health monitoring
    pub health_monitoring: bool,

    /// Enable automatic restart
    pub auto_restart: bool,

    /// Enable communication logging
    pub communication_logging: bool,

    /// Enable performance metrics collection
    pub performance_metrics: bool,

    /// Enable resource monitoring
    pub resource_monitoring: bool,

    /// Enable experimental features
    pub experimental: bool,

    /// Enable circuit breaker pattern
    pub circuit_breaker: bool,

    /// Enable request batching
    pub request_batching: bool,

    /// Enable caching
    pub caching: bool,
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("state.db"),

            startup_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(10),

            health_check_interval: Duration::from_secs(60),
            health_check_timeout: Duration::from_secs(5),
            max_consecutive_failures: 5,

            enable_auto_restart: true,
            max_restart_attempts: 5,
            restart_base_delay: Duration::from_secs(1),
            restart_max_delay: Duration::from_secs(300),
            restart_backoff_multiplier: 2.0,

            max_memory_mb: Some(512),
            max_cpu_percent: Some(50.0),
            max_concurrent_requests: 100,

            log_communication: false,
            log_level: "info".to_string(),
            max_log_retention_days: 7,

            language_configs: language_config::default_language_configs(),
            server_configs: server_config::default_server_configs(),
            global_environment: HashMap::new(),

            features: LspFeatures::default(),
        }
    }
}

impl Default for LspFeatures {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_detection: true,
            health_monitoring: true,
            auto_restart: true,
            communication_logging: false,
            performance_metrics: true,
            resource_monitoring: false,
            experimental: false,
            circuit_breaker: true,
            request_batching: false,
            caching: false,
        }
    }
}

impl LspConfig {
    /// Load configuration from file
    pub async fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        let content = tokio::fs::read_to_string(path_ref).await?;
        let config: LspConfig = if path_ref.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yml::from_str(&content)?
        } else {
            serde_json::from_str(&content)?
        };
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        let content = if path_ref.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yml::to_string(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        tokio::fs::write(path_ref, content).await?;
        Ok(())
    }

    /// Get configuration for a specific language
    pub fn get_language_config(&self, language: &Language) -> Option<&LanguageConfig> {
        self.language_configs.get(language)
    }

    /// Get configuration for a specific server
    pub fn get_server_config(&self, server_name: &str) -> Option<&ServerConfig> {
        self.server_configs.get(server_name)
    }

    /// Set language configuration
    pub fn set_language_config(&mut self, language: Language, config: LanguageConfig) {
        self.language_configs.insert(language, config);
    }

    /// Set server configuration
    pub fn set_server_config(&mut self, server_name: String, config: ServerConfig) {
        self.server_configs.insert(server_name, config);
    }

    /// Check if a feature is enabled
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        match feature {
            "auto_detection" => self.features.auto_detection,
            "health_monitoring" => self.features.health_monitoring,
            "auto_restart" => self.features.auto_restart,
            "communication_logging" => self.features.communication_logging,
            "performance_metrics" => self.features.performance_metrics,
            "resource_monitoring" => self.features.resource_monitoring,
            "experimental" => self.features.experimental,
            "circuit_breaker" => self.features.circuit_breaker,
            "request_batching" => self.features.request_batching,
            "caching" => self.features.caching,
            _ => false,
        }
    }

    /// Enable a feature
    pub fn enable_feature(&mut self, feature: &str) {
        match feature {
            "auto_detection" => self.features.auto_detection = true,
            "health_monitoring" => self.features.health_monitoring = true,
            "auto_restart" => self.features.auto_restart = true,
            "communication_logging" => self.features.communication_logging = true,
            "performance_metrics" => self.features.performance_metrics = true,
            "resource_monitoring" => self.features.resource_monitoring = true,
            "experimental" => self.features.experimental = true,
            "circuit_breaker" => self.features.circuit_breaker = true,
            "request_batching" => self.features.request_batching = true,
            "caching" => self.features.caching = true,
            _ => {}
        }
    }

    /// Disable a feature
    pub fn disable_feature(&mut self, feature: &str) {
        match feature {
            "auto_detection" => self.features.auto_detection = false,
            "health_monitoring" => self.features.health_monitoring = false,
            "auto_restart" => self.features.auto_restart = false,
            "communication_logging" => self.features.communication_logging = false,
            "performance_metrics" => self.features.performance_metrics = false,
            "resource_monitoring" => self.features.resource_monitoring = false,
            "experimental" => self.features.experimental = false,
            "circuit_breaker" => self.features.circuit_breaker = false,
            "request_batching" => self.features.request_batching = false,
            "caching" => self.features.caching = false,
            _ => {}
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        self.validate_timeouts()?;
        self.validate_resource_limits()?;
        self.validate_restart_policy()?;
        self.validate_language_configs()?;
        Ok(())
    }

    fn validate_timeouts(&self) -> Result<(), String> {
        if self.startup_timeout < Duration::from_secs(1) {
            return Err("Startup timeout too short".to_string());
        }
        if self.request_timeout < Duration::from_millis(100) {
            return Err("Request timeout too short".to_string());
        }
        Ok(())
    }

    fn validate_resource_limits(&self) -> Result<(), String> {
        if let Some(memory_mb) = self.max_memory_mb {
            if memory_mb < 64 {
                return Err("Memory limit too low (minimum 64MB)".to_string());
            }
        }
        if let Some(cpu_percent) = self.max_cpu_percent {
            if cpu_percent <= 0.0 || cpu_percent > 100.0 {
                return Err("CPU limit must be between 0 and 100 percent".to_string());
            }
        }
        Ok(())
    }

    fn validate_restart_policy(&self) -> Result<(), String> {
        if self.max_restart_attempts == 0 {
            return Err("Max restart attempts cannot be zero".to_string());
        }
        if self.restart_base_delay < Duration::from_millis(100) {
            return Err("Restart base delay too short".to_string());
        }
        Ok(())
    }

    fn validate_language_configs(&self) -> Result<(), String> {
        for (language, config) in &self.language_configs {
            if config.preferred_servers.is_empty() {
                return Err(format!("No preferred servers for language: {:?}", language));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config_creation() {
        let config = LspConfig::default();
        assert!(config.features.enabled);
        assert!(config.features.auto_detection);
        assert!(!config.language_configs.is_empty());
        assert!(!config.server_configs.is_empty());
    }

    #[test]
    fn test_config_validation() {
        let mut config = LspConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid timeout
        config.startup_timeout = Duration::from_millis(100);
        assert!(config.validate().is_err());

        // Fix timeout and test invalid memory limit
        config.startup_timeout = Duration::from_secs(10);
        config.max_memory_mb = Some(32);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_management() {
        let mut config = LspConfig::default();

        assert!(config.is_feature_enabled("auto_detection"));
        config.disable_feature("auto_detection");
        assert!(!config.is_feature_enabled("auto_detection"));

        config.enable_feature("experimental");
        assert!(config.is_feature_enabled("experimental"));
    }

    #[test]
    fn test_language_config_access() {
        let config = LspConfig::default();

        let python_config = config.get_language_config(&Language::Python);
        assert!(python_config.is_some());
        assert!(!python_config.unwrap().preferred_servers.is_empty());

        let unknown_config =
            config.get_language_config(&Language::Other("unknown".to_string()));
        assert!(unknown_config.is_none());
    }

    #[test]
    fn test_server_config_access() {
        let config = LspConfig::default();

        let rust_analyzer = config.get_server_config("rust-analyzer");
        assert!(rust_analyzer.is_some());
        assert!(rust_analyzer.unwrap().enabled);

        let unknown_server = config.get_server_config("unknown-server");
        assert!(unknown_server.is_none());
    }

    #[tokio::test]
    async fn test_config_file_operations() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");

        let mut original_config = LspConfig::default();
        original_config.max_memory_mb = Some(1024);
        original_config.enable_feature("experimental");

        // Save config
        original_config.save_to_file(&config_path).await.unwrap();

        // Load config
        let loaded_config = LspConfig::load_from_file(&config_path).await.unwrap();
        assert_eq!(loaded_config.max_memory_mb, Some(1024));
        assert!(loaded_config.is_feature_enabled("experimental"));
    }

    #[tokio::test]
    async fn test_yaml_config_operations() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.yaml");

        let mut original_config = LspConfig::default();
        original_config.log_level = "debug".to_string();

        // Save as YAML
        original_config.save_to_file(&config_path).await.unwrap();

        // Load from YAML
        let loaded_config = LspConfig::load_from_file(&config_path).await.unwrap();
        assert_eq!(loaded_config.log_level, "debug");
    }
}
