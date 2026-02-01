//! LSP Configuration Module
//!
//! This module handles configuration management for LSP servers including
//! language-specific settings, server parameters, and timeout configurations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::lsp::Language;

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

/// Language-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Preferred LSP servers in priority order
    pub preferred_servers: Vec<String>,
    
    /// Language-specific settings
    pub settings: HashMap<String, JsonValue>,
    
    /// File patterns for this language
    pub file_patterns: Vec<String>,
    
    /// Workspace configuration
    pub workspace_settings: HashMap<String, JsonValue>,
    
    /// Enable/disable features for this language
    pub enabled_features: Vec<String>,
    pub disabled_features: Vec<String>,
    
    /// Timeout overrides
    pub request_timeout_override: Option<Duration>,
    pub startup_timeout_override: Option<Duration>,
}

/// Server-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Executable path override
    pub executable_path: Option<PathBuf>,
    
    /// Command line arguments
    pub arguments: Vec<String>,
    
    /// Environment variables for this server
    pub environment: HashMap<String, String>,
    
    /// Working directory override
    pub working_directory: Option<PathBuf>,
    
    /// Server-specific initialization options
    pub initialization_options: HashMap<String, JsonValue>,
    
    /// Server capabilities overrides
    pub capabilities_override: Option<ServerCapabilitiesOverride>,
    
    /// Resource limits for this server
    pub memory_limit_mb: Option<u64>,
    pub cpu_limit_percent: Option<f32>,
    
    /// Restart policy overrides
    pub restart_policy: Option<RestartPolicyOverride>,
    
    /// Communication settings
    pub use_stdio: bool,
    pub tcp_port: Option<u16>,
    
    /// Enable/disable this server
    pub enabled: bool,
}

/// Server capabilities override
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilitiesOverride {
    /// Force enable/disable specific capabilities
    pub force_enable: Vec<String>,
    pub force_disable: Vec<String>,
    
    /// Custom capability values
    pub custom_values: HashMap<String, JsonValue>,
}

/// Restart policy override for specific servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartPolicyOverride {
    pub enabled: Option<bool>,
    pub max_attempts: Option<u32>,
    pub base_delay: Option<Duration>,
    pub max_delay: Option<Duration>,
    pub backoff_multiplier: Option<f64>,
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
            
            language_configs: Self::default_language_configs(),
            server_configs: Self::default_server_configs(),
            global_environment: HashMap::new(),
            
            features: LspFeatures::default(),
        }
    }
}

impl LspConfig {
    /// Create default language configurations
    fn default_language_configs() -> HashMap<Language, LanguageConfig> {
        let mut configs = HashMap::new();

        // Python configuration
        configs.insert(Language::Python, LanguageConfig {
            preferred_servers: vec!["ruff-lsp".to_string(), "pylsp".to_string(), "pyright-langserver".to_string()],
            settings: {
                let mut settings = HashMap::new();
                settings.insert("python.analysis.autoSearchPaths".to_string(), JsonValue::Bool(true));
                settings.insert("python.analysis.useLibraryCodeForTypes".to_string(), JsonValue::Bool(true));
                settings
            },
            file_patterns: vec!["*.py".to_string(), "*.pyw".to_string(), "*.pyi".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        // Rust configuration
        configs.insert(Language::Rust, LanguageConfig {
            preferred_servers: vec!["rust-analyzer".to_string()],
            settings: {
                let mut settings = HashMap::new();
                settings.insert("rust-analyzer.checkOnSave.command".to_string(), 
                               JsonValue::String("clippy".to_string()));
                settings.insert("rust-analyzer.cargo.allFeatures".to_string(), JsonValue::Bool(true));
                settings
            },
            file_patterns: vec!["*.rs".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
                "code_action".to_string(),
                "formatting".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: Some(Duration::from_secs(60)), // Rust-analyzer can be slow
            startup_timeout_override: Some(Duration::from_secs(60)),
        });

        // TypeScript configuration
        configs.insert(Language::TypeScript, LanguageConfig {
            preferred_servers: vec!["typescript-language-server".to_string()],
            settings: {
                let mut settings = HashMap::new();
                settings.insert("typescript.preferences.includePackageJsonAutoImports".to_string(), 
                               JsonValue::String("on".to_string()));
                settings
            },
            file_patterns: vec!["*.ts".to_string(), "*.tsx".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
                "formatting".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        // JavaScript configuration
        configs.insert(Language::JavaScript, LanguageConfig {
            preferred_servers: vec!["typescript-language-server".to_string()],
            settings: HashMap::new(),
            file_patterns: vec!["*.js".to_string(), "*.jsx".to_string(), "*.mjs".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        // Go configuration
        configs.insert(Language::Go, LanguageConfig {
            preferred_servers: vec!["gopls".to_string()],
            settings: {
                let mut settings = HashMap::new();
                settings.insert("gopls.analyses.unusedparams".to_string(), JsonValue::Bool(true));
                settings.insert("gopls.analyses.shadow".to_string(), JsonValue::Bool(true));
                settings
            },
            file_patterns: vec!["*.go".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
                "formatting".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        // C/C++ configuration
        configs.insert(Language::C, LanguageConfig {
            preferred_servers: vec!["clangd".to_string(), "ccls".to_string()],
            settings: HashMap::new(),
            file_patterns: vec!["*.c".to_string(), "*.h".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        configs.insert(Language::Cpp, LanguageConfig {
            preferred_servers: vec!["clangd".to_string(), "ccls".to_string()],
            settings: HashMap::new(),
            file_patterns: vec!["*.cpp".to_string(), "*.cc".to_string(), "*.cxx".to_string(), "*.hpp".to_string()],
            workspace_settings: HashMap::new(),
            enabled_features: vec![
                "completion".to_string(),
                "hover".to_string(),
                "definition".to_string(),
                "references".to_string(),
                "diagnostics".to_string(),
            ],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        });

        configs
    }

    /// Create default server configurations
    fn default_server_configs() -> HashMap<String, ServerConfig> {
        let mut configs = HashMap::new();

        // ruff-lsp configuration
        configs.insert("ruff-lsp".to_string(), ServerConfig {
            executable_path: None,
            arguments: vec![],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: {
                let mut opts = HashMap::new();
                opts.insert("settings".to_string(), serde_json::json!({
                    "args": ["--config", "pyproject.toml"]
                }));
                opts
            },
            capabilities_override: None,
            memory_limit_mb: Some(256),
            cpu_limit_percent: Some(25.0),
            restart_policy: None,
            use_stdio: true,
            tcp_port: None,
            enabled: true,
        });

        // rust-analyzer configuration
        configs.insert("rust-analyzer".to_string(), ServerConfig {
            executable_path: None,
            arguments: vec![],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: {
                let mut opts = HashMap::new();
                opts.insert("checkOnSave".to_string(), serde_json::json!({
                    "command": "clippy"
                }));
                opts.insert("cargo".to_string(), serde_json::json!({
                    "allFeatures": true
                }));
                opts
            },
            capabilities_override: None,
            memory_limit_mb: Some(512),
            cpu_limit_percent: Some(50.0),
            restart_policy: Some(RestartPolicyOverride {
                enabled: Some(true),
                max_attempts: Some(3),
                base_delay: Some(Duration::from_secs(2)),
                max_delay: Some(Duration::from_secs(60)),
                backoff_multiplier: Some(1.5),
            }),
            use_stdio: true,
            tcp_port: None,
            enabled: true,
        });

        // clangd configuration
        configs.insert("clangd".to_string(), ServerConfig {
            executable_path: None,
            arguments: vec![
                "--background-index".to_string(),
                "--clang-tidy".to_string(),
                "--completion-style=detailed".to_string(),
                "--header-insertion=iwyu".to_string(),
            ],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: HashMap::new(),
            capabilities_override: None,
            memory_limit_mb: Some(1024),
            cpu_limit_percent: Some(50.0),
            restart_policy: None,
            use_stdio: true,
            tcp_port: None,
            enabled: true,
        });

        // typescript-language-server configuration
        configs.insert("typescript-language-server".to_string(), ServerConfig {
            executable_path: None,
            arguments: vec!["--stdio".to_string()],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: {
                let mut opts = HashMap::new();
                opts.insert("preferences".to_string(), serde_json::json!({
                    "includePackageJsonAutoImports": "on"
                }));
                opts
            },
            capabilities_override: None,
            memory_limit_mb: Some(512),
            cpu_limit_percent: Some(30.0),
            restart_policy: None,
            use_stdio: true,
            tcp_port: None,
            enabled: true,
        });

        // gopls configuration
        configs.insert("gopls".to_string(), ServerConfig {
            executable_path: None,
            arguments: vec![],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: {
                let mut opts = HashMap::new();
                opts.insert("analyses".to_string(), serde_json::json!({
                    "unusedparams": true,
                    "shadow": true
                }));
                opts
            },
            capabilities_override: None,
            memory_limit_mb: Some(512),
            cpu_limit_percent: Some(40.0),
            restart_policy: None,
            use_stdio: true,
            tcp_port: None,
            enabled: true,
        });

        configs
    }

    /// Load configuration from file
    pub async fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
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
    pub async fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
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
        // Check timeouts are reasonable
        if self.startup_timeout < Duration::from_secs(1) {
            return Err("Startup timeout too short".to_string());
        }
        if self.request_timeout < Duration::from_millis(100) {
            return Err("Request timeout too short".to_string());
        }

        // Check resource limits
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

        // Check restart policy
        if self.max_restart_attempts == 0 {
            return Err("Max restart attempts cannot be zero".to_string());
        }
        if self.restart_base_delay < Duration::from_millis(100) {
            return Err("Restart base delay too short".to_string());
        }

        // Validate language configs
        for (language, config) in &self.language_configs {
            if config.preferred_servers.is_empty() {
                return Err(format!("No preferred servers for language: {:?}", language));
            }
        }

        Ok(())
    }
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            preferred_servers: vec![],
            settings: HashMap::new(),
            file_patterns: vec![],
            workspace_settings: HashMap::new(),
            enabled_features: vec![],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            executable_path: None,
            arguments: vec![],
            environment: HashMap::new(),
            working_directory: None,
            initialization_options: HashMap::new(),
            capabilities_override: None,
            memory_limit_mb: None,
            cpu_limit_percent: None,
            restart_policy: None,
            use_stdio: true,
            tcp_port: None,
            enabled: true,
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
        
        let unknown_config = config.get_language_config(&Language::Other("unknown".to_string()));
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