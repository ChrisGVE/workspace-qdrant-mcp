//! Server-specific LSP configuration
//!
//! Contains per-server configuration types (executable paths, arguments,
//! initialization options, resource limits, restart policies) and default
//! server configurations for known LSP servers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

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

/// Create default server configurations for known LSP servers.
pub fn default_server_configs() -> HashMap<String, ServerConfig> {
    let mut configs = HashMap::new();

    configs.insert("ruff-lsp".to_string(), ruff_lsp_config());
    configs.insert("rust-analyzer".to_string(), rust_analyzer_config());
    configs.insert("clangd".to_string(), clangd_config());
    configs.insert(
        "typescript-language-server".to_string(),
        typescript_language_server_config(),
    );
    configs.insert("gopls".to_string(), gopls_config());

    configs
}

fn ruff_lsp_config() -> ServerConfig {
    ServerConfig {
        initialization_options: {
            let mut opts = HashMap::new();
            opts.insert(
                "settings".to_string(),
                serde_json::json!({
                    "args": ["--config", "pyproject.toml"]
                }),
            );
            opts
        },
        memory_limit_mb: Some(256),
        cpu_limit_percent: Some(25.0),
        ..ServerConfig::default()
    }
}

fn rust_analyzer_config() -> ServerConfig {
    ServerConfig {
        initialization_options: {
            let mut opts = HashMap::new();
            opts.insert(
                "checkOnSave".to_string(),
                serde_json::json!({
                    "command": "clippy"
                }),
            );
            opts.insert(
                "cargo".to_string(),
                serde_json::json!({
                    "allFeatures": true
                }),
            );
            opts
        },
        memory_limit_mb: Some(512),
        cpu_limit_percent: Some(50.0),
        restart_policy: Some(RestartPolicyOverride {
            enabled: Some(true),
            max_attempts: Some(3),
            base_delay: Some(Duration::from_secs(2)),
            max_delay: Some(Duration::from_secs(60)),
            backoff_multiplier: Some(1.5),
        }),
        ..ServerConfig::default()
    }
}

fn clangd_config() -> ServerConfig {
    ServerConfig {
        arguments: vec![
            "--background-index".to_string(),
            "--clang-tidy".to_string(),
            "--completion-style=detailed".to_string(),
            "--header-insertion=iwyu".to_string(),
        ],
        memory_limit_mb: Some(1024),
        cpu_limit_percent: Some(50.0),
        ..ServerConfig::default()
    }
}

fn typescript_language_server_config() -> ServerConfig {
    ServerConfig {
        arguments: vec!["--stdio".to_string()],
        initialization_options: {
            let mut opts = HashMap::new();
            opts.insert(
                "preferences".to_string(),
                serde_json::json!({
                    "includePackageJsonAutoImports": "on"
                }),
            );
            opts
        },
        memory_limit_mb: Some(512),
        cpu_limit_percent: Some(30.0),
        ..ServerConfig::default()
    }
}

fn gopls_config() -> ServerConfig {
    ServerConfig {
        initialization_options: {
            let mut opts = HashMap::new();
            opts.insert(
                "analyses".to_string(),
                serde_json::json!({
                    "unusedparams": true,
                    "shadow": true
                }),
            );
            opts
        },
        memory_limit_mb: Some(512),
        cpu_limit_percent: Some(40.0),
        ..ServerConfig::default()
    }
}
