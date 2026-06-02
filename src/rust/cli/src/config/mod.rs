//! CLI configuration module
//!
//! Handles daemon address, output format preferences, and connection settings.
//! Configuration can be loaded from environment variables and YAML config file.
//!
//! Note: Some builder methods are infrastructure for future CLI features.

#![allow(dead_code)]

mod path_env;
#[cfg(test)]
mod tests;

pub use path_env::{capture_user_path, setup_environment_path};

use std::env;

use wqm_common::cli_profiles::{load_cli_config, Profile};
use wqm_common::config::{apply_env_overrides, validate_timeout, validate_url, EnvOverride};
use wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG;

// Re-export shared path functions from wqm-common (single source of truth)
pub use wqm_common::paths::{
    get_config_dir, get_database_path, get_database_path_checked,
    ConfigPathError as DatabasePathError,
};

/// Get the config file path (~/.config/workspace-qdrant/config.yaml)
pub fn get_config_file_path() -> Result<std::path::PathBuf, DatabasePathError> {
    get_config_dir().map(|dir| dir.join("config.yaml"))
}

/// Output format for CLI responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable table format (default)
    #[default]
    Table,
    /// Machine-readable JSON format
    Json,
    /// Plain text (minimal formatting)
    Plain,
    /// Script-friendly space-separated output (no ANSI, categorical only)
    Script,
}

impl OutputFormat {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "table" => Some(Self::Table),
            "json" => Some(Self::Json),
            "plain" | "text" => Some(Self::Plain),
            "script" => Some(Self::Script),
            _ => None,
        }
    }
}

/// CLI configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Daemon gRPC address (e.g., "http://127.0.0.1:50051")
    pub daemon_address: String,

    /// Qdrant HTTP base URL (e.g., "http://localhost:6333")
    pub qdrant_url: String,

    /// Environment variable holding the Qdrant API key, if any.
    pub qdrant_api_key_env: String,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Default output format
    pub output_format: OutputFormat,

    /// Enable colored output
    pub color_enabled: bool,

    /// Verbose mode
    pub verbose: bool,

    /// Active profile name (empty if no cli-config.toml was consulted).
    pub active_profile: String,
}

impl Default for Config {
    fn default() -> Self {
        let yaml = &*DEFAULT_YAML_CONFIG;
        Self {
            daemon_address: format!("http://{}:{}", yaml.grpc.host, yaml.grpc.port),
            qdrant_url: wqm_common::constants::DEFAULT_QDRANT_URL.to_string(),
            qdrant_api_key_env: String::new(),
            connection_timeout_secs: 5,
            output_format: OutputFormat::Table,
            color_enabled: true,
            verbose: false,
            active_profile: String::new(),
        }
    }
}

impl Config {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a profile on top of the default config. Env overrides still win.
    pub fn with_profile(mut self, profile: &Profile) -> Self {
        self.daemon_address = profile.daemon_address.clone();
        self.qdrant_url = profile.qdrant_url.clone();
        self.qdrant_api_key_env = profile.qdrant_api_key_env.clone();
        self.active_profile = profile.name.clone();
        self
    }

    /// Load configuration from environment variables
    ///
    /// Environment variables (highest priority, override any profile value):
    /// - `WQM_PROFILE`: Name of the cli-config.toml profile to activate.
    /// - `WQM_DAEMON_ADDR`: Daemon gRPC address
    /// - `WQM_QDRANT_URL`: Qdrant HTTP base URL
    /// - `WQM_TIMEOUT`: Connection timeout in seconds
    /// - `WQM_OUTPUT_FORMAT`: Output format (table, json, plain)
    /// - `NO_COLOR`: Disable colored output (any value)
    /// - `WQM_VERBOSE`: Enable verbose mode (any value)
    pub fn from_env() -> Self {
        Self::from_env_with(&|key| env::var(key).ok())
    }

    /// Getter-injectable core of [`from_env`].
    ///
    /// The active cli-config profile is applied first (so explicit env vars win),
    /// then the env overrides run through the shared declarative
    /// [`apply_env_overrides`] engine. Profile loading still consults the real
    /// process environment for `WQM_CLI_CONFIG`; only the override layer uses the
    /// injected getter.
    pub(crate) fn from_env_with(env_getter: &dyn Fn(&str) -> Option<String>) -> Self {
        let mut config = Self::default();

        // Apply the active profile first, so explicit env vars still override.
        match load_cli_config() {
            Ok(Some((file, _path))) => {
                let profile_name = env_getter("WQM_PROFILE")
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| file.active.clone());
                if let Some(profile) = file.find(&profile_name) {
                    config = config.with_profile(profile);
                }
            }
            Ok(None) => {}
            Err(_) => {
                // Malformed cli-config.toml is visible via `wqm config show`;
                // here we prefer a working CLI over a hard failure.
            }
        }

        // Env overrides via the shared declarative engine. Parsing failures and
        // presence-only flags (NO_COLOR / WQM_VERBOSE) are handled inside each
        // setter, preserving the prior behaviour exactly.
        let specs: Vec<EnvOverride<Config>> = vec![
            EnvOverride::single("WQM_DAEMON_ADDR", |c: &mut Config, v| c.daemon_address = v),
            EnvOverride::single("WQM_QDRANT_URL", |c: &mut Config, v| c.qdrant_url = v),
            EnvOverride::single("WQM_TIMEOUT", |c: &mut Config, v| {
                if let Ok(secs) = v.parse::<u64>() {
                    c.connection_timeout_secs = secs;
                }
            }),
            EnvOverride::single("WQM_OUTPUT_FORMAT", |c: &mut Config, v| {
                if let Some(fmt) = OutputFormat::from_str(&v) {
                    c.output_format = fmt;
                }
            }),
            EnvOverride::single("NO_COLOR", |c: &mut Config, _v| c.color_enabled = false),
            EnvOverride::single("WQM_VERBOSE", |c: &mut Config, _v| c.verbose = true),
        ];
        apply_env_overrides(&mut config, env_getter, &specs);

        config
    }

    /// Builder: set daemon address
    pub fn with_daemon_address(mut self, addr: impl Into<String>) -> Self {
        self.daemon_address = addr.into();
        self
    }

    /// Builder: set connection timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.connection_timeout_secs = secs;
        self
    }

    /// Builder: set output format
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Builder: set color enabled
    pub fn with_color(mut self, enabled: bool) -> Self {
        self.color_enabled = enabled;
        self
    }

    /// Builder: set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Daemon address: shared http(s)-scheme check, CLI-specific message.
        validate_url(&self.daemon_address).map_err(|_| {
            format!(
                "Invalid daemon address: {}. Must start with http:// or https://",
                self.daemon_address
            )
        })?;

        // Timeout: shared non-zero check, plus the CLI-specific upper bound.
        validate_timeout(self.connection_timeout_secs)
            .map_err(|_| "Connection timeout must be greater than 0".to_string())?;
        if self.connection_timeout_secs > 300 {
            return Err("Connection timeout must be less than 300 seconds".to_string());
        }

        Ok(())
    }
}

/// Resolve the daemon gRPC address using env + active cli-config profile.
///
/// Priority: `WQM_DAEMON_ADDR` > active profile > workspace default.
pub fn resolve_daemon_address() -> String {
    Config::from_env().daemon_address
}

/// Resolve the Qdrant HTTP base URL using env + active cli-config profile.
///
/// Priority: `QDRANT_URL` > `WQM_QDRANT_URL` > active profile > workspace
/// default (`http://localhost:6333`). `QDRANT_URL` is honored for
/// compatibility with docker-compose deployments.
pub fn resolve_qdrant_url() -> String {
    if let Ok(url) = env::var("QDRANT_URL") {
        if !url.is_empty() {
            return url;
        }
    }
    Config::from_env().qdrant_url
}

/// Resolve the Qdrant API key (if the active profile names an env var holding
/// it, or `QDRANT_API_KEY` is set directly).
pub fn resolve_qdrant_api_key() -> Option<String> {
    if let Ok(key) = env::var("QDRANT_API_KEY") {
        if !key.is_empty() {
            return Some(key);
        }
    }
    let cfg = Config::from_env();
    if cfg.qdrant_api_key_env.is_empty() {
        return None;
    }
    env::var(&cfg.qdrant_api_key_env)
        .ok()
        .filter(|v| !v.is_empty())
}
