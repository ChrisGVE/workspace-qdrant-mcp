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

pub use path_env::{setup_environment_path, capture_user_path};

use std::env;

use wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG;

// Re-export shared path functions from wqm-common (single source of truth)
pub use wqm_common::paths::{
    ConfigPathError as DatabasePathError,
    get_database_path,
    get_database_path_checked,
    get_config_dir,
};

/// Get the config file path (~/.workspace-qdrant/config.yaml)
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
}

impl OutputFormat {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "table" => Some(Self::Table),
            "json" => Some(Self::Json),
            "plain" | "text" => Some(Self::Plain),
            _ => None,
        }
    }
}

/// CLI configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Daemon gRPC address (e.g., "http://127.0.0.1:50051")
    pub daemon_address: String,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Default output format
    pub output_format: OutputFormat,

    /// Enable colored output
    pub color_enabled: bool,

    /// Verbose mode
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        let yaml = &*DEFAULT_YAML_CONFIG;
        Self {
            daemon_address: format!("http://{}:{}", yaml.grpc.host, yaml.grpc.port),
            connection_timeout_secs: 5,
            output_format: OutputFormat::Table,
            color_enabled: true,
            verbose: false,
        }
    }
}

impl Config {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    ///
    /// Environment variables:
    /// - `WQM_DAEMON_ADDR`: Daemon gRPC address
    /// - `WQM_TIMEOUT`: Connection timeout in seconds
    /// - `WQM_OUTPUT_FORMAT`: Output format (table, json, plain)
    /// - `NO_COLOR`: Disable colored output (any value)
    /// - `WQM_VERBOSE`: Enable verbose mode (any value)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Daemon address
        if let Ok(addr) = env::var("WQM_DAEMON_ADDR") {
            config.daemon_address = addr;
        }

        // Connection timeout
        if let Ok(timeout) = env::var("WQM_TIMEOUT") {
            if let Ok(secs) = timeout.parse::<u64>() {
                config.connection_timeout_secs = secs;
            }
        }

        // Output format
        if let Ok(format) = env::var("WQM_OUTPUT_FORMAT") {
            if let Some(fmt) = OutputFormat::from_str(&format) {
                config.output_format = fmt;
            }
        }

        // Color enabled (NO_COLOR is a standard env var)
        if env::var("NO_COLOR").is_ok() {
            config.color_enabled = false;
        }

        // Verbose mode
        if env::var("WQM_VERBOSE").is_ok() {
            config.verbose = true;
        }

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
        // Validate daemon address format
        if !self.daemon_address.starts_with("http://")
            && !self.daemon_address.starts_with("https://")
        {
            return Err(format!(
                "Invalid daemon address: {}. Must start with http:// or https://",
                self.daemon_address
            ));
        }

        // Validate timeout
        if self.connection_timeout_secs == 0 {
            return Err("Connection timeout must be greater than 0".to_string());
        }

        if self.connection_timeout_secs > 300 {
            return Err("Connection timeout must be less than 300 seconds".to_string());
        }

        Ok(())
    }
}
