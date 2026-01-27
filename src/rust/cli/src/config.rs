//! CLI configuration module
//!
//! Handles daemon address, output format preferences, and connection settings.
//! Configuration can be loaded from environment variables.
//!
//! Note: Some builder methods are infrastructure for future CLI features.

#![allow(dead_code)]

use std::env;

use crate::grpc::client::DEFAULT_GRPC_PORT;

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
        Self {
            daemon_address: format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.daemon_address.contains("50051"));
        assert_eq!(config.connection_timeout_secs, 5);
        assert_eq!(config.output_format, OutputFormat::Table);
        assert!(config.color_enabled);
        assert!(!config.verbose);
    }

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(OutputFormat::from_str("table"), Some(OutputFormat::Table));
        assert_eq!(OutputFormat::from_str("TABLE"), Some(OutputFormat::Table));
        assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("JSON"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("plain"), Some(OutputFormat::Plain));
        assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Plain));
        assert_eq!(OutputFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_builder_pattern() {
        let config = Config::new()
            .with_daemon_address("http://localhost:9999")
            .with_timeout(10)
            .with_output_format(OutputFormat::Json)
            .with_color(false)
            .with_verbose(true);

        assert_eq!(config.daemon_address, "http://localhost:9999");
        assert_eq!(config.connection_timeout_secs, 10);
        assert_eq!(config.output_format, OutputFormat::Json);
        assert!(!config.color_enabled);
        assert!(config.verbose);
    }

    #[test]
    fn test_validation_success() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_address() {
        let config = Config::new().with_daemon_address("localhost:50051");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_timeout() {
        let config = Config::new().with_timeout(0);
        assert!(config.validate().is_err());

        let config = Config::new().with_timeout(500);
        assert!(config.validate().is_err());
    }
}
