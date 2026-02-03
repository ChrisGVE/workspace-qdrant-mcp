//! CLI configuration module
//!
//! Handles daemon address, output format preferences, and connection settings.
//! Configuration can be loaded from environment variables and YAML config file.
//!
//! Note: Some builder methods are infrastructure for future CLI features.

#![allow(dead_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::grpc::client::DEFAULT_GRPC_PORT;

/// Database path error with helpful message
#[derive(Debug)]
pub struct DatabasePathError {
    pub message: String,
    pub path: PathBuf,
}

impl std::fmt::Display for DatabasePathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DatabasePathError {}

/// Get the canonical path to the state database
///
/// Per spec, the canonical path is `~/.workspace-qdrant/state.db`
/// This is shared between the daemon and CLI for consistency.
pub fn get_database_path() -> Result<PathBuf, DatabasePathError> {
    // First check environment variable override
    if let Ok(path) = env::var("WQM_DATABASE_PATH") {
        return Ok(PathBuf::from(path));
    }

    // Use canonical path: ~/.workspace-qdrant/state.db
    if let Ok(home) = env::var("HOME") {
        return Ok(PathBuf::from(format!("{}/.workspace-qdrant/state.db", home)));
    }

    // Fallback for systems without HOME
    if let Some(home_dir) = dirs::home_dir() {
        return Ok(home_dir.join(".workspace-qdrant").join("state.db"));
    }

    Err(DatabasePathError {
        message: "Could not determine home directory for database path".to_string(),
        path: PathBuf::new(),
    })
}

/// Get the database path, checking if it exists
///
/// Returns an error with a helpful message if the database doesn't exist,
/// indicating the user should start the daemon first.
pub fn get_database_path_checked() -> Result<PathBuf, DatabasePathError> {
    let path = get_database_path()?;

    if !path.exists() {
        return Err(DatabasePathError {
            message: format!(
                "Database not found at {}. Run daemon first: wqm service start",
                path.display()
            ),
            path,
        });
    }

    Ok(path)
}

/// Get the config directory path (~/.workspace-qdrant/)
pub fn get_config_dir() -> Result<PathBuf, DatabasePathError> {
    if let Ok(home) = env::var("HOME") {
        return Ok(PathBuf::from(format!("{}/.workspace-qdrant", home)));
    }

    if let Some(home_dir) = dirs::home_dir() {
        return Ok(home_dir.join(".workspace-qdrant"));
    }

    Err(DatabasePathError {
        message: "Could not determine home directory for config path".to_string(),
        path: PathBuf::new(),
    })
}

/// Get the config file path (~/.workspace-qdrant/config.yaml)
pub fn get_config_file_path() -> Result<PathBuf, DatabasePathError> {
    get_config_dir().map(|dir| dir.join("config.yaml"))
}

/// Get unified config search paths (in priority order)
///
/// Returns all potential config file paths that should be checked:
/// 1. WQM_CONFIG_PATH environment variable (if set)
/// 2. User home: ~/.workspace-qdrant/config.yaml
/// 3. XDG: ~/.config/workspace-qdrant/config.yaml
/// 4. macOS: ~/Library/Application Support/workspace-qdrant/config.yaml
///
/// Note: Unlike the daemon, CLI does not search project-local configs
/// since CLI commands operate system-wide.
pub fn get_unified_config_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Explicit path via environment variable (highest priority)
    if let Ok(explicit_path) = env::var("WQM_CONFIG_PATH") {
        paths.push(PathBuf::from(explicit_path));
    }

    // 2. User home config: ~/.workspace-qdrant/config.yaml
    if let Ok(home) = env::var("HOME") {
        paths.push(PathBuf::from(format!("{}/.workspace-qdrant/config.yaml", home)));
        paths.push(PathBuf::from(format!("{}/.workspace-qdrant/config.yml", home)));
    } else if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".workspace-qdrant").join("config.yaml"));
        paths.push(home_dir.join(".workspace-qdrant").join("config.yml"));
    }

    // 3. XDG config: ~/.config/workspace-qdrant/config.yaml
    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join("workspace-qdrant").join("config.yaml"));
        paths.push(config_dir.join("workspace-qdrant").join("config.yml"));
    }

    // 4. macOS Application Support (only on macOS)
    #[cfg(target_os = "macos")]
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join("Library").join("Application Support")
            .join("workspace-qdrant").join("config.yaml"));
    }

    paths
}

/// Find the first existing config file from unified search paths
pub fn find_config_file() -> Option<PathBuf> {
    get_unified_config_search_paths()
        .into_iter()
        .find(|path| path.exists())
}

/// Read user PATH from config file
///
/// Returns None if the config file doesn't exist or doesn't have user_path set.
pub fn read_user_path() -> Option<String> {
    let config_path = get_config_file_path().ok()?;

    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;

    // Simple YAML parsing for environment.user_path
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("user_path:") {
            let value = trimmed.strip_prefix("user_path:")?.trim();
            // Handle quoted strings
            let value = value.trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }

    None
}

/// Write user PATH to config file
///
/// Creates the config directory and file if they don't exist.
/// Updates the environment.user_path value if the file exists.
pub fn write_user_path(path: &str) -> Result<(), String> {
    let config_dir = get_config_dir()
        .map_err(|e| format!("Failed to get config directory: {}", e))?;

    // Create config directory if it doesn't exist
    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    let config_path = config_dir.join("config.yaml");

    // Read existing config or create new
    let mut config_content = if config_path.exists() {
        fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config file: {}", e))?
    } else {
        String::new()
    };

    // Check if environment section exists
    let has_environment = config_content.contains("\nenvironment:") || config_content.starts_with("environment:");
    let has_user_path = config_content.contains("user_path:");

    if has_user_path {
        // Update existing user_path line
        let mut new_content = String::new();
        for line in config_content.lines() {
            if line.trim().starts_with("user_path:") {
                new_content.push_str(&format!("  user_path: \"{}\"\n", path));
            } else {
                new_content.push_str(line);
                new_content.push('\n');
            }
        }
        config_content = new_content;
    } else if has_environment {
        // Add user_path under existing environment section
        let mut new_content = String::new();
        for line in config_content.lines() {
            new_content.push_str(line);
            new_content.push('\n');
            if line.trim() == "environment:" {
                new_content.push_str(&format!("  user_path: \"{}\"\n", path));
            }
        }
        config_content = new_content;
    } else {
        // Add new environment section
        if !config_content.is_empty() && !config_content.ends_with('\n') {
            config_content.push('\n');
        }
        config_content.push_str(&format!("\nenvironment:\n  user_path: \"{}\"\n", path));
    }

    // Write config file
    let mut file = fs::File::create(&config_path)
        .map_err(|e| format!("Failed to create config file: {}", e))?;
    file.write_all(config_content.as_bytes())
        .map_err(|e| format!("Failed to write config file: {}", e))?;

    Ok(())
}

/// Get current system PATH
pub fn get_current_path() -> String {
    env::var("PATH").unwrap_or_default()
}

/// Capture and store current PATH to config
///
/// Returns true if PATH was captured, false if it was already stored.
pub fn capture_user_path() -> Result<bool, String> {
    // Check if PATH is already stored
    if read_user_path().is_some() {
        return Ok(false);
    }

    let current_path = get_current_path();
    if current_path.is_empty() {
        return Err("Current PATH is empty".to_string());
    }

    write_user_path(&current_path)?;
    Ok(true)
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

    #[test]
    fn test_get_database_path() {
        // Clear any env override first to ensure we test the default path
        let prev = std::env::var("WQM_DATABASE_PATH").ok();
        std::env::remove_var("WQM_DATABASE_PATH");

        // Should return a valid path (assuming HOME is set)
        let result = get_database_path();
        assert!(result.is_ok());

        let path = result.unwrap();
        assert!(path.to_string_lossy().contains(".workspace-qdrant"));
        assert!(path.to_string_lossy().ends_with("state.db"));

        // Restore env var if it was set
        if let Some(val) = prev {
            std::env::set_var("WQM_DATABASE_PATH", val);
        }
    }

    #[test]
    fn test_get_database_path_with_env_override() {
        // Save and restore to avoid test interference
        let prev = std::env::var("WQM_DATABASE_PATH").ok();

        // Test environment variable override
        std::env::set_var("WQM_DATABASE_PATH", "/custom/path/state.db");
        let result = get_database_path();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from("/custom/path/state.db"));

        // Restore previous state
        match prev {
            Some(val) => std::env::set_var("WQM_DATABASE_PATH", val),
            None => std::env::remove_var("WQM_DATABASE_PATH"),
        }
    }

    #[test]
    fn test_database_path_error_display() {
        let error = DatabasePathError {
            message: "Test error message".to_string(),
            path: PathBuf::from("/test/path"),
        };
        assert_eq!(error.to_string(), "Test error message");
    }
}
