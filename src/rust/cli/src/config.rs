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

use wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG;

// Re-export shared path functions from wqm-common (single source of truth)
pub use wqm_common::paths::{
    ConfigPathError as DatabasePathError,
    get_database_path,
    get_database_path_checked,
    get_config_dir,
};

/// Get the config file path (~/.workspace-qdrant/config.yaml)
pub fn get_config_file_path() -> Result<PathBuf, DatabasePathError> {
    get_config_dir().map(|dir| dir.join("config.yaml"))
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

/// Platform-specific PATH separator
#[cfg(not(target_os = "windows"))]
pub const PATH_SEPARATOR: char = ':';
#[cfg(target_os = "windows")]
pub const PATH_SEPARATOR: char = ';';

/// Expand a single path segment, resolving `~` and environment variables.
///
/// Handles:
/// - `~` or `~/...` → home directory
/// - `$VAR` or `${VAR}` → environment variable value
/// - Recursive expansion up to a depth limit to prevent infinite loops
fn expand_path_segment(segment: &str) -> String {
    expand_path_segment_recursive(segment, 0)
}

fn expand_path_segment_recursive(segment: &str, depth: u8) -> String {
    if depth > 10 || segment.is_empty() {
        return segment.to_string();
    }

    let mut result = segment.to_string();

    // Expand ~ at start
    if result == "~" || result.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            result = if result == "~" {
                home.to_string_lossy().to_string()
            } else {
                format!("{}{}", home.display(), &result[1..])
            };
        }
    }

    // Expand ${VAR} patterns
    let mut expanded = String::with_capacity(result.len());
    let chars: Vec<char> = result.chars().collect();
    let mut i = 0;
    let mut changed = false;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            if chars[i + 1] == '{' {
                // ${VAR} form
                if let Some(close) = chars[i + 2..].iter().position(|&c| c == '}') {
                    let var_name: String = chars[i + 2..i + 2 + close].iter().collect();
                    if let Ok(val) = env::var(&var_name) {
                        expanded.push_str(&val);
                        changed = true;
                    }
                    // Skip past the closing brace
                    i = i + 2 + close + 1;
                    continue;
                }
            } else if chars[i + 1].is_ascii_alphanumeric() || chars[i + 1] == '_' {
                // $VAR form - collect alphanumeric + underscore
                let start = i + 1;
                let mut end = start;
                while end < chars.len()
                    && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let var_name: String = chars[start..end].iter().collect();
                if let Ok(val) = env::var(&var_name) {
                    expanded.push_str(&val);
                    changed = true;
                }
                i = end;
                continue;
            }
        }
        expanded.push(chars[i]);
        i += 1;
    }

    // Recurse if we made substitutions (to handle nested vars)
    if changed {
        expand_path_segment_recursive(&expanded, depth + 1)
    } else {
        expanded
    }
}

/// Expand all segments in a PATH string.
///
/// Splits by platform separator, expands each segment, and returns the
/// expanded segments.
fn expand_path_segments(path: &str) -> Vec<String> {
    path.split(PATH_SEPARATOR)
        .filter(|s| !s.is_empty())
        .map(|s| expand_path_segment(s))
        .collect()
}

/// Merge and deduplicate PATH segments.
///
/// Combines current PATH with saved user_path, keeping first occurrence
/// of each entry. Current PATH entries take precedence.
fn merge_and_dedup(current_segments: &[String], saved_segments: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();

    // Current PATH first (higher precedence)
    for seg in current_segments {
        if !seg.is_empty() && seen.insert(seg.clone()) {
            merged.push(seg.clone());
        }
    }

    // Then saved user_path
    for seg in saved_segments {
        if !seg.is_empty() && seen.insert(seg.clone()) {
            merged.push(seg.clone());
        }
    }

    merged
}

/// Join PATH segments with platform separator.
fn join_path_segments(segments: &[String]) -> String {
    segments.join(&PATH_SEPARATOR.to_string())
}

/// Set up the environment PATH on CLI invocation.
///
/// Per specification:
/// 1. Expand: Retrieve $PATH and expand all env vars recursively
/// 2. Merge: Append existing user_path from config to expanded $PATH
/// 3. Deduplicate: Remove duplicates, keeping first occurrence
/// 4. Save: Write to config only if different from current value
///
/// Returns Ok(true) if user_path was updated, Ok(false) if unchanged.
pub fn setup_environment_path() -> Result<bool, String> {
    // Step 1: Expand current $PATH
    let current_path = get_current_path();
    let current_segments = expand_path_segments(&current_path);

    // Step 2: Read saved user_path and expand it too
    let saved_path = read_user_path().unwrap_or_default();
    let saved_segments = expand_path_segments(&saved_path);

    // Step 3: Merge and deduplicate
    let merged = merge_and_dedup(&current_segments, &saved_segments);
    let new_user_path = join_path_segments(&merged);

    // Step 4: Save only if different
    if new_user_path == saved_path {
        return Ok(false);
    }

    write_user_path(&new_user_path)?;
    Ok(true)
}

/// Capture and store current PATH to config (legacy compatibility).
///
/// Returns true if PATH was captured, false if it was already stored.
pub fn capture_user_path() -> Result<bool, String> {
    setup_environment_path()
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
        let error = DatabasePathError::NoHomeDirectory;
        assert_eq!(error.to_string(), "could not determine home directory");

        let error = DatabasePathError::DatabaseNotFound {
            path: PathBuf::from("/test/state.db"),
        };
        assert!(error.to_string().contains("run daemon first"));
    }

    // =========================================================================
    // PATH expansion tests
    // =========================================================================

    #[test]
    fn test_expand_path_segment_tilde() {
        let expanded = expand_path_segment("~/bin");
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, format!("{}/bin", home.display()));
        }
    }

    #[test]
    fn test_expand_path_segment_tilde_alone() {
        let expanded = expand_path_segment("~");
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home.to_string_lossy().to_string());
        }
    }

    #[test]
    fn test_expand_path_segment_env_var_dollar() {
        // Use HOME which is always set on Unix
        let expanded = expand_path_segment("$HOME/bin");
        if let Ok(home) = env::var("HOME") {
            assert_eq!(expanded, format!("{}/bin", home));
        }
    }

    #[test]
    fn test_expand_path_segment_env_var_braces() {
        let expanded = expand_path_segment("${HOME}/bin");
        if let Ok(home) = env::var("HOME") {
            assert_eq!(expanded, format!("{}/bin", home));
        }
    }

    #[test]
    fn test_expand_path_segment_no_expansion() {
        let expanded = expand_path_segment("/usr/local/bin");
        assert_eq!(expanded, "/usr/local/bin");
    }

    #[test]
    fn test_expand_path_segment_empty() {
        let expanded = expand_path_segment("");
        assert_eq!(expanded, "");
    }

    #[test]
    fn test_expand_path_segment_unknown_var() {
        // Unknown vars should be removed (no value to substitute)
        let expanded =
            expand_path_segment("$WQM_TEST_NONEXISTENT_VAR_12345/bin");
        // The $VAR is consumed but env::var fails, so nothing is written
        assert_eq!(expanded, "/bin");
    }

    #[test]
    fn test_expand_path_segment_recursive() {
        // Set up nested env vars for recursive expansion
        env::set_var("WQM_TEST_INNER", "/resolved");
        env::set_var("WQM_TEST_OUTER", "$WQM_TEST_INNER/path");
        let expanded = expand_path_segment("$WQM_TEST_OUTER");
        assert_eq!(expanded, "/resolved/path");
        env::remove_var("WQM_TEST_INNER");
        env::remove_var("WQM_TEST_OUTER");
    }

    #[test]
    fn test_expand_path_segment_depth_limit() {
        // Set up a self-referencing var to test depth limit
        env::set_var("WQM_TEST_LOOP", "$WQM_TEST_LOOP");
        let expanded = expand_path_segment("$WQM_TEST_LOOP");
        // Should terminate without infinite recursion
        assert!(!expanded.is_empty() || expanded.is_empty()); // just prove it returns
        env::remove_var("WQM_TEST_LOOP");
    }

    // =========================================================================
    // PATH segments expansion tests
    // =========================================================================

    #[test]
    fn test_expand_path_segments_basic() {
        let segments = expand_path_segments("/usr/bin:/usr/local/bin");
        assert_eq!(segments, vec!["/usr/bin", "/usr/local/bin"]);
    }

    #[test]
    fn test_expand_path_segments_with_tilde() {
        let segments = expand_path_segments("~/bin:/usr/local/bin");
        assert_eq!(segments.len(), 2);
        if let Some(home) = dirs::home_dir() {
            assert_eq!(segments[0], format!("{}/bin", home.display()));
        }
        assert_eq!(segments[1], "/usr/local/bin");
    }

    #[test]
    fn test_expand_path_segments_empty() {
        let segments = expand_path_segments("");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_expand_path_segments_filters_empty() {
        // Double separator produces empty segments which should be filtered
        let segments = expand_path_segments("/usr/bin::/usr/local/bin");
        assert_eq!(segments, vec!["/usr/bin", "/usr/local/bin"]);
    }

    // =========================================================================
    // Merge and dedup tests
    // =========================================================================

    #[test]
    fn test_merge_and_dedup_no_overlap() {
        let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
        let saved = vec!["/opt/bin".to_string()];
        let result = merge_and_dedup(&current, &saved);
        assert_eq!(result, vec!["/usr/bin", "/usr/local/bin", "/opt/bin"]);
    }

    #[test]
    fn test_merge_and_dedup_with_overlap() {
        let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
        let saved = vec![
            "/usr/bin".to_string(),
            "/opt/bin".to_string(),
        ];
        let result = merge_and_dedup(&current, &saved);
        // /usr/bin appears only once, from current (first occurrence wins)
        assert_eq!(result, vec!["/usr/bin", "/usr/local/bin", "/opt/bin"]);
    }

    #[test]
    fn test_merge_and_dedup_preserves_order() {
        let current = vec![
            "/c".to_string(),
            "/a".to_string(),
            "/b".to_string(),
        ];
        let saved = vec![
            "/d".to_string(),
            "/a".to_string(), // duplicate
        ];
        let result = merge_and_dedup(&current, &saved);
        assert_eq!(result, vec!["/c", "/a", "/b", "/d"]);
    }

    #[test]
    fn test_merge_and_dedup_both_empty() {
        let result = merge_and_dedup(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_and_dedup_current_empty() {
        let saved = vec!["/opt/bin".to_string()];
        let result = merge_and_dedup(&[], &saved);
        assert_eq!(result, vec!["/opt/bin"]);
    }

    #[test]
    fn test_merge_and_dedup_saved_empty() {
        let current = vec!["/usr/bin".to_string()];
        let result = merge_and_dedup(&current, &[]);
        assert_eq!(result, vec!["/usr/bin"]);
    }

    #[test]
    fn test_merge_and_dedup_filters_empty_entries() {
        let current = vec!["".to_string(), "/usr/bin".to_string()];
        let saved = vec!["".to_string(), "/opt/bin".to_string()];
        let result = merge_and_dedup(&current, &saved);
        assert_eq!(result, vec!["/usr/bin", "/opt/bin"]);
    }

    #[test]
    fn test_merge_and_dedup_all_duplicates() {
        let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
        let saved = vec!["/usr/local/bin".to_string(), "/usr/bin".to_string()];
        let result = merge_and_dedup(&current, &saved);
        assert_eq!(result, vec!["/usr/bin", "/usr/local/bin"]);
    }

    // =========================================================================
    // Join and separator tests
    // =========================================================================

    #[test]
    fn test_join_path_segments() {
        let segments = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
        let joined = join_path_segments(&segments);
        #[cfg(not(target_os = "windows"))]
        assert_eq!(joined, "/usr/bin:/usr/local/bin");
        #[cfg(target_os = "windows")]
        assert_eq!(joined, "/usr/bin;/usr/local/bin");
    }

    #[test]
    fn test_join_path_segments_empty() {
        let segments: Vec<String> = vec![];
        let joined = join_path_segments(&segments);
        assert_eq!(joined, "");
    }

    #[test]
    fn test_join_path_segments_single() {
        let segments = vec!["/usr/bin".to_string()];
        let joined = join_path_segments(&segments);
        assert_eq!(joined, "/usr/bin");
    }

    #[test]
    fn test_path_separator_value() {
        #[cfg(not(target_os = "windows"))]
        assert_eq!(PATH_SEPARATOR, ':');
        #[cfg(target_os = "windows")]
        assert_eq!(PATH_SEPARATOR, ';');
    }

    // =========================================================================
    // Integration: setup_environment_path
    // =========================================================================

    #[test]
    fn test_setup_environment_path_captures_path() {
        // Test the full flow using component functions
        let current = get_current_path();
        assert!(!current.is_empty(), "System PATH should not be empty");

        let segments = expand_path_segments(&current);
        assert!(!segments.is_empty());

        let merged = merge_and_dedup(&segments, &[]);
        // Dedup may reduce count if system PATH has duplicates
        assert!(merged.len() <= segments.len());
        assert!(!merged.is_empty());

        let joined = join_path_segments(&merged);
        assert!(!joined.is_empty());
    }
}
