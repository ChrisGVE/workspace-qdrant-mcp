//! Unified configuration integration for Rust daemon
//!
//! This module provides integration with the Python unified configuration system,
//! allowing the Rust daemon to load configuration from the same sources as the
//! Python MCP server and supports existing TOML configuration files.

use crate::config::DaemonConfig;
use crate::storage::TransportMode;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use tracing::{debug, info, error};

/// Expand environment variables in a string.
///
/// Supports both ${VAR} and $VAR syntax. If a variable is not set,
/// the reference is left unchanged (literal text preserved).
///
/// # Examples
/// ```
/// use workspace_qdrant_core::unified_config::expand_env_vars;
///
/// std::env::set_var("HOME", "/home/user");
/// assert_eq!(expand_env_vars("$HOME/cache"), "/home/user/cache");
/// assert_eq!(expand_env_vars("${HOME}/cache"), "/home/user/cache");
/// assert_eq!(expand_env_vars("$NONEXISTENT/path"), "$NONEXISTENT/path");
/// ```
pub fn expand_env_vars(s: &str) -> String {
    shellexpand::env_with_context_no_errors(s, |var| {
        std::env::var(var).ok()
    }).to_string()
}

/// Expand environment variables in an optional path.
///
/// Returns None if input is None, otherwise expands environment variables
/// in the path string and returns a new PathBuf.
pub fn expand_path_env_vars(path: Option<&PathBuf>) -> Option<PathBuf> {
    path.map(|p| {
        let expanded = expand_env_vars(&p.to_string_lossy());
        PathBuf::from(expanded)
    })
}

/// Error types for unified configuration operations
#[derive(Debug, thiserror::Error)]
pub enum UnifiedConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(PathBuf),
    
    #[error("Configuration parsing error: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("YAML parsing error: {0}")]
    YamlError(String),
    
    #[error("Configuration validation error: {0}")]
    ValidationError(String),
}

/// Supported configuration formats (YAML only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension (YAML only)
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => ConfigFormat::Yaml,
            _ => ConfigFormat::Yaml, // Default to YAML
        }
    }
}

/// Unified configuration manager for Rust daemon
pub struct UnifiedConfigManager {
    config_dir: PathBuf,
    config_file_patterns: Vec<String>,
}

impl UnifiedConfigManager {
    /// Create new unified configuration manager
    ///
    /// Search paths (in priority order):
    /// 1. Explicit config via WQM_CONFIG_PATH environment variable
    /// 2. Project-local: .workspace-qdrant.yaml (in config_dir)
    /// 3. User config: ~/.workspace-qdrant/config.yaml
    /// 4. Platform config: $XDG_CONFIG_HOME/workspace-qdrant/config.yaml (Linux),
    ///    %APPDATA%/workspace-qdrant/config.yaml (Windows)
    /// 5. macOS: ~/Library/Application Support/workspace-qdrant/config.yaml
    pub fn new<P: Into<PathBuf>>(config_dir: Option<P>) -> Self {
        let config_dir = config_dir.map(|p| p.into()).unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });

        // Project-local config file patterns (searched in config_dir)
        let config_file_patterns = vec![
            ".workspace-qdrant.yaml".to_string(),
            ".workspace-qdrant.yml".to_string(),
        ];

        Self {
            config_dir,
            config_file_patterns,
        }
    }

    /// Get unified config search paths (in priority order)
    ///
    /// Returns all potential config file paths that should be checked:
    /// 1. WQM_CONFIG_PATH environment variable (if set)
    /// 2. Project-local configs in config_dir
    /// 3. User home: ~/.workspace-qdrant/config.yaml
    /// 4. XDG: ~/.config/workspace-qdrant/config.yaml
    /// 5. macOS: ~/Library/Application Support/workspace-qdrant/config.yaml
    pub fn get_unified_search_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. Explicit path via environment variable (highest priority)
        if let Ok(explicit_path) = std::env::var("WQM_CONFIG_PATH") {
            paths.push(PathBuf::from(explicit_path));
        }

        // 2. Project-local configs
        for pattern in &self.config_file_patterns {
            paths.push(self.config_dir.join(pattern));
        }

        // 3. User home config: ~/.workspace-qdrant/config.yaml
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join(".workspace-qdrant").join("config.yaml"));
            paths.push(home.join(".workspace-qdrant").join("config.yml"));
        }

        // 4. XDG config: ~/.config/workspace-qdrant/config.yaml
        if let Some(config_dir) = dirs::config_dir() {
            paths.push(config_dir.join("workspace-qdrant").join("config.yaml"));
            paths.push(config_dir.join("workspace-qdrant").join("config.yml"));
        }

        // 5. macOS Application Support (only on macOS)
        #[cfg(target_os = "macos")]
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join("Library").join("Application Support")
                .join("workspace-qdrant").join("config.yaml"));
        }

        paths
    }

    /// Discover available configuration files from all unified search paths
    ///
    /// Returns tuples of (path, format, exists) for all potential config locations.
    /// The order reflects priority - first existing file wins.
    pub fn discover_config_sources(&self) -> Vec<(PathBuf, ConfigFormat, bool)> {
        let paths = self.get_unified_search_paths();
        let mut sources = Vec::new();

        for config_file in paths {
            let format = ConfigFormat::from_path(&config_file);
            let exists = config_file.exists();
            sources.push((config_file, format, exists));
        }

        debug!("Discovered {} configuration sources ({} exist)",
               sources.len(),
               sources.iter().filter(|(_, _, e)| *e).count());
        sources
    }

    /// Get preferred configuration source from unified search paths
    ///
    /// Returns the first existing config file from the priority-ordered search paths.
    /// Priority order:
    /// 1. WQM_CONFIG_PATH environment variable
    /// 2. Project-local configs
    /// 3. ~/.workspace-qdrant/config.yaml
    /// 4. ~/.config/workspace-qdrant/config.yaml
    /// 5. ~/Library/Application Support/workspace-qdrant/config.yaml (macOS)
    pub fn get_preferred_config_source(&self, prefer_format: Option<ConfigFormat>) -> Option<(PathBuf, ConfigFormat)> {
        let sources = self.discover_config_sources();
        let existing_sources: Vec<_> = sources.into_iter()
            .filter(|(_, _, exists)| *exists)
            .collect();

        if existing_sources.is_empty() {
            return None;
        }

        // If format preference specified, try to find that format first
        if let Some(prefer_fmt) = prefer_format {
            for (path, format, _) in &existing_sources {
                if *format == prefer_fmt {
                    return Some((path.clone(), *format));
                }
            }
        }

        // Return first existing source (follows discovery order = priority order)
        existing_sources.first()
            .map(|(path, format, _)| (path.clone(), *format))
    }

    /// Load unified configuration
    pub fn load_config(&self, config_file: Option<&Path>) -> Result<DaemonConfig, UnifiedConfigError> {
        let (source_file, format) = if let Some(file) = config_file {
            if !file.exists() {
                return Err(UnifiedConfigError::FileNotFound(file.to_path_buf()));
            }
            (file.to_path_buf(), ConfigFormat::from_path(file))
        } else {
            // Auto-discover configuration
            match self.get_preferred_config_source(Some(ConfigFormat::Yaml)) {
                Some((path, fmt)) => {
                    info!("Loading configuration from: {}", path.display());
                    (path, fmt)
                }
                None => {
                    info!("No configuration file found, using defaults");
                    return Ok(DaemonConfig::default());
                }
            }
        };

        // Load configuration data based on format
        let config_data = self.load_config_file(&source_file, format)?;

        // Apply environment variable overrides
        let config_with_env = self.apply_env_overrides(config_data)?;

        // Expand environment variables in path values
        let config_expanded = self.expand_config_paths(config_with_env);

        // Validate configuration
        self.validate_config(&config_expanded)?;

        info!("Configuration loaded and validated successfully");
        Ok(config_expanded)
    }

    /// Load configuration file based on format
    fn load_config_file(&self, file_path: &Path, format: ConfigFormat) -> Result<DaemonConfig, UnifiedConfigError> {
        let content = fs::read_to_string(file_path)?;

        match format {
            ConfigFormat::Yaml => {
                serde_yml::from_str(&content)
                    .map_err(|e| UnifiedConfigError::YamlError(e.to_string()))
            }
        }
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&self, mut config: DaemonConfig) -> Result<DaemonConfig, UnifiedConfigError> {
        const ENV_PREFIX: &str = "WORKSPACE_QDRANT_";

        // Server-level overrides
        if let Ok(log_file) = std::env::var(format!("{}LOG_FILE", ENV_PREFIX)) {
            config.log_file = Some(PathBuf::from(log_file));
        }

        if let Ok(max_concurrent) = std::env::var(format!("{}MAX_CONCURRENT_TASKS", ENV_PREFIX)) {
            config.max_concurrent_tasks = Some(max_concurrent.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid max_concurrent_tasks: {}", e)))?);
        }

        if let Ok(timeout) = std::env::var(format!("{}DEFAULT_TIMEOUT_MS", ENV_PREFIX)) {
            config.default_timeout_ms = Some(timeout.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid default_timeout_ms: {}", e)))?);
        }

        if let Ok(preemption) = std::env::var(format!("{}ENABLE_PREEMPTION", ENV_PREFIX)) {
            config.enable_preemption = preemption.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid enable_preemption: {}", e)))?;
        }

        if let Ok(chunk_size) = std::env::var(format!("{}CHUNK_SIZE", ENV_PREFIX)) {
            config.chunk_size = chunk_size.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid chunk_size: {}", e)))?;
        }


        if let Ok(log_level) = std::env::var(format!("{}LOG_LEVEL", ENV_PREFIX)) {
            config.log_level = log_level;
        }

        if let Ok(enable_metrics) = std::env::var(format!("{}ENABLE_METRICS", ENV_PREFIX)) {
            config.observability.metrics.enabled = enable_metrics.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid enable_metrics: {}", e)))?;
        }

        if let Ok(metrics_interval) = std::env::var(format!("{}METRICS_INTERVAL_SECS", ENV_PREFIX)) {
            config.observability.collection_interval = metrics_interval.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid metrics_interval_secs: {}", e)))?;
        }

        // Qdrant configuration overrides
        if let Ok(url) = std::env::var(format!("{}QDRANT__URL", ENV_PREFIX)) {
            config.qdrant.url = url;
        }

        if let Ok(transport) = std::env::var(format!("{}QDRANT__TRANSPORT", ENV_PREFIX)) {
            config.qdrant.transport = match transport.to_lowercase().as_str() {
                "grpc" => TransportMode::Grpc,
                "http" => TransportMode::Http,
                _ => return Err(UnifiedConfigError::ValidationError(
                    format!("Invalid transport mode: {}", transport)
                )),
            };
        }

        if let Ok(timeout) = std::env::var(format!("{}QDRANT__TIMEOUT_MS", ENV_PREFIX)) {
            config.qdrant.timeout_ms = timeout.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant timeout_ms: {}", e)))?;
        }

        if let Ok(max_retries) = std::env::var(format!("{}QDRANT__MAX_RETRIES", ENV_PREFIX)) {
            config.qdrant.max_retries = max_retries.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant max_retries: {}", e)))?;
        }

        if let Ok(retry_delay) = std::env::var(format!("{}QDRANT__RETRY_DELAY_MS", ENV_PREFIX)) {
            config.qdrant.retry_delay_ms = retry_delay.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant retry_delay_ms: {}", e)))?;
        }

        if let Ok(pool_size) = std::env::var(format!("{}QDRANT__POOL_SIZE", ENV_PREFIX)) {
            config.qdrant.pool_size = pool_size.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant pool_size: {}", e)))?;
        }

        if let Ok(tls) = std::env::var(format!("{}QDRANT__TLS", ENV_PREFIX)) {
            config.qdrant.tls = tls.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant tls: {}", e)))?;
        }

        if let Ok(vector_size) = std::env::var(format!("{}QDRANT__DENSE_VECTOR_SIZE", ENV_PREFIX)) {
            config.qdrant.dense_vector_size = vector_size.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid qdrant dense_vector_size: {}", e)))?;
        }

        // Auto-ingestion configuration overrides
        if let Ok(enabled) = std::env::var(format!("{}AUTO_INGESTION__ENABLED", ENV_PREFIX)) {
            config.auto_ingestion.enabled = enabled.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid auto_ingestion enabled: {}", e)))?;
        }

        if let Ok(auto_watches) = std::env::var(format!("{}AUTO_INGESTION__AUTO_CREATE_WATCHES", ENV_PREFIX)) {
            config.auto_ingestion.auto_create_watches = auto_watches.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid auto_ingestion auto_create_watches: {}", e)))?;
        }

        if let Ok(suffix) = std::env::var(format!("{}AUTO_INGESTION__TARGET_COLLECTION_SUFFIX", ENV_PREFIX)) {
            config.auto_ingestion.target_collection_suffix = suffix;
        }

        if let Ok(max_files) = std::env::var(format!("{}AUTO_INGESTION__MAX_FILES_PER_BATCH", ENV_PREFIX)) {
            config.auto_ingestion.max_files_per_batch = max_files.parse()
                .map_err(|e| UnifiedConfigError::ValidationError(format!("Invalid auto_ingestion max_files_per_batch: {}", e)))?;
        }

        Ok(config)
    }

    /// Expand environment variables in path-like configuration values.
    ///
    /// Supports ${VAR} and $VAR syntax. Unset variables are left unchanged.
    fn expand_config_paths(&self, mut config: DaemonConfig) -> DaemonConfig {
        // Expand log_file path
        if let Some(ref path) = config.log_file {
            config.log_file = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
        }

        // Expand project_path
        if let Some(ref path) = config.project_path {
            config.project_path = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
        }

        // Expand embedding.model_cache_dir
        if let Some(ref path) = config.embedding.model_cache_dir {
            config.embedding.model_cache_dir = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
        }

        debug!("Expanded environment variables in path configuration values");
        config
    }

    /// Validate configuration
    fn validate_config(&self, config: &DaemonConfig) -> Result<(), UnifiedConfigError> {
        // Validate concurrent tasks
        if let Some(max_concurrent) = config.max_concurrent_tasks {
            if max_concurrent == 0 {
                return Err(UnifiedConfigError::ValidationError(
                    "max_concurrent_tasks must be greater than 0".to_string()
                ));
            }
            if max_concurrent > 100 {
                return Err(UnifiedConfigError::ValidationError(
                    "max_concurrent_tasks should not exceed 100".to_string()
                ));
            }
        }

        // Validate timeout
        if let Some(timeout) = config.default_timeout_ms {
            if timeout == 0 {
                return Err(UnifiedConfigError::ValidationError(
                    "default_timeout_ms must be greater than 0".to_string()
                ));
            }
            if timeout > 300_000 {
                return Err(UnifiedConfigError::ValidationError(
                    "default_timeout_ms should not exceed 5 minutes".to_string()
                ));
            }
        }

        // Validate chunk size
        if config.chunk_size == 0 {
            return Err(UnifiedConfigError::ValidationError(
                "chunk_size must be greater than 0".to_string()
            ));
        }
        if config.chunk_size > 10_000 {
            return Err(UnifiedConfigError::ValidationError(
                "chunk_size should not exceed 10,000".to_string()
            ));
        }

        // Validate log level
        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&config.log_level.as_str()) {
            return Err(UnifiedConfigError::ValidationError(
                format!("log_level must be one of: {}", valid_log_levels.join(", "))
            ));
        }

        // Validate Qdrant URL
        if config.qdrant.url.is_empty() {
            return Err(UnifiedConfigError::ValidationError(
                "Qdrant URL is required".to_string()
            ));
        }

        if !config.qdrant.url.starts_with("http://") && !config.qdrant.url.starts_with("https://") {
            return Err(UnifiedConfigError::ValidationError(
                "Qdrant URL must start with http:// or https://".to_string()
            ));
        }

        Ok(())
    }

    /// Save configuration to file
    pub fn save_config(&self, config: &DaemonConfig, file_path: &Path, format: ConfigFormat) -> Result<(), UnifiedConfigError> {
        // Create parent directory if needed
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = match format {
            ConfigFormat::Yaml => {
                serde_yml::to_string(config)
                    .map_err(|e| UnifiedConfigError::YamlError(e.to_string()))?
            }
        };

        fs::write(file_path, content)?;
        info!("Configuration saved to {} in {:?} format", file_path.display(), format);
        Ok(())
    }

    /// Convert configuration file between formats
    pub fn convert_config(&self, source_file: &Path, target_file: &Path, target_format: Option<ConfigFormat>) -> Result<(), UnifiedConfigError> {
        if !source_file.exists() {
            return Err(UnifiedConfigError::FileNotFound(source_file.to_path_buf()));
        }

        let source_format = ConfigFormat::from_path(source_file);
        let target_format = target_format.unwrap_or_else(|| ConfigFormat::from_path(target_file));

        info!("Converting configuration from {} to {} ({:?})", 
              source_file.display(), target_file.display(), target_format);

        // Load source configuration
        let config = self.load_config_file(source_file, source_format)?;

        // Save in target format
        self.save_config(&config, target_file, target_format)?;

        info!("Configuration conversion completed: {} -> {}", 
              source_file.display(), target_file.display());
        Ok(())
    }

    /// Get configuration information
    pub fn get_config_info(&self) -> HashMap<String, serde_json::Value> {
        let sources = self.discover_config_sources();
        let mut info = HashMap::new();

        info.insert("config_dir".to_string(), 
                   serde_json::Value::String(self.config_dir.display().to_string()));
        info.insert("env_prefix".to_string(), 
                   serde_json::Value::String("WORKSPACE_QDRANT_".to_string()));

        let sources_json: Vec<serde_json::Value> = sources.into_iter().map(|(path, format, exists)| {
            let mut source = serde_json::Map::new();
            source.insert("file_path".to_string(), 
                         serde_json::Value::String(path.display().to_string()));
            source.insert("format".to_string(), 
                         serde_json::Value::String(format!("{:?}", format).to_lowercase()));
            source.insert("exists".to_string(), 
                         serde_json::Value::Bool(exists));
            
            if exists {
                if let Ok(metadata) = path.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                            source.insert("last_modified".to_string(), 
                                         serde_json::Value::Number(serde_json::Number::from(duration.as_secs())));
                        }
                    }
                }
            }
            
            serde_json::Value::Object(source)
        }).collect();

        info.insert("sources".to_string(), serde_json::Value::Array(sources_json));

        if let Some((preferred_path, _)) = self.get_preferred_config_source(Some(ConfigFormat::Yaml)) {
            info.insert("preferred_source".to_string(), 
                       serde_json::Value::String(preferred_path.display().to_string()));
        }

        info
    }
}

impl Default for UnifiedConfigManager {
    fn default() -> Self {
        Self::new(None::<PathBuf>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_config_format_detection() {
        assert_eq!(ConfigFormat::from_path("config.yaml"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config.yml"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config"), ConfigFormat::Yaml);
        assert_eq!(ConfigFormat::from_path("config.toml"), ConfigFormat::Yaml); // Should default to YAML
        assert_eq!(ConfigFormat::from_path("config.json"), ConfigFormat::Yaml); // Should default to YAML
    }

    #[test]
    fn test_config_discovery() {
        let temp_dir = TempDir::new().unwrap();
        let config_manager = UnifiedConfigManager::new(Some(temp_dir.path()));

        // Create config file with current preferred name
        let preferred_path = temp_dir.path().join(".workspace-qdrant.yaml");
        fs::write(&preferred_path, "# Config").unwrap();

        let sources = config_manager.discover_config_sources();
        let existing_sources: Vec<_> = sources.into_iter()
            .filter(|(_, _, exists)| *exists)
            .collect();

        assert_eq!(existing_sources.len(), 1);

        // Should find .workspace-qdrant.yaml
        let preferred = config_manager.get_preferred_config_source(Some(ConfigFormat::Yaml));
        assert!(preferred.is_some());
        let (path, format) = preferred.unwrap();
        assert_eq!(format, ConfigFormat::Yaml);
        assert!(path.ends_with(".workspace-qdrant.yaml"));
    }

    #[test]
    fn test_default_config_creation() {
        let config = DaemonConfig::default();
        assert_eq!(config.max_concurrent_tasks, Some(4));
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        let config_manager = UnifiedConfigManager::new(Some(temp_dir.path()));

        // Test valid config
        let valid_config = DaemonConfig::default();
        assert!(config_manager.validate_config(&valid_config).is_ok());

        // Test invalid chunk size
        let mut invalid_config = DaemonConfig::default();
        invalid_config.chunk_size = 0;
        assert!(config_manager.validate_config(&invalid_config).is_err());

        // Test invalid max concurrent tasks
        invalid_config = DaemonConfig::default();
        invalid_config.max_concurrent_tasks = Some(0);
        assert!(config_manager.validate_config(&invalid_config).is_err());

        // Test invalid log level
        invalid_config = DaemonConfig::default();
        invalid_config.log_level = "invalid".to_string();
        assert!(config_manager.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_expand_env_vars() {
        // Set a test environment variable
        std::env::set_var("WQM_TEST_VAR", "/test/path");

        // Test ${VAR} syntax
        assert_eq!(expand_env_vars("${WQM_TEST_VAR}/cache"), "/test/path/cache");

        // Test $VAR syntax
        assert_eq!(expand_env_vars("$WQM_TEST_VAR/cache"), "/test/path/cache");

        // Test unset variable (should be left unchanged)
        let result = expand_env_vars("$WQM_NONEXISTENT_VAR/path");
        assert!(result.contains("WQM_NONEXISTENT_VAR"));

        // Test string without variables
        assert_eq!(expand_env_vars("/static/path"), "/static/path");

        // Test multiple variables
        std::env::set_var("WQM_TEST_VAR2", "subdir");
        assert_eq!(
            expand_env_vars("${WQM_TEST_VAR}/${WQM_TEST_VAR2}"),
            "/test/path/subdir"
        );

        // Cleanup
        std::env::remove_var("WQM_TEST_VAR");
        std::env::remove_var("WQM_TEST_VAR2");
    }

    #[test]
    fn test_expand_path_env_vars() {
        std::env::set_var("WQM_TEST_HOME", "/home/testuser");

        // Test with Some path
        let path = PathBuf::from("${WQM_TEST_HOME}/.cache/models");
        let expanded = expand_path_env_vars(Some(&path));
        assert!(expanded.is_some());
        assert_eq!(expanded.unwrap(), PathBuf::from("/home/testuser/.cache/models"));

        // Test with None
        let expanded_none = expand_path_env_vars(None);
        assert!(expanded_none.is_none());

        // Cleanup
        std::env::remove_var("WQM_TEST_HOME");
    }

    #[test]
    fn test_expand_config_paths() {
        let temp_dir = TempDir::new().unwrap();
        let config_manager = UnifiedConfigManager::new(Some(temp_dir.path()));

        std::env::set_var("WQM_TEST_DIR", "/expanded/dir");

        let mut config = DaemonConfig::default();
        config.log_file = Some(PathBuf::from("${WQM_TEST_DIR}/daemon.log"));
        config.project_path = Some(PathBuf::from("$WQM_TEST_DIR/project"));
        config.embedding.model_cache_dir = Some(PathBuf::from("${WQM_TEST_DIR}/models"));

        let expanded = config_manager.expand_config_paths(config);

        assert_eq!(
            expanded.log_file.unwrap(),
            PathBuf::from("/expanded/dir/daemon.log")
        );
        assert_eq!(
            expanded.project_path.unwrap(),
            PathBuf::from("/expanded/dir/project")
        );
        assert_eq!(
            expanded.embedding.model_cache_dir.unwrap(),
            PathBuf::from("/expanded/dir/models")
        );

        // Cleanup
        std::env::remove_var("WQM_TEST_DIR");
    }
}