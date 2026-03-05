//! UnifiedConfigManager: config discovery, loading, persistence, and introspection

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

use crate::config::DaemonConfig;
use crate::unified_config::env_overrides::apply_env_overrides;
use crate::unified_config::types::{ConfigFormat, UnifiedConfigError};
use crate::unified_config::validation::{expand_config_paths, validate_config};

/// Unified configuration manager for Rust daemon
///
/// Uses `wqm_common::paths::get_config_search_paths()` for the canonical
/// config file cascade. No project-local config is searched.
pub struct UnifiedConfigManager;

impl UnifiedConfigManager {
    /// Create new unified configuration manager.
    ///
    /// The `_config_dir` parameter is accepted for backward compatibility
    /// but ignored — config search paths are determined by `wqm_common::paths`.
    pub fn new<P: Into<PathBuf>>(_config_dir: Option<P>) -> Self {
        Self
    }

    /// Get unified config search paths (in priority order).
    ///
    /// Delegates to `wqm_common::paths::get_config_search_paths()`.
    pub fn get_unified_search_paths(&self) -> Vec<PathBuf> {
        wqm_common::paths::get_config_search_paths()
    }

    /// Discover available configuration files from all unified search paths.
    ///
    /// Returns tuples of (path, format, exists) for all potential config locations.
    pub fn discover_config_sources(&self) -> Vec<(PathBuf, ConfigFormat, bool)> {
        use tracing::debug;

        let paths = self.get_unified_search_paths();
        let mut sources = Vec::new();

        for config_file in paths {
            let format = ConfigFormat::from_path(&config_file);
            let exists = config_file.exists();
            sources.push((config_file, format, exists));
        }

        debug!(
            "Discovered {} configuration sources ({} exist)",
            sources.len(),
            sources.iter().filter(|(_, _, e)| *e).count()
        );
        sources
    }

    /// Get preferred configuration source from unified search paths.
    ///
    /// Returns the first existing config file, optionally preferring a format.
    pub fn get_preferred_config_source(
        &self,
        prefer_format: Option<ConfigFormat>,
    ) -> Option<(PathBuf, ConfigFormat)> {
        let existing_sources: Vec<_> = self
            .discover_config_sources()
            .into_iter()
            .filter(|(_, _, exists)| *exists)
            .collect();

        if existing_sources.is_empty() {
            return None;
        }

        if let Some(prefer_fmt) = prefer_format {
            for (path, format, _) in &existing_sources {
                if *format == prefer_fmt {
                    return Some((path.clone(), *format));
                }
            }
        }

        existing_sources
            .first()
            .map(|(path, format, _)| (path.clone(), *format))
    }

    /// Load unified configuration, applying env overrides, path expansion, and validation.
    pub fn load_config(
        &self,
        config_file: Option<&Path>,
    ) -> Result<DaemonConfig, UnifiedConfigError> {
        let (source_file, format) = if let Some(file) = config_file {
            if !file.exists() {
                return Err(UnifiedConfigError::FileNotFound(file.to_path_buf()));
            }
            (file.to_path_buf(), ConfigFormat::from_path(file))
        } else {
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

        let raw = self.load_config_file(&source_file, format)?;
        let with_env = apply_env_overrides(raw)?;
        let expanded = expand_config_paths(with_env);
        validate_config(&expanded)?;

        info!("Configuration loaded and validated successfully");
        Ok(expanded)
    }

    /// Load and deserialise a configuration file.
    fn load_config_file(
        &self,
        file_path: &Path,
        format: ConfigFormat,
    ) -> Result<DaemonConfig, UnifiedConfigError> {
        let content = fs::read_to_string(file_path)?;
        match format {
            ConfigFormat::Yaml => serde_yaml_ng::from_str(&content)
                .map_err(|e| UnifiedConfigError::YamlError(e.to_string())),
        }
    }

    /// Expose path expansion publicly (used by tests).
    pub fn expand_config_paths(&self, config: DaemonConfig) -> DaemonConfig {
        expand_config_paths(config)
    }

    /// Expose validation publicly (used by tests).
    pub fn validate_config(&self, config: &DaemonConfig) -> Result<(), UnifiedConfigError> {
        validate_config(config)
    }

    /// Save configuration to file.
    pub fn save_config(
        &self,
        config: &DaemonConfig,
        file_path: &Path,
        format: ConfigFormat,
    ) -> Result<(), UnifiedConfigError> {
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = match format {
            ConfigFormat::Yaml => serde_yaml_ng::to_string(config)
                .map_err(|e| UnifiedConfigError::YamlError(e.to_string()))?,
        };

        fs::write(file_path, content)?;
        info!(
            "Configuration saved to {} in {:?} format",
            file_path.display(),
            format
        );
        Ok(())
    }

    /// Convert a configuration file to another format (YAML-only, kept for API compat).
    pub fn convert_config(
        &self,
        source_file: &Path,
        target_file: &Path,
        target_format: Option<ConfigFormat>,
    ) -> Result<(), UnifiedConfigError> {
        if !source_file.exists() {
            return Err(UnifiedConfigError::FileNotFound(source_file.to_path_buf()));
        }

        let source_format = ConfigFormat::from_path(source_file);
        let target_format = target_format.unwrap_or_else(|| ConfigFormat::from_path(target_file));

        info!(
            "Converting configuration from {} to {} ({:?})",
            source_file.display(),
            target_file.display(),
            target_format
        );

        let config = self.load_config_file(source_file, source_format)?;
        self.save_config(&config, target_file, target_format)?;

        info!(
            "Configuration conversion completed: {} -> {}",
            source_file.display(),
            target_file.display()
        );
        Ok(())
    }

    /// Return a JSON map describing available config sources and the preferred one.
    pub fn get_config_info(&self) -> HashMap<String, serde_json::Value> {
        let sources = self.discover_config_sources();
        let mut info = HashMap::new();

        info.insert(
            "env_prefix".to_string(),
            serde_json::Value::String("WORKSPACE_QDRANT_".to_string()),
        );

        let sources_json: Vec<serde_json::Value> = sources
            .into_iter()
            .map(|(path, format, exists)| {
                let mut source = serde_json::Map::new();
                source.insert(
                    "file_path".to_string(),
                    serde_json::Value::String(path.display().to_string()),
                );
                source.insert(
                    "format".to_string(),
                    serde_json::Value::String(format!("{:?}", format).to_lowercase()),
                );
                source.insert("exists".to_string(), serde_json::Value::Bool(exists));

                if exists {
                    if let Ok(metadata) = path.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                                source.insert(
                                    "last_modified".to_string(),
                                    serde_json::Value::Number(serde_json::Number::from(
                                        duration.as_secs(),
                                    )),
                                );
                            }
                        }
                    }
                }

                serde_json::Value::Object(source)
            })
            .collect();

        info.insert(
            "sources".to_string(),
            serde_json::Value::Array(sources_json),
        );

        if let Some((preferred_path, _)) =
            self.get_preferred_config_source(Some(ConfigFormat::Yaml))
        {
            info.insert(
                "preferred_source".to_string(),
                serde_json::Value::String(preferred_path.display().to_string()),
            );
        }

        info
    }
}

impl Default for UnifiedConfigManager {
    fn default() -> Self {
        Self
    }
}
