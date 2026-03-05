//! Config validation and path expansion

use std::path::PathBuf;

use tracing::debug;

use crate::config::DaemonConfig;
use crate::unified_config::types::UnifiedConfigError;
use wqm_common::env_expand::expand_env_vars;

/// Expand environment variables in path-like configuration values.
pub(super) fn expand_config_paths(mut config: DaemonConfig) -> DaemonConfig {
    if let Some(ref path) = config.log_file {
        config.log_file = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }
    if let Some(ref path) = config.project_path {
        config.project_path = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }
    if let Some(ref path) = config.embedding.model_cache_dir {
        config.embedding.model_cache_dir =
            Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }

    debug!("Expanded environment variables in path configuration values");
    config
}

/// Validate a `DaemonConfig` for correctness.
pub(super) fn validate_config(config: &DaemonConfig) -> Result<(), UnifiedConfigError> {
    if let Some(max_concurrent) = config.max_concurrent_tasks {
        if max_concurrent == 0 {
            return Err(UnifiedConfigError::ValidationError(
                "max_concurrent_tasks must be greater than 0".to_string(),
            ));
        }
        if max_concurrent > 100 {
            return Err(UnifiedConfigError::ValidationError(
                "max_concurrent_tasks should not exceed 100".to_string(),
            ));
        }
    }

    if let Some(timeout) = config.default_timeout_ms {
        if timeout == 0 {
            return Err(UnifiedConfigError::ValidationError(
                "default_timeout_ms must be greater than 0".to_string(),
            ));
        }
        if timeout > 300_000 {
            return Err(UnifiedConfigError::ValidationError(
                "default_timeout_ms should not exceed 5 minutes".to_string(),
            ));
        }
    }

    if config.chunk_size == 0 {
        return Err(UnifiedConfigError::ValidationError(
            "chunk_size must be greater than 0".to_string(),
        ));
    }
    if config.chunk_size > 10_000 {
        return Err(UnifiedConfigError::ValidationError(
            "chunk_size should not exceed 10,000".to_string(),
        ));
    }

    let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
    if !valid_log_levels.contains(&config.log_level.as_str()) {
        return Err(UnifiedConfigError::ValidationError(format!(
            "log_level must be one of: {}",
            valid_log_levels.join(", ")
        )));
    }

    if config.qdrant.url.is_empty() {
        return Err(UnifiedConfigError::ValidationError(
            "Qdrant URL is required".to_string(),
        ));
    }

    if !config.qdrant.url.starts_with("http://") && !config.qdrant.url.starts_with("https://") {
        return Err(UnifiedConfigError::ValidationError(
            "Qdrant URL must start with http:// or https://".to_string(),
        ));
    }

    Ok(())
}
