//! Environment variable overrides for DaemonConfig

use std::path::PathBuf;

use crate::config::DaemonConfig;
use crate::storage::TransportMode;
use crate::unified_config::types::UnifiedConfigError;

const ENV_PREFIX: &str = "WORKSPACE_QDRANT_";

/// Apply environment variable overrides to `config`, returning the mutated value.
pub(super) fn apply_env_overrides(
    mut config: DaemonConfig,
) -> Result<DaemonConfig, UnifiedConfigError> {
    apply_general_overrides(&mut config)?;
    apply_qdrant_overrides(&mut config)?;
    apply_auto_ingestion_overrides(&mut config)?;
    apply_daemon_endpoint_overrides(&mut config)?;
    Ok(config)
}

fn apply_general_overrides(config: &mut DaemonConfig) -> Result<(), UnifiedConfigError> {
    if let Ok(log_file) = std::env::var(format!("{}LOG_FILE", ENV_PREFIX)) {
        config.log_file = Some(PathBuf::from(log_file));
    }

    if let Ok(max_concurrent) =
        std::env::var(format!("{}MAX_CONCURRENT_TASKS", ENV_PREFIX))
    {
        config.max_concurrent_tasks = Some(max_concurrent.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid max_concurrent_tasks: {}",
                e
            ))
        })?);
    }

    if let Ok(timeout) = std::env::var(format!("{}DEFAULT_TIMEOUT_MS", ENV_PREFIX)) {
        config.default_timeout_ms = Some(timeout.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid default_timeout_ms: {}", e))
        })?);
    }

    if let Ok(preemption) = std::env::var(format!("{}ENABLE_PREEMPTION", ENV_PREFIX)) {
        config.enable_preemption = preemption.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid enable_preemption: {}", e))
        })?;
    }

    if let Ok(chunk_size) = std::env::var(format!("{}CHUNK_SIZE", ENV_PREFIX)) {
        config.chunk_size = chunk_size.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid chunk_size: {}", e))
        })?;
    }

    if let Ok(log_level) = std::env::var(format!("{}LOG_LEVEL", ENV_PREFIX)) {
        config.log_level = log_level;
    }

    if let Ok(enable_metrics) = std::env::var(format!("{}ENABLE_METRICS", ENV_PREFIX)) {
        config.observability.metrics.enabled = enable_metrics.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid enable_metrics: {}", e))
        })?;
    }

    if let Ok(metrics_interval) =
        std::env::var(format!("{}METRICS_INTERVAL_SECS", ENV_PREFIX))
    {
        config.observability.collection_interval =
            metrics_interval.parse().map_err(|e| {
                UnifiedConfigError::ValidationError(format!(
                    "Invalid metrics_interval_secs: {}",
                    e
                ))
            })?;
    }

    Ok(())
}

fn apply_qdrant_overrides(config: &mut DaemonConfig) -> Result<(), UnifiedConfigError> {
    if let Ok(url) = std::env::var(format!("{}QDRANT__URL", ENV_PREFIX)) {
        config.qdrant.url = url;
    }

    if let Ok(transport) = std::env::var(format!("{}QDRANT__TRANSPORT", ENV_PREFIX)) {
        config.qdrant.transport = match transport.to_lowercase().as_str() {
            "grpc" => TransportMode::Grpc,
            "http" => TransportMode::Http,
            _ => {
                return Err(UnifiedConfigError::ValidationError(format!(
                    "Invalid transport mode: {}",
                    transport
                )))
            }
        };
    }

    if let Ok(timeout) = std::env::var(format!("{}QDRANT__TIMEOUT_MS", ENV_PREFIX)) {
        config.qdrant.timeout_ms = timeout.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid qdrant timeout_ms: {}", e))
        })?;
    }

    if let Ok(max_retries) = std::env::var(format!("{}QDRANT__MAX_RETRIES", ENV_PREFIX)) {
        config.qdrant.max_retries = max_retries.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid qdrant max_retries: {}", e))
        })?;
    }

    if let Ok(retry_delay) =
        std::env::var(format!("{}QDRANT__RETRY_DELAY_MS", ENV_PREFIX))
    {
        config.qdrant.retry_delay_ms = retry_delay.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid qdrant retry_delay_ms: {}",
                e
            ))
        })?;
    }

    if let Ok(pool_size) = std::env::var(format!("{}QDRANT__POOL_SIZE", ENV_PREFIX)) {
        config.qdrant.pool_size = pool_size.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid qdrant pool_size: {}", e))
        })?;
    }

    if let Ok(tls) = std::env::var(format!("{}QDRANT__TLS", ENV_PREFIX)) {
        config.qdrant.tls = tls.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid qdrant tls: {}", e))
        })?;
    }

    if let Ok(vector_size) =
        std::env::var(format!("{}QDRANT__DENSE_VECTOR_SIZE", ENV_PREFIX))
    {
        config.qdrant.dense_vector_size = vector_size.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid qdrant dense_vector_size: {}",
                e
            ))
        })?;
    }

    Ok(())
}

fn apply_auto_ingestion_overrides(
    config: &mut DaemonConfig,
) -> Result<(), UnifiedConfigError> {
    if let Ok(enabled) =
        std::env::var(format!("{}AUTO_INGESTION__ENABLED", ENV_PREFIX))
    {
        config.auto_ingestion.enabled = enabled.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid auto_ingestion enabled: {}",
                e
            ))
        })?;
    }

    if let Ok(auto_watches) =
        std::env::var(format!("{}AUTO_INGESTION__AUTO_CREATE_WATCHES", ENV_PREFIX))
    {
        config.auto_ingestion.auto_create_watches = auto_watches.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid auto_ingestion auto_create_watches: {}",
                e
            ))
        })?;
    }

    if let Ok(suffix) =
        std::env::var(format!("{}AUTO_INGESTION__TARGET_COLLECTION_SUFFIX", ENV_PREFIX))
    {
        config.auto_ingestion.target_collection_suffix = suffix;
    }

    if let Ok(max_files) =
        std::env::var(format!("{}AUTO_INGESTION__MAX_FILES_PER_BATCH", ENV_PREFIX))
    {
        config.auto_ingestion.max_files_per_batch = max_files.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!(
                "Invalid auto_ingestion max_files_per_batch: {}",
                e
            ))
        })?;
    }

    Ok(())
}

fn apply_daemon_endpoint_overrides(
    config: &mut DaemonConfig,
) -> Result<(), UnifiedConfigError> {
    if let Ok(host) = std::env::var(format!("{}DAEMON_HOST", ENV_PREFIX)) {
        config.daemon_endpoint.host = host;
    }

    if let Ok(port) = std::env::var(format!("{}DAEMON_PORT", ENV_PREFIX)) {
        config.daemon_endpoint.grpc_port = port.parse().map_err(|e| {
            UnifiedConfigError::ValidationError(format!("Invalid daemon_port: {}", e))
        })?;
    }

    if let Ok(token) = std::env::var(format!("{}DAEMON_AUTH_TOKEN", ENV_PREFIX)) {
        config.daemon_endpoint.auth_token = Some(token);
    }

    Ok(())
}
