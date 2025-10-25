//! Configuration for Qdrant client operations

use crate::qdrant::error::QdrantResult;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for Qdrant client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantClientConfig {
    /// Qdrant server URL
    pub url: String,

    /// API key for authentication (optional)
    pub api_key: Option<String>,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Maximum number of retries for failed operations
    pub max_retries: u32,

    /// Initial retry delay in milliseconds
    pub retry_delay_ms: u64,

    /// Maximum retry delay in milliseconds (for exponential backoff)
    pub max_retry_delay_ms: u64,

    /// Connection pool configuration
    pub pool_config: PoolConfig,

    /// Circuit breaker configuration
    pub circuit_breaker_config: CircuitBreakerConfig,

    /// Default collection configuration
    pub default_collection_config: DefaultCollectionConfig,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,

    /// Minimum number of idle connections to maintain
    pub min_idle_connections: usize,

    /// Maximum idle time for connections before closing
    pub max_idle_time_secs: u64,

    /// Maximum connection lifetime
    pub max_connection_lifetime_secs: u64,

    /// Connection acquisition timeout
    pub acquisition_timeout_secs: u64,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,

    /// Failure threshold to open circuit
    pub failure_threshold: u32,

    /// Success threshold to close circuit
    pub success_threshold: u32,

    /// Circuit breaker timeout in seconds
    pub timeout_secs: u64,

    /// Half-open state duration in seconds
    pub half_open_timeout_secs: u64,
}

/// Default collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultCollectionConfig {
    /// Default vector size
    pub vector_size: u64,

    /// Default distance metric
    pub distance_metric: String,

    /// Enable indexing by default
    pub enable_indexing: bool,

    /// Default replication factor
    pub replication_factor: u32,

    /// Default number of shards
    pub shard_number: u32,

    /// Enable on-disk vectors
    pub on_disk_vectors: bool,

    /// HNSW configuration
    pub hnsw_config: HnswConfig,
}

/// HNSW (Hierarchical Navigable Small World) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links created for each node
    pub m: u64,

    /// Size of the dynamic candidate list
    pub ef_construct: u64,

    /// Number of nearest neighbors to return during search
    pub ef: u64,

    /// Enable full scan for small segments
    pub full_scan_threshold: u64,
}

impl Default for QdrantClientConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            connection_timeout_secs: 30,
            request_timeout_secs: 60,
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            pool_config: PoolConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            default_collection_config: DefaultCollectionConfig::default(),
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_idle_connections: 2,
            max_idle_time_secs: 300,
            max_connection_lifetime_secs: 3600,
            acquisition_timeout_secs: 30,
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout_secs: 60,
            half_open_timeout_secs: 30,
        }
    }
}

impl Default for DefaultCollectionConfig {
    fn default() -> Self {
        Self {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            replication_factor: 1,
            shard_number: 1,
            on_disk_vectors: false,
            hnsw_config: HnswConfig::default(),
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construct: 100,
            ef: 64,
            full_scan_threshold: 10000,
        }
    }
}

impl QdrantClientConfig {
    /// Validate the configuration
    pub fn validate(&self) -> QdrantResult<()> {
        use crate::qdrant::error::QdrantError;

        if self.url.is_empty() {
            return Err(QdrantError::Configuration {
                message: "URL cannot be empty".to_string(),
            });
        }

        if self.connection_timeout_secs == 0 {
            return Err(QdrantError::Configuration {
                message: "Connection timeout must be greater than 0".to_string(),
            });
        }

        if self.request_timeout_secs == 0 {
            return Err(QdrantError::Configuration {
                message: "Request timeout must be greater than 0".to_string(),
            });
        }

        if self.max_retries > 10 {
            return Err(QdrantError::Configuration {
                message: "Maximum retries should not exceed 10".to_string(),
            });
        }

        if self.pool_config.max_connections == 0 {
            return Err(QdrantError::Configuration {
                message: "Pool max connections must be greater than 0".to_string(),
            });
        }

        if self.pool_config.min_idle_connections > self.pool_config.max_connections {
            return Err(QdrantError::Configuration {
                message: "Pool min idle connections cannot exceed max connections".to_string(),
            });
        }

        if self.default_collection_config.vector_size == 0 {
            return Err(QdrantError::Configuration {
                message: "Default vector size must be greater than 0".to_string(),
            });
        }

        // Validate distance metric
        match self.default_collection_config.distance_metric.as_str() {
            "Cosine" | "Euclid" | "Dot" => {}
            _ => {
                return Err(QdrantError::Configuration {
                    message: format!("Invalid distance metric: {}", self.default_collection_config.distance_metric),
                });
            }
        }

        Ok(())
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.connection_timeout_secs)
    }

    /// Get request timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    /// Get initial retry delay as Duration
    pub fn retry_delay(&self) -> Duration {
        Duration::from_millis(self.retry_delay_ms)
    }

    /// Get maximum retry delay as Duration
    pub fn max_retry_delay(&self) -> Duration {
        Duration::from_millis(self.max_retry_delay_ms)
    }

    /// Create a test configuration for testing purposes
    #[cfg(any(test, feature = "test-utils"))]
    pub fn test_config() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            connection_timeout_secs: 5,
            request_timeout_secs: 10,
            max_retries: 2,
            retry_delay_ms: 100,
            max_retry_delay_ms: 1000,
            pool_config: PoolConfig {
                max_connections: 2,
                min_idle_connections: 1,
                max_idle_time_secs: 60,
                max_connection_lifetime_secs: 300,
                acquisition_timeout_secs: 5,
            },
            circuit_breaker_config: CircuitBreakerConfig {
                enabled: false, // Disable for tests
                failure_threshold: 2,
                success_threshold: 1,
                timeout_secs: 10,
                half_open_timeout_secs: 5,
            },
            default_collection_config: DefaultCollectionConfig {
                vector_size: 128, // Smaller for tests
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
                on_disk_vectors: false,
                hnsw_config: HnswConfig {
                    m: 8,
                    ef_construct: 50,
                    ef: 32,
                    full_scan_threshold: 1000,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = QdrantClientConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_test_config_validation() {
        let config = QdrantClientConfig::test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_url() {
        let mut config = QdrantClientConfig::default();
        config.url = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_timeout() {
        let mut config = QdrantClientConfig::default();
        config.connection_timeout_secs = 0;
        assert!(config.validate().is_err());

        config.connection_timeout_secs = 30;
        config.request_timeout_secs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_pool_config() {
        let mut config = QdrantClientConfig::default();
        config.pool_config.max_connections = 0;
        assert!(config.validate().is_err());

        config.pool_config.max_connections = 5;
        config.pool_config.min_idle_connections = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_distance_metric() {
        let mut config = QdrantClientConfig::default();
        config.default_collection_config.distance_metric = "Invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_duration_conversions() {
        let config = QdrantClientConfig::default();
        assert_eq!(config.connection_timeout(), Duration::from_secs(30));
        assert_eq!(config.request_timeout(), Duration::from_secs(60));
        assert_eq!(config.retry_delay(), Duration::from_millis(1000));
        assert_eq!(config.max_retry_delay(), Duration::from_millis(30000));
    }
}