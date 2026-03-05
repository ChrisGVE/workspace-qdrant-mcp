//! Storage configuration types
//!
//! Configuration for Qdrant client transport, HTTP/2 settings, and
//! multi-tenant collection parameters.

use serde::{Deserialize, Serialize};

/// Transport mode for Qdrant connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMode {
    /// gRPC transport (default, more efficient)
    Grpc,
    /// HTTP transport (fallback)
    Http,
}

impl Default for TransportMode {
    fn default() -> Self {
        Self::Grpc
    }
}

/// HTTP/2 configuration for gRPC transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Http2Config {
    /// Maximum frame size (bytes)
    pub max_frame_size: Option<u32>,
    /// Initial connection window size (bytes)
    pub initial_window_size: Option<u32>,
    /// Maximum header list size (bytes)
    pub max_header_list_size: Option<u32>,
    /// Enable HTTP/2 server push
    pub enable_push: bool,
    /// Enable TCP keepalive
    pub tcp_keepalive: bool,
    /// Keepalive interval in milliseconds
    pub keepalive_interval_ms: Option<u32>,
    /// Keepalive timeout in milliseconds
    pub keepalive_timeout_ms: Option<u32>,
    /// Enable HTTP/2 adaptive window sizing
    pub http2_adaptive_window: bool,
}

impl Default for Http2Config {
    fn default() -> Self {
        Self {
            max_frame_size: Some(8192), // Conservative default (vs 16384)
            initial_window_size: Some(32768),
            max_header_list_size: Some(8192),
            enable_push: false,
            tcp_keepalive: true,
            keepalive_interval_ms: Some(30000),
            keepalive_timeout_ms: Some(5000),
            http2_adaptive_window: false,
        }
    }
}

/// Storage client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Qdrant server URL
    pub url: String,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    /// Connection timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Base retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Transport mode (gRPC or HTTP)
    pub transport: TransportMode,
    /// Connection pool size
    pub pool_size: usize,
    /// Enable TLS (for production)
    pub tls: bool,
    /// Default vector size for dense vectors
    pub dense_vector_size: u64,
    /// Default sparse vector size
    pub sparse_vector_size: Option<u64>,
    /// HTTP/2 configuration for gRPC transport
    pub http2: Http2Config,
    /// Skip compatibility checks during connection
    pub check_compatibility: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            transport: TransportMode::default(),
            pool_size: 10,
            tls: false,
            dense_vector_size: 384, // Default for all-MiniLM-L6-v2
            sparse_vector_size: None,
            http2: Http2Config::default(),
            check_compatibility: true,
        }
    }
}

impl StorageConfig {
    /// Create a daemon-mode configuration with compatibility checking disabled
    /// to ensure complete console silence for MCP stdio protocol compliance.
    /// Uses gRPC transport on port 6334 (Qdrant's gRPC port).
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.check_compatibility = false; // Disable to suppress Qdrant client output
        config.transport = TransportMode::Grpc; // gRPC is required by qdrant-client
                                                // qdrant-client uses gRPC protocol - ensure we use port 6334
                                                // Use 127.0.0.1 explicitly to avoid IPv6 resolution issues
        config.url = "http://127.0.0.1:6334".to_string();
        config
    }
}

/// Multi-tenant collection configuration
#[derive(Debug, Clone)]
pub struct MultiTenantConfig {
    /// Dense vector size (default: 384 for all-MiniLM-L6-v2)
    pub vector_size: u64,
    /// HNSW m parameter (default: 16)
    pub hnsw_m: u64,
    /// HNSW ef_construct parameter (default: 100)
    pub hnsw_ef_construct: u64,
    /// Enable on_disk_payload for large collections
    pub on_disk_payload: bool,
}

impl Default for MultiTenantConfig {
    fn default() -> Self {
        Self {
            vector_size: 384, // all-MiniLM-L6-v2
            hnsw_m: 16,
            hnsw_ef_construct: 100,
            on_disk_payload: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_tenant_config_default() {
        let config = MultiTenantConfig::default();
        assert_eq!(config.vector_size, 384);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_ef_construct, 100);
        assert!(config.on_disk_payload);
    }

    #[test]
    fn test_multi_tenant_config_custom() {
        let config = MultiTenantConfig {
            vector_size: 768,
            hnsw_m: 32,
            hnsw_ef_construct: 200,
            on_disk_payload: false,
        };
        assert_eq!(config.vector_size, 768);
        assert_eq!(config.hnsw_m, 32);
        assert_eq!(config.hnsw_ef_construct, 200);
        assert!(!config.on_disk_payload);
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.url, "http://localhost:6333");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
        assert_eq!(config.dense_vector_size, 384);
        assert!(config.check_compatibility);
    }

    #[test]
    fn test_storage_config_daemon_mode() {
        let config = StorageConfig::daemon_mode();
        assert!(!config.check_compatibility);
    }

    #[test]
    fn test_transport_mode_default() {
        let mode = TransportMode::default();
        assert!(matches!(mode, TransportMode::Grpc));
    }

    #[test]
    fn test_http2_config_default() {
        let config = Http2Config::default();
        assert_eq!(config.max_frame_size, Some(8192));
        assert_eq!(config.initial_window_size, Some(32768));
        assert_eq!(config.max_header_list_size, Some(8192));
        assert!(!config.enable_push);
        assert!(config.tcp_keepalive);
        assert_eq!(config.keepalive_interval_ms, Some(30000));
        assert_eq!(config.keepalive_timeout_ms, Some(5000));
        assert!(!config.http2_adaptive_window);
    }
}
