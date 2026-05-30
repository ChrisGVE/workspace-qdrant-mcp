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
///
/// `#[serde(default)]` lets a user config specify only the fields it cares
/// about (e.g. just `url`/`api_key`); omitted fields such as `timeout_ms` fall
/// back to [`StorageConfig::default()`] instead of failing the whole daemon
/// config parse (`qdrant:` is deserialized as part of `DaemonConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
    ///
    /// The Qdrant URL is resolved in order:
    ///   1. `QDRANT_URL` environment variable
    ///   2. Hard-coded default `http://127.0.0.1:6334`
    ///
    /// The API key is read from `QDRANT_API_KEY` when present so local
    /// Docker runs stay aligned with the qdrant service auth settings.
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.check_compatibility = false; // Disable to suppress Qdrant client output
        config.transport = TransportMode::Grpc; // gRPC is required by qdrant-client
                                                // qdrant-client uses gRPC protocol - ensure we use port 6334
                                                // Use 127.0.0.1 explicitly to avoid IPv6 resolution issues
        config.url =
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());
        config.api_key = std::env::var("QDRANT_API_KEY").ok().and_then(|key| {
            if key.trim().is_empty() {
                None
            } else {
                Some(key)
            }
        });
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

    // Env-var tests mutate process-global state; serialise with `#[serial]` so
    // parallel test threads cannot observe each other's QDRANT_URL mutations.
    #[test]
    #[serial_test::serial]
    fn daemon_mode_uses_qdrant_url_env_var() {
        let prev = std::env::var("QDRANT_URL").ok();
        std::env::set_var("QDRANT_URL", "http://qdrant:6333");
        let config = StorageConfig::daemon_mode();
        match prev {
            Some(v) => std::env::set_var("QDRANT_URL", v),
            None => std::env::remove_var("QDRANT_URL"),
        }
        assert_eq!(config.url, "http://qdrant:6333");
        assert!(!config.check_compatibility);
    }

    #[test]
    #[serial_test::serial]
    fn daemon_mode_uses_qdrant_api_key_env_var() {
        let prev = std::env::var("QDRANT_API_KEY").ok();
        std::env::set_var("QDRANT_API_KEY", "sekret");
        let config = StorageConfig::daemon_mode();
        match prev {
            Some(v) => std::env::set_var("QDRANT_API_KEY", v),
            None => std::env::remove_var("QDRANT_API_KEY"),
        }
        assert_eq!(config.api_key.as_deref(), Some("sekret"));
        assert!(!config.check_compatibility);
    }

    #[test]
    #[serial_test::serial]
    fn daemon_mode_ignores_empty_qdrant_api_key() {
        let prev = std::env::var("QDRANT_API_KEY").ok();
        std::env::set_var("QDRANT_API_KEY", "   ");
        let config = StorageConfig::daemon_mode();
        match prev {
            Some(v) => std::env::set_var("QDRANT_API_KEY", v),
            None => std::env::remove_var("QDRANT_API_KEY"),
        }
        assert!(config.api_key.is_none());
        assert!(!config.check_compatibility);
    }

    #[test]
    #[serial_test::serial]
    fn daemon_mode_defaults_when_no_qdrant_url() {
        let prev = std::env::var("QDRANT_URL").ok();
        std::env::remove_var("QDRANT_URL");
        let config = StorageConfig::daemon_mode();
        if let Some(v) = prev {
            std::env::set_var("QDRANT_URL", v);
        }
        assert_eq!(config.url, "http://127.0.0.1:6334");
        assert!(!config.check_compatibility);
    }
}
