//! Storage configuration types
//!
//! Configuration for Qdrant client transport, HTTP/2 settings, and
//! multi-tenant collection parameters.

use serde::{Deserialize, Serialize};

/// Transport mode for Qdrant connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMode {
    /// gRPC transport (default, more efficient)
    #[serde(alias = "grpc")]
    Grpc,
    /// HTTP transport (fallback)
    #[serde(alias = "http")]
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
#[derive(Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Qdrant server URL
    pub url: String,
    /// API key for authentication (optional).
    ///
    /// Never serialized: secrets must come from the environment, not be written
    /// back to a config file by `save_config` (WI-g1). Still deserialized so an
    /// operator-provided file value loads into memory.
    #[serde(skip_serializing)]
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

/// Manual `Debug` (WI-g2): `api_key` is rendered as `Some("[REDACTED]")`/`None`
/// so the secret never appears in `{:?}` / `{:#?}` output, tracing spans, or any
/// log line that debug-prints the config. All other fields print verbatim.
impl std::fmt::Debug for StorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("timeout_ms", &self.timeout_ms)
            .field("max_retries", &self.max_retries)
            .field("retry_delay_ms", &self.retry_delay_ms)
            .field("transport", &self.transport)
            .field("pool_size", &self.pool_size)
            .field("tls", &self.tls)
            .field("dense_vector_size", &self.dense_vector_size)
            .field("sparse_vector_size", &self.sparse_vector_size)
            .field("http2", &self.http2)
            .field("check_compatibility", &self.check_compatibility)
            .finish()
    }
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
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.check_compatibility = false; // Disable to suppress Qdrant client output
        config.transport = TransportMode::Grpc; // gRPC is required by qdrant-client
                                                // qdrant-client uses gRPC protocol - ensure we use port 6334
                                                // Use 127.0.0.1 explicitly to avoid IPv6 resolution issues
        config.url =
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());
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
    fn storage_api_key_never_serialized() {
        // WI-g1 / AC-g1.1: a configured api_key must not be written back.
        let config = StorageConfig {
            api_key: Some("sk-storage-secret".to_string()),
            ..StorageConfig::default()
        };
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(
            !json.contains("sk-storage-secret"),
            "api_key leaked into serialized StorageConfig: {json}"
        );
        // AC-g1.2: deserialization still resolves an operator-provided value.
        // Build a valid object from a key-less serialization, then inject the
        // key (avoids hand-listing every field of StorageConfig).
        let mut value = serde_json::to_value(StorageConfig::default()).expect("config to value");
        value["api_key"] = serde_json::Value::String("sk-loaded".to_string());
        let loaded: StorageConfig = serde_json::from_value(value).expect("load");
        assert_eq!(loaded.api_key.as_deref(), Some("sk-loaded"));
    }

    #[test]
    fn storage_api_key_redacted_in_debug() {
        // WI-g2 / AC-g2.1: a configured api_key must never appear in Debug output
        // (plain or alternate), so a loader that debug-prints the config cannot leak it.
        let config = StorageConfig {
            api_key: Some("sk-debug-secret".to_string()),
            ..StorageConfig::default()
        };
        let plain = format!("{config:?}");
        let alt = format!("{config:#?}");
        assert!(
            !plain.contains("sk-debug-secret"),
            "api_key leaked into {{:?}}: {plain}"
        );
        assert!(
            !alt.contains("sk-debug-secret"),
            "api_key leaked into {{:#?}}: {alt}"
        );
        assert!(plain.contains("[REDACTED]"), "expected redaction marker");
        // None must render as `None`, not `Some(...)`.
        let none_dbg = format!("{:?}", StorageConfig::default());
        assert!(none_dbg.contains("api_key: None"), "got: {none_dbg}");
    }

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
