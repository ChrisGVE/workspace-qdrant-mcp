//! Test helper utilities for Rust tests
//!
//! Provides simplified configuration builders and utilities for tests

use workspace_qdrant_daemon::config::*;

/// Create a minimal ServerConfig for testing
pub fn test_server_config() -> ServerConfig {
    ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8000,
        max_connections: 100,
        request_timeout_secs: 60,
        connection_timeout_secs: 30,
        streaming: StreamingConfig::default(),
        message: MessageConfig::default(),
        enable_tls: false,
        security: test_security_config(),
        transport: TransportConfig::default(),
        compression: CompressionConfig::default(),
    }
}

/// Create a minimal SecurityConfig for testing
pub fn test_security_config() -> SecurityConfig {
    SecurityConfig {
        tls: TlsConfig {
            enabled: false,
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
        },
        jwt: JwtConfig {
            enabled: false,
        },
        api_key: ApiKeyConfig {
            enabled: false,
        },
        authorization: AuthorizationConfig {
            enabled: false,
        },
    }
}

/// Create a minimal QdrantConfig for testing
pub fn test_qdrant_config() -> QdrantConfig {
    QdrantConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        default_collection: test_collection_config(),
        max_retries: 3,
    }
}

/// Create a minimal CollectionConfig for testing
pub fn test_collection_config() -> CollectionConfig {
    CollectionConfig {
        vector_size: 384,
        shard_number: 1,
        replication_factor: 1,
        distance_metric: "Cosine".to_string(),
        enable_indexing: true,
    }
}

/// Create a minimal ProcessingConfig for testing
pub fn test_processing_config() -> ProcessingConfig {
    ProcessingConfig {
        max_concurrent_tasks: 4,
        supported_extensions: vec![
            "rs".to_string(),
            "py".to_string(),
            "js".to_string(),
            "ts".to_string(),
        ],
        default_chunk_size: 1000,
        default_chunk_overlap: 200,
        max_file_size_bytes: 10 * 1024 * 1024, // 10 MB
        enable_lsp: true,
        lsp_timeout_secs: 10,
    }
}

/// Create a minimal FileWatcherConfig for testing
pub fn test_file_watcher_config() -> FileWatcherConfig {
    FileWatcherConfig {
        enabled: true,
        ignore_patterns: vec![
            ".git".to_string(),
            "node_modules".to_string(),
            "target".to_string(),
        ],
        recursive: true,
        max_watched_dirs: 100,
        debounce_ms: 100,
    }
}

/// Create a minimal LoggingConfig for testing
pub fn test_logging_config() -> LoggingConfig {
    LoggingConfig {
        enabled: true,
        level: "info".to_string(),
        file_path: None,
        max_file_size: SizeUnit(10 * 1024 * 1024), // 10 MB
        max_files: 10,
        enable_json: false,
        enable_structured: true,
        enable_console: true,
    }
}

/// Create a minimal MetricsConfig for testing
pub fn test_metrics_config() -> MetricsConfig {
    MetricsConfig {
        enabled: true,
        collection_interval_secs: 60,
    }
}

/// Create a minimal DatabaseConfig for testing
pub fn test_database_config() -> DatabaseConfig {
    DatabaseConfig {
        url: "sqlite://test.db".to_string(),
        max_connections: 10,
    }
}
