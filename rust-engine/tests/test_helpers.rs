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
    use std::collections::HashMap;
    SecurityConfig {
        tls: TlsConfig {
            enabled: false,
            cert_file: None,
            key_file: None,
            ca_cert_file: None,
            client_cert_verification: ClientCertVerification::None,
        },
        auth: AuthConfig {
            jwt: JwtConfig {
                secret_or_key_file: "test_secret".to_string(),
                expiration_secs: 3600,
                algorithm: "HS256".to_string(),
                issuer: "test-issuer".to_string(),
                audience: "test-audience".to_string(),
            },
            api_key: ApiKeyConfig {
                enabled: false,
                key_permissions: HashMap::new(),
                valid_keys: vec![],
                header_name: "X-API-Key".to_string(),
            },
            enable_service_auth: false,
            authorization: AuthorizationConfig {
                enabled: false,
                service_permissions: ServicePermissions {
                    document_processor: vec![],
                    search_service: vec![],
                    memory_service: vec![],
                    system_service: vec![],
                },
                default_permissions: vec![],
            },
        },
        audit: SecurityAuditConfig {
            enabled: false,
            log_auth_events: false,
            log_auth_failures: false,
            log_rate_limit_events: false,
            log_suspicious_patterns: false,
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
        max_connections: 10,
        sqlite_path: "test.db".to_string(),
        connection_timeout_secs: 30,
        enable_wal: true,
    }
}
