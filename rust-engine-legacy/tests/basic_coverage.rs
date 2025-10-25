//! Basic coverage tests for Rust engine components
//! These tests provide basic coverage measurement without complex protobuf interactions

use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::error::*;

#[test]
fn test_daemon_config_creation() {
    let config = DaemonConfig::default();
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 50051);
    assert_eq!(config.database.max_connections, 10);
    assert_eq!(config.qdrant.url, "http://localhost:6333");
}

#[test]
fn test_server_config_creation() {
    let config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8080,
        max_connections: 500,
        connection_timeout_secs: 60,
        request_timeout_secs: 300,
        enable_tls: true,
        security: SecurityConfig::default(),
        transport: TransportConfig::default(),
        message: MessageConfig::default(),
        compression: CompressionConfig::default(),
        streaming: StreamingConfig::default(),
    };

    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 8080);
    assert!(config.enable_tls);
}

#[test]
fn test_database_config_creation() {
    let config = DatabaseConfig {
        sqlite_path: "/tmp/test.db".to_string(),
        max_connections: 20,
        connection_timeout_secs: 45,
        enable_wal: false,
    };

    assert_eq!(config.sqlite_path, "/tmp/test.db");
    assert_eq!(config.max_connections, 20);
    assert!(!config.enable_wal);
}

#[test]
fn test_qdrant_config_creation() {
    let collection_config = CollectionConfig {
        vector_size: 768,
        distance_metric: "Euclidean".to_string(),
        enable_indexing: false,
        replication_factor: 2,
        shard_number: 3,
    };

    let config = QdrantConfig {
        url: "http://remote:6333".to_string(),
        api_key: Some("test-key".to_string()),
        max_retries: 5,
        default_collection: collection_config,
    };

    assert_eq!(config.url, "http://remote:6333");
    assert_eq!(config.api_key, Some("test-key".to_string()));
    assert_eq!(config.max_retries, 5);
    assert_eq!(config.default_collection.vector_size, 768);
}

#[test]
fn test_processing_config_creation() {
    let config = ProcessingConfig {
        max_concurrent_tasks: 8,
        default_chunk_size: 2000,
        default_chunk_overlap: 400,
        max_file_size_bytes: 200_000_000,
        supported_extensions: vec!["rs".to_string(), "py".to_string()],
        enable_lsp: false,
        lsp_timeout_secs: 20,
    };

    assert_eq!(config.max_concurrent_tasks, 8);
    assert_eq!(config.default_chunk_size, 2000);
    assert!(!config.enable_lsp);
    assert!(config.supported_extensions.contains(&"rs".to_string()));
}

#[test]
fn test_file_watcher_config_creation() {
    let config = FileWatcherConfig {
        enabled: true,
        debounce_ms: 1000,
        max_watched_dirs: 50,
        ignore_patterns: vec!["*.log".to_string(), "target/**".to_string()],
        recursive: false,
    };

    assert!(config.enabled);
    assert_eq!(config.debounce_ms, 1000);
    assert!(!config.recursive);
    assert!(config.ignore_patterns.contains(&"*.log".to_string()));
}

#[test]
fn test_metrics_config_creation() {
    let config = MetricsConfig {
        enabled: false,
        collection_interval_secs: 120,
    };

    assert!(!config.enabled);
    assert_eq!(config.collection_interval_secs, 120);
}

#[test]
fn test_logging_config_creation() {
    let config = LoggingConfig {
        enabled: true,
        level: "debug".to_string(),
        file_path: Some("/custom/log.log".to_string()),
        max_file_size: SizeUnit(200 * 1024 * 1024), // 200 MB
        max_files: 10,
        enable_json: true,
        enable_structured: true,
        enable_console: true,
    };

    assert_eq!(config.level, "debug");
    assert_eq!(config.file_path, Some("/custom/log.log".to_string()));
    assert!(config.enable_json);
    assert_eq!(config.max_file_size.0, 200 * 1024 * 1024);
}

#[test]
fn test_config_validation_valid() {
    let config = DaemonConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_invalid_port() {
    let mut config = DaemonConfig::default();
    config.server.port = 0;

    let result = config.validate();
    assert!(result.is_err());

    match result {
        Err(DaemonError::Config(_)) => {
            // Expected error type
        },
        _ => panic!("Expected Config error for invalid port"),
    }
}

#[test]
fn test_config_validation_empty_qdrant_url() {
    let mut config = DaemonConfig::default();
    config.qdrant.url = String::new();

    let result = config.validate();
    assert!(result.is_err());

    match result {
        Err(DaemonError::Config(_)) => {
            // Expected error type
        },
        _ => panic!("Expected Config error for empty Qdrant URL"),
    }
}

#[test]
fn test_config_validation_empty_database_path() {
    let mut config = DaemonConfig::default();
    config.database.sqlite_path = String::new();

    let result = config.validate();
    assert!(result.is_err());

    match result {
        Err(DaemonError::Config(_)) => {
            // Expected error type
        },
        _ => panic!("Expected Config error for empty database path"),
    }
}

#[test]
fn test_config_validation_zero_chunk_size() {
    let mut config = DaemonConfig::default();
    config.processing.default_chunk_size = 0;

    let result = config.validate();
    assert!(result.is_err());

    match result {
        Err(DaemonError::Config(_)) => {
            // Expected error type
        },
        _ => panic!("Expected Config error for zero chunk size"),
    }
}

#[test]
fn test_config_clone_debug() {
    let config = DaemonConfig::default();
    let cloned = config.clone();

    assert_eq!(config.server.host, cloned.server.host);
    assert_eq!(config.qdrant.url, cloned.qdrant.url);

    // Test debug format
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("DaemonConfig"));
    assert!(debug_str.contains("ServerConfig"));
}

#[test]
fn test_collection_config_standalone() {
    let collection_config = CollectionConfig {
        vector_size: 768,
        distance_metric: "Dot".to_string(),
        enable_indexing: false,
        replication_factor: 3,
        shard_number: 2,
    };

    let debug_str = format!("{:?}", collection_config);
    assert!(debug_str.contains("CollectionConfig"));
    assert!(debug_str.contains("768"));
    assert!(debug_str.contains("Dot"));

    let cloned = collection_config.clone();
    assert_eq!(collection_config.vector_size, cloned.vector_size);
    assert_eq!(collection_config.distance_metric, cloned.distance_metric);
}

#[test]
fn test_error_types() {
    // Test that we can create different error types
    let config_error = DaemonError::Config(
        config::ConfigError::Message("Test config error".to_string())
    );

    let io_error = DaemonError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Test IO error"
    ));

    // Test that they can be formatted
    let config_msg = format!("{}", config_error);
    assert!(config_msg.contains("Configuration error"));

    let io_msg = format!("{}", io_error);
    assert!(io_msg.contains("IO error"));
}

#[test]
fn test_config_structs_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<DaemonConfig>();
    assert_send_sync::<ServerConfig>();
    assert_send_sync::<DatabaseConfig>();
    assert_send_sync::<QdrantConfig>();
    assert_send_sync::<CollectionConfig>();
    assert_send_sync::<ProcessingConfig>();
    assert_send_sync::<FileWatcherConfig>();
    assert_send_sync::<MetricsConfig>();
    assert_send_sync::<LoggingConfig>();
}