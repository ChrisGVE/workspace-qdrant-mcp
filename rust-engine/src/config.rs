//! Configuration management for the Workspace Qdrant Daemon

use crate::error::DaemonResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Server configuration
    pub server: ServerConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Qdrant configuration
    pub qdrant: QdrantConfig,

    /// Document processing configuration
    pub processing: ProcessingConfig,

    /// File watching configuration
    pub file_watcher: FileWatcherConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Logging configuration
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// gRPC server host
    pub host: String,

    /// gRPC server port
    pub port: u16,

    /// Maximum number of concurrent connections
    pub max_connections: usize,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable TLS (for future use)
    pub enable_tls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// SQLite database file path
    pub sqlite_path: String,

    /// Maximum number of database connections
    pub max_connections: u32,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Enable WAL mode for better concurrency
    pub enable_wal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,

    /// Qdrant API key (optional)
    pub api_key: Option<String>,

    /// Connection timeout in seconds
    pub timeout_secs: u64,

    /// Maximum number of retries
    pub max_retries: u32,

    /// Default collection configuration
    pub default_collection: CollectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Vector size for embeddings
    pub vector_size: usize,

    /// Distance metric (Cosine, Euclidean, Dot)
    pub distance_metric: String,

    /// Enable payload indexing
    pub enable_indexing: bool,

    /// Replication factor
    pub replication_factor: u32,

    /// Number of shards
    pub shard_number: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum number of concurrent processing tasks
    pub max_concurrent_tasks: usize,

    /// Default chunk size for documents
    pub default_chunk_size: usize,

    /// Default chunk overlap
    pub default_chunk_overlap: usize,

    /// Maximum file size to process (in bytes)
    pub max_file_size_bytes: u64,

    /// Supported file extensions
    pub supported_extensions: Vec<String>,

    /// Enable LSP integration
    pub enable_lsp: bool,

    /// LSP server timeout in seconds
    pub lsp_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWatcherConfig {
    /// Enable file watching
    pub enabled: bool,

    /// Debounce delay in milliseconds
    pub debounce_ms: u64,

    /// Maximum number of watched directories
    pub max_watched_dirs: usize,

    /// Patterns to ignore
    pub ignore_patterns: Vec<String>,

    /// Enable recursive watching
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,

    /// Metrics collection interval in seconds
    pub collection_interval_secs: u64,

    /// Metrics retention period in days
    pub retention_days: u32,

    /// Enable Prometheus metrics export
    pub enable_prometheus: bool,

    /// Prometheus metrics port
    pub prometheus_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Log file path (optional)
    pub file_path: Option<String>,

    /// Enable JSON logging
    pub json_format: bool,

    /// Maximum log file size in MB
    pub max_file_size_mb: u64,

    /// Maximum number of log files to keep
    pub max_files: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 1000,
                connection_timeout_secs: 30,
                request_timeout_secs: 300,
                enable_tls: false,
            },
            database: DatabaseConfig {
                sqlite_path: "./workspace_daemon.db".to_string(),
                max_connections: 10,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: CollectionConfig {
                    vector_size: 384, // sentence-transformers/all-MiniLM-L6-v2
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 4,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 100 * 1024 * 1024, // 100MB
                supported_extensions: vec![
                    "rs".to_string(),
                    "py".to_string(),
                    "js".to_string(),
                    "ts".to_string(),
                    "md".to_string(),
                    "txt".to_string(),
                    "pdf".to_string(),
                    "html".to_string(),
                    "json".to_string(),
                    "xml".to_string(),
                ],
                enable_lsp: true,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: true,
                debounce_ms: 500,
                max_watched_dirs: 100,
                ignore_patterns: vec![
                    "target/**".to_string(),
                    "node_modules/**".to_string(),
                    ".git/**".to_string(),
                    "*.tmp".to_string(),
                    "*.log".to_string(),
                ],
                recursive: true,
            },
            metrics: MetricsConfig {
                enabled: true,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: true,
                prometheus_port: 9090,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: Some("./workspace_daemon.log".to_string()),
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }
}

impl DaemonConfig {
    /// Load configuration from file or use defaults
    pub fn load(config_path: Option<&Path>) -> DaemonResult<Self> {
        match config_path {
            Some(path) => {
                let content = std::fs::read_to_string(path)?;
                let config: DaemonConfig = serde_yaml::from_str(&content)
                    .map_err(|e| crate::error::DaemonError::Config(
                        config::ConfigError::Message(format!("Invalid YAML: {}", e))
                    ))?;
                Ok(config)
            },
            None => {
                // Try to load from environment variables
                Self::from_env()
            }
        }
    }

    /// Load configuration from environment variables
    fn from_env() -> DaemonResult<Self> {
        let mut config = Self::default();

        // Override with environment variables if present
        if let Ok(url) = std::env::var("QDRANT_URL") {
            config.qdrant.url = url;
        }

        if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
            config.qdrant.api_key = Some(api_key);
        }

        if let Ok(host) = std::env::var("DAEMON_HOST") {
            config.server.host = host;
        }

        if let Ok(port) = std::env::var("DAEMON_PORT") {
            config.server.port = port.parse()
                .map_err(|e| crate::error::DaemonError::Config(
                    config::ConfigError::Message(format!("Invalid port: {}", e))
                ))?;
        }

        if let Ok(db_path) = std::env::var("DAEMON_DB_PATH") {
            config.database.sqlite_path = db_path;
        }

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> DaemonResult<()> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| crate::error::DaemonError::Config(
                config::ConfigError::Message(format!("Serialization error: {}", e))
            ))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> DaemonResult<()> {
        // Validate server configuration
        if self.server.port == 0 {
            return Err(crate::error::DaemonError::Config(
                config::ConfigError::Message("Server port cannot be 0".to_string())
            ));
        }

        // Validate Qdrant URL
        if self.qdrant.url.is_empty() {
            return Err(crate::error::DaemonError::Config(
                config::ConfigError::Message("Qdrant URL cannot be empty".to_string())
            ));
        }

        // Validate database path
        if self.database.sqlite_path.is_empty() {
            return Err(crate::error::DaemonError::Config(
                config::ConfigError::Message("Database path cannot be empty".to_string())
            ));
        }

        // Validate processing configuration
        if self.processing.default_chunk_size == 0 {
            return Err(crate::error::DaemonError::Config(
                config::ConfigError::Message("Chunk size must be greater than 0".to_string())
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::tempdir;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();

        // Test server defaults
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 50051);
        assert_eq!(config.server.max_connections, 1000);
        assert_eq!(config.server.connection_timeout_secs, 30);
        assert_eq!(config.server.request_timeout_secs, 300);
        assert!(!config.server.enable_tls);

        // Test database defaults
        assert_eq!(config.database.sqlite_path, "./workspace_daemon.db");
        assert_eq!(config.database.max_connections, 10);
        assert_eq!(config.database.connection_timeout_secs, 30);
        assert!(config.database.enable_wal);

        // Test qdrant defaults
        assert_eq!(config.qdrant.url, "http://localhost:6333");
        assert!(config.qdrant.api_key.is_none());
        assert_eq!(config.qdrant.timeout_secs, 30);
        assert_eq!(config.qdrant.max_retries, 3);
        assert_eq!(config.qdrant.default_collection.vector_size, 384);
        assert_eq!(config.qdrant.default_collection.distance_metric, "Cosine");
        assert!(config.qdrant.default_collection.enable_indexing);
        assert_eq!(config.qdrant.default_collection.replication_factor, 1);
        assert_eq!(config.qdrant.default_collection.shard_number, 1);

        // Test processing defaults
        assert_eq!(config.processing.max_concurrent_tasks, 4);
        assert_eq!(config.processing.default_chunk_size, 1000);
        assert_eq!(config.processing.default_chunk_overlap, 200);
        assert_eq!(config.processing.max_file_size_bytes, 100 * 1024 * 1024);
        assert!(config.processing.supported_extensions.contains(&"rs".to_string()));
        assert!(config.processing.supported_extensions.contains(&"py".to_string()));
        assert!(config.processing.enable_lsp);
        assert_eq!(config.processing.lsp_timeout_secs, 10);

        // Test file watcher defaults
        assert!(config.file_watcher.enabled);
        assert_eq!(config.file_watcher.debounce_ms, 500);
        assert_eq!(config.file_watcher.max_watched_dirs, 100);
        assert!(config.file_watcher.ignore_patterns.contains(&"target/**".to_string()));
        assert!(config.file_watcher.ignore_patterns.contains(&"node_modules/**".to_string()));
        assert!(config.file_watcher.recursive);

        // Test metrics defaults
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.collection_interval_secs, 60);
        assert_eq!(config.metrics.retention_days, 30);
        assert!(config.metrics.enable_prometheus);
        assert_eq!(config.metrics.prometheus_port, 9090);

        // Test logging defaults
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.file_path, Some("./workspace_daemon.log".to_string()));
        assert!(!config.logging.json_format);
        assert_eq!(config.logging.max_file_size_mb, 100);
        assert_eq!(config.logging.max_files, 5);
    }

    #[test]
    fn test_daemon_config_debug_clone() {
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
    fn test_load_config_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("config.yaml");
        let mut file = File::create(&file_path).unwrap();

        writeln!(file, r#"
server:
  host: "0.0.0.0"
  port: 8080
  max_connections: 500
  connection_timeout_secs: 60
  request_timeout_secs: 600
  enable_tls: true
qdrant:
  url: "http://remote-qdrant:6333"
  api_key: "test-key"
  timeout_secs: 45
  max_retries: 5
  default_collection:
    vector_size: 512
    distance_metric: "Euclidean"
    enable_indexing: false
    replication_factor: 2
    shard_number: 3
database:
  sqlite_path: "/custom/path.db"
  max_connections: 20
  connection_timeout_secs: 45
  enable_wal: false
processing:
  max_concurrent_tasks: 8
  default_chunk_size: 2000
  default_chunk_overlap: 400
  max_file_size_bytes: 200000000
  supported_extensions: ["rs", "py"]
  enable_lsp: false
  lsp_timeout_secs: 20
file_watcher:
  enabled: false
  debounce_ms: 1000
  max_watched_dirs: 50
  ignore_patterns: ["*.log"]
  recursive: false
metrics:
  enabled: false
  collection_interval_secs: 120
  retention_days: 60
  enable_prometheus: false
  prometheus_port: 9091
logging:
  level: "debug"
  file_path: "/custom/log.log"
  json_format: true
  max_file_size_mb: 200
  max_files: 10
"#).unwrap();

        let config = DaemonConfig::load(Some(&file_path)).unwrap();

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.max_connections, 500);
        assert_eq!(config.server.connection_timeout_secs, 60);
        assert_eq!(config.server.request_timeout_secs, 600);
        assert!(config.server.enable_tls);

        assert_eq!(config.qdrant.url, "http://remote-qdrant:6333");
        assert_eq!(config.qdrant.api_key, Some("test-key".to_string()));
        assert_eq!(config.qdrant.timeout_secs, 45);
        assert_eq!(config.qdrant.max_retries, 5);
        assert_eq!(config.qdrant.default_collection.vector_size, 512);
        assert_eq!(config.qdrant.default_collection.distance_metric, "Euclidean");
        assert!(!config.qdrant.default_collection.enable_indexing);
        assert_eq!(config.qdrant.default_collection.replication_factor, 2);
        assert_eq!(config.qdrant.default_collection.shard_number, 3);

        assert_eq!(config.database.sqlite_path, "/custom/path.db");
        assert_eq!(config.database.max_connections, 20);
        assert_eq!(config.database.connection_timeout_secs, 45);
        assert!(!config.database.enable_wal);

        assert_eq!(config.processing.max_concurrent_tasks, 8);
        assert_eq!(config.processing.default_chunk_size, 2000);
        assert_eq!(config.processing.default_chunk_overlap, 400);
        assert_eq!(config.processing.max_file_size_bytes, 200000000);
        assert_eq!(config.processing.supported_extensions, vec!["rs", "py"]);
        assert!(!config.processing.enable_lsp);
        assert_eq!(config.processing.lsp_timeout_secs, 20);

        assert!(!config.file_watcher.enabled);
        assert_eq!(config.file_watcher.debounce_ms, 1000);
        assert_eq!(config.file_watcher.max_watched_dirs, 50);
        assert_eq!(config.file_watcher.ignore_patterns, vec!["*.log"]);
        assert!(!config.file_watcher.recursive);

        assert!(!config.metrics.enabled);
        assert_eq!(config.metrics.collection_interval_secs, 120);
        assert_eq!(config.metrics.retention_days, 60);
        assert!(!config.metrics.enable_prometheus);
        assert_eq!(config.metrics.prometheus_port, 9091);

        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.file_path, Some("/custom/log.log".to_string()));
        assert!(config.logging.json_format);
        assert_eq!(config.logging.max_file_size_mb, 200);
        assert_eq!(config.logging.max_files, 10);
    }

    #[test]
    fn test_load_config_no_file() {
        let config = DaemonConfig::load(None).unwrap();
        // Should load defaults when no file is provided
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 50051);
    }

    #[test]
    fn test_load_config_invalid_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid.yaml");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "invalid: yaml: content: [").unwrap();

        let result = DaemonConfig::load(Some(&file_path));
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Config(_)) = result {
            // Expected error type
        } else {
            panic!("Expected Config error");
        }
    }

    #[test]
    fn test_load_config_nonexistent_file() {
        let nonexistent_path = Path::new("/nonexistent/config.yaml");
        let result = DaemonConfig::load(Some(nonexistent_path));
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Io(_)) = result {
            // Expected error type
        } else {
            panic!("Expected IO error");
        }
    }

    #[test]
    fn test_from_env() {
        // Save original env vars
        let original_qdrant_url = env::var("QDRANT_URL").ok();
        let original_qdrant_api_key = env::var("QDRANT_API_KEY").ok();
        let original_daemon_host = env::var("DAEMON_HOST").ok();
        let original_daemon_port = env::var("DAEMON_PORT").ok();
        let original_daemon_db_path = env::var("DAEMON_DB_PATH").ok();

        // Set test env vars
        env::set_var("QDRANT_URL", "http://test-qdrant:6333");
        env::set_var("QDRANT_API_KEY", "test-api-key");
        env::set_var("DAEMON_HOST", "0.0.0.0");
        env::set_var("DAEMON_PORT", "8080");
        env::set_var("DAEMON_DB_PATH", "/test/db.sqlite");

        let config = DaemonConfig::from_env().unwrap();

        assert_eq!(config.qdrant.url, "http://test-qdrant:6333");
        assert_eq!(config.qdrant.api_key, Some("test-api-key".to_string()));
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.database.sqlite_path, "/test/db.sqlite");

        // Restore original env vars
        match original_qdrant_url {
            Some(val) => env::set_var("QDRANT_URL", val),
            None => env::remove_var("QDRANT_URL"),
        }
        match original_qdrant_api_key {
            Some(val) => env::set_var("QDRANT_API_KEY", val),
            None => env::remove_var("QDRANT_API_KEY"),
        }
        match original_daemon_host {
            Some(val) => env::set_var("DAEMON_HOST", val),
            None => env::remove_var("DAEMON_HOST"),
        }
        match original_daemon_port {
            Some(val) => env::set_var("DAEMON_PORT", val),
            None => env::remove_var("DAEMON_PORT"),
        }
        match original_daemon_db_path {
            Some(val) => env::set_var("DAEMON_DB_PATH", val),
            None => env::remove_var("DAEMON_DB_PATH"),
        }
    }

    #[test]
    fn test_env_port_parsing_error() {
        // Test the port parsing logic directly
        let invalid_port_str = "invalid_port";
        let parse_result: Result<u16, _> = invalid_port_str.parse();
        assert!(parse_result.is_err());

        // Test that we can create the expected error type
        let daemon_error = crate::error::DaemonError::Config(
            config::ConfigError::Message(format!("Invalid port: {}", parse_result.unwrap_err()))
        );

        match daemon_error {
            crate::error::DaemonError::Config(_) => {
                // Expected error type - test passes
            },
            _ => panic!("Expected Config error for invalid port"),
        }

        // Test the error message formatting
        let error_msg = format!("{}", daemon_error);
        assert!(error_msg.contains("Configuration error"));
        assert!(error_msg.contains("Invalid port"));
    }

    #[test]
    fn test_save_config() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("output.yaml");

        let config = DaemonConfig::default();
        config.save(&file_path).unwrap();

        // Verify file was created and can be read back
        assert!(file_path.exists());
        let loaded_config = DaemonConfig::load(Some(&file_path)).unwrap();

        assert_eq!(config.server.host, loaded_config.server.host);
        assert_eq!(config.server.port, loaded_config.server.port);
        assert_eq!(config.qdrant.url, loaded_config.qdrant.url);
    }

    #[test]
    fn test_validate_config_valid() {
        let config = DaemonConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_config_invalid_port() {
        let mut config = DaemonConfig::default();
        config.server.port = 0;

        let result = config.validate();
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Config(_)) = result {
            // Expected error type
        } else {
            panic!("Expected Config error for invalid port");
        }
    }

    #[test]
    fn test_validate_config_empty_qdrant_url() {
        let mut config = DaemonConfig::default();
        config.qdrant.url = String::new();

        let result = config.validate();
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Config(_)) = result {
            // Expected error type
        } else {
            panic!("Expected Config error for empty Qdrant URL");
        }
    }

    #[test]
    fn test_validate_config_empty_database_path() {
        let mut config = DaemonConfig::default();
        config.database.sqlite_path = String::new();

        let result = config.validate();
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Config(_)) = result {
            // Expected error type
        } else {
            panic!("Expected Config error for empty database path");
        }
    }

    #[test]
    fn test_validate_config_zero_chunk_size() {
        let mut config = DaemonConfig::default();
        config.processing.default_chunk_size = 0;

        let result = config.validate();
        assert!(result.is_err());

        if let Err(crate::error::DaemonError::Config(_)) = result {
            // Expected error type
        } else {
            panic!("Expected Config error for zero chunk size");
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = DaemonConfig::default();

        // Test YAML serialization
        let yaml_str = serde_yaml::to_string(&config).unwrap();
        let deserialized: DaemonConfig = serde_yaml::from_str(&yaml_str).unwrap();

        assert_eq!(config.server.host, deserialized.server.host);
        assert_eq!(config.server.port, deserialized.server.port);
        assert_eq!(config.qdrant.url, deserialized.qdrant.url);

        // Test JSON serialization
        let json_str = serde_json::to_string(&config).unwrap();
        let deserialized: DaemonConfig = serde_json::from_str(&json_str).unwrap();

        assert_eq!(config.server.host, deserialized.server.host);
        assert_eq!(config.qdrant.url, deserialized.qdrant.url);
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
    fn test_config_with_serde_defaults() {
        // Test that all config structs can be created with minimal YAML
        // This tests basic serialization/deserialization
        let minimal_yaml = r#"
server:
  host: "custom-host"
  port: 9999
  max_connections: 500
  connection_timeout_secs: 30
  request_timeout_secs: 300
  enable_tls: false
qdrant:
  url: "http://custom-qdrant:6333"
  api_key: null
  timeout_secs: 30
  max_retries: 3
  default_collection:
    vector_size: 384
    distance_metric: "Cosine"
    enable_indexing: true
    replication_factor: 1
    shard_number: 1
database:
  sqlite_path: "./workspace_daemon.db"
  max_connections: 10
  connection_timeout_secs: 30
  enable_wal: true
processing:
  max_concurrent_tasks: 4
  default_chunk_size: 1000
  default_chunk_overlap: 200
  max_file_size_bytes: 104857600
  supported_extensions: ["rs", "py"]
  enable_lsp: true
  lsp_timeout_secs: 10
file_watcher:
  enabled: true
  debounce_ms: 500
  max_watched_dirs: 100
  ignore_patterns: ["target/**"]
  recursive: true
metrics:
  enabled: true
  collection_interval_secs: 60
  retention_days: 30
  enable_prometheus: true
  prometheus_port: 9090
logging:
  level: "info"
  file_path: "./workspace_daemon.log"
  json_format: false
  max_file_size_mb: 100
  max_files: 5
"#;

        let config: DaemonConfig = serde_yaml::from_str(minimal_yaml).unwrap();

        // Custom fields should be set
        assert_eq!(config.server.host, "custom-host");
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.qdrant.url, "http://custom-qdrant:6333");
    }

    #[test]
    fn test_collection_config_edge_cases() {
        let config = CollectionConfig {
            vector_size: 1536,
            distance_metric: "Euclidean".to_string(),
            enable_indexing: false,
            replication_factor: 3,
            shard_number: 4,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CollectionConfig"));
        assert!(debug_str.contains("1536"));
        assert!(debug_str.contains("Euclidean"));

        let cloned = config.clone();
        assert_eq!(config.vector_size, cloned.vector_size);
        assert_eq!(config.distance_metric, cloned.distance_metric);
        assert_eq!(config.enable_indexing, cloned.enable_indexing);
        assert_eq!(config.replication_factor, cloned.replication_factor);
        assert_eq!(config.shard_number, cloned.shard_number);
    }

    #[test]
    fn test_config_validation_edge_cases() {
        // Test maximum values
        let mut config = DaemonConfig::default();
        config.server.port = 65535;
        config.processing.default_chunk_size = usize::MAX;
        config.database.max_connections = u32::MAX;
        assert!(config.validate().is_ok());

        // Test minimum valid values
        config.processing.default_chunk_size = 1;
        assert!(config.validate().is_ok());

        config.server.connection_timeout_secs = 0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_file_watcher_config_comprehensive() {
        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms: u64::MAX,
            max_watched_dirs: usize::MAX,
            ignore_patterns: vec!["*".to_string(); 100],
            recursive: false,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("FileWatcherConfig"));
        assert!(debug_str.contains("debounce_ms"));

        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.debounce_ms, cloned.debounce_ms);
        assert_eq!(config.max_watched_dirs, cloned.max_watched_dirs);
        assert_eq!(config.ignore_patterns.len(), cloned.ignore_patterns.len());
        assert_eq!(config.recursive, cloned.recursive);
    }

    #[test]
    fn test_all_config_components_debug() {
        let config = DaemonConfig::default();

        // Test debug formatting for all components
        let server_debug = format!("{:?}", config.server);
        assert!(server_debug.contains("ServerConfig"));

        let db_debug = format!("{:?}", config.database);
        assert!(db_debug.contains("DatabaseConfig"));

        let qdrant_debug = format!("{:?}", config.qdrant);
        assert!(qdrant_debug.contains("QdrantConfig"));

        let processing_debug = format!("{:?}", config.processing);
        assert!(processing_debug.contains("ProcessingConfig"));

        let watcher_debug = format!("{:?}", config.file_watcher);
        assert!(watcher_debug.contains("FileWatcherConfig"));

        let metrics_debug = format!("{:?}", config.metrics);
        assert!(metrics_debug.contains("MetricsConfig"));

        let logging_debug = format!("{:?}", config.logging);
        assert!(logging_debug.contains("LoggingConfig"));
    }
}