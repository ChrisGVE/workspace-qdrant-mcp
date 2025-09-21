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