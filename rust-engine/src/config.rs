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

    /// Auto ingestion configuration
    #[serde(default)]
    pub auto_ingestion: AutoIngestionConfig,

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

    /// Enable TLS
    pub enable_tls: bool,

    /// Security configuration
    pub security: SecurityConfig,

    /// Transport configuration
    pub transport: TransportConfig,

    /// Message configuration
    pub message: MessageConfig,

    /// Compression configuration
    pub compression: CompressionConfig,

    /// Streaming configuration
    pub streaming: StreamingConfig,
}

/// Message size and validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageConfig {
    /// Maximum message size for incoming requests (bytes)
    pub max_incoming_message_size: usize,

    /// Maximum message size for outgoing responses (bytes)
    pub max_outgoing_message_size: usize,

    /// Enable message size validation
    pub enable_size_validation: bool,

    /// Maximum frame size for HTTP/2
    pub max_frame_size: u32,

    /// Initial window size for HTTP/2
    pub initial_window_size: u32,

    /// Service-specific message size limits
    pub service_limits: ServiceMessageLimits,

    /// Size monitoring and alerting configuration
    pub monitoring: MessageMonitoringConfig,
}

/// Service-specific message size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMessageLimits {
    /// Document processor service limits
    pub document_processor: ServiceLimit,

    /// Search service limits
    pub search_service: ServiceLimit,

    /// Memory service limits
    pub memory_service: ServiceLimit,

    /// System service limits
    pub system_service: ServiceLimit,
}

/// Individual service message limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLimit {
    /// Maximum incoming message size (bytes)
    pub max_incoming: usize,

    /// Maximum outgoing message size (bytes)
    pub max_outgoing: usize,

    /// Enable size validation for this service
    pub enable_validation: bool,
}

/// Message monitoring and alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMonitoringConfig {
    /// Enable detailed message size monitoring
    pub enable_detailed_monitoring: bool,

    /// Alert threshold for oversized messages (percentage of limit)
    pub oversized_alert_threshold: f64,

    /// Enable real-time metrics collection
    pub enable_realtime_metrics: bool,

    /// Metrics collection interval (seconds)
    pub metrics_interval_secs: u64,
}

/// Compression configuration for gRPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable gzip compression
    pub enable_gzip: bool,

    /// Minimum message size to trigger compression (bytes)
    pub compression_threshold: usize,

    /// Compression level (1-9, where 9 is maximum compression)
    pub compression_level: u32,

    /// Enable compression for streaming responses
    pub enable_streaming_compression: bool,

    /// Monitor compression efficiency
    pub enable_compression_monitoring: bool,

    /// Adaptive compression configuration
    pub adaptive: AdaptiveCompressionConfig,

    /// Compression performance monitoring
    pub performance: CompressionPerformanceConfig,
}

/// Adaptive compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompressionConfig {
    /// Enable adaptive compression based on content type
    pub enable_adaptive: bool,

    /// Text content compression level (1-9)
    pub text_compression_level: u32,

    /// Binary content compression level (1-9)
    pub binary_compression_level: u32,

    /// JSON/structured data compression level (1-9)
    pub structured_compression_level: u32,

    /// Maximum time to spend on compression (milliseconds)
    pub max_compression_time_ms: u64,
}

/// Compression performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPerformanceConfig {
    /// Enable compression ratio tracking
    pub enable_ratio_tracking: bool,

    /// Alert threshold for poor compression ratio
    pub poor_ratio_threshold: f64,

    /// Enable compression time monitoring
    pub enable_time_monitoring: bool,

    /// Alert threshold for slow compression (milliseconds)
    pub slow_compression_threshold_ms: u64,

    /// Enable compression failure alerting
    pub enable_failure_alerting: bool,
}

/// Streaming configuration for large operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable server-side streaming
    pub enable_server_streaming: bool,

    /// Enable client-side streaming
    pub enable_client_streaming: bool,

    /// Maximum concurrent streams per connection
    pub max_concurrent_streams: u32,

    /// Stream buffer size (number of items)
    pub stream_buffer_size: usize,

    /// Stream timeout in seconds
    pub stream_timeout_secs: u64,

    /// Enable stream flow control
    pub enable_flow_control: bool,

    /// Progress tracking configuration
    pub progress: StreamProgressConfig,

    /// Stream health and recovery configuration
    pub health: StreamHealthConfig,

    /// Large operation streaming configuration
    pub large_operations: LargeOperationStreamConfig,
}

/// Stream progress tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgressConfig {
    /// Enable progress tracking for streams
    pub enable_progress_tracking: bool,

    /// Progress update interval (milliseconds)
    pub progress_update_interval_ms: u64,

    /// Enable progress callbacks
    pub enable_progress_callbacks: bool,

    /// Minimum operation size to enable progress tracking (bytes)
    pub progress_threshold: usize,
}

/// Stream health monitoring and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamHealthConfig {
    /// Enable stream health monitoring
    pub enable_health_monitoring: bool,

    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,

    /// Enable automatic stream recovery
    pub enable_auto_recovery: bool,

    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,

    /// Recovery backoff multiplier
    pub recovery_backoff_multiplier: f64,

    /// Initial recovery delay (milliseconds)
    pub initial_recovery_delay_ms: u64,
}

/// Large operation streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeOperationStreamConfig {
    /// Enable streaming for large document uploads
    pub enable_large_document_streaming: bool,

    /// Chunk size for large operations (bytes)
    pub large_operation_chunk_size: usize,

    /// Enable streaming for bulk operations
    pub enable_bulk_streaming: bool,

    /// Maximum memory usage for streaming operations (bytes)
    pub max_streaming_memory: usize,

    /// Enable bidirectional streaming optimization
    pub enable_bidirectional_optimization: bool,
}

/// Security configuration for gRPC server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// TLS configuration
    pub tls: TlsConfig,

    /// Authentication configuration
    pub auth: AuthConfig,

    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,

    /// Security audit configuration
    pub audit: SecurityAuditConfig,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Server certificate file path
    pub cert_file: Option<String>,

    /// Server private key file path
    pub key_file: Option<String>,

    /// CA certificate file path for client verification
    pub ca_cert_file: Option<String>,

    /// Enable mutual TLS (mTLS)
    pub enable_mtls: bool,

    /// Client certificate verification mode
    pub client_cert_verification: ClientCertVerification,

    /// TLS protocol versions to support
    pub supported_protocols: Vec<String>,

    /// Cipher suites to use
    pub cipher_suites: Vec<String>,
}

/// Client certificate verification modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCertVerification {
    /// No client certificate required
    None,
    /// Client certificate optional
    Optional,
    /// Client certificate required
    Required,
}

/// Authentication and authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable service-to-service authentication
    pub enable_service_auth: bool,

    /// JWT token configuration
    pub jwt: JwtConfig,

    /// API key configuration
    pub api_key: ApiKeyConfig,

    /// Service authorization rules
    pub authorization: AuthorizationConfig,
}

/// JWT token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// JWT signing secret or public key file path
    pub secret_or_key_file: String,

    /// Token issuer
    pub issuer: String,

    /// Token audience
    pub audience: String,

    /// Token expiration time in seconds
    pub expiration_secs: u64,

    /// Algorithm to use (HS256, RS256, etc.)
    pub algorithm: String,
}

/// API key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Enable API key authentication
    pub enabled: bool,

    /// API key header name
    pub header_name: String,

    /// Valid API keys (in production, load from secure storage)
    pub valid_keys: Vec<String>,

    /// API key permissions mapping
    pub key_permissions: std::collections::HashMap<String, Vec<String>>,
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    /// Enable authorization checks
    pub enabled: bool,

    /// Default permissions for authenticated users
    pub default_permissions: Vec<String>,

    /// Service-specific permissions
    pub service_permissions: ServicePermissions,
}

/// Service-specific permission configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePermissions {
    /// Document processor permissions
    pub document_processor: Vec<String>,

    /// Search service permissions
    pub search_service: Vec<String>,

    /// Memory service permissions
    pub memory_service: Vec<String>,

    /// System service permissions
    pub system_service: Vec<String>,
}

/// Enhanced rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,

    /// Requests per second per client
    pub requests_per_second: u32,

    /// Burst capacity
    pub burst_capacity: u32,

    /// Connection pool limits per service
    pub connection_pool_limits: ConnectionPoolLimits,

    /// Request queue depth limits
    pub queue_depth_limit: u32,

    /// Memory usage protection
    pub memory_protection: MemoryProtectionConfig,

    /// Resource exhaustion protection
    pub resource_protection: ResourceProtectionConfig,
}

/// Connection pool limits per service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolLimits {
    /// Document processor connection limit
    pub document_processor: u32,

    /// Search service connection limit
    pub search_service: u32,

    /// Memory service connection limit
    pub memory_service: u32,

    /// System service connection limit
    pub system_service: u32,
}

/// Memory usage protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProtectionConfig {
    /// Enable memory usage monitoring
    pub enabled: bool,

    /// Maximum memory usage per connection (bytes)
    pub max_memory_per_connection: u64,

    /// Total memory usage limit (bytes)
    pub total_memory_limit: u64,

    /// Memory cleanup interval (seconds)
    pub cleanup_interval_secs: u64,
}

/// Resource exhaustion protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProtectionConfig {
    /// Enable resource protection
    pub enabled: bool,

    /// CPU usage threshold for throttling (percentage)
    pub cpu_threshold_percent: f64,

    /// Disk space threshold for throttling (percentage)
    pub disk_threshold_percent: f64,

    /// Circuit breaker failure threshold
    pub circuit_breaker_failure_threshold: u32,

    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout_secs: u64,
}

/// Security audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditConfig {
    /// Enable security audit logging
    pub enabled: bool,

    /// Audit log file path
    pub log_file_path: String,

    /// Log authentication events
    pub log_auth_events: bool,

    /// Log authorization failures
    pub log_auth_failures: bool,

    /// Log rate limiting events
    pub log_rate_limit_events: bool,

    /// Log suspicious patterns
    pub log_suspicious_patterns: bool,

    /// Audit log rotation configuration
    pub rotation: AuditLogRotation,
}

/// Audit log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogRotation {
    /// Maximum file size before rotation (MB)
    pub max_file_size_mb: u64,

    /// Maximum number of files to keep
    pub max_files: u32,

    /// Compress rotated files
    pub compress: bool,
}

/// Transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// Unix domain socket configuration
    pub unix_socket: UnixSocketConfig,

    /// Local communication optimizations
    pub local_optimization: LocalOptimizationConfig,

    /// Transport selection strategy
    pub transport_strategy: TransportStrategy,
}

/// Unix domain socket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnixSocketConfig {
    /// Enable Unix domain socket support
    pub enabled: bool,

    /// Unix socket file path
    pub socket_path: String,

    /// Socket file permissions (octal)
    pub permissions: u32,

    /// Enable Unix socket for local development
    pub prefer_for_local: bool,
}

/// Local communication optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalOptimizationConfig {
    /// Enable local transport optimizations
    pub enabled: bool,

    /// Use larger buffers for local communication
    pub use_large_buffers: bool,

    /// Buffer size for local communication (bytes)
    pub local_buffer_size: usize,

    /// Enable memory-efficient serialization
    pub memory_efficient_serialization: bool,

    /// Reduced latency settings for local calls
    pub reduce_latency: LocalLatencyConfig,
}

/// Local latency reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLatencyConfig {
    /// Disable Nagle's algorithm for local connections
    pub disable_nagle: bool,

    /// Use custom local connection pooling
    pub custom_connection_pooling: bool,

    /// Local connection pool size
    pub connection_pool_size: u32,

    /// Local connection keep-alive interval (seconds)
    pub keepalive_interval_secs: u64,
}

/// Transport selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportStrategy {
    /// Automatically detect and use best transport
    Auto,
    /// Force TCP transport
    ForceTcp,
    /// Force Unix socket (local only)
    ForceUnixSocket,
    /// Use Unix socket with TCP fallback
    UnixSocketWithTcpFallback,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIngestionConfig {
    /// Enable automatic ingestion and watch creation
    pub enabled: bool,

    /// Automatically create watches for detected projects
    pub auto_create_watches: bool,

    /// Project path to watch (if specified)
    pub project_path: Option<String>,

    /// Target collection suffix for auto-created collections
    pub target_collection_suffix: String,

    /// Include source files in watching
    pub include_source_files: bool,

    /// Include common files (README, docs, etc.)
    pub include_common_files: bool,

    /// File patterns to include
    pub include_patterns: Vec<String>,

    /// File patterns to exclude (in addition to standard ignores)
    pub exclude_patterns: Vec<String>,

    /// Enable recursive watching of subdirectories
    pub recursive: bool,

    /// Maximum depth for recursive watching (0 = unlimited)
    pub max_depth: u32,
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
                security: SecurityConfig::default(),
                transport: TransportConfig::default(),
                message: MessageConfig::default(),
                compression: CompressionConfig::default(),
                streaming: StreamingConfig::default(),
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
            auto_ingestion: AutoIngestionConfig::default(),
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
                let config: DaemonConfig = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                    // Parse as TOML
                    toml::from_str(&content)
                        .map_err(|e| crate::error::DaemonError::Config(
                            config::ConfigError::Message(format!("Invalid TOML: {}", e))
                        ))?
                } else {
                    // Parse as YAML (default)
                    serde_yaml::from_str(&content)
                        .map_err(|e| crate::error::DaemonError::Config(
                            config::ConfigError::Message(format!("Invalid YAML: {}", e))
                        ))?
                };
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
    #[allow(dead_code)]
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

impl Default for AutoIngestionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_create_watches: true,
            project_path: None,
            target_collection_suffix: "repo".to_string(),
            include_source_files: true,
            include_common_files: true,
            include_patterns: vec![
                "*.rs".to_string(),
                "*.py".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.md".to_string(),
                "*.txt".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            exclude_patterns: vec![
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                "build/**".to_string(),
                "dist/**".to_string(),
                "*.log".to_string(),
                "*.tmp".to_string(),
                "*.lock".to_string(),
            ],
            recursive: true,
            max_depth: 10,
        }
    }
}

impl Default for MessageConfig {
    fn default() -> Self {
        Self {
            // 16MB default limit (existing baseline mentioned in requirements)
            max_incoming_message_size: 16 * 1024 * 1024,
            max_outgoing_message_size: 16 * 1024 * 1024,
            enable_size_validation: true,
            // 16KB frame size for HTTP/2
            max_frame_size: 16 * 1024,
            // 64KB initial window for HTTP/2
            initial_window_size: 64 * 1024,
            service_limits: ServiceMessageLimits::default(),
            monitoring: MessageMonitoringConfig::default(),
        }
    }
}

impl Default for ServiceMessageLimits {
    fn default() -> Self {
        Self {
            document_processor: ServiceLimit::default_document_processor(),
            search_service: ServiceLimit::default_search(),
            memory_service: ServiceLimit::default_memory(),
            system_service: ServiceLimit::default_system(),
        }
    }
}

impl ServiceLimit {
    fn default_document_processor() -> Self {
        Self {
            max_incoming: 64 * 1024 * 1024, // 64MB for large documents
            max_outgoing: 32 * 1024 * 1024, // 32MB for processed responses
            enable_validation: true,
        }
    }

    fn default_search() -> Self {
        Self {
            max_incoming: 4 * 1024 * 1024,  // 4MB for search queries
            max_outgoing: 16 * 1024 * 1024, // 16MB for search results
            enable_validation: true,
        }
    }

    fn default_memory() -> Self {
        Self {
            max_incoming: 8 * 1024 * 1024,  // 8MB for memory operations
            max_outgoing: 8 * 1024 * 1024,  // 8MB for memory responses
            enable_validation: true,
        }
    }

    fn default_system() -> Self {
        Self {
            max_incoming: 1024 * 1024,      // 1MB for system commands
            max_outgoing: 4 * 1024 * 1024,  // 4MB for system responses
            enable_validation: true,
        }
    }
}

impl Default for MessageMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_detailed_monitoring: true,
            oversized_alert_threshold: 0.8, // Alert at 80% of limit
            enable_realtime_metrics: true,
            metrics_interval_secs: 60,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_gzip: true,
            // Compress messages larger than 1KB
            compression_threshold: 1024,
            // Medium compression level (6) for balance of speed/size
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
            adaptive: AdaptiveCompressionConfig::default(),
            performance: CompressionPerformanceConfig::default(),
        }
    }
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            enable_adaptive: true,
            text_compression_level: 9,   // High compression for text
            binary_compression_level: 3, // Low compression for binary
            structured_compression_level: 6, // Medium for JSON/structured
            max_compression_time_ms: 100,
        }
    }
}

impl Default for CompressionPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_ratio_tracking: true,
            poor_ratio_threshold: 0.9, // Alert if compression ratio > 90%
            enable_time_monitoring: true,
            slow_compression_threshold_ms: 200, // Alert if compression > 200ms
            enable_failure_alerting: true,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_server_streaming: true,
            enable_client_streaming: true,
            // 128 concurrent streams per connection
            max_concurrent_streams: 128,
            // Buffer 1000 items for streaming
            stream_buffer_size: 1000,
            // 5 minute stream timeout
            stream_timeout_secs: 300,
            enable_flow_control: true,
            progress: StreamProgressConfig::default(),
            health: StreamHealthConfig::default(),
            large_operations: LargeOperationStreamConfig::default(),
        }
    }
}

impl Default for StreamProgressConfig {
    fn default() -> Self {
        Self {
            enable_progress_tracking: true,
            progress_update_interval_ms: 1000, // 1 second updates
            enable_progress_callbacks: true,
            progress_threshold: 1024 * 1024, // 1MB minimum for progress tracking
        }
    }
}

impl Default for StreamHealthConfig {
    fn default() -> Self {
        Self {
            enable_health_monitoring: true,
            health_check_interval_secs: 30,
            enable_auto_recovery: true,
            max_recovery_attempts: 3,
            recovery_backoff_multiplier: 2.0,
            initial_recovery_delay_ms: 500,
        }
    }
}

impl Default for LargeOperationStreamConfig {
    fn default() -> Self {
        Self {
            enable_large_document_streaming: true,
            large_operation_chunk_size: 1024 * 1024, // 1MB chunks
            enable_bulk_streaming: true,
            max_streaming_memory: 128 * 1024 * 1024, // 128MB memory limit
            enable_bidirectional_optimization: true,
        }
    }
}

// Default implementations for new configuration structs

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tls: TlsConfig::default(),
            auth: AuthConfig::default(),
            rate_limiting: RateLimitConfig::default(),
            audit: SecurityAuditConfig::default(),
        }
    }
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_file: None,
            key_file: None,
            ca_cert_file: None,
            enable_mtls: false,
            client_cert_verification: ClientCertVerification::None,
            supported_protocols: vec!["TLSv1.2".to_string(), "TLSv1.3".to_string()],
            cipher_suites: vec![], // Use system defaults
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enable_service_auth: false,
            jwt: JwtConfig::default(),
            api_key: ApiKeyConfig::default(),
            authorization: AuthorizationConfig::default(),
        }
    }
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret_or_key_file: "changeme_jwt_secret".to_string(),
            issuer: "workspace-qdrant-mcp".to_string(),
            audience: "workspace-qdrant-clients".to_string(),
            expiration_secs: 3600, // 1 hour
            algorithm: "HS256".to_string(),
        }
    }
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            header_name: "X-API-Key".to_string(),
            valid_keys: vec![],
            key_permissions: std::collections::HashMap::new(),
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_permissions: vec![
                "read".to_string(),
                "write".to_string(),
            ],
            service_permissions: ServicePermissions::default(),
        }
    }
}

impl Default for ServicePermissions {
    fn default() -> Self {
        Self {
            document_processor: vec!["process".to_string(), "read".to_string()],
            search_service: vec!["search".to_string(), "read".to_string()],
            memory_service: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
            system_service: vec!["admin".to_string(), "read".to_string()],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_capacity: 200,
            connection_pool_limits: ConnectionPoolLimits::default(),
            queue_depth_limit: 1000,
            memory_protection: MemoryProtectionConfig::default(),
            resource_protection: ResourceProtectionConfig::default(),
        }
    }
}

impl Default for ConnectionPoolLimits {
    fn default() -> Self {
        Self {
            document_processor: 50,
            search_service: 100,
            memory_service: 75,
            system_service: 25,
        }
    }
}

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_per_connection: 100 * 1024 * 1024, // 100MB
            total_memory_limit: 1024 * 1024 * 1024, // 1GB
            cleanup_interval_secs: 300, // 5 minutes
        }
    }
}

impl Default for ResourceProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_threshold_percent: 80.0,
            disk_threshold_percent: 90.0,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout_secs: 60,
        }
    }
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_file_path: "./security_audit.log".to_string(),
            log_auth_events: true,
            log_auth_failures: true,
            log_rate_limit_events: true,
            log_suspicious_patterns: true,
            rotation: AuditLogRotation::default(),
        }
    }
}

impl Default for AuditLogRotation {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            max_files: 10,
            compress: true,
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            unix_socket: UnixSocketConfig::default(),
            local_optimization: LocalOptimizationConfig::default(),
            transport_strategy: TransportStrategy::Auto,
        }
    }
}

impl Default for UnixSocketConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            socket_path: "/tmp/workspace-qdrant-mcp.sock".to_string(),
            permissions: 0o600, // Owner read/write only
            prefer_for_local: true,
        }
    }
}

impl Default for LocalOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_large_buffers: true,
            local_buffer_size: 64 * 1024, // 64KB
            memory_efficient_serialization: true,
            reduce_latency: LocalLatencyConfig::default(),
        }
    }
}

impl Default for LocalLatencyConfig {
    fn default() -> Self {
        Self {
            disable_nagle: true,
            custom_connection_pooling: true,
            connection_pool_size: 10,
            keepalive_interval_secs: 30,
        }
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

        // Test message configuration defaults
        assert_eq!(config.server.message.max_incoming_message_size, 16 * 1024 * 1024);
        assert_eq!(config.server.message.max_outgoing_message_size, 16 * 1024 * 1024);
        assert!(config.server.message.enable_size_validation);
        assert_eq!(config.server.message.max_frame_size, 16 * 1024);
        assert_eq!(config.server.message.initial_window_size, 64 * 1024);

        // Test compression configuration defaults
        assert!(config.server.compression.enable_gzip);
        assert_eq!(config.server.compression.compression_threshold, 1024);
        assert_eq!(config.server.compression.compression_level, 6);
        assert!(config.server.compression.enable_streaming_compression);
        assert!(config.server.compression.enable_compression_monitoring);

        // Test streaming configuration defaults
        assert!(config.server.streaming.enable_server_streaming);
        assert!(config.server.streaming.enable_client_streaming);
        assert_eq!(config.server.streaming.max_concurrent_streams, 128);
        assert_eq!(config.server.streaming.stream_buffer_size, 1000);
        assert_eq!(config.server.streaming.stream_timeout_secs, 300);
        assert!(config.server.streaming.enable_flow_control);

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
  message:
    max_incoming_message_size: 33554432
    max_outgoing_message_size: 33554432
    enable_size_validation: false
    max_frame_size: 32768
    initial_window_size: 131072
  compression:
    enable_gzip: false
    compression_threshold: 2048
    compression_level: 9
    enable_streaming_compression: false
    enable_compression_monitoring: false
  streaming:
    enable_server_streaming: false
    enable_client_streaming: false
    max_concurrent_streams: 64
    stream_buffer_size: 500
    stream_timeout_secs: 120
    enable_flow_control: false
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

        // Test message configuration from YAML
        assert_eq!(config.server.message.max_incoming_message_size, 33554432);
        assert_eq!(config.server.message.max_outgoing_message_size, 33554432);
        assert!(!config.server.message.enable_size_validation);
        assert_eq!(config.server.message.max_frame_size, 32768);
        assert_eq!(config.server.message.initial_window_size, 131072);

        // Test compression configuration from YAML
        assert!(!config.server.compression.enable_gzip);
        assert_eq!(config.server.compression.compression_threshold, 2048);
        assert_eq!(config.server.compression.compression_level, 9);
        assert!(!config.server.compression.enable_streaming_compression);
        assert!(!config.server.compression.enable_compression_monitoring);

        // Test streaming configuration from YAML
        assert!(!config.server.streaming.enable_server_streaming);
        assert!(!config.server.streaming.enable_client_streaming);
        assert_eq!(config.server.streaming.max_concurrent_streams, 64);
        assert_eq!(config.server.streaming.stream_buffer_size, 500);
        assert_eq!(config.server.streaming.stream_timeout_secs, 120);
        assert!(!config.server.streaming.enable_flow_control);

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
        assert_send_sync::<MessageConfig>();
        assert_send_sync::<CompressionConfig>();
        assert_send_sync::<StreamingConfig>();
        assert_send_sync::<SecurityConfig>();
        assert_send_sync::<TlsConfig>();
        assert_send_sync::<AuthConfig>();
        assert_send_sync::<TransportConfig>();
        assert_send_sync::<UnixSocketConfig>();
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
  message:
    max_incoming_message_size: 16777216
    max_outgoing_message_size: 16777216
    enable_size_validation: true
    max_frame_size: 16384
    initial_window_size: 65536
  compression:
    enable_gzip: true
    compression_threshold: 1024
    compression_level: 6
    enable_streaming_compression: true
    enable_compression_monitoring: true
  streaming:
    enable_server_streaming: true
    enable_client_streaming: true
    max_concurrent_streams: 128
    stream_buffer_size: 1000
    stream_timeout_secs: 300
    enable_flow_control: true
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