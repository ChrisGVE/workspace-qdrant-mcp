//! gRPC service for workspace-qdrant-mcp ingestion engine
//!
//! This crate provides the gRPC server implementation for communication
//! between the Python MCP server and the Rust processing engine.
//!
//! # Module layout
//!
//! - [`auth`] - Authentication interceptor and TLS/auth configuration types
//! - [`builder`] - Fluent builder methods for `GrpcServer` dependency injection
//! - [`factory`] - Service instantiation, TLS setup, and server startup
//! - [`services`] - Individual gRPC service implementations

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use sqlx::SqlitePool;
use thiserror::Error;
use tokio::sync::{Notify, RwLock};
use workspace_qdrant_core::adaptive_resources::AdaptiveResourceState;
use workspace_qdrant_core::LanguageServerManager;
use workspace_qdrant_core::SearchDbManager;

pub mod auth;
mod builder;
mod factory;
pub mod services;

// Re-export auth types at crate root for backward compatibility
pub use auth::{AuthConfig, AuthInterceptor, TlsConfig};

pub mod proto {
    // Generated protobuf definitions from build.rs
    // Package: workspace_daemon - defines SystemService, CollectionService, DocumentService,
    // EmbeddingService, ProjectService, TextSearchService, GraphService
    tonic::include_proto!("workspace_daemon");
}

/// gRPC service errors
#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("Service error: {0}")]
    Service(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Authorization error: {0}")]
    Authorization(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// gRPC server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub tls_config: Option<TlsConfig>,
    pub auth_config: Option<AuthConfig>,
    pub timeout_config: TimeoutConfig,
    pub performance_config: PerformanceConfig,
    pub health_check_config: HealthCheckConfig,
}

/// Timeout configuration
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub request_timeout: Duration,
    pub connection_timeout: Duration,
    pub keepalive_interval: Duration,
    pub keepalive_timeout: Duration,
}

/// Performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub max_concurrent_streams: u32,
    pub max_message_size: usize,
    pub max_connection_idle: Duration,
    pub max_connection_age: Duration,
    pub tcp_nodelay: bool,
    pub tcp_keepalive: Option<Duration>,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub enabled: bool,
    pub interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: u32,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(10),
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(5),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 1000,
            max_message_size: 16 * 1024 * 1024, // 16MB
            max_connection_idle: Duration::from_secs(300),   // 5 minutes
            max_connection_age: Duration::from_secs(3600),   // 1 hour
            tcp_nodelay: true,
            tcp_keepalive: Some(Duration::from_secs(600)),   // 10 minutes
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
        }
    }
}

impl ServerConfig {
    pub fn new(bind_addr: SocketAddr) -> Self {
        Self {
            bind_addr,
            tls_config: None,
            auth_config: Some(AuthConfig::default()),
            timeout_config: TimeoutConfig::default(),
            performance_config: PerformanceConfig::default(),
            health_check_config: HealthCheckConfig::default(),
        }
    }

    /// Create a secure configuration with TLS and authentication required
    pub fn new_secure(
        bind_addr: SocketAddr,
        cert_path: String,
        key_path: String,
        api_key: String,
    ) -> Self {
        Self {
            bind_addr,
            tls_config: Some(TlsConfig {
                cert_path,
                key_path,
                ca_cert_path: None,
                require_client_cert: false,
            }),
            auth_config: Some(AuthConfig {
                enabled: true,
                api_key: Some(api_key),
                jwt_secret: None,
                allowed_origins: vec![],
            }),
            timeout_config: TimeoutConfig::default(),
            performance_config: PerformanceConfig::default(),
            health_check_config: HealthCheckConfig::default(),
        }
    }

    /// Create a mutual TLS configuration for maximum security
    pub fn new_mutual_tls(
        bind_addr: SocketAddr,
        cert_path: String,
        key_path: String,
        ca_cert_path: String,
    ) -> Self {
        Self {
            bind_addr,
            tls_config: Some(TlsConfig {
                cert_path,
                key_path,
                ca_cert_path: Some(ca_cert_path),
                require_client_cert: true,
            }),
            auth_config: Some(AuthConfig {
                enabled: false, // mTLS provides authentication
                api_key: None,
                jwt_secret: None,
                allowed_origins: vec![],
            }),
            timeout_config: TimeoutConfig::default(),
            performance_config: PerformanceConfig::default(),
            health_check_config: HealthCheckConfig::default(),
        }
    }

    /// Check if the configuration is secure (has TLS or requires authentication)
    pub fn is_secure(&self) -> bool {
        self.tls_config.is_some()
            || self.auth_config.as_ref().map(|a| a.enabled).unwrap_or(false)
    }

    /// Get security warnings for the current configuration
    pub fn get_security_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.tls_config.is_none() {
            warnings.push(
                "TLS is not enabled - all communication will be unencrypted".to_string(),
            );
        }

        self.check_auth_warnings(&mut warnings);

        // Check if binding to all interfaces without TLS
        if self.bind_addr.ip().is_unspecified() && self.tls_config.is_none() {
            warnings.push(
                "Binding to 0.0.0.0 without TLS - server exposed to network without encryption"
                    .to_string(),
            );
        }

        warnings
    }

    fn check_auth_warnings(&self, warnings: &mut Vec<String>) {
        let Some(auth) = &self.auth_config else {
            warnings.push("No authentication configured".to_string());
            return;
        };

        if !auth.enabled {
            warnings.push(
                "Authentication is disabled - anyone can access the gRPC server".to_string(),
            );
        }
        if auth.enabled && auth.api_key.is_none() && auth.jwt_secret.is_none() {
            warnings
                .push("Authentication enabled but no credentials configured".to_string());
        }
        if auth.allowed_origins.contains(&"*".to_string()) {
            warnings
                .push("Wildcard origin allowed - CORS protection disabled".to_string());
        }
    }

    pub fn with_tls(mut self, tls_config: TlsConfig) -> Self {
        self.tls_config = Some(tls_config);
        self
    }

    pub fn with_auth(mut self, auth_config: AuthConfig) -> Self {
        self.auth_config = Some(auth_config);
        self
    }

    pub fn with_timeouts(mut self, timeout_config: TimeoutConfig) -> Self {
        self.timeout_config = timeout_config;
        self
    }

    pub fn with_performance(mut self, performance_config: PerformanceConfig) -> Self {
        self.performance_config = performance_config;
        self
    }
}

/// gRPC server instance with enhanced security and performance
pub struct GrpcServer {
    pub(crate) config: ServerConfig,
    pub(crate) shutdown_signal: Option<tokio::sync::oneshot::Receiver<()>>,
    pub(crate) metrics: Arc<ServerMetrics>,
    /// Optional database pool for ProjectService
    pub(crate) db_pool: Option<SqlitePool>,
    /// Whether to enable LSP lifecycle management in ProjectService
    pub(crate) enable_lsp: bool,
    /// External LSP manager for lifecycle control (optional)
    pub(crate) lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    /// Shared pause flag for watcher pause/resume propagation
    pub(crate) pause_flag: Option<Arc<AtomicBool>>,
    /// Signal to trigger immediate WatchManager refresh on config changes
    pub(crate) watch_refresh_signal: Option<Arc<Notify>>,
    /// Queue processor health state for monitoring
    pub(crate) queue_health: Option<Arc<workspace_qdrant_core::QueueProcessorHealth>>,
    /// Adaptive resource state for idle/burst mode reporting
    pub(crate) adaptive_state: Option<Arc<AdaptiveResourceState>>,
    /// Search database manager for TextSearchService
    pub(crate) search_db: Option<Arc<SearchDbManager>>,
    /// Graph store for GraphService (code relationship queries)
    pub(crate) graph_store: Option<
        workspace_qdrant_core::graph::SharedGraphStore<
            workspace_qdrant_core::graph::SqliteGraphStore,
        >,
    >,
    /// Hierarchy builder for tag hierarchy rebuild via gRPC
    pub(crate) hierarchy_builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    /// Lexicon manager for vocabulary rebuild via gRPC
    pub(crate) lexicon_manager: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    /// Storage client for Qdrant operations (rules rebuild)
    pub(crate) storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
}

/// Server metrics for monitoring
#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub total_requests: std::sync::atomic::AtomicU64,
    pub active_connections: std::sync::atomic::AtomicU32,
    pub failed_requests: std::sync::atomic::AtomicU64,
    pub auth_failures: std::sync::atomic::AtomicU64,
    pub avg_response_time: std::sync::atomic::AtomicU64, // in milliseconds
}

impl GrpcServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            shutdown_signal: None,
            metrics: Arc::new(ServerMetrics::default()),
            db_pool: None,
            enable_lsp: false,
            lsp_manager: None,
            pause_flag: None,
            watch_refresh_signal: None,
            queue_health: None,
            adaptive_state: None,
            search_db: None,
            graph_store: None,
            hierarchy_builder: None,
            lexicon_manager: None,
            storage_client: None,
        }
    }
}

/// Basic health check function
pub fn health_check() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    #[test]
    fn test_health_check() {
        assert!(health_check());
    }

    #[test]
    fn test_server_config() {
        let addr = "127.0.0.1:50051".parse::<SocketAddr>().unwrap();
        let config = ServerConfig::new(addr);
        assert_eq!(config.bind_addr, addr);
    }

    #[test]
    fn test_protobuf_serialization() {
        use crate::proto::*;
        use prost::Message;

        let request = CreateCollectionRequest {
            collection_name: "test_collection".to_string(),
            project_id: "test_project".to_string(),
            config: Some(CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                metadata_schema: std::collections::HashMap::new(),
            }),
        };

        let bytes = prost::Message::encode_to_vec(&request);
        assert!(!bytes.is_empty());

        let decoded = CreateCollectionRequest::decode(&bytes[..]).unwrap();
        assert_eq!(decoded.collection_name, "test_collection");
        assert_eq!(decoded.project_id, "test_project");
        assert!(decoded.config.is_some());
        let config = decoded.config.unwrap();
        assert_eq!(config.vector_size, 384);
        assert_eq!(config.distance_metric, "Cosine");
    }

    #[test]
    fn test_grpc_service_instantiation() {
        use crate::services::SystemServiceImpl;

        let _system_service = SystemServiceImpl::new();
    }
}
