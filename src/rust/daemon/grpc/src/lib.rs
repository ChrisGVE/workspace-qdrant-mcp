//! gRPC service for workspace-qdrant-mcp ingestion engine
//!
//! This crate provides the gRPC server implementation for communication
//! between the Python MCP server and the Rust processing engine.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::signal;
use tonic::transport::{Server, ServerTlsConfig, Identity};
use tonic::{Request, Status, metadata::MetadataValue};
use uuid::Uuid;

pub mod proto {
    // Generated protobuf definitions from build.rs
    tonic::include_proto!("workspace_qdrant.v1");
}

pub mod service;

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

/// TLS configuration for secure connections
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
    pub require_client_cert: bool,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub jwt_secret: Option<String>,
    pub allowed_origins: Vec<String>,
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

// Macro to implement Default for configuration structs
macro_rules! implement_defaults {
    ($($struct_name:ident { $($field:ident: $value:expr),* $(,)? }),* $(,)?) => {
        $(
            impl Default for $struct_name {
                fn default() -> Self {
                    Self {
                        $($field: $value,)*
                    }
                }
            }
        )*
    };
}

implement_defaults! {
    AuthConfig {
        enabled: false,
        api_key: None,
        jwt_secret: None,
        allowed_origins: vec!["*".to_string()],
    },
    TimeoutConfig {
        request_timeout: Duration::from_secs(30),
        connection_timeout: Duration::from_secs(10),
        keepalive_interval: Duration::from_secs(30),
        keepalive_timeout: Duration::from_secs(5),
    },
    PerformanceConfig {
        max_concurrent_streams: 1000,
        max_message_size: 16 * 1024 * 1024, // 16MB
        max_connection_idle: Duration::from_secs(300), // 5 minutes
        max_connection_age: Duration::from_secs(3600), // 1 hour
        tcp_nodelay: true,
        tcp_keepalive: Some(Duration::from_secs(600)), // 10 minutes
    },
    HealthCheckConfig {
        enabled: true,
        interval: Duration::from_secs(30),
        timeout: Duration::from_secs(5),
        failure_threshold: 3,
    },
}

/// gRPC server instance with enhanced security and performance
pub struct GrpcServer {
    config: ServerConfig,
    shutdown_signal: Option<tokio::sync::oneshot::Receiver<()>>,
    metrics: Arc<ServerMetrics>,
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

/// Authentication interceptor
#[derive(Clone)]
pub struct AuthInterceptor {
    config: Option<AuthConfig>,
}

impl AuthInterceptor {
    pub fn new(config: Option<AuthConfig>) -> Self {
        Self { config }
    }

    pub fn check(&self, req: &Request<()>) -> Result<(), Status> {
        let Some(auth_config) = &self.config else {
            return Ok(()); // No auth configured
        };

        if !auth_config.enabled {
            return Ok(());
        }

        // Check API key if configured
        if let Some(expected_key) = &auth_config.api_key {
            let auth_header = req.metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| Status::unauthenticated("Missing authorization header"))?;

            if !auth_header.starts_with("Bearer ") {
                return Err(Status::unauthenticated("Invalid authorization format"));
            }

            let token = &auth_header[7..]; // Remove "Bearer " prefix
            if token != expected_key {
                return Err(Status::unauthenticated("Invalid API key"));
            }
        }

        // Check origin if configured
        if !auth_config.allowed_origins.contains(&"*".to_string()) {
            let origin = req.metadata()
                .get("origin")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");

            if !auth_config.allowed_origins.contains(&origin.to_string()) {
                return Err(Status::permission_denied("Origin not allowed"));
            }
        }

        Ok(())
    }
}

impl GrpcServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            shutdown_signal: None,
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    pub fn with_shutdown_signal(mut self, receiver: tokio::sync::oneshot::Receiver<()>) -> Self {
        self.shutdown_signal = Some(receiver);
        self
    }

    pub async fn start(&mut self) -> Result<(), GrpcError> {
        let service = service::IngestionService::new_with_auth(
            self.config.auth_config.clone()
        );

        tracing::info!("Starting gRPC server on {}", self.config.bind_addr);
        tracing::info!("gRPC server configuration: TLS={}, Auth={}, Timeouts={:?}",
            self.config.tls_config.is_some(),
            self.config.auth_config.as_ref().map(|a| a.enabled).unwrap_or(false),
            self.config.timeout_config
        );

        let mut server_builder = Server::builder()
            .timeout(self.config.timeout_config.request_timeout)
            .concurrency_limit_per_connection(self.config.performance_config.max_concurrent_streams as usize)
            .tcp_nodelay(self.config.performance_config.tcp_nodelay);

        // Configure TLS if enabled
        if let Some(tls_config) = &self.config.tls_config {
            let cert = tokio::fs::read(&tls_config.cert_path).await
                .map_err(|e| GrpcError::Configuration(
                    format!("Failed to read TLS certificate: {}", e)
                ))?;

            let key = tokio::fs::read(&tls_config.key_path).await
                .map_err(|e| GrpcError::Configuration(
                    format!("Failed to read TLS key: {}", e)
                ))?;

            let identity = Identity::from_pem(cert, key);
            let mut tls = ServerTlsConfig::new().identity(identity);

            if let Some(ca_cert_path) = &tls_config.ca_cert_path {
                let ca_cert = tokio::fs::read(ca_cert_path).await
                    .map_err(|e| GrpcError::Configuration(
                        format!("Failed to read CA certificate: {}", e)
                    ))?;
                tls = tls.client_ca_root(tonic::transport::Certificate::from_pem(ca_cert));
            }

            if tls_config.require_client_cert {
                tls = tls.client_auth_optional(false);
            }

            server_builder = server_builder.tls_config(tls)
                .map_err(|e| GrpcError::Configuration(
                    format!("Failed to configure TLS: {}", e)
                ))?;
        }

        // Add the service without interceptors (auth handled in service)
        let svc = proto::ingest_service_server::IngestServiceServer::new(service);

        let server = server_builder.add_service(svc);

        // Start server with graceful shutdown
        match self.shutdown_signal.take() {
            Some(shutdown) => {
                tracing::info!("gRPC server started with graceful shutdown support");
                server
                    .serve_with_shutdown(self.config.bind_addr, async {
                        shutdown.await.ok();
                        tracing::info!("gRPC server received shutdown signal");
                    })
                    .await?
            }
            None => {
                tracing::info!("gRPC server started without shutdown signal");
                server.serve(self.config.bind_addr).await?
            }
        }

        tracing::info!("gRPC server stopped");
        Ok(())
    }

    pub fn get_metrics(&self) -> Arc<ServerMetrics> {
        Arc::clone(&self.metrics)
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
        
        // Test ProcessDocumentRequest serialization
        let request = ProcessDocumentRequest {
            file_path: "/test/path.txt".to_string(),
            collection: "test_collection".to_string(),
            metadata: std::collections::HashMap::new(),
            document_id: Some("test_doc_id".to_string()),
            chunk_text: true,
        };
        
        // Should be able to serialize and deserialize
        let bytes = prost::Message::encode_to_vec(&request);
        assert!(!bytes.is_empty());
        
        let decoded = ProcessDocumentRequest::decode(&bytes[..]).unwrap();
        assert_eq!(decoded.file_path, "/test/path.txt");
        assert_eq!(decoded.collection, "test_collection");
        assert_eq!(decoded.document_id, Some("test_doc_id".to_string()));
        assert_eq!(decoded.chunk_text, true);
    }

    #[test]
    fn test_grpc_service_instantiation() {
        use crate::service::IngestionService;
        
        // Test that the service can be created
        let _service = IngestionService::new();
        // Service should be valid (no panic on creation)
        let _service2 = IngestionService::default();
    }
}
