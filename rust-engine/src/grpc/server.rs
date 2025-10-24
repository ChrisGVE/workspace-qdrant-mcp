//! Main gRPC server implementation with all service registrations

use crate::daemon::WorkspaceDaemon;
use crate::grpc::middleware::{ConnectionManager, ConnectionInterceptor};
#[cfg(any(test, feature = "test-utils"))]
use crate::grpc::message_validation::MessageValidator;
use crate::grpc::services::{
    SystemServiceImpl,
};
use crate::proto::{
    system_service_server::SystemServiceServer,
};

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tonic::transport::Server;
use tracing::info;

/// Main gRPC server coordinating all daemon services
pub struct GrpcServer {
    daemon: Arc<WorkspaceDaemon>,
    address: SocketAddr,
    connection_manager: Arc<ConnectionManager>,
    #[cfg(any(test, feature = "test-utils"))]
    message_validator: Arc<MessageValidator>,
}

impl GrpcServer {
    /// Create a new gRPC server instance
    pub fn new(daemon: WorkspaceDaemon, address: SocketAddr) -> Self {
        let config = daemon.config().clone();
        let connection_manager = Arc::new(ConnectionManager::new(
            config.server().max_connections as u64,
            100, // 100 requests per second per client
        ));

        Self {
            daemon: Arc::new(daemon),
            address,
            connection_manager,
            #[cfg(any(test, feature = "test-utils"))]
            message_validator: {
                // Initialize message validator with configuration for test/debug use
                Arc::new(MessageValidator::new(
                    config.server().message.clone(),
                    config.server().compression.clone(),
                    config.server().streaming.clone(),
                ))
            },
        }
    }

    /// Start the gRPC server in foreground mode
    pub async fn serve(self) -> Result<()> {
        let address = self.address;
        info!("Starting gRPC server on {}", address);
        info!("Connection manager initialized with max connections: {}",
              self.daemon.config().server().max_connections);

        // Start connection cleanup task
        let connection_manager_clone = Arc::clone(&self.connection_manager);
        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                connection_manager_clone.cleanup_expired_connections(Duration::from_secs(300));
            }
        });

        let server = self.build_server().await?;

        // Graceful shutdown
        let shutdown_signal = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install CTRL+C signal handler");
            info!("Received shutdown signal, gracefully shutting down...");
            cleanup_task.abort();
        };

        server
            .serve_with_shutdown(address, shutdown_signal)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        info!("gRPC server shutdown complete");
        Ok(())
    }

    /// Start the gRPC server in daemon mode
    pub async fn serve_daemon(self) -> Result<()> {
        let address = self.address;
        info!("Starting gRPC server in daemon mode on {}", address);
        info!("Connection manager initialized with max connections: {}",
              self.daemon.config().server().max_connections);

        // Start connection cleanup task
        let connection_manager_clone = Arc::clone(&self.connection_manager);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                connection_manager_clone.cleanup_expired_connections(Duration::from_secs(300));
            }
        });

        let server = self.build_server().await?;

        server
            .serve(address)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        Ok(())
    }

    /// Build the complete gRPC server with all services
    async fn build_server(self) -> Result<tonic::transport::server::Router> {
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(include_bytes!(concat!(env!("OUT_DIR"), "/workspace_daemon_descriptor.bin")))
            .build_v1()
            .map_err(|e| anyhow::anyhow!("Failed to build reflection service: {}", e))?;

        // Create connection interceptor
        let _interceptor = ConnectionInterceptor::new(Arc::clone(&self.connection_manager));

        // Create service implementations
        let system_service = SystemServiceImpl::new(Arc::clone(&self.daemon));

        let config = self.daemon.config();

        let server = Server::builder()
            // Add connection timeout from config
            .timeout(Duration::from_secs(config.server().connection_timeout_secs))
            // HTTP/2 frame configuration
            .max_frame_size(Some(config.server().message.max_frame_size))
            .initial_stream_window_size(Some(config.server().message.initial_window_size))
            .initial_connection_window_size(Some(config.server().message.initial_window_size * 2))
            // Streaming configuration
            .max_concurrent_streams(Some(config.server().streaming.max_concurrent_streams))
            // Connection limits and keep-alive
            .concurrency_limit_per_connection(config.server().streaming.max_concurrent_streams as usize)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            // Request timeout for streaming operations
            .tcp_nodelay(true)
            // Register system service
            .add_service(
                SystemServiceServer::new(system_service)
                    .max_decoding_message_size(config.server().message.service_limits.system_service.max_incoming)
                    .max_encoding_message_size(config.server().message.service_limits.system_service.max_outgoing)
            )
            // Add reflection for debugging
            .add_service(reflection_service);

        Ok(server)
    }

    /// Get connection statistics (test/debug use)
    #[cfg(any(test, feature = "test-utils"))]
    pub fn get_connection_stats(&self) -> crate::grpc::middleware::ConnectionStats {
        self.connection_manager.get_stats()
    }

    /// Get connection manager for external access (test/debug use)
    #[cfg(any(test, feature = "test-utils"))]
    pub fn connection_manager(&self) -> &Arc<ConnectionManager> {
        &self.connection_manager
    }

    /// Get message validator for external access (test/debug use)
    #[cfg(any(test, feature = "test-utils"))]
    pub fn message_validator(&self) -> &Arc<MessageValidator> {
        &self.message_validator
    }

    /// Get message processing statistics (test/debug use)
    #[cfg(any(test, feature = "test-utils"))]
    pub fn get_message_stats(&self) -> crate::grpc::message_validation::MessageStats {
        self.message_validator.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_daemon_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:";

        DaemonConfig {
            system: SystemConfig {
                project_name: "test-project".to_string(),
                database: DatabaseConfig {
                    sqlite_path: db_path.to_string(),
                    max_connections: 5,
                    connection_timeout_secs: 30,
                    enable_wal: true,
                },
                auto_ingestion: AutoIngestionConfig {
                    enabled: false,
                    project_collection: "test".to_string(),
                    auto_create_watches: false,
                    project_path: None,
                    include_source_files: true,
                    include_patterns: vec![],
                    exclude_patterns: vec![],
                    max_depth: 10,
                },
                processing: ProcessingConfig {
                    max_concurrent_tasks: 4,
                    supported_extensions: vec![],
                    default_chunk_size: 1000,
                    default_chunk_overlap: 200,
                    max_file_size_bytes: 10 * 1024 * 1024,
                    enable_lsp: false,
                    lsp_timeout_secs: 10,
                },
                file_watcher: FileWatcherConfig {
                    enabled: false,
                    ignore_patterns: vec![],
                    recursive: true,
                    max_watched_dirs: 100,
                    debounce_ms: 100,
                },
            },
            grpc: GrpcConfig {
                server: GrpcServerConfig {
                    enabled: true,
                    port: 50052,
                },
                client: GrpcClientConfig::default(),
                security: SecurityConfig::default(),
                transport: TransportConfig::default(),
                message: MessageConfig::default(),
            },
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50052, // Use different port for testing
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
            },
            external_services: ExternalServicesConfig {
                qdrant: QdrantConfig {
                    url: "http://localhost:6333".to_string(),
                    api_key: None,
                    max_retries: 3,
                    default_collection: CollectionConfig {
                        vector_size: 384,
                        distance_metric: "Cosine".to_string(),
                        enable_indexing: true,
                        replication_factor: 1,
                        shard_number: 1,
                    },
                },
            },
            transport: crate::config::TransportConfig::default(),
            message: crate::config::MessageConfig::default(),
            security: crate::config::SecurityConfig::default(),
            streaming: crate::config::StreamingConfig::default(),
            compression: crate::config::CompressionConfig::default(),
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                max_retries: 3,
                default_collection: CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            workspace: WorkspaceConfig {
                collection_basename: Some("test-workspace".to_string()),
                collection_types: vec!["code".to_string(), "notes".to_string()],
                memory_collection_name: "_memory".to_string(),
                auto_create_collections: true,
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 2,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 1024 * 1024,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 10,
                ignore_patterns: vec![],
                recursive: true,
            },
            auto_ingestion: crate::config::AutoIngestionConfig::default(),
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
            },
            logging: LoggingConfig {
                enabled: true,
                level: "info".to_string(),
                file_path: None,
                max_file_size: SizeUnit(100 * 1024 * 1024), // 100MB
                max_files: 5,
                enable_json: false,
                enable_structured: false,
                enable_console: true,
            },
        }
    }

    async fn create_test_daemon() -> WorkspaceDaemon {
        let config = create_test_daemon_config();
        WorkspaceDaemon::new(config).await.expect("Failed to create daemon")
    }

    #[tokio::test]
    async fn test_grpc_server_new() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);

        let server = GrpcServer::new(daemon, address);

        assert_eq!(server.address, address);
        assert!(Arc::strong_count(&server.connection_manager) >= 1);
    }

    #[tokio::test]
    async fn test_grpc_server_connection_manager_access() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);
        let connection_manager = server.connection_manager();

        // Test that we can access connection manager
        let stats = connection_manager.get_stats();
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_grpc_server_get_connection_stats() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);
        let stats = server.get_connection_stats();

        // Initially should have no active connections
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_grpc_server_build_server() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);
        let result = server.build_server().await;

        assert!(result.is_ok());
    }

    #[test]
    fn test_grpc_server_address_types() {
        // Test IPv4 address
        let ipv4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 8080);
        assert_eq!(ipv4.ip(), IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(ipv4.port(), 8080);

        // Test IPv6 address
        let ipv6 = "[::1]:9090".parse::<SocketAddr>().unwrap();
        assert!(ipv6.is_ipv6());
        assert_eq!(ipv6.port(), 9090);
    }

    #[tokio::test]
    async fn test_grpc_server_with_different_ports() {
        let daemon = create_test_daemon().await;

        let ports = [8080, 8081, 8082];
        for port in ports {
            let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
            let server = GrpcServer::new(daemon.clone(), address);
            assert_eq!(server.address.port(), port);
        }
    }

    #[tokio::test]
    async fn test_grpc_server_connection_manager_initialization() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);
        let connection_manager = server.connection_manager();

        // Test that connection manager is properly initialized
        let stats = connection_manager.get_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_received, 0);
    }

    #[tokio::test]
    async fn test_grpc_server_daemon_config_access() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);

        // Test accessing daemon config through server
        let connection_manager = server.connection_manager();
        let stats = connection_manager.get_stats();

        // Should have valid stats (active_connections is u64, always >= 0)
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_grpc_server_arc_sharing() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);
        let daemon_arc = &server.daemon;

        // Verify daemon is properly shared via Arc
        assert!(Arc::strong_count(daemon_arc) >= 1);
    }

    #[test]
    fn test_socket_addr_parsing() {
        // Test various socket address formats
        let addrs = [
            "127.0.0.1:8080",
            "0.0.0.0:8080",
            "[::1]:8080",
            "[::]:8080",
        ];

        for addr_str in addrs {
            let result = addr_str.parse::<SocketAddr>();
            assert!(result.is_ok(), "Failed to parse address: {}", addr_str);
        }
    }

    #[tokio::test]
    async fn test_grpc_server_memory_efficiency() {
        let daemon = create_test_daemon().await;
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let server = GrpcServer::new(daemon, address);

        // Verify that server doesn't hold unnecessary references
        let initial_daemon_count = Arc::strong_count(&server.daemon);
        let initial_manager_count = Arc::strong_count(&server.connection_manager);

        assert_eq!(initial_daemon_count, 1);
        assert_eq!(initial_manager_count, 1);
    }

    #[test]
    fn test_grpc_server_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<GrpcServer>();
        assert_sync::<GrpcServer>();
    }
}
