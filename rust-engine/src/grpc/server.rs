//! Main gRPC server implementation with all service registrations

use crate::daemon::WorkspaceDaemon;
use crate::error::DaemonError;
use crate::grpc::middleware::{ConnectionManager, ConnectionInterceptor};
use crate::grpc::services::{
    DocumentProcessorImpl,
    SearchServiceImpl,
    MemoryServiceImpl,
    SystemServiceImpl,
};
use crate::proto::{
    document_processor_server::DocumentProcessorServer,
    search_service_server::SearchServiceServer,
    memory_service_server::MemoryServiceServer,
    system_service_server::SystemServiceServer,
};

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tonic::transport::Server;
use tracing::{error, info, warn};

/// Main gRPC server coordinating all daemon services
pub struct GrpcServer {
    daemon: Arc<WorkspaceDaemon>,
    address: SocketAddr,
    connection_manager: Arc<ConnectionManager>,
}

impl GrpcServer {
    /// Create a new gRPC server instance
    pub fn new(daemon: WorkspaceDaemon, address: SocketAddr) -> Self {
        let config = daemon.config();
        let connection_manager = Arc::new(ConnectionManager::new(
            config.server.max_connections as u64,
            100, // 100 requests per second per client
        ));

        Self {
            daemon: Arc::new(daemon),
            address,
            connection_manager,
        }
    }

    /// Start the gRPC server in foreground mode
    pub async fn serve(self) -> Result<()> {
        info!("Starting gRPC server on {}", self.address);
        info!("Connection manager initialized with max connections: {}",
              self.daemon.config().server.max_connections);

        let server = self.build_server().await?;

        // Start connection cleanup task
        let connection_manager_clone = Arc::clone(&self.connection_manager);
        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                connection_manager_clone.cleanup_expired_connections(Duration::from_secs(300));
            }
        });

        // Graceful shutdown
        let shutdown_signal = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install CTRL+C signal handler");
            info!("Received shutdown signal, gracefully shutting down...");
            cleanup_task.abort();
        };

        server
            .serve_with_shutdown(self.address, shutdown_signal)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        info!("gRPC server shutdown complete");
        Ok(())
    }

    /// Start the gRPC server in daemon mode
    pub async fn serve_daemon(self) -> Result<()> {
        info!("Starting gRPC server in daemon mode on {}", self.address);
        info!("Connection manager initialized with max connections: {}",
              self.daemon.config().server.max_connections);

        let server = self.build_server().await?;

        // Start connection cleanup task
        let connection_manager_clone = Arc::clone(&self.connection_manager);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                connection_manager_clone.cleanup_expired_connections(Duration::from_secs(300));
            }
        });

        server
            .serve(self.address)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        Ok(())
    }

    /// Build the complete gRPC server with all services
    async fn build_server(self) -> Result<tonic::transport::server::Router> {
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(include_bytes!(concat!(env!("OUT_DIR"), "/workspace_daemon_descriptor.bin")))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build reflection service: {}", e))?;

        // Create connection interceptor
        let interceptor = ConnectionInterceptor::new(Arc::clone(&self.connection_manager));

        // Create service implementations
        let document_processor = DocumentProcessorImpl::new(Arc::clone(&self.daemon));
        let search_service = SearchServiceImpl::new(Arc::clone(&self.daemon));
        let memory_service = MemoryServiceImpl::new(Arc::clone(&self.daemon));
        let system_service = SystemServiceImpl::new(Arc::clone(&self.daemon));

        let config = self.daemon.config();

        let server = Server::builder()
            // Add compression support
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip)
            // Set message size limits (16MB)
            .max_decoding_message_size(16 * 1024 * 1024)
            .max_encoding_message_size(16 * 1024 * 1024)
            // Add connection timeout from config
            .timeout(Duration::from_secs(config.server.connection_timeout_secs))
            // Set concurrency limits
            .concurrency_limit_per_connection(256)
            // Add keep-alive settings
            .tcp_keepalive(Some(Duration::from_secs(60)))
            // Register all services with interceptors
            .add_service(
                tonic::service::interceptor(
                    DocumentProcessorServer::new(document_processor),
                    move |req| interceptor.intercept(req)
                )
            )
            .add_service(
                tonic::service::interceptor(
                    SearchServiceServer::new(search_service),
                    move |req| interceptor.intercept(req)
                )
            )
            .add_service(
                tonic::service::interceptor(
                    MemoryServiceServer::new(memory_service),
                    move |req| interceptor.intercept(req)
                )
            )
            .add_service(
                tonic::service::interceptor(
                    SystemServiceServer::new(system_service),
                    move |req| interceptor.intercept(req)
                )
            )
            // Add reflection for debugging
            .add_service(reflection_service);

        Ok(server)
    }

    /// Get connection statistics
    pub fn get_connection_stats(&self) -> crate::grpc::middleware::ConnectionStats {
        self.connection_manager.get_stats()
    }

    /// Get connection manager for external access
    pub fn connection_manager(&self) -> &Arc<ConnectionManager> {
        &self.connection_manager
    }
}