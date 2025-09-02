//! gRPC service for workspace-qdrant-mcp ingestion engine
//!
//! This crate provides the gRPC server implementation for communication
//! between the Python MCP server and the Rust processing engine.

use std::net::SocketAddr;
use thiserror::Error;
use tonic::transport::Server;

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
}

/// gRPC server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
}

impl ServerConfig {
    pub fn new(bind_addr: SocketAddr) -> Self {
        Self { bind_addr }
    }
}

/// gRPC server instance
pub struct GrpcServer {
    config: ServerConfig,
}

impl GrpcServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    pub async fn start(&self) -> Result<(), GrpcError> {
        let service = service::IngestionService::new();

        tracing::info!("Starting gRPC server on {}", self.config.bind_addr);

        Server::builder()
            .add_service(proto::ingest_service_server::IngestServiceServer::new(service))
            .serve(self.config.bind_addr)
            .await?;

        Ok(())
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

    #[tokio::test]
    async fn test_grpc_server() {
        let addr = "127.0.0.1:50052".parse::<SocketAddr>().unwrap();
        let config = ServerConfig::new(addr);
        let server = GrpcServer::new(config);
        // Basic instantiation test
        assert!(server.start().await.is_ok());
    }
}
