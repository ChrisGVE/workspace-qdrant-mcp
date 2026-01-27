//! DaemonClient wrapper
//!
//! Connects to memexd daemon and provides access to all 4 gRPC services:
//! - SystemService (7 RPCs): Health monitoring, status, lifecycle
//! - CollectionService (5 RPCs): Collection CRUD, alias management
//! - DocumentService (3 RPCs): Text ingestion
//! - ProjectService (5 RPCs): Project registration, sessions
//!
//! Note: Some service client methods are infrastructure for future CLI commands.

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::time::Duration;
use tonic::transport::Channel;

/// Generated from workspace_daemon.proto
pub mod workspace_daemon {
    tonic::include_proto!("workspace_daemon");
}

use workspace_daemon::{
    collection_service_client::CollectionServiceClient,
    document_service_client::DocumentServiceClient,
    project_service_client::ProjectServiceClient,
    system_service_client::SystemServiceClient,
};

/// Default gRPC port for memexd daemon
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// Default connection timeout in seconds
pub const DEFAULT_CONNECTION_TIMEOUT_SECS: u64 = 5;

/// Client wrapper for memexd daemon gRPC services
pub struct DaemonClient {
    system: SystemServiceClient<Channel>,
    collection: CollectionServiceClient<Channel>,
    document: DocumentServiceClient<Channel>,
    project: ProjectServiceClient<Channel>,
}

impl DaemonClient {
    /// Connect to the memexd daemon at the specified address
    ///
    /// # Arguments
    /// * `addr` - gRPC endpoint (e.g., "http://127.0.0.1:50051")
    ///
    /// # Errors
    /// Returns error if connection fails or times out
    pub async fn connect(addr: &str) -> Result<Self> {
        let channel = Channel::from_shared(addr.to_string())
            .context("Invalid daemon address")?
            .connect_timeout(Duration::from_secs(DEFAULT_CONNECTION_TIMEOUT_SECS))
            .connect()
            .await
            .context("Failed to connect to memexd daemon. Is it running?")?;

        Ok(Self {
            system: SystemServiceClient::new(channel.clone()),
            collection: CollectionServiceClient::new(channel.clone()),
            document: DocumentServiceClient::new(channel.clone()),
            project: ProjectServiceClient::new(channel),
        })
    }

    /// Connect to daemon at default address (localhost:50051)
    pub async fn connect_default() -> Result<Self> {
        Self::connect(&format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT)).await
    }

    /// Get mutable reference to SystemService client
    ///
    /// Provides access to:
    /// - HealthCheck
    /// - GetStatus
    /// - GetMetrics
    /// - SendRefreshSignal
    /// - NotifyServerStatus
    /// - PauseAllWatchers
    /// - ResumeAllWatchers
    pub fn system(&mut self) -> &mut SystemServiceClient<Channel> {
        &mut self.system
    }

    /// Get mutable reference to CollectionService client
    ///
    /// Provides access to:
    /// - CreateCollection
    /// - DeleteCollection
    /// - CreateCollectionAlias
    /// - DeleteCollectionAlias
    /// - RenameCollectionAlias
    pub fn collection(&mut self) -> &mut CollectionServiceClient<Channel> {
        &mut self.collection
    }

    /// Get mutable reference to DocumentService client
    ///
    /// Provides access to:
    /// - IngestText
    /// - UpdateText
    /// - DeleteText
    pub fn document(&mut self) -> &mut DocumentServiceClient<Channel> {
        &mut self.document
    }

    /// Get mutable reference to ProjectService client
    ///
    /// Provides access to:
    /// - RegisterProject
    /// - DeprioritizeProject
    /// - GetProjectStatus
    /// - ListProjects
    /// - Heartbeat
    pub fn project(&mut self) -> &mut ProjectServiceClient<Channel> {
        &mut self.project
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
        assert_eq!(DEFAULT_CONNECTION_TIMEOUT_SECS, 5);
    }
}
