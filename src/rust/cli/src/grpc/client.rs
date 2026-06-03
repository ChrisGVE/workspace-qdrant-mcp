//! DaemonClient wrapper
//!
//! Connects to memexd daemon and provides access to all gRPC services:
//! - SystemService: Health monitoring, status, lifecycle
//! - CollectionService: Collection CRUD, alias management
//! - DocumentService: Text ingestion
//! - ProjectService: Project registration, sessions
//! - GraphService: Code relationship graph queries
//! - QueueWriteService: Queue item lifecycle mutations
//! - WatchWriteService: Watch folder state mutations
//! - LibraryWriteService: Library management mutations
//! - TrackingWriteService: Observability and mirror mutations
//! - AdminWriteService: Administrative mutations

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::time::Duration;
use tonic::transport::Channel;

/// Generated client stubs, now sourced from the shared `wqm-proto` crate
/// (WI-c1, #82) instead of compiled in this crate's build.rs.
pub use wqm_proto::workspace_daemon;

use workspace_daemon::{
    admin_write_service_client::AdminWriteServiceClient,
    collection_service_client::CollectionServiceClient,
    document_service_client::DocumentServiceClient,
    embedding_service_client::EmbeddingServiceClient, graph_service_client::GraphServiceClient,
    library_write_service_client::LibraryWriteServiceClient,
    project_service_client::ProjectServiceClient,
    queue_write_service_client::QueueWriteServiceClient,
    system_service_client::SystemServiceClient,
    text_search_service_client::TextSearchServiceClient,
    tracking_write_service_client::TrackingWriteServiceClient,
    watch_write_service_client::WatchWriteServiceClient,
};

/// Default gRPC port for memexd daemon (canonical source: wqm_common::constants)
pub use wqm_common::constants::DEFAULT_GRPC_PORT;

/// Default connection timeout in seconds
pub const DEFAULT_CONNECTION_TIMEOUT_SECS: u64 = 5;

/// Client wrapper for memexd daemon gRPC services
pub struct DaemonClient {
    system: SystemServiceClient<Channel>,
    collection: CollectionServiceClient<Channel>,
    document: DocumentServiceClient<Channel>,
    project: ProjectServiceClient<Channel>,
    graph: GraphServiceClient<Channel>,
    queue_write: QueueWriteServiceClient<Channel>,
    watch_write: WatchWriteServiceClient<Channel>,
    library_write: LibraryWriteServiceClient<Channel>,
    tracking_write: TrackingWriteServiceClient<Channel>,
    admin_write: AdminWriteServiceClient<Channel>,
    embedding: EmbeddingServiceClient<Channel>,
    text_search: TextSearchServiceClient<Channel>,
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
            project: ProjectServiceClient::new(channel.clone()),
            graph: GraphServiceClient::new(channel.clone()),
            queue_write: QueueWriteServiceClient::new(channel.clone()),
            watch_write: WatchWriteServiceClient::new(channel.clone()),
            library_write: LibraryWriteServiceClient::new(channel.clone()),
            tracking_write: TrackingWriteServiceClient::new(channel.clone()),
            admin_write: AdminWriteServiceClient::new(channel.clone()),
            embedding: EmbeddingServiceClient::new(channel.clone()),
            text_search: TextSearchServiceClient::new(channel),
        })
    }

    /// Connect to daemon at the default address resolved from environment and
    /// the active cli-config.toml profile.
    ///
    /// Priority: `WQM_DAEMON_ADDR` > active profile daemon_address > built-in
    /// default (`http://127.0.0.1:50051`). Keeping the constant around lets
    /// tests and diagnostics quote the hard-coded fallback.
    pub async fn connect_default() -> Result<Self> {
        let addr = crate::config::resolve_daemon_address();
        if addr.is_empty() {
            Self::connect(&format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT)).await
        } else {
            Self::connect(&addr).await
        }
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

    /// Get mutable reference to GraphService client
    pub fn graph(&mut self) -> &mut GraphServiceClient<Channel> {
        &mut self.graph
    }

    /// Get mutable reference to QueueWriteService client
    pub fn queue_write(&mut self) -> &mut QueueWriteServiceClient<Channel> {
        &mut self.queue_write
    }

    /// Get mutable reference to WatchWriteService client
    pub fn watch_write(&mut self) -> &mut WatchWriteServiceClient<Channel> {
        &mut self.watch_write
    }

    /// Get mutable reference to LibraryWriteService client
    pub fn library_write(&mut self) -> &mut LibraryWriteServiceClient<Channel> {
        &mut self.library_write
    }

    /// Get mutable reference to TrackingWriteService client
    pub fn tracking_write(&mut self) -> &mut TrackingWriteServiceClient<Channel> {
        &mut self.tracking_write
    }

    /// Get mutable reference to AdminWriteService client
    pub fn admin_write(&mut self) -> &mut AdminWriteServiceClient<Channel> {
        &mut self.admin_write
    }

    /// Get mutable reference to EmbeddingService client
    ///
    /// Provides access to:
    /// - EmbedText
    /// - GenerateSparseVector
    pub fn embedding(&mut self) -> &mut EmbeddingServiceClient<Channel> {
        &mut self.embedding
    }

    /// Get mutable reference to TextSearchService client
    ///
    /// Provides access to:
    /// - Search
    /// - CountMatches
    pub fn text_search(&mut self) -> &mut TextSearchServiceClient<Channel> {
        &mut self.text_search
    }
}

/// Check that the daemon is running and return a connected client.
///
/// Returns a clear error message if the daemon is not available.
/// Use this before any write operation that requires the daemon.
pub async fn ensure_daemon_available() -> Result<DaemonClient> {
    DaemonClient::connect_default().await.map_err(|e| {
        anyhow::anyhow!(
            "Daemon not running. Start memexd to execute this command.\n\
             Hint: wqm service start\n\
             Cause: {}",
            e
        )
    })
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
