//! gRPC client implementations with connection management and service discovery
//! Provides high-level client abstractions for all gRPC services

use crate::proto::*;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Connection pool for managing gRPC client connections
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<String, Channel>>>,
    default_timeout: Duration,
    connect_timeout: Duration,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            default_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
        }
    }

    /// Create a new connection pool with custom timeouts
    pub fn with_timeouts(default_timeout: Duration, connect_timeout: Duration) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            default_timeout,
            connect_timeout,
        }
    }

    /// Get or create a connection to the specified address
    pub async fn get_connection(&self, address: &str) -> Result<Channel> {
        // Try to get existing connection
        {
            let connections = self.connections.read().await;
            if let Some(channel) = connections.get(address) {
                debug!("Reusing existing connection to {}", address);
                return Ok(channel.clone());
            }
        }

        // Create new connection
        let endpoint = Endpoint::from_shared(address.to_string())?
            .timeout(self.default_timeout)
            .connect_timeout(self.connect_timeout);

        let channel = endpoint.connect().await
            .map_err(|e| anyhow!("Failed to connect to {}: {}", address, e))?;

        // Store the connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(address.to_string(), channel.clone());
        }

        info!("Created new connection to {}", address);
        Ok(channel)
    }

    /// Remove a connection from the pool
    pub async fn remove_connection(&self, address: &str) {
        let mut connections = self.connections.write().await;
        if connections.remove(address).is_some() {
            debug!("Removed connection to {}", address);
        }
    }

    /// Clear all connections
    pub async fn clear(&self) {
        let mut connections = self.connections.write().await;
        connections.clear();
        info!("Cleared all connections from pool");
    }

    /// Get the number of active connections
    pub async fn connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level gRPC client for workspace daemon services
#[derive(Debug, Clone)]
pub struct WorkspaceDaemonClient {
    pool: ConnectionPool,
    address: String,
}

impl WorkspaceDaemonClient {
    /// Create a new client
    pub fn new(address: String) -> Self {
        Self {
            pool: ConnectionPool::new(),
            address,
        }
    }

    /// Create a new client with custom connection pool
    pub fn with_pool(address: String, pool: ConnectionPool) -> Self {
        Self {
            pool,
            address,
        }
    }

    /// Get the connection channel
    async fn get_channel(&self) -> Result<Channel> {
        self.pool.get_connection(&self.address).await
    }

    // ================================
    // Document Processor Client
    // ================================

    /// Process a single document
    pub async fn process_document(
        &self,
        request: ProcessDocumentRequest,
    ) -> Result<ProcessDocumentResponse> {
        let channel = self.get_channel().await?;
        let mut client = document_processor_client::DocumentProcessorClient::new(channel);

        let response = client.process_document(Request::new(request)).await
            .map_err(|e| anyhow!("Document processing failed: {}", e))?;

        Ok(response.into_inner())
    }

    /// Get processing status
    pub async fn get_processing_status(
        &self,
        operation_id: String,
    ) -> Result<ProcessingStatusResponse> {
        let channel = self.get_channel().await?;
        let mut client = document_processor_client::DocumentProcessorClient::new(channel);

        let request = ProcessingStatusRequest { operation_id };
        let response = client.get_processing_status(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to get processing status: {}", e))?;

        Ok(response.into_inner())
    }

    /// Cancel processing operation
    pub async fn cancel_processing(&self, operation_id: String) -> Result<()> {
        let channel = self.get_channel().await?;
        let mut client = document_processor_client::DocumentProcessorClient::new(channel);

        let request = CancelProcessingRequest { operation_id };
        client.cancel_processing(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to cancel processing: {}", e))?;

        Ok(())
    }

    // ================================
    // Search Service Client
    // ================================

    /// Perform hybrid search
    pub async fn hybrid_search(
        &self,
        query: String,
        context: SearchContext,
        project_id: Option<String>,
        collection_names: Vec<String>,
        options: Option<SearchOptions>,
    ) -> Result<HybridSearchResponse> {
        let channel = self.get_channel().await?;
        let mut client = search_service_client::SearchServiceClient::new(channel);

        let request = HybridSearchRequest {
            query,
            context: context as i32,
            project_id: project_id.unwrap_or_default(),
            collection_names,
            options,
        };

        let response = client.hybrid_search(Request::new(request)).await
            .map_err(|e| anyhow!("Hybrid search failed: {}", e))?;

        Ok(response.into_inner())
    }

    /// Perform semantic search
    pub async fn semantic_search(
        &self,
        query: String,
        context: SearchContext,
        project_id: Option<String>,
        collection_names: Vec<String>,
        options: Option<SearchOptions>,
    ) -> Result<SemanticSearchResponse> {
        let channel = self.get_channel().await?;
        let mut client = search_service_client::SearchServiceClient::new(channel);

        let request = SemanticSearchRequest {
            query,
            context: context as i32,
            project_id: project_id.unwrap_or_default(),
            collection_names,
            options,
        };

        let response = client.semantic_search(Request::new(request)).await
            .map_err(|e| anyhow!("Semantic search failed: {}", e))?;

        Ok(response.into_inner())
    }

    /// Get search suggestions
    pub async fn get_suggestions(
        &self,
        partial_query: String,
        context: SearchContext,
        max_suggestions: i32,
        project_id: Option<String>,
    ) -> Result<SuggestionsResponse> {
        let channel = self.get_channel().await?;
        let mut client = search_service_client::SearchServiceClient::new(channel);

        let request = SuggestionsRequest {
            partial_query,
            context: context as i32,
            max_suggestions,
            project_id: project_id.unwrap_or_default(),
        };

        let response = client.get_suggestions(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to get suggestions: {}", e))?;

        Ok(response.into_inner())
    }

    // ================================
    // Memory Service Client
    // ================================

    /// Add document to memory
    pub async fn add_document(
        &self,
        file_path: String,
        collection_name: String,
        project_id: String,
        content: Option<DocumentContent>,
        metadata: HashMap<String, String>,
    ) -> Result<AddDocumentResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = AddDocumentRequest {
            file_path,
            collection_name,
            project_id,
            content,
            metadata,
        };

        let response = client.add_document(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to add document: {}", e))?;

        Ok(response.into_inner())
    }

    /// Update existing document
    pub async fn update_document(
        &self,
        document_id: String,
        content: Option<DocumentContent>,
        metadata: HashMap<String, String>,
    ) -> Result<UpdateDocumentResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = UpdateDocumentRequest {
            document_id,
            content,
            metadata,
        };

        let response = client.update_document(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to update document: {}", e))?;

        Ok(response.into_inner())
    }

    /// Remove document from memory
    pub async fn remove_document(
        &self,
        document_id: String,
        collection_name: String,
    ) -> Result<()> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = RemoveDocumentRequest {
            document_id,
            collection_name,
        };

        client.remove_document(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to remove document: {}", e))?;

        Ok(())
    }

    /// Get document metadata
    pub async fn get_document(
        &self,
        document_id: String,
        collection_name: String,
    ) -> Result<GetDocumentResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = GetDocumentRequest {
            document_id,
            collection_name,
        };

        let response = client.get_document(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to get document: {}", e))?;

        Ok(response.into_inner())
    }

    /// List documents with filtering
    pub async fn list_documents(
        &self,
        collection_name: String,
        project_id: String,
        limit: i32,
        offset: i32,
        filter: Option<DocumentFilter>,
    ) -> Result<ListDocumentsResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = ListDocumentsRequest {
            collection_name,
            project_id,
            limit,
            offset,
            filter,
        };

        let response = client.list_documents(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to list documents: {}", e))?;

        Ok(response.into_inner())
    }

    /// Create collection
    pub async fn create_collection(
        &self,
        collection_name: String,
        project_id: String,
        config: Option<CollectionConfig>,
    ) -> Result<CreateCollectionResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = CreateCollectionRequest {
            collection_name,
            project_id,
            config,
        };

        let response = client.create_collection(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to create collection: {}", e))?;

        Ok(response.into_inner())
    }

    /// Delete collection
    pub async fn delete_collection(
        &self,
        collection_name: String,
        project_id: String,
        force: bool,
    ) -> Result<()> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = DeleteCollectionRequest {
            collection_name,
            project_id,
            force,
        };

        client.delete_collection(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to delete collection: {}", e))?;

        Ok(())
    }

    /// List collections
    pub async fn list_collections(&self, project_id: String) -> Result<ListCollectionsResponse> {
        let channel = self.get_channel().await?;
        let mut client = memory_service_client::MemoryServiceClient::new(channel);

        let request = ListCollectionsRequest { project_id };

        let response = client.list_collections(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to list collections: {}", e))?;

        Ok(response.into_inner())
    }

    // ================================
    // System Service Client
    // ================================

    /// Health check
    pub async fn health_check(&self) -> Result<HealthCheckResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let response = client.health_check(Request::new(())).await
            .map_err(|e| anyhow!("Health check failed: {}", e))?;

        Ok(response.into_inner())
    }

    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatusResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let response = client.get_status(Request::new(())).await
            .map_err(|e| anyhow!("Failed to get status: {}", e))?;

        Ok(response.into_inner())
    }

    /// Get metrics
    pub async fn get_metrics(
        &self,
        since: Option<prost_types::Timestamp>,
        metric_names: Vec<String>,
    ) -> Result<MetricsResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let request = MetricsRequest {
            since,
            metric_names,
        };

        let response = client.get_metrics(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to get metrics: {}", e))?;

        Ok(response.into_inner())
    }

    /// Get configuration
    pub async fn get_config(&self) -> Result<ConfigResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let response = client.get_config(Request::new(())).await
            .map_err(|e| anyhow!("Failed to get config: {}", e))?;

        Ok(response.into_inner())
    }

    /// Detect project
    pub async fn detect_project(&self, path: String) -> Result<DetectProjectResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let request = DetectProjectRequest { path };

        let response = client.detect_project(Request::new(request)).await
            .map_err(|e| anyhow!("Failed to detect project: {}", e))?;

        Ok(response.into_inner())
    }

    /// List projects
    pub async fn list_projects(&self) -> Result<ListProjectsResponse> {
        let channel = self.get_channel().await?;
        let mut client = system_service_client::SystemServiceClient::new(channel);

        let response = client.list_projects(Request::new(())).await
            .map_err(|e| anyhow!("Failed to list projects: {}", e))?;

        Ok(response.into_inner())
    }

    // ================================
    // Connection Management
    // ================================

    /// Test connectivity to the server
    pub async fn test_connection(&self) -> Result<bool> {
        match self.health_check().await {
            Ok(response) => {
                let is_healthy = response.status == ServiceStatus::Healthy as i32;
                debug!("Connection test result: healthy={}", is_healthy);
                Ok(is_healthy)
            },
            Err(e) => {
                warn!("Connection test failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Disconnect and clear cached connections
    pub async fn disconnect(&self) {
        self.pool.clear().await;
        info!("Disconnected from {}", self.address);
    }

    /// Get connection statistics
    pub async fn connection_stats(&self) -> ConnectionStats {
        ConnectionStats {
            address: self.address.clone(),
            active_connections: self.pool.connection_count().await,
        }
    }
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub address: String,
    pub active_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    #[tokio::test]
    async fn test_connection_pool_basic_operations() {
        let pool = ConnectionPool::new();

        // Test connection count
        assert_eq!(pool.connection_count().await, 0);

        // Test clear
        pool.clear().await;
        assert_eq!(pool.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_workspace_daemon_client_creation() {
        let address = "http://127.0.0.1:50051".to_string();
        let client = WorkspaceDaemonClient::new(address.clone());

        let stats = client.connection_stats().await;
        assert_eq!(stats.address, address);
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_connection_pool_with_timeouts() {
        let pool = ConnectionPool::with_timeouts(
            Duration::from_secs(10),
            Duration::from_secs(5),
        );

        assert_eq!(pool.default_timeout, Duration::from_secs(10));
        assert_eq!(pool.connect_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_connection_stats() {
        let stats = ConnectionStats {
            address: "http://127.0.0.1:50051".to_string(),
            active_connections: 3,
        };

        assert_eq!(stats.address, "http://127.0.0.1:50051");
        assert_eq!(stats.active_connections, 3);
    }

    #[tokio::test]
    async fn test_client_with_custom_pool() {
        let address = "http://127.0.0.1:50051".to_string();
        let pool = ConnectionPool::with_timeouts(
            Duration::from_secs(15),
            Duration::from_secs(8),
        );

        let client = WorkspaceDaemonClient::with_pool(address.clone(), pool);

        let stats = client.connection_stats().await;
        assert_eq!(stats.address, address);
    }
}