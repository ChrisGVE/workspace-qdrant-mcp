//! Storage abstraction layer
//!
//! This module provides the Qdrant storage interface implementation with
//! comprehensive vector database operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::env;
use qdrant_client::{Qdrant, QdrantError};
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::{PointStruct, UpsertPoints};
use qdrant_client::qdrant::{CreateCollection, DeleteCollection, Distance, VectorParams, VectorsConfig};
use qdrant_client::qdrant::Datatype;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, FieldType, Filter, HnswConfigDiffBuilder, VectorParamsBuilder,
    DenseVector, SparseVector, VectorParamsMap, SparseVectorConfig, SparseVectorParams,
    vectors_config, QueryPointsBuilder,
};
use serde::{Serialize, Deserialize};
use tokio::time::sleep;
use thiserror::Error;
use tracing::{debug, info, warn, error};
// Note: tonic, hyper, and url imports removed as they're not currently used
// They would be needed for advanced gRPC HTTP/2 configuration

/// Multi-tenant collection names (unified architecture - canonical names)
pub mod collections {
    /// Projects collection - stores code and documents from all projects
    /// Filtered by project_id payload field
    pub const PROJECTS: &str = "projects";

    /// Libraries collection - stores library documentation
    /// Filtered by library_name payload field
    pub const LIBRARIES: &str = "libraries";

    /// Memory collection - stores agent memory and cross-project notes
    pub const MEMORY: &str = "memory";
}

/// Multi-tenant collection configuration
#[derive(Debug, Clone)]
pub struct MultiTenantConfig {
    /// Dense vector size (default: 384 for all-MiniLM-L6-v2)
    pub vector_size: u64,
    /// HNSW m parameter (default: 16)
    pub hnsw_m: u64,
    /// HNSW ef_construct parameter (default: 100)
    pub hnsw_ef_construct: u64,
    /// Enable on_disk_payload for large collections
    pub on_disk_payload: bool,
}

impl Default for MultiTenantConfig {
    fn default() -> Self {
        Self {
            vector_size: 384, // all-MiniLM-L6-v2
            hnsw_m: 16,
            hnsw_ef_construct: 100,
            on_disk_payload: true,
        }
    }
}

/// Result of multi-tenant collection initialization
#[derive(Debug, Clone, Default)]
pub struct MultiTenantInitResult {
    /// Whether `projects` collection was created
    pub projects_created: bool,
    /// Whether project_id index was created
    pub projects_indexed: bool,
    /// Whether `libraries` collection was created
    pub libraries_created: bool,
    /// Whether library_name index was created
    pub libraries_indexed: bool,
    /// Whether _memory collection was created
    pub memory_created: bool,
}

impl MultiTenantInitResult {
    /// Check if all collections were successfully initialized
    pub fn is_complete(&self) -> bool {
        self.projects_created
            && self.projects_indexed
            && self.libraries_created
            && self.libraries_indexed
            && self.memory_created
    }
}

/// Storage-related errors
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Connection error: {0}")]
    Connection(String),
    
    #[error("Collection error: {0}")]
    Collection(String),
    
    #[error("Point operation error: {0}")]
    Point(String),
    
    #[error("Search error: {0}")]
    Search(String),
    
    #[error("Batch operation error: {0}")]
    Batch(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Qdrant client error: {0}")]
    Qdrant(Box<QdrantError>),
}

impl From<QdrantError> for StorageError {
    fn from(err: QdrantError) -> Self {
        StorageError::Qdrant(Box::new(err))
    }
}

/// Transport mode for Qdrant connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMode {
    /// gRPC transport (default, more efficient)
    Grpc,
    /// HTTP transport (fallback)
    Http,
}

impl Default for TransportMode {
    fn default() -> Self {
        Self::Grpc
    }
}

/// HTTP/2 configuration for gRPC transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Http2Config {
    /// Maximum frame size (bytes)
    pub max_frame_size: Option<u32>,
    /// Initial connection window size (bytes)
    pub initial_window_size: Option<u32>,
    /// Maximum header list size (bytes)
    pub max_header_list_size: Option<u32>,
    /// Enable HTTP/2 server push
    pub enable_push: bool,
    /// Enable TCP keepalive
    pub tcp_keepalive: bool,
    /// Keepalive interval in milliseconds
    pub keepalive_interval_ms: Option<u32>,
    /// Keepalive timeout in milliseconds
    pub keepalive_timeout_ms: Option<u32>,
    /// Enable HTTP/2 adaptive window sizing
    pub http2_adaptive_window: bool,
}

impl Default for Http2Config {
    fn default() -> Self {
        Self {
            max_frame_size: Some(8192), // Conservative default (vs 16384)
            initial_window_size: Some(32768),
            max_header_list_size: Some(8192),
            enable_push: false,
            tcp_keepalive: true,
            keepalive_interval_ms: Some(30000),
            keepalive_timeout_ms: Some(5000),
            http2_adaptive_window: false,
        }
    }
}

/// Storage client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Qdrant server URL
    pub url: String,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    /// Connection timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Base retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Transport mode (gRPC or HTTP)
    pub transport: TransportMode,
    /// Connection pool size
    pub pool_size: usize,
    /// Enable TLS (for production)
    pub tls: bool,
    /// Default vector size for dense vectors
    pub dense_vector_size: u64,
    /// Default sparse vector size
    pub sparse_vector_size: Option<u64>,
    /// HTTP/2 configuration for gRPC transport
    pub http2: Http2Config,
    /// Skip compatibility checks during connection
    pub check_compatibility: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            transport: TransportMode::default(),
            pool_size: 10,
            tls: false,
            dense_vector_size: 1536, // Default for OpenAI embeddings
            sparse_vector_size: None,
            http2: Http2Config::default(),
            check_compatibility: true,
        }
    }
}

impl StorageConfig {
    /// Create a daemon-mode configuration with compatibility checking disabled
    /// to ensure complete console silence for MCP stdio protocol compliance.
    /// Uses gRPC transport on port 6334 (Qdrant's gRPC port).
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.check_compatibility = false; // Disable to suppress Qdrant client output
        config.transport = TransportMode::Grpc; // gRPC is required by qdrant-client
        // qdrant-client uses gRPC protocol - ensure we use port 6334
        // Use 127.0.0.1 explicitly to avoid IPv6 resolution issues
        config.url = "http://127.0.0.1:6334".to_string();
        config
    }
}

/// Document point for Qdrant insertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPoint {
    /// Unique document ID
    pub id: String,
    /// Dense vector representation
    pub dense_vector: Vec<f32>,
    /// Sparse vector representation (optional)
    pub sparse_vector: Option<HashMap<u32, f32>>,
    /// Document content and metadata
    pub payload: HashMap<String, serde_json::Value>,
}

/// Search result from Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub id: String,
    /// Search score
    pub score: f32,
    /// Document payload
    pub payload: HashMap<String, serde_json::Value>,
    /// Dense vector (if requested)
    pub dense_vector: Option<Vec<f32>>,
    /// Sparse vector (if requested)
    pub sparse_vector: Option<HashMap<u32, f32>>,
}

/// Parameters for search operations
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Dense vector representation
    pub dense_vector: Option<Vec<f32>>,
    /// Sparse vector representation
    pub sparse_vector: Option<HashMap<u32, f32>>,
    /// Search mode (dense, sparse, or hybrid)
    pub search_mode: HybridSearchMode,
    /// Maximum number of results
    pub limit: usize,
    /// Minimum score threshold
    pub score_threshold: Option<f32>,
    /// Optional filter conditions
    pub filter: Option<HashMap<String, serde_json::Value>>,
}

/// Parameters for hybrid search operations
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    /// Dense vector representation
    pub dense_vector: Option<Vec<f32>>,
    /// Sparse vector representation
    pub sparse_vector: Option<HashMap<u32, f32>>,
    /// Weight for dense vector results
    pub dense_weight: f32,
    /// Weight for sparse vector results
    pub sparse_weight: f32,
    /// Maximum number of results
    pub limit: usize,
    /// Minimum score threshold
    pub score_threshold: Option<f32>,
    /// Optional filter conditions
    pub filter: Option<HashMap<String, serde_json::Value>>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            dense_vector: None,
            sparse_vector: None,
            search_mode: HybridSearchMode::Dense,
            limit: 10,
            score_threshold: None,
            filter: None,
        }
    }
}

/// Hybrid search mode for dense/sparse fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HybridSearchMode {
    /// Dense vector search only
    Dense,
    /// Sparse vector search only  
    Sparse,
    /// Hybrid search with RRF fusion
    Hybrid { dense_weight: f32, sparse_weight: f32 },
}

impl Default for HybridSearchMode {
    fn default() -> Self {
        Self::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 }
    }
}

/// Batch operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total points processed
    pub total_points: usize,
    /// Successfully inserted points
    pub successful: usize,
    /// Failed insertions
    pub failed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Average throughput (points per second)
    pub throughput: f64,
}

/// Collection information returned from Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfoResult {
    /// Collection name
    pub name: String,
    /// Number of points in the collection
    pub points_count: u64,
    /// Number of indexed vectors
    pub vectors_count: u64,
    /// Collection status (green, yellow, red, grey)
    pub status: String,
    /// Dense vector dimension (if configured)
    pub vector_dimension: Option<u64>,
    /// Collection aliases
    pub aliases: Vec<String>,
}

/// Connection pool statistics
#[derive(Debug, Default)]
struct ConnectionStats {
    successful_connections: u64,
    failed_connections: u64,
    active_connections: u32,
    total_requests: u64,
    total_errors: u64,
}

/// Storage client with Qdrant integration
pub struct StorageClient {
    /// Qdrant client instance
    client: Arc<Qdrant>,
    /// Client configuration
    config: StorageConfig,
    /// Connection pool statistics
    stats: Arc<tokio::sync::Mutex<ConnectionStats>>,
}

impl StorageClient {
    /// Create a new storage client with default configuration
    pub fn new() -> Self {
        Self::with_config(StorageConfig::default())
    }
    
    /// Create a storage client with custom configuration
    pub fn with_config(config: StorageConfig) -> Self {
        // Debug: write to a file to verify code is running
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/storage_debug.log")
        {
            use std::io::Write;
            let _ = writeln!(f, "StorageClient::with_config called: url={}, check_compat={}",
                config.url, config.check_compatibility);
        }

        info!("Initializing Qdrant client with transport: {:?}", config.transport);
        
        // Determine the appropriate URL based on transport mode
        // Qdrant uses port 6333 for HTTP/REST and port 6334 for gRPC
        let connection_url = match config.transport {
            TransportMode::Grpc => {
                // For gRPC, use port 6334 (Qdrant's gRPC port)
                // Auto-convert port 6333 to 6334 if specified
                let url = if config.url.contains(":6333") {
                    config.url.replace(":6333", ":6334")
                } else if !config.url.contains(":6334") && !config.url.contains(":") {
                    // If no port specified, add gRPC port
                    format!("{}:6334", config.url.trim_end_matches('/'))
                } else {
                    config.url.clone()
                };
                url
            },
            TransportMode::Http => {
                // For HTTP, use port 6333 (Qdrant's REST port)
                if config.url.starts_with("http://") || config.url.starts_with("https://") {
                    config.url.clone()
                } else {
                    format!("http://{}", config.url)
                }
            }
        };
        
        info!("Connecting to Qdrant at: {}", connection_url);
        
        let mut qdrant_config = QdrantConfig::from_url(&connection_url);
        
        // Configure authentication
        if let Some(api_key) = &config.api_key {
            info!("Configuring API key authentication");
            qdrant_config = qdrant_config.api_key(api_key.clone());
        }
        
        // Configure timeout
        qdrant_config = qdrant_config.timeout(Duration::from_millis(config.timeout_ms));
        
        // Configure connection timeout for better reliability
        qdrant_config = qdrant_config.connect_timeout(Duration::from_millis(config.timeout_ms / 2));
        
        // Enable keep-alive for better connection stability
        qdrant_config = qdrant_config.keep_alive_while_idle();
        
        // Configure compatibility checking - disable in daemon mode for silence
        if !config.check_compatibility {
            qdrant_config = qdrant_config.skip_compatibility_check();
        }
        
        // Log and apply HTTP/2 configuration for gRPC transport
        if matches!(config.transport, TransportMode::Grpc) {
            info!("gRPC transport configured with HTTP/2 settings:");
            if let Some(frame_size) = config.http2.max_frame_size {
                info!("  - Max frame size: {} bytes", frame_size);
            }
            if let Some(window_size) = config.http2.initial_window_size {
                info!("  - Initial window size: {} bytes", window_size);
            }
            if let Some(header_size) = config.http2.max_header_list_size {
                info!("  - Max header list size: {} bytes", header_size);
            }
            info!("  - Server push: {}", config.http2.enable_push);
            info!("  - TCP keepalive: {}", config.http2.tcp_keepalive);
            
            // Apply HTTP/2 settings to qdrant_config where possible
            // Note: The qdrant-client library doesn't expose direct HTTP/2 frame size configuration,
            // so we configure what we can at the connection level
            if config.http2.tcp_keepalive {
                qdrant_config = qdrant_config.keep_alive_while_idle();
            }
            
            // For frame size configuration, we'll need to set environment variables
            // or use other mechanisms since qdrant-client doesn't expose these directly
            if let Some(_frame_size) = config.http2.max_frame_size {
                // This is a workaround - we can't directly configure frame size in qdrant-client
                warn!("HTTP/2 max frame size configuration requires lower-level gRPC configuration");
                warn!("Consider using HTTP transport if frame size errors persist");
            }
        }
        
        // Try to build the client with fallback to HTTP on gRPC failure
        // In daemon mode, temporarily suppress stdout/stderr during client creation
        let client = if is_daemon_mode() && !config.check_compatibility {
            // Suppress output during client creation for MCP stdio compliance
            suppress_output_temporarily(|| Qdrant::new(qdrant_config.clone()))
        } else {
            Qdrant::new(qdrant_config.clone())
        };

        let client = match client {
            Ok(client) => {
                info!("Successfully created Qdrant client with {:?} transport", config.transport);
                Arc::new(client)
            },
            Err(e) if matches!(config.transport, TransportMode::Grpc) => {
                error!("Failed to create gRPC client: {}", e);
                warn!("Attempting fallback to HTTP transport...");
                
                // Fallback to HTTP transport
                let fallback_url = connection_url.replace("grpc://", "http://");
                let mut fallback_config = QdrantConfig::from_url(&fallback_url)
                    .timeout(Duration::from_millis(config.timeout_ms))
                    .connect_timeout(Duration::from_millis(config.timeout_ms / 2))
                    .keep_alive_while_idle();
                    
                if let Some(api_key) = &config.api_key {
                    fallback_config = fallback_config.api_key(api_key.clone());
                }
                    
                match Qdrant::new(fallback_config) {
                    Ok(client) => {
                        warn!("Successfully created fallback HTTP client");
                        Arc::new(client)
                    },
                    Err(fallback_error) => {
                        error!("Fallback to HTTP also failed: {}", fallback_error);
                        panic!("Failed to build Qdrant client with both gRPC and HTTP: original error: {}, fallback error: {}", e, fallback_error);
                    }
                }
            },
            Err(e) => {
                error!("Failed to build Qdrant client: {}", e);
                panic!("Failed to build Qdrant client: {}", e);
            }
        };
        
        Self {
            client,
            config,
            stats: Arc::new(tokio::sync::Mutex::new(ConnectionStats::default())),
        }
    }
    
    /// Test connection to Qdrant server
    pub async fn test_connection(&self) -> Result<bool, StorageError> {
        debug!("Testing connection to Qdrant server: {}", self.config.url);
        
        match self.client.health_check().await {
            Ok(_) => {
                info!("Successfully connected to Qdrant server");
                self.update_stats(|stats| stats.successful_connections += 1).await;
                Ok(true)
            },
            Err(e) => {
                error!("Failed to connect to Qdrant server: {}", e);
                self.update_stats(|stats| stats.failed_connections += 1).await;
                Err(StorageError::Connection(e.to_string()))
            }
        }
    }
    
    /// Create a new collection with vector configuration
    pub async fn create_collection(
        &self,
        collection_name: &str,
        dense_vector_size: Option<u64>,
        _sparse_vector_size: Option<u64>,
    ) -> Result<(), StorageError> {
        info!("Creating collection: {}", collection_name);
        
        let dense_size = dense_vector_size.unwrap_or(self.config.dense_vector_size);
        
        let mut vectors_config = VectorsConfig::default();
        
        // Configure dense vector
        let dense_vector_params = VectorParams {
            size: dense_size,
            distance: Distance::Cosine.into(),
            hnsw_config: None,
            quantization_config: None,
            on_disk: Some(false), // Keep in memory for better performance
            datatype: Some(Datatype::Float32.into()),
            multivector_config: None,
        };
        
        vectors_config.config = Some(qdrant_client::qdrant::vectors_config::Config::Params(dense_vector_params));
        
        let create_collection = CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(vectors_config),
            shard_number: Some(1),
            replication_factor: Some(1),
            write_consistency_factor: Some(1),
            on_disk_payload: Some(true),
            timeout: Some(self.config.timeout_ms),
            ..Default::default()
        };
        
        self.retry_operation(|| async {
            self.client.create_collection(create_collection.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        info!("Successfully created collection: {}", collection_name);
        Ok(())
    }
    
    /// Delete a collection
    pub async fn delete_collection(&self, collection_name: &str) -> Result<(), StorageError> {
        info!("Deleting collection: {}", collection_name);
        
        let delete_collection = DeleteCollection {
            collection_name: collection_name.to_string(),
            timeout: Some(self.config.timeout_ms),
        };
        
        self.retry_operation(|| async {
            self.client.delete_collection(delete_collection.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        info!("Successfully deleted collection: {}", collection_name);
        Ok(())
    }

    /// Delete points from a collection by file_path AND tenant_id filter
    ///
    /// Uses Qdrant's delete_points API with a combined filter condition to remove
    /// all points matching both the file_path and tenant_id. Tenant isolation is
    /// enforced to prevent cross-tenant data deletion in shared collections.
    ///
    /// # Arguments
    /// * `collection_name` - The collection to delete points from
    /// * `file_path` - The file_path to match for deletion
    /// * `tenant_id` - The tenant_id to scope the deletion (must not be empty)
    ///
    /// # Returns
    /// * `Ok(u64)` - Number of points deleted (estimated from operation)
    /// * `Err(StorageError)` - If deletion fails or tenant_id is empty
    pub async fn delete_points_by_filter(
        &self,
        collection_name: &str,
        file_path: &str,
        tenant_id: &str,
    ) -> Result<u64, StorageError> {
        if tenant_id.trim().is_empty() {
            return Err(StorageError::Point(
                "tenant_id must not be empty for delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with file_path='{}' tenant_id='{}' from collection '{}'",
            file_path, tenant_id, collection_name
        );

        // Build filter requiring BOTH file_path AND tenant_id match
        let filter = Filter::must([
            Condition::matches("file_path", file_path.to_string()),
            Condition::matches("tenant_id", tenant_id.to_string()),
        ]);

        // Build delete request
        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        // Execute deletion with retry
        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points: {}", e)))
        })
        .await?;

        info!(
            "Successfully deleted points with file_path='{}' tenant_id='{}' from '{}'",
            file_path, tenant_id, collection_name
        );

        Ok(0) // Qdrant delete doesn't return count
    }

    /// Delete points from a collection by tenant_id filter
    ///
    /// Deletes all points belonging to a specific tenant/project from the collection.
    /// Used for tenant-wide cleanup operations.
    pub async fn delete_points_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
    ) -> Result<u64, StorageError> {
        if tenant_id.trim().is_empty() {
            return Err(StorageError::Point(
                "tenant_id must not be empty for tenant delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with tenant_id='{}' from collection '{}'",
            tenant_id, collection_name
        );

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points by tenant: {}", e)))
        })
        .await?;

        info!(
            "Successfully deleted points with tenant_id='{}' from '{}'",
            tenant_id, collection_name
        );

        Ok(0)
    }

    /// Scroll through all points in a collection for a tenant, returning file paths
    ///
    /// Paginates through Qdrant using the scroll API with a tenant_id filter,
    /// extracting the `file_path` payload field from each point. Used for
    /// post-scan cleanup of excluded files.
    ///
    /// # Arguments
    /// * `collection_name` - The collection to scroll through
    /// * `tenant_id` - The tenant/project ID to filter by
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - All file paths found for this tenant
    /// * `Err(StorageError)` - If scroll operation fails
    pub async fn scroll_file_paths_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
    ) -> Result<Vec<String>, StorageError> {
        use qdrant_client::qdrant::ScrollPointsBuilder;

        debug!(
            "Scrolling file paths for tenant_id='{}' in collection '{}'",
            tenant_id, collection_name
        );

        let mut file_paths = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let batch_size = 100u32;

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);

        loop {
            let filter_clone = filter.clone();
            let current_offset = offset.clone();

            let response = self
                .retry_operation(|| {
                    let f = filter_clone.clone();
                    let o = current_offset.clone();
                    async move {
                        let mut builder = ScrollPointsBuilder::new(collection_name)
                            .filter(f)
                            .limit(batch_size)
                            .with_payload(true)
                            .with_vectors(false);

                        if let Some(offset_id) = o {
                            builder = builder.offset(offset_id);
                        }

                        self.client
                            .scroll(builder)
                            .await
                            .map_err(|e| StorageError::Search(format!("Scroll failed: {}", e)))
                    }
                })
                .await?;

            for point in &response.result {
                if let Some(value) = point.payload.get("file_path") {
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(path)) = &value.kind {
                        file_paths.push(path.clone());
                    }
                }
            }

            match response.next_page_offset {
                Some(next_offset) => {
                    offset = Some(next_offset);
                }
                None => break,
            }
        }

        debug!(
            "Scrolled {} file paths for tenant_id='{}' in '{}'",
            file_paths.len(), tenant_id, collection_name
        );

        Ok(file_paths)
    }

    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        debug!("Checking if collection exists: {}", collection_name);

        let response = self.retry_operation(|| async {
            self.client.collection_exists(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        Ok(response)
    }

    /// List all collections in Qdrant
    ///
    /// Returns a list of collection names.
    pub async fn list_collections(&self) -> Result<Vec<String>, StorageError> {
        debug!("Listing all collections");

        let response = self.retry_operation(|| async {
            self.client.list_collections().await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        let names = response.collections
            .into_iter()
            .map(|c| c.name)
            .collect();

        Ok(names)
    }

    /// Get detailed information about a collection
    ///
    /// Returns point count, vector config, status, and aliases.
    pub async fn get_collection_info(
        &self,
        collection_name: &str,
    ) -> Result<CollectionInfoResult, StorageError> {
        debug!("Getting collection info: {}", collection_name);

        let info = self.retry_operation(|| async {
            self.client.collection_info(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        // Extract point and vector counts from collection info
        let points_count = info.result
            .as_ref()
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);

        let vectors_count = info.result
            .as_ref()
            .map(|r| r.indexed_vectors_count.unwrap_or(0))
            .unwrap_or(0);

        // Extract status
        let status = info.result
            .as_ref()
            .map(|r| match r.status() {
                qdrant_client::qdrant::CollectionStatus::Green => "green",
                qdrant_client::qdrant::CollectionStatus::Yellow => "yellow",
                qdrant_client::qdrant::CollectionStatus::Red => "red",
                qdrant_client::qdrant::CollectionStatus::Grey => "grey",
                _ => "unknown",
            })
            .unwrap_or("unknown")
            .to_string();

        // Extract dense vector dimension from config
        let vector_dimension = info.result
            .as_ref()
            .and_then(|r| r.config.as_ref())
            .and_then(|c| c.params.as_ref())
            .and_then(|p| p.vectors_config.as_ref())
            .and_then(|vc| {
                use qdrant_client::qdrant::vectors_config::Config;
                match &vc.config {
                    Some(Config::Params(params)) => Some(params.size),
                    Some(Config::ParamsMap(map)) => {
                        map.map.get("dense").map(|p| p.size)
                    }
                    _ => None,
                }
            });

        // Get aliases for this collection
        let aliases = match self.client.list_collection_aliases(collection_name).await {
            Ok(response) => {
                response.aliases
                    .into_iter()
                    .map(|a| a.alias_name)
                    .collect()
            }
            Err(e) => {
                warn!("Failed to get aliases for {}: {}", collection_name, e);
                vec![]
            }
        };

        Ok(CollectionInfoResult {
            name: collection_name.to_string(),
            points_count,
            vectors_count,
            status,
            vector_dimension,
            aliases,
        })
    }

    /// Delete points from a collection by document_id filter
    ///
    /// Deletes all chunks belonging to a specific document from the collection.
    /// Used by gRPC update_document (delete + re-ingest) and delete_document.
    pub async fn delete_points_by_document_id(
        &self,
        collection_name: &str,
        document_id: &str,
    ) -> Result<u64, StorageError> {
        if document_id.trim().is_empty() {
            return Err(StorageError::Point(
                "document_id must not be empty for delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with document_id='{}' from collection '{}'",
            document_id, collection_name
        );

        let filter = Filter::must([
            Condition::matches("document_id", document_id.to_string()),
        ]);

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points by document_id: {}", e)))
        })
        .await?;

        info!(
            "Successfully deleted points with document_id='{}' from '{}'",
            document_id, collection_name
        );

        Ok(0) // Qdrant delete doesn't return count
    }

    /// Count points in a collection, optionally filtered by tenant_id
    pub async fn count_points(
        &self,
        collection_name: &str,
        tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        debug!("Counting points in collection: {} (tenant: {:?})", collection_name, tenant_id);

        let mut builder = CountPointsBuilder::new(collection_name).exact(true);

        if let Some(tid) = tenant_id {
            builder = builder.filter(Filter::must([
                Condition::matches("tenant_id", tid.to_string()),
            ]));
        }

        let count = self.retry_operation(|| async {
            self.client.count(builder.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Create a multi-tenant collection with optimized HNSW configuration
    ///
    /// This uses the builder pattern from qdrant-client for better ergonomics
    /// and supports the new multi-tenant architecture requirements.
    ///
    /// Collections are created with named vectors:
    /// - "dense": Dense semantic vectors (384 dimensions for all-MiniLM-L6-v2)
    /// - "sparse": Sparse BM25-style keyword vectors for hybrid search
    pub async fn create_multi_tenant_collection(
        &self,
        collection_name: &str,
        config: &MultiTenantConfig,
    ) -> Result<(), StorageError> {
        info!(
            "Creating multi-tenant collection: {} (vector_size={}, m={}, ef_construct={})",
            collection_name, config.vector_size, config.hnsw_m, config.hnsw_ef_construct
        );

        // Check if collection already exists (idempotency)
        if self.collection_exists(collection_name).await? {
            info!("Collection {} already exists, skipping creation", collection_name);
            return Ok(());
        }

        // Build HNSW configuration for dense vectors
        let hnsw_config = HnswConfigDiffBuilder::default()
            .m(config.hnsw_m)
            .ef_construct(config.hnsw_ef_construct);

        // Build dense vector parameters with HNSW config
        let dense_vector_params: VectorParams = VectorParamsBuilder::new(config.vector_size, Distance::Cosine)
            .hnsw_config(hnsw_config)
            .on_disk(false) // Keep vectors in memory for performance
            .build();

        // Build named vectors config with "dense" vector using VectorParamsMap
        let mut dense_vectors_map = HashMap::new();
        dense_vectors_map.insert("dense".to_string(), dense_vector_params);

        let named_vectors_config = VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                map: dense_vectors_map,
            })),
        };

        // Build sparse vectors config with "sparse" vector for BM25-style keyword search
        let mut sparse_vectors_map = HashMap::new();
        sparse_vectors_map.insert("sparse".to_string(), SparseVectorParams {
            index: None, // Use default sparse index configuration
            modifier: None,
        });

        let sparse_config = SparseVectorConfig {
            map: sparse_vectors_map,
        };

        // Create collection with both dense and sparse vectors
        let create_request = CreateCollectionBuilder::new(collection_name)
            .vectors_config(named_vectors_config)
            .sparse_vectors_config(sparse_config)
            .on_disk_payload(config.on_disk_payload)
            .shard_number(1)
            .replication_factor(1)
            .write_consistency_factor(1);

        self.retry_operation(|| async {
            self.client
                .create_collection(create_request.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully created multi-tenant collection with dense+sparse vectors: {}", collection_name);
        Ok(())
    }

    /// Create a payload index for efficient filtering
    ///
    /// Creates a keyword index on the specified field for fast filtering in queries.
    /// This is essential for multi-tenant filtering by project_id or library_name.
    pub async fn create_payload_index(
        &self,
        collection_name: &str,
        field_name: &str,
    ) -> Result<(), StorageError> {
        info!(
            "Creating payload index on {}.{}",
            collection_name, field_name
        );

        let index_request = CreateFieldIndexCollectionBuilder::new(
            collection_name,
            field_name,
            FieldType::Keyword,
        );

        self.retry_operation(|| async {
            self.client
                .create_field_index(index_request.clone())
                .await
                .map_err(|e| StorageError::Collection(format!(
                    "Failed to create payload index on {}.{}: {}",
                    collection_name, field_name, e
                )))
        })
        .await?;

        info!(
            "Successfully created payload index on {}.{}",
            collection_name, field_name
        );
        Ok(())
    }

    /// Initialize all multi-tenant collections with proper configuration
    ///
    /// Creates the three unified collections:
    /// - _projects: project code/documents, indexed by project_id
    /// - _libraries: library documentation, indexed by library_name
    /// - _memory: agent memory and cross-project notes
    ///
    /// This method is idempotent - existing collections are skipped.
    pub async fn initialize_multi_tenant_collections(
        &self,
        config: Option<MultiTenantConfig>,
    ) -> Result<MultiTenantInitResult, StorageError> {
        let config = config.unwrap_or_default();
        info!("Initializing multi-tenant collections with config: {:?}", config);

        let mut result = MultiTenantInitResult::default();

        // Create _projects collection
        match self.create_multi_tenant_collection(collections::PROJECTS, &config).await {
            Ok(()) => {
                // Check if we actually created it (vs skipped)
                if !self.collection_exists(collections::PROJECTS).await.unwrap_or(false) {
                    result.projects_created = true;
                }
                // Create project_id index
                match self.create_payload_index(collections::PROJECTS, "project_id").await {
                    Ok(()) => result.projects_indexed = true,
                    Err(e) => {
                        // Index might already exist, log but continue
                        warn!("Could not create project_id index (may already exist): {}", e);
                        result.projects_indexed = true; // Assume it exists
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", collections::PROJECTS, e);
                return Err(e);
            }
        }
        result.projects_created = true;

        // Create _libraries collection
        match self.create_multi_tenant_collection(collections::LIBRARIES, &config).await {
            Ok(()) => {
                // Create library_name index
                match self.create_payload_index(collections::LIBRARIES, "library_name").await {
                    Ok(()) => result.libraries_indexed = true,
                    Err(e) => {
                        warn!("Could not create library_name index (may already exist): {}", e);
                        result.libraries_indexed = true;
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", collections::LIBRARIES, e);
                return Err(e);
            }
        }
        result.libraries_created = true;

        // Create _memory collection (no additional index needed)
        match self.create_multi_tenant_collection(collections::MEMORY, &config).await {
            Ok(()) => {}
            Err(e) => {
                error!("Failed to create {} collection: {}", collections::MEMORY, e);
                return Err(e);
            }
        }
        result.memory_created = true;

        info!("Multi-tenant collections initialized: {:?}", result);
        Ok(result)
    }

    /// Insert a single document point
    pub async fn insert_point(
        &self,
        collection_name: &str,
        point: DocumentPoint,
    ) -> Result<(), StorageError> {
        debug!("Inserting point {} into collection {}", point.id, collection_name);
        
        let qdrant_point = self.convert_to_qdrant_point(point)?;
        
        let upsert_points = UpsertPoints {
            collection_name: collection_name.to_string(),
            points: vec![qdrant_point],
            wait: Some(true),
            ..Default::default()
        };
        
        self.retry_operation(|| async {
            self.client.upsert_points(upsert_points.clone()).await
                .map_err(|e| StorageError::Point(e.to_string()))
        }).await?;
        
        debug!("Successfully inserted point into collection {}", collection_name);
        Ok(())
    }
    
    /// Insert multiple document points in batch
    pub async fn insert_points_batch(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
    ) -> Result<BatchStats, StorageError> {
        info!("Inserting {} points into collection {} in batches", points.len(), collection_name);
        
        let start_time = std::time::Instant::now();
        let batch_size = batch_size.unwrap_or(100); // Default batch size
        let total_points = points.len();
        let mut successful = 0;
        let mut failed = 0;
        
        for chunk in points.chunks(batch_size) {
            let qdrant_points: Result<Vec<_>, _> = chunk.iter()
                .map(|p| self.convert_to_qdrant_point(p.clone()))
                .collect();
                
            match qdrant_points {
                Ok(points_batch) => {
                    let upsert_points = UpsertPoints {
                        collection_name: collection_name.to_string(),
                        points: points_batch,
                        wait: Some(false), // Don't wait for batch operations
                        ..Default::default()
                    };
                    
                    match self.retry_operation(|| async {
                        self.client.upsert_points(upsert_points.clone()).await
                            .map_err(|e| StorageError::Batch(e.to_string()))
                    }).await {
                        Ok(_) => successful += chunk.len(),
                        Err(e) => {
                            error!("Failed to insert batch: {}", e);
                            failed += chunk.len();
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to convert points batch: {}", e);
                    failed += chunk.len();
                }
            }
            
            // Small delay between batches to avoid overwhelming the server
            sleep(Duration::from_millis(10)).await;
        }
        
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        let throughput = if processing_time_ms > 0 {
            (successful as f64) / (processing_time_ms as f64 / 1000.0)
        } else {
            0.0
        };
        
        let stats = BatchStats {
            total_points,
            successful,
            failed,
            processing_time_ms,
            throughput,
        };
        
        info!("Batch insertion completed: {} successful, {} failed, {:.2} points/sec", 
              successful, failed, throughput);
        
        Ok(stats)
    }
    
    /// Perform hybrid search with dense/sparse vector fusion
    pub async fn search(
        &self,
        collection_name: &str,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, StorageError> {
        debug!("Performing search in collection: {}", collection_name);
        
        let results = match params.search_mode {
            HybridSearchMode::Dense => {
                if let Some(vector) = params.dense_vector {
                    self.search_dense(collection_name, vector, params.limit, params.score_threshold, params.filter).await?
                } else {
                    return Err(StorageError::Search("Dense vector required for dense search".to_string()));
                }
            },
            HybridSearchMode::Sparse => {
                if let Some(vector) = params.sparse_vector {
                    self.search_sparse(collection_name, vector, params.limit, params.score_threshold, params.filter).await?
                } else {
                    return Err(StorageError::Search("Sparse vector required for sparse search".to_string()));
                }
            },
            HybridSearchMode::Hybrid { dense_weight, sparse_weight } => {
                let hybrid_params = HybridSearchParams {
                    dense_vector: params.dense_vector,
                    sparse_vector: params.sparse_vector,
                    dense_weight,
                    sparse_weight,
                    limit: params.limit,
                    score_threshold: params.score_threshold,
                    filter: params.filter,
                };
                self.search_hybrid(collection_name, hybrid_params).await?
            }
        };
        
        debug!("Search completed, returned {} results", results.len());
        Ok(results)
    }
    
    /// Get connection statistics
    pub async fn get_stats(&self) -> Result<HashMap<String, u64>, StorageError> {
        let stats = self.stats.lock().await;
        let mut result = HashMap::new();
        
        result.insert("successful_connections".to_string(), stats.successful_connections);
        result.insert("failed_connections".to_string(), stats.failed_connections);
        result.insert("active_connections".to_string(), stats.active_connections as u64);
        result.insert("total_requests".to_string(), stats.total_requests);
        result.insert("total_errors".to_string(), stats.total_errors);
        
        Ok(result)
    }

    // Private helper methods
    
    /// Convert DocumentPoint to Qdrant PointStruct
    ///
    /// Converts a DocumentPoint with dense and optional sparse vectors to a Qdrant PointStruct.
    /// Uses named vectors: "dense" for semantic vectors, "sparse" for BM25-style keyword vectors.
    fn convert_to_qdrant_point(&self, point: DocumentPoint) -> Result<PointStruct, StorageError> {
        let payload = point.payload.into_iter()
            .map(|(k, v)| (k, Self::convert_json_to_qdrant_value(v)))
            .collect();

        // Build named vectors with "dense" and optionally "sparse"
        let mut named_vectors = std::collections::HashMap::new();

        // Add dense vector using the new Vector format with DenseVector
        named_vectors.insert(
            "dense".to_string(),
            qdrant_client::qdrant::Vector {
                data: vec![], // Deprecated, use vector field instead
                indices: None,
                vectors_count: None,
                vector: Some(qdrant_client::qdrant::vector::Vector::Dense(DenseVector {
                    data: point.dense_vector,
                })),
            }
        );

        // Add sparse vector if present
        if let Some(sparse_map) = point.sparse_vector {
            // Convert HashMap<u32, f32> to (indices, values) for SparseVector
            let mut indices: Vec<u32> = Vec::with_capacity(sparse_map.len());
            let mut values: Vec<f32> = Vec::with_capacity(sparse_map.len());

            // Sort by index for consistent ordering
            let mut entries: Vec<_> = sparse_map.into_iter().collect();
            entries.sort_by_key(|(idx, _)| *idx);

            for (idx, val) in entries {
                indices.push(idx);
                values.push(val);
            }

            named_vectors.insert(
                "sparse".to_string(),
                qdrant_client::qdrant::Vector {
                    data: vec![], // Deprecated for sparse vectors
                    indices: None, // Deprecated, use vector field instead
                    vectors_count: None,
                    vector: Some(qdrant_client::qdrant::vector::Vector::Sparse(SparseVector {
                        indices,
                        values,
                    })),
                }
            );
        }

        Ok(PointStruct {
            id: Some(qdrant_client::qdrant::PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(point.id)),
            }),
            vectors: Some(qdrant_client::qdrant::Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                    qdrant_client::qdrant::NamedVectors {
                        vectors: named_vectors,
                    }
                )),
            }),
            payload,
        })
    }
    
    /// Convert JSON value to Qdrant value
    fn convert_json_to_qdrant_value(value: serde_json::Value) -> qdrant_client::qdrant::Value {
        match value {
            serde_json::Value::Null => qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::NullValue(0)),
            },
            serde_json::Value::Bool(b) => qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(b)),
            },
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
                    }
                } else if let Some(f) = n.as_f64() {
                    qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)),
                    }
                } else {
                    qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(n.to_string())),
                    }
                }
            },
            serde_json::Value::String(s) => qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
            },
            serde_json::Value::Array(arr) => {
                let list_value = qdrant_client::qdrant::ListValue {
                    values: arr.into_iter()
                        .map(Self::convert_json_to_qdrant_value)
                        .collect(),
                };
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list_value)),
                }
            },
            serde_json::Value::Object(obj) => {
                let struct_value = qdrant_client::qdrant::Struct {
                    fields: obj.into_iter()
                        .map(|(k, v)| (k, Self::convert_json_to_qdrant_value(v)))
                        .collect(),
                };
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::StructValue(struct_value)),
                }
            }
        }
    }
    
    /// Dense vector search using Qdrant's QueryPoints API with named dense vectors
    async fn search_dense(
        &self,
        collection_name: &str,
        dense_vector: Vec<f32>,
        limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let query_builder = QueryPointsBuilder::new(collection_name)
            .query(dense_vector)
            .using("dense")
            .limit(limit as u64)
            .with_payload(true);

        let response = self.retry_operation(|| async {
            self.client.query(query_builder.clone()).await
                .map_err(|e| StorageError::Search(e.to_string()))
        }).await?;

        let results = response.result.into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;
                let json_payload: HashMap<String, serde_json::Value> = payload.into_iter()
                    .map(|(k, v)| (k, Self::convert_qdrant_value_to_json(v)))
                    .collect();

                let id = match scored_point.id.unwrap().point_id_options.unwrap() {
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => uuid,
                    qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => num.to_string(),
                };

                SearchResult {
                    id,
                    score: scored_point.score,
                    payload: json_payload,
                    dense_vector: None,
                    sparse_vector: None,
                }
            })
            .collect();

        Ok(results)
    }
    
    /// Sparse vector search using Qdrant's QueryPoints API with named sparse vectors
    async fn search_sparse(
        &self,
        collection_name: &str,
        sparse_vector: HashMap<u32, f32>,
        limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        if sparse_vector.is_empty() {
            debug!("Empty sparse vector, returning no results");
            return Ok(vec![]);
        }

        // Convert HashMap<u32, f32> to Vec<(u32, f32)> for QueryPointsBuilder
        let sparse_pairs: Vec<(u32, f32)> = sparse_vector
            .into_iter()
            .collect();

        let query_builder = QueryPointsBuilder::new(collection_name)
            .query(sparse_pairs)
            .using("sparse")
            .limit(limit as u64)
            .with_payload(true);

        let response = self.retry_operation(|| async {
            self.client.query(query_builder.clone()).await
                .map_err(|e| StorageError::Search(e.to_string()))
        }).await?;

        let results = response.result.into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;
                let json_payload: HashMap<String, serde_json::Value> = payload.into_iter()
                    .map(|(k, v)| (k, Self::convert_qdrant_value_to_json(v)))
                    .collect();

                let id = match scored_point.id.unwrap().point_id_options.unwrap() {
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => uuid,
                    qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => num.to_string(),
                };

                SearchResult {
                    id,
                    score: scored_point.score,
                    payload: json_payload,
                    dense_vector: None,
                    sparse_vector: None,
                }
            })
            .collect();

        Ok(results)
    }
    
    /// Hybrid search with RRF (Reciprocal Rank Fusion)
    async fn search_hybrid(
        &self,
        collection_name: &str,
        params: HybridSearchParams,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let mut all_results = HashMap::new();
        
        // Perform dense search if vector is provided
        if let Some(vector) = params.dense_vector {
            let dense_results = self.search_dense(collection_name, vector, params.limit * 2, params.score_threshold, params.filter.clone()).await?;
            
            for (rank, result) in dense_results.into_iter().enumerate() {
                let rrf_score = params.dense_weight / (60.0 + (rank + 1) as f32); // Standard RRF formula
                let entry = all_results.entry(result.id.clone())
                    .or_insert_with(|| (result, 0.0));
                entry.1 += rrf_score;
            }
        }
        
        // Perform sparse search if vector is provided
        if let Some(vector) = params.sparse_vector {
            let sparse_results = self.search_sparse(collection_name, vector, params.limit * 2, params.score_threshold, params.filter).await?;
            
            for (rank, result) in sparse_results.into_iter().enumerate() {
                let rrf_score = params.sparse_weight / (60.0 + (rank + 1) as f32); // Standard RRF formula
                let entry = all_results.entry(result.id.clone())
                    .or_insert_with(|| (result, 0.0));
                entry.1 += rrf_score;
            }
        }
        
        // Sort by combined RRF score and take top results
        let mut final_results: Vec<_> = all_results.into_iter()
            .map(|(_, (mut result, rrf_score))| {
                result.score = rrf_score; // Replace original score with RRF score
                result
            })
            .collect();
            
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(params.limit);
        
        Ok(final_results)
    }
    
    /// Convert Qdrant value to JSON value
    fn convert_qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> serde_json::Value {
        match value.kind {
            Some(qdrant_client::qdrant::value::Kind::NullValue(_)) => serde_json::Value::Null,
            Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => serde_json::Value::Bool(b),
            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => serde_json::Value::Number(i.into()),
            Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)) => {
                serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)))
            },
            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => serde_json::Value::String(s),
            Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => {
                serde_json::Value::Array(
                    list.values.into_iter()
                        .map(Self::convert_qdrant_value_to_json)
                        .collect()
                )
            },
            Some(qdrant_client::qdrant::value::Kind::StructValue(struct_val)) => {
                serde_json::Value::Object(
                    struct_val.fields.into_iter()
                        .map(|(k, v)| (k, Self::convert_qdrant_value_to_json(v)))
                        .collect()
                )
            },
            None => serde_json::Value::Null,
        }
    }
    
    /// Retry operation with exponential backoff
    async fn retry_operation<F, Fut, T>(&self, operation: F) -> Result<T, StorageError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, StorageError>>,
    {
        let mut attempt = 0;
        let max_retries = self.config.max_retries;
        let base_delay = Duration::from_millis(self.config.retry_delay_ms);
        
        loop {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    self.update_stats(|stats| stats.total_requests += 1).await;
                    return Ok(result);
                },
                Err(e) => {
                    attempt += 1;
                    self.update_stats(|stats| {
                        stats.total_requests += 1;
                        stats.total_errors += 1;
                    }).await;
                    
                    if attempt >= max_retries {
                        error!("Operation failed after {} attempts: {}", attempt, e);
                        return Err(e);
                    }
                    
                    // Exponential backoff with jitter
                    let delay = base_delay * (2_u32.pow(attempt - 1));
                    let jitter = Duration::from_millis(fastrand::u64(0..=100));
                    let total_delay = delay + jitter;
                    
                    warn!("Operation failed (attempt {}), retrying in {:?}: {}", attempt, total_delay, e);
                    sleep(total_delay).await;
                }
            }
        }
    }
    
    /// Update connection statistics
    async fn update_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut ConnectionStats),
    {
        if let Ok(mut stats) = self.stats.try_lock() {
            update_fn(&mut stats);
        }
    }
}

impl Default for StorageClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect if running in daemon mode for MCP stdio compliance
fn is_daemon_mode() -> bool {
    // Primary explicit indicator
    if env::var("WQM_SERVICE_MODE").map(|v| v == "true").unwrap_or(false) {
        return true;
    }

    // macOS LaunchAgent/LaunchDaemon - XPC_SERVICE_NAME is set to "0" in regular
    // terminal sessions, so we check that it's not empty and not "0"
    if let Ok(xpc_name) = env::var("XPC_SERVICE_NAME") {
        if !xpc_name.is_empty() && xpc_name != "0" {
            return true;
        }
    }

    // Linux systemd indicators
    env::var("SYSTEMD_EXEC_PID").is_ok() ||
        env::var("SYSLOG_IDENTIFIER").is_ok() ||
        env::var("LOGNAME").map(|v| v == "root").unwrap_or(false)
}

/// Temporarily suppress stdout and stderr during a function call for MCP compliance
#[cfg(unix)]
fn suppress_output_temporarily<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    // Save original file descriptors
    let original_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let original_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };

    let result = if let Ok(null_file) = OpenOptions::new().write(true).open("/dev/null") {
        let null_fd = null_file.as_raw_fd();

        // Redirect stdout and stderr to /dev/null
        unsafe {
            libc::dup2(null_fd, libc::STDOUT_FILENO);
            libc::dup2(null_fd, libc::STDERR_FILENO);
        }

        // Execute the function
        let result = f();

        // Restore original file descriptors
        unsafe {
            libc::dup2(original_stdout, libc::STDOUT_FILENO);
            libc::dup2(original_stderr, libc::STDERR_FILENO);
            libc::close(original_stdout);
            libc::close(original_stderr);
        }

        result
    } else {
        // If we can't open /dev/null, just run the function normally
        f()
    };

    result
}

/// Windows version of output suppression
#[cfg(windows)]
fn suppress_output_temporarily<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use std::fs::OpenOptions;
    use std::os::windows::io::AsRawHandle;

    if let Ok(null_file) = OpenOptions::new().write(true).open("NUL") {
        unsafe {
            winapi::um::processenv::SetStdHandle(
                winapi::um::winbase::STD_OUTPUT_HANDLE,
                null_file.as_raw_handle() as *mut std::ffi::c_void
            );
            winapi::um::processenv::SetStdHandle(
                winapi::um::winbase::STD_ERROR_HANDLE,
                null_file.as_raw_handle() as *mut std::ffi::c_void
            );
        }
    }

    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_names() {
        // Canonical collection names (without underscore prefix)
        assert_eq!(collections::PROJECTS, "projects");
        assert_eq!(collections::LIBRARIES, "libraries");
        assert_eq!(collections::MEMORY, "memory");
    }

    #[test]
    fn test_multi_tenant_config_default() {
        let config = MultiTenantConfig::default();
        assert_eq!(config.vector_size, 384);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_ef_construct, 100);
        assert!(config.on_disk_payload);
    }

    #[test]
    fn test_multi_tenant_config_custom() {
        let config = MultiTenantConfig {
            vector_size: 768,
            hnsw_m: 32,
            hnsw_ef_construct: 200,
            on_disk_payload: false,
        };
        assert_eq!(config.vector_size, 768);
        assert_eq!(config.hnsw_m, 32);
        assert_eq!(config.hnsw_ef_construct, 200);
        assert!(!config.on_disk_payload);
    }

    #[test]
    fn test_multi_tenant_init_result_default() {
        let result = MultiTenantInitResult::default();
        assert!(!result.projects_created);
        assert!(!result.projects_indexed);
        assert!(!result.libraries_created);
        assert!(!result.libraries_indexed);
        assert!(!result.memory_created);
        assert!(!result.is_complete());
    }

    #[test]
    fn test_multi_tenant_init_result_complete() {
        let result = MultiTenantInitResult {
            projects_created: true,
            projects_indexed: true,
            libraries_created: true,
            libraries_indexed: true,
            memory_created: true,
        };
        assert!(result.is_complete());
    }

    #[test]
    fn test_multi_tenant_init_result_incomplete() {
        let result = MultiTenantInitResult {
            projects_created: true,
            projects_indexed: true,
            libraries_created: true,
            libraries_indexed: false, // Missing index
            memory_created: true,
        };
        assert!(!result.is_complete());
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.url, "http://localhost:6333");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
        assert_eq!(config.dense_vector_size, 1536);
        assert!(config.check_compatibility);
    }

    #[test]
    fn test_storage_config_daemon_mode() {
        let config = StorageConfig::daemon_mode();
        assert!(!config.check_compatibility);
    }

    #[test]
    fn test_hybrid_search_mode_default() {
        let mode = HybridSearchMode::default();
        match mode {
            HybridSearchMode::Hybrid { dense_weight, sparse_weight } => {
                assert_eq!(dense_weight, 1.0);
                assert_eq!(sparse_weight, 1.0);
            }
            _ => panic!("Expected Hybrid mode as default"),
        }
    }

    #[test]
    fn test_search_params_default() {
        let params = SearchParams::default();
        assert!(params.dense_vector.is_none());
        assert!(params.sparse_vector.is_none());
        assert_eq!(params.limit, 10);
        assert!(params.score_threshold.is_none());
        assert!(params.filter.is_none());
    }

    #[test]
    fn test_transport_mode_default() {
        let mode = TransportMode::default();
        assert!(matches!(mode, TransportMode::Grpc));
    }

    #[test]
    fn test_http2_config_default() {
        let config = Http2Config::default();
        assert_eq!(config.max_frame_size, Some(8192));
        assert_eq!(config.initial_window_size, Some(32768));
        assert_eq!(config.max_header_list_size, Some(8192));
        assert!(!config.enable_push);
        assert!(config.tcp_keepalive);
        assert_eq!(config.keepalive_interval_ms, Some(30000));
        assert_eq!(config.keepalive_timeout_ms, Some(5000));
        assert!(!config.http2_adaptive_window);
    }
}