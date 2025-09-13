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
use qdrant_client::qdrant::{PointStruct, SearchPoints, UpsertPoints};
use qdrant_client::qdrant::{CreateCollection, DeleteCollection, Distance, VectorParams, VectorsConfig};
use qdrant_client::qdrant::Datatype;
use serde::{Serialize, Deserialize};
use tokio::time::sleep;
use thiserror::Error;
use tracing::{debug, info, warn, error};
// Note: tonic, hyper, and url imports removed as they're not currently used
// They would be needed for advanced gRPC HTTP/2 configuration

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
    /// to ensure complete console silence for MCP stdio protocol compliance
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.check_compatibility = false; // Disable to suppress Qdrant client output
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
        
        info!("Initializing Qdrant client with transport: {:?}", config.transport);
        
        // Determine the appropriate URL based on transport mode
        let connection_url = match config.transport {
            TransportMode::Grpc => {
                // Use gRPC endpoint (typically different port or scheme)
                if config.url.contains("://") {
                    config.url.clone()
                } else {
                    // Default to HTTP for now, actual gRPC configuration would need different setup
                    format!("http://{}", config.url.trim_start_matches("http://"))
                }
            },
            TransportMode::Http => {
                // Ensure HTTP scheme
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
            debug!("Disabling compatibility check as requested");
            // Set environment variables that the Qdrant client might recognize
            std::env::set_var("QDRANT_SKIP_VERSION_CHECK", "true");
            std::env::set_var("QDRANT_CHECK_COMPATIBILITY", "false");
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
            if let Some(frame_size) = config.http2.max_frame_size {
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
    
    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        debug!("Checking if collection exists: {}", collection_name);
        
        let response = self.retry_operation(|| async {
            self.client.collection_exists(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        Ok(response)
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
    fn convert_to_qdrant_point(&self, point: DocumentPoint) -> Result<PointStruct, StorageError> {
        let payload = point.payload.into_iter()
            .map(|(k, v)| (k, Self::convert_json_to_qdrant_value(v)))
            .collect();
        
        Ok(PointStruct {
            id: Some(qdrant_client::qdrant::PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(point.id)),
            }),
            vectors: Some(qdrant_client::qdrant::Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                    qdrant_client::qdrant::Vector {
                        data: point.dense_vector,
                        indices: None,
                        vectors_count: None,
                        vector: None,
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
    
    /// Dense vector search
    async fn search_dense(
        &self,
        collection_name: &str,
        dense_vector: Vec<f32>,
        limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let search_points = SearchPoints {
            collection_name: collection_name.to_string(),
            vector: dense_vector,
            limit: limit as u64,
            with_payload: Some(true.into()),
            with_vectors: Some(false.into()),
            ..Default::default()
        };
        
        let response = self.retry_operation(|| async {
            self.client.search_points(search_points.clone()).await
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
    
    /// Sparse vector search (placeholder implementation)
    async fn search_sparse(
        &self,
        _collection_name: &str,
        _sparse_vector: HashMap<u32, f32>,
        _limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        // Note: Sparse vector search implementation depends on Qdrant version
        warn!("Sparse vector search not fully implemented in this version");
        Ok(vec![])
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
    env::var("WQM_SERVICE_MODE").map(|v| v == "true").unwrap_or(false) ||
        env::var("XPC_SERVICE_NAME").is_ok() ||
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