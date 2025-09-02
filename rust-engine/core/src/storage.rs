//! Storage abstraction layer
//!
//! This module provides the Qdrant storage interface implementation with
//! comprehensive vector database operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{PointStruct, SearchPoints, UpsertPoints, FieldType, PayloadSchemaParams};
use qdrant_client::qdrant::{CreateCollection, DeleteCollection, CollectionExists, Distance, VectorParams, VectorsConfig};
use qdrant_client::qdrant::{SparseVectorParams, SparseIndices};
use serde::{Serialize, Deserialize};
use tokio::time::{sleep, timeout};
use thiserror::Error;
use tracing::{debug, info, warn, error};

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
    Qdrant(#[from] qdrant_client::QdrantError),
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
        }
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

/// Storage client with Qdrant integration
pub struct StorageClient {
    /// Qdrant client instance
    client: Arc<QdrantClient>,
    /// Client configuration
    config: StorageConfig,
    /// Connection pool statistics
    stats: Arc<tokio::sync::Mutex<ConnectionStats>>,
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

impl StorageClient {
    /// Create a new storage client with default configuration
    pub fn new() -> Self {
        Self::with_config(StorageConfig::default())
    }
    
    /// Create a storage client with custom configuration
    pub fn with_config(config: StorageConfig) -> Self {
        let mut client_builder = QdrantClient::from_url(&config.url);
        
        // Configure authentication
        if let Some(api_key) = &config.api_key {
            client_builder = client_builder.with_api_key(api_key);
        }
        
        // Configure timeout
        client_builder = client_builder.with_timeout(Duration::from_millis(config.timeout_ms));
        
        let client = Arc::new(client_builder.build().expect(\"Failed to build Qdrant client\"));
        
        Self {
            client,
            config,
            stats: Arc::new(tokio::sync::Mutex::new(ConnectionStats::default())),
        }\n    }
    
    /// Test connection to Qdrant server
    pub async fn test_connection(&self) -> Result<bool, StorageError> {
        debug!(\"Testing connection to Qdrant server: {}\", self.config.url);
        
        match self.client.health_check().await {
            Ok(_) => {
                info!(\"Successfully connected to Qdrant server\");
                self.update_stats(|stats| stats.successful_connections += 1).await;
                Ok(true)
            },
            Err(e) => {
                error!(\"Failed to connect to Qdrant server: {}\", e);
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
        sparse_vector_size: Option<u64>,
    ) -> Result<(), StorageError> {
        info!(\"Creating collection: {}\", collection_name);
        
        let dense_size = dense_vector_size.unwrap_or(self.config.dense_vector_size);
        
        let mut vectors_config = VectorsConfig::default();
        
        // Configure dense vector
        let dense_vector_params = VectorParams {
            size: dense_size,
            distance: Distance::Cosine.into(),
            hnsw_config: None,
            quantization_config: None,
            on_disk: Some(false), // Keep in memory for better performance
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
            self.client.create_collection(&create_collection).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        // Configure sparse vectors if requested
        if let Some(sparse_size) = sparse_vector_size.or(self.config.sparse_vector_size) {
            let sparse_params = SparseVectorParams {
                map: [(\"sparse\".to_string(), qdrant_client::qdrant::SparseVectorConfig {
                    map: HashMap::new(),
                })]
                .into_iter()
                .collect(),
            };
            
            // Note: Sparse vector configuration would be added here in a production implementation
            // The qdrant-client API for sparse vectors may vary by version
        }
        
        info!(\"Successfully created collection: {}\", collection_name);
        Ok(())
    }
    
    /// Delete a collection
    pub async fn delete_collection(&self, collection_name: &str) -> Result<(), StorageError> {
        info!(\"Deleting collection: {}\", collection_name);
        
        let delete_collection = DeleteCollection {
            collection_name: collection_name.to_string(),
            timeout: Some(self.config.timeout_ms),
        };
        
        self.retry_operation(|| async {
            self.client.delete_collection(&delete_collection).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        info!(\"Successfully deleted collection: {}\", collection_name);
        Ok(())
    }
    
    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        debug!(\"Checking if collection exists: {}\", collection_name);
        
        let request = CollectionExists {
            collection_name: collection_name.to_string(),
        };
        
        let response = self.retry_operation(|| async {
            self.client.collection_exists(&request).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        Ok(response.result.unwrap_or(false))
    }
    
    /// Insert a single document point
    pub async fn insert_point(
        &self,
        collection_name: &str,
        point: DocumentPoint,
    ) -> Result<(), StorageError> {
        debug!(\"Inserting point {} into collection {}\", point.id, collection_name);
        
        let qdrant_point = self.convert_to_qdrant_point(point)?;
        
        let upsert_points = UpsertPoints {
            collection_name: collection_name.to_string(),
            points: vec![qdrant_point],
            wait: Some(true),
            ..Default::default()
        };
        
        self.retry_operation(|| async {
            self.client.upsert_points(&upsert_points).await
                .map_err(|e| StorageError::Point(e.to_string()))
        }).await?;
        
        debug!(\"Successfully inserted point into collection {}\", collection_name);
        Ok(())
    }
    
    /// Insert multiple document points in batch
    pub async fn insert_points_batch(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
    ) -> Result<BatchStats, StorageError> {
        info!(\"Inserting {} points into collection {} in batches\", points.len(), collection_name);
        
        let start_time = std::time::Instant::now();
        let batch_size = batch_size.unwrap_or(100); // Default batch size
        let total_points = points.len();
        let mut successful = 0;\n        let mut failed = 0;
        
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
                        self.client.upsert_points(&upsert_points).await
                            .map_err(|e| StorageError::Batch(e.to_string()))
                    }).await {
                        Ok(_) => successful += chunk.len(),
                        Err(e) => {
                            error!(\"Failed to insert batch: {}\", e);
                            failed += chunk.len();
                        }
                    }
                },
                Err(e) => {
                    error!(\"Failed to convert points batch: {}\", e);
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
        
        info!(\"Batch insertion completed: {} successful, {} failed, {:.2} points/sec\", 
              successful, failed, throughput);
        
        Ok(stats)
    }
