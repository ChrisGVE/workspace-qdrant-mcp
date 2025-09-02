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
        
        let client = Arc::new(client_builder.build().expect("Failed to build Qdrant client"));
        
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
        sparse_vector_size: Option<u64>,
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
                map: [("sparse".to_string(), qdrant_client::qdrant::SparseVectorConfig {
                    map: HashMap::new(),
                })]
                .into_iter()
                .collect(),
            };
            
            // Note: Sparse vector configuration would be added here in a production implementation
            // The qdrant-client API for sparse vectors may vary by version
        }
        
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
            self.client.delete_collection(&delete_collection).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        info!("Successfully deleted collection: {}", collection_name);
        Ok(())
    }
    
    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        debug!("Checking if collection exists: {}", collection_name);
        
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
        debug!("Inserting point {} into collection {}", point.id, collection_name);
        
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
        info!(\"Inserting {} points into collection {} in batches\", points.len(), collection_name);
        
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
    
    /// Perform hybrid search with dense/sparse vector fusion
    pub async fn search(
        &self,
        collection_name: &str,
        dense_vector: Option<Vec<f32>>,
        sparse_vector: Option<HashMap<u32, f32>>,
        search_mode: HybridSearchMode,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        debug!("Performing search in collection: {}", collection_name);
        
        let results = match search_mode {
            HybridSearchMode::Dense => {
                if let Some(vector) = dense_vector {
                    self.search_dense(collection_name, vector, limit, score_threshold, filter).await?
                } else {
                    return Err(StorageError::Search("Dense vector required for dense search ".to_string()));
                }
            },
            HybridSearchMode::Sparse => {
                if let Some(vector) = sparse_vector {
                    self.search_sparse(collection_name, vector, limit, score_threshold, filter).await?
                } else {
                    return Err(StorageError::Search("Sparse vector required for sparse search ".to_string()));
                }
            },
            HybridSearchMode::Hybrid { dense_weight, sparse_weight } => {
                self.search_hybrid(collection_name, dense_vector, sparse_vector, 
                                 dense_weight, sparse_weight, limit, score_threshold, filter).await?
            }
        };
        
        debug!("Search completed, returned {} results ", results.len());
        Ok(results)
    }
    
    /// Dense vector search
    async fn search_dense(
        &self,
        collection_name: &str,
        dense_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let search_points = SearchPoints {
            collection_name: collection_name.to_string(),
            vector: dense_vector,
            limit: limit as u64,
            score_threshold,
            filter: self.build_filter(filter)?,
            with_payload: Some(true.into()),
            with_vectors: Some(false.into()), // Don't return vectors by default for performance
            ..Default::default()
        };
        
        let response = self.retry_operation(|| async {
            self.client.search_points(&search_points).await
                .map_err(|e| StorageError::Search(e.to_string()))
        }).await?;
        
        let results = response.result.into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;
                let json_payload: HashMap<String, serde_json::Value> = payload.into_iter()
                    .map(|(k, v)| (k, self.convert_qdrant_value_to_json(v)))
                    .collect();
                    
                SearchResult {
                    id: scored_point.id.unwrap().point_id_options.unwrap().to_string(),
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
        collection_name: &str,
        _sparse_vector: HashMap<u32, f32>,
        _limit: usize,
        _score_threshold: Option<f32>,
        _filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        // Note: Sparse vector search implementation depends on Qdrant version
        // This is a placeholder implementation
        warn!("Sparse vector search not fully implemented in this version ");
        Ok(vec![])
    }
    
    /// Hybrid search with RRF (Reciprocal Rank Fusion)
    async fn search_hybrid(
        &self,
        collection_name: &str,
        dense_vector: Option<Vec<f32>>,
        sparse_vector: Option<HashMap<u32, f32>>,
        dense_weight: f32,
        sparse_weight: f32,
        limit: usize,
        score_threshold: Option<f32>,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let mut all_results = HashMap::new();
        
        // Perform dense search if vector is provided
        if let Some(vector) = dense_vector {
            let dense_results = self.search_dense(collection_name, vector, limit * 2, score_threshold, filter.clone()).await?;
            
            for (rank, result) in dense_results.into_iter().enumerate() {
                let rrf_score = dense_weight / (60.0 + (rank + 1) as f32); // Standard RRF formula
                let entry = all_results.entry(result.id.clone())
                    .or_insert_with(|| (result, 0.0));
                entry.1 += rrf_score;
            }
        }
        
        // Perform sparse search if vector is provided
        if let Some(vector) = sparse_vector {
            let sparse_results = self.search_sparse(collection_name, vector, limit * 2, score_threshold, filter).await?;
            
            for (rank, result) in sparse_results.into_iter().enumerate() {
                let rrf_score = sparse_weight / (60.0 + (rank + 1) as f32); // Standard RRF formula
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
        final_results.truncate(limit);
        
        Ok(final_results)
    }
    
    /// Get collection information
    pub async fn get_collection_info(&self, collection_name: &str) -> Result<HashMap<String, serde_json::Value>, StorageError> {
        debug!("Getting collection info for: {}", collection_name);
        
        let response = self.retry_operation(|| async {
            self.client.collection_info(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        // Convert collection info to JSON-serializable format
        let mut info = HashMap::new();
        if let Some(result) = response.result {
            info.insert("status".to_string(), serde_json::Value::String(format!("{:?}", result.status)));
            info.insert("vectors_count".to_string(), serde_json::Value::Number(result.vectors_count.unwrap_or(0).into()));
            info.insert("indexed_vectors_count".to_string(), serde_json::Value::Number(result.indexed_vectors_count.unwrap_or(0).into()));
            info.insert("points_count".to_string(), serde_json::Value::Number(result.points_count.unwrap_or(0).into()));
            info.insert("segments_count".to_string(), serde_json::Value::Number(result.segments_count.unwrap_or(0).into()));
        }
        
        Ok(info)
    }
    
    /// List all collections
    pub async fn list_collections(&self) -> Result<Vec<String>, StorageError> {
        debug!("Listing all collections ");
        
        let response = self.retry_operation(|| async {
            self.client.list_collections().await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;
        
        let collections = response.collections.into_iter()
            .map(|collection| collection.name)
            .collect();
            
        Ok(collections)
    }
    
    /// Delete points from collection by ID
    pub async fn delete_points(
        &self,
        collection_name: &str,
        point_ids: Vec<String>,
    ) -> Result<(), StorageError> {
        info!("Deleting {} points from collection {}", point_ids.len(), collection_name);
        
        let point_selector = qdrant_client::qdrant::PointsSelector {
            points_selector_one_of: Some(qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                qdrant_client::qdrant::PointsIdsList {
                    ids: point_ids.into_iter()
                        .map(|id| qdrant_client::qdrant::PointId {
                            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id)),
                        })
                        .collect(),
                }
            )),
        };
        
        let delete_points = qdrant_client::qdrant::DeletePoints {
            collection_name: collection_name.to_string(),
            points: Some(point_selector),
            wait: Some(true),
            ..Default::default()
        };
        
        self.retry_operation(|| async {
            self.client.delete_points(&delete_points).await
                .map_err(|e| StorageError::Point(e.to_string()))
        }).await?;
        
        info!("Successfully deleted points from collection {}", collection_name);
        Ok(())
    }
    
    // Private helper methods
    
    /// Convert DocumentPoint to Qdrant PointStruct
    fn convert_to_qdrant_point(&self, point: DocumentPoint) -> Result<PointStruct, StorageError> {
        let mut vectors = HashMap::new();
        
        // Add dense vector
        vectors.insert("".to_string(), qdrant_client::qdrant::Vector {
            data: point.dense_vector,
        });
        
        // Add sparse vector if present
        if let Some(sparse) = point.sparse_vector {
            let sparse_vector = qdrant_client::qdrant::SparseVector {
                indices: sparse.keys().cloned().collect(),
                values: sparse.values().cloned().collect(),
            };
            // Note: Sparse vector integration depends on Qdrant client version
            // This is a simplified implementation
        }
        
        // Convert payload
        let payload = point.payload.into_iter()
            .map(|(k, v)| (k, self.convert_json_to_qdrant_value(v)))
            .collect();
        
        Ok(PointStruct {
            id: Some(qdrant_client::qdrant::PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(point.id)),
            }),
            vectors: Some(qdrant_client::qdrant::Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                    qdrant_client::qdrant::Vector {
                        data: vectors.get("").unwrap().data.clone(),
                    }
                )),
            }),
            payload,
        })
    }
    
    /// Convert JSON value to Qdrant value
    fn convert_json_to_qdrant_value(&self, value: serde_json::Value) -> qdrant_client::qdrant::Value {
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
                        .map(|v| self.convert_json_to_qdrant_value(v))
                        .collect(),
                };
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list_value)),
                }
            },
            serde_json::Value::Object(obj) => {
                let struct_value = qdrant_client::qdrant::Struct {
                    fields: obj.into_iter()
                        .map(|(k, v)| (k, self.convert_json_to_qdrant_value(v)))
                        .collect(),
                };
                qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::StructValue(struct_value)),
                }
            }
        }
    }
    
    /// Convert Qdrant value to JSON value
    fn convert_qdrant_value_to_json(&self, value: qdrant_client::qdrant::Value) -> serde_json::Value {
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
                        .map(|v| self.convert_qdrant_value_to_json(v))
                        .collect()
                )
            },
            Some(qdrant_client::qdrant::value::Kind::StructValue(struct_val)) => {
                serde_json::Value::Object(
                    struct_val.fields.into_iter()
                        .map(|(k, v)| (k, self.convert_qdrant_value_to_json(v)))
                        .collect()
                )
            },
            None => serde_json::Value::Null,
        }
    }
    
    /// Build Qdrant filter from JSON filter
    fn build_filter(&self, filter: Option<HashMap<String, serde_json::Value>>) -> Result<Option<qdrant_client::qdrant::Filter>, StorageError> {
        if let Some(filter_map) = filter {
            // Simple implementation - convert to must conditions
            let conditions: Vec<_> = filter_map.into_iter()
                .map(|(key, value)| {
                    let match_value = qdrant_client::qdrant::r#match::MatchValue::Keyword(value.to_string());
                    qdrant_client::qdrant::Condition {
                        condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                            qdrant_client::qdrant::FieldCondition {
                                key,
                                r#match: Some(qdrant_client::qdrant::Match {
                                    match_value: Some(match_value),
                                }),
                                range: None,
                                geo_bounding_box: None,
                                geo_radius: None,
                                values_count: None,
                            }
                        )),
                    }
                })
                .collect();
                
            if conditions.is_empty() {
                Ok(None)
            } else {
                Ok(Some(qdrant_client::qdrant::Filter {
                    should: vec![],
                    must: conditions,
                    must_not: vec![],
                }))
            }
        } else {
            Ok(None)
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
                        info!("Operation succeeded after {} retries ", attempt);
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
}

impl Default for StorageClient {
    fn default() -> Self {
        Self::new()
    }
