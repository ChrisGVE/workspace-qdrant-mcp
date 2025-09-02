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
