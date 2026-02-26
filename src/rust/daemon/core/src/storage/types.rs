//! Storage data types
//!
//! Error types, document points, search results, and collection info
//! structures used across the storage module.

use std::collections::HashMap;
use qdrant_client::QdrantError;
use serde::{Serialize, Deserialize};
use thiserror::Error;

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
    /// Whether `rules` collection was created
    pub rules_created: bool,
    /// Whether scratchpad collection was created
    pub scratchpad_created: bool,
    /// Whether images collection was created (512-dim CLIP vectors)
    pub images_created: bool,
}

impl MultiTenantInitResult {
    /// Check if all collections were successfully initialized
    pub fn is_complete(&self) -> bool {
        self.projects_created
            && self.projects_indexed
            && self.libraries_created
            && self.libraries_indexed
            && self.rules_created
            && self.scratchpad_created
            && self.images_created
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_tenant_init_result_default() {
        let result = MultiTenantInitResult::default();
        assert!(!result.projects_created);
        assert!(!result.projects_indexed);
        assert!(!result.libraries_created);
        assert!(!result.libraries_indexed);
        assert!(!result.rules_created);
        assert!(!result.images_created);
        assert!(!result.is_complete());
    }

    #[test]
    fn test_multi_tenant_init_result_complete() {
        let result = MultiTenantInitResult {
            projects_created: true,
            projects_indexed: true,
            libraries_created: true,
            libraries_indexed: true,
            rules_created: true,
            scratchpad_created: true,
            images_created: true,
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
            rules_created: true,
            scratchpad_created: true,
            images_created: true,
        };
        assert!(!result.is_complete());
    }

    #[test]
    fn test_multi_tenant_init_result_missing_images() {
        let result = MultiTenantInitResult {
            projects_created: true,
            projects_indexed: true,
            libraries_created: true,
            libraries_indexed: true,
            rules_created: true,
            scratchpad_created: true,
            images_created: false,
        };
        assert!(!result.is_complete());
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
    fn test_collection_names() {
        use wqm_common::constants::{
            COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
        };
        // Canonical collection names (without underscore prefix)
        assert_eq!(COLLECTION_PROJECTS, "projects");
        assert_eq!(COLLECTION_LIBRARIES, "libraries");
        assert_eq!(COLLECTION_RULES, "rules");
        assert_eq!(COLLECTION_SCRATCHPAD, "scratchpad");
    }
}
