//! Search-result data types shared between the storage crates and daemon-core.
//!
//! Location: `wqm-common/src/search/types.rs`. Context: canonical home (F0) of
//! `SearchResult`, the per-hit result of a Qdrant search/scroll — relocated from
//! `daemon-core/storage/types.rs` so the read crate (`wqm-storage`) and daemon-core
//! share ONE definition (FP-2 / DR GP-9). Daemon-core re-exports it from
//! `crate::storage::types` so existing call sites are unchanged.
//! Neighbors: `super::rrf` (fuses `Vec<SearchResult>` across collections).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Search result from Qdrant.
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
