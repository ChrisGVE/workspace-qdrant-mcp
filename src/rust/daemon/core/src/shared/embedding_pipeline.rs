//! Semaphore-gated embedding pipeline extracted from duplicated inline patterns.
//!
//! The pattern of acquiring the embedding semaphore, calling `generate_embedding`,
//! and converting the sparse result to a `HashMap<u32, f32>` was duplicated in
//! every `process_*_item` function. This module provides canonical helpers.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::warn;

use crate::embedding::{EmbeddingGenerator, SparseEmbedding};
use crate::unified_queue_processor::UnifiedProcessorError;

/// Result of a semaphore-gated embedding operation.
pub struct EmbedResult {
    /// Dense embedding vector.
    pub dense_vector: Vec<f32>,
    /// Sparse BM25 vector as index→weight map, or `None` if indices were empty.
    pub sparse_vector: Option<HashMap<u32, f32>>,
}

/// Generate a dense + sparse embedding with semaphore gating.
///
/// This is the canonical embedding path for content items (memory, scratchpad,
/// generic content, URLs) that use the `EmbeddingGenerator`'s built-in BM25
/// for sparse vectors.
///
/// For file items that use `LexiconManager` IDF-weighted sparse vectors,
/// use `embed_dense_only` and compute sparse separately via `LexiconManager`.
pub async fn embed_with_sparse(
    generator: &Arc<EmbeddingGenerator>,
    semaphore: &Arc<Semaphore>,
    text: &str,
    model_hint: &str,
) -> Result<EmbedResult, UnifiedProcessorError> {
    let _permit = semaphore
        .acquire()
        .await
        .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;

    let embedding_result = generator
        .generate_embedding(text, model_hint)
        .await
        .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;

    drop(_permit);

    Ok(EmbedResult {
        dense_vector: embedding_result.dense.vector,
        sparse_vector: sparse_embedding_to_map(&embedding_result.sparse),
    })
}

/// Generate only a dense embedding with semaphore gating.
///
/// Used by file processing where sparse vectors are computed separately
/// via `LexiconManager` for IDF-weighted BM25.
pub async fn embed_dense_only(
    generator: &Arc<EmbeddingGenerator>,
    semaphore: &Arc<Semaphore>,
    text: &str,
    model_hint: &str,
) -> Result<Vec<f32>, UnifiedProcessorError> {
    let _permit = semaphore
        .acquire()
        .await
        .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;

    let embedding_result = generator
        .generate_embedding(text, model_hint)
        .await
        .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;

    drop(_permit);

    Ok(embedding_result.dense.vector)
}

/// Convert a `SparseEmbedding` to the `HashMap<u32, f32>` format expected by `DocumentPoint`.
///
/// Returns `None` if the sparse embedding has no indices (warns about missing BM25 coverage).
pub fn sparse_embedding_to_map(sparse: &SparseEmbedding) -> Option<HashMap<u32, f32>> {
    if sparse.indices.is_empty() {
        warn!(
            "Sparse embedding has empty indices — point will be stored \
             without sparse vector (BM25 search won't match)"
        );
        return None;
    }
    let map: HashMap<u32, f32> = sparse
        .indices
        .iter()
        .zip(sparse.values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();
    Some(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_embedding_to_map_empty() {
        let sparse = SparseEmbedding {
            indices: vec![],
            values: vec![],
            vocab_size: 0,
        };
        assert!(sparse_embedding_to_map(&sparse).is_none());
    }

    #[test]
    fn test_sparse_embedding_to_map_populated() {
        let sparse = SparseEmbedding {
            indices: vec![1, 5, 10],
            values: vec![0.5, 0.3, 0.8],
            vocab_size: 100,
        };
        let map = sparse_embedding_to_map(&sparse).expect("should produce map");
        assert_eq!(map.len(), 3);
        assert!((map[&1] - 0.5).abs() < f32::EPSILON);
        assert!((map[&5] - 0.3).abs() < f32::EPSILON);
        assert!((map[&10] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparse_embedding_to_map_single() {
        let sparse = SparseEmbedding {
            indices: vec![42],
            values: vec![1.0],
            vocab_size: 50,
        };
        let map = sparse_embedding_to_map(&sparse).expect("should produce map");
        assert_eq!(map.len(), 1);
        assert!((map[&42] - 1.0).abs() < f32::EPSILON);
    }
}
