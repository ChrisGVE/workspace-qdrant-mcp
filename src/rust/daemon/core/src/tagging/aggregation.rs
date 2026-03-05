//! Document embedding aggregation for Tier 2 tagging.
//!
//! Computes aggregate embeddings for documents from their chunk embeddings
//! using weighted or uniform mean.

use crate::keyword_extraction::semantic_rerank::weighted_mean_vector;

/// Compute an aggregate embedding for a document from its chunk embeddings.
///
/// Uses weighted mean: each chunk is weighted equally (weight = 1.0).
/// Returns `None` if no embeddings are provided.
pub fn aggregate_document_embedding(chunk_embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    if chunk_embeddings.is_empty() {
        return None;
    }

    let weighted: Vec<(Vec<f32>, f64)> =
        chunk_embeddings.iter().map(|e| (e.clone(), 1.0)).collect();

    weighted_mean_vector(&weighted)
}

/// Compute a weighted aggregate embedding for a document.
///
/// Weights allow emphasizing certain chunks (e.g. title, first paragraph).
/// `chunk_weights` must have the same length as `chunk_embeddings`.
pub fn aggregate_document_embedding_weighted(
    chunk_embeddings: &[Vec<f32>],
    chunk_weights: &[f64],
) -> Option<Vec<f32>> {
    if chunk_embeddings.is_empty()
        || chunk_weights.is_empty()
        || chunk_embeddings.len() != chunk_weights.len()
    {
        return None;
    }

    let weighted: Vec<(Vec<f32>, f64)> = chunk_embeddings
        .iter()
        .cloned()
        .zip(chunk_weights.iter().copied())
        .collect();

    weighted_mean_vector(&weighted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_single_chunk() {
        let chunks = vec![vec![1.0, 0.0, 0.0]];
        let agg = aggregate_document_embedding(&chunks).unwrap();
        assert_eq!(agg, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_aggregate_two_chunks() {
        let chunks = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let agg = aggregate_document_embedding(&chunks).unwrap();
        assert!((agg[0] - 0.5).abs() < 1e-5);
        assert!((agg[1] - 0.5).abs() < 1e-5);
        assert!((agg[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_empty() {
        let chunks: Vec<Vec<f32>> = Vec::new();
        assert!(aggregate_document_embedding(&chunks).is_none());
    }

    #[test]
    fn test_aggregate_weighted() {
        let chunks = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let weights = vec![3.0, 1.0];
        let agg = aggregate_document_embedding_weighted(&chunks, &weights).unwrap();
        assert!((agg[0] - 0.75).abs() < 1e-5);
        assert!((agg[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_weighted_mismatch() {
        let chunks = vec![vec![1.0, 0.0]];
        let weights = vec![1.0, 2.0]; // length mismatch
        assert!(aggregate_document_embedding_weighted(&chunks, &weights).is_none());
    }
}
