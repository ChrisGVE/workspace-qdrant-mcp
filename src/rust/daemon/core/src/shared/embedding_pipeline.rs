//! Semaphore-gated embedding pipeline extracted from duplicated inline patterns.
//!
//! The pattern of acquiring the embedding semaphore, calling `generate_embedding`,
//! and converting the sparse result to a `HashMap<u32, f32>` was duplicated in
//! every `process_*_item` function. This module provides canonical helpers.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::warn;

use crate::document_processor::chunking::floor_char_boundary;
use crate::embedding::{EmbeddingGenerator, SparseEmbedding};
use crate::unified_queue_processor::UnifiedProcessorError;

/// Result of a semaphore-gated embedding operation.
pub struct EmbedResult {
    /// Dense embedding vector.
    pub dense_vector: Vec<f32>,
    /// Sparse BM25 vector as index→weight map, or `None` if indices were empty.
    pub sparse_vector: Option<HashMap<u32, f32>>,
}

/// Clamp `text` to at most `max_bytes` UTF-8 bytes, cutting on a char boundary.
///
/// Returns the input untouched when it already fits — including the
/// `usize::MAX` case, which is how a provider says it imposes no caller-side
/// limit. When the cap lands inside a multi-byte character the cut falls back
/// to the preceding boundary, so the result is always a valid prefix.
fn bound_input(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    &text[..floor_char_boundary(text, max_bytes)]
}

/// Bound a single embedding input to the active provider's byte budget.
///
/// Content items (rules, scratchpad, generic content) reach the provider as one
/// undivided text, unlike the file path which splits oversized chunks before it
/// gets here. A provider with a finite context rejects an overlong input
/// outright, so the text is clamped and the loss is reported once.
fn bound_to_provider_budget<'a>(
    generator: &Arc<EmbeddingGenerator>,
    text: &'a str,
    model_hint: &str,
) -> &'a str {
    let max_bytes = generator.max_input_bytes();
    let bounded = bound_input(text, max_bytes);
    if bounded.len() < text.len() {
        warn!(
            original_bytes = text.len(),
            max_input_bytes = max_bytes,
            model_hint = model_hint,
            "Embedding input exceeds the provider's byte budget — truncating; \
             the tail of this item will not be embedded"
        );
    }
    bounded
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

    let bounded_text = bound_to_provider_budget(generator, text, model_hint);
    let embedding_result = generator
        .generate_embedding(bounded_text, model_hint)
        .await
        .map_err(UnifiedProcessorError::from)?;

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

    let bounded_text = bound_to_provider_budget(generator, text, model_hint);
    let embedding_result = generator
        .generate_embedding(bounded_text, model_hint)
        .await
        .map_err(UnifiedProcessorError::from)?;

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

    #[test]
    fn test_bound_input_below_cap_is_unchanged() {
        let text = "a short scratchpad note";
        assert_eq!(bound_input(text, 1024), text);
        // Exactly at the cap is also unchanged.
        assert_eq!(bound_input(text, text.len()), text);
    }

    #[test]
    fn test_bound_input_above_cap_truncates_on_char_boundary() {
        // "abc" is 3 bytes, "\u{e9}" is 2 bytes, so a cap of 4 lands inside the
        // multi-byte char and must fall back to the boundary at byte 3.
        let text = "abc\u{e9}def";
        let bounded = bound_input(text, 4);
        assert!(bounded.len() <= 4);
        assert_eq!(bounded, "abc");
        assert!(text.starts_with(bounded));

        // An em dash is 3 bytes; a cap of 4 lands inside it.
        let dashed = "ab\u{2014}cd";
        let bounded = bound_input(dashed, 4);
        assert!(bounded.len() <= 4);
        assert_eq!(bounded, "ab");
        assert!(dashed.starts_with(bounded));

        // A cap landing exactly after the multi-byte char keeps it whole.
        assert_eq!(bound_input(dashed, 5), "ab\u{2014}");
    }

    #[test]
    fn test_bound_input_unlimited_cap_is_unchanged() {
        let text = "unbounded provider, no caller-side limit \u{2014} keep it all";
        assert_eq!(bound_input(text, usize::MAX), text);
    }
}
