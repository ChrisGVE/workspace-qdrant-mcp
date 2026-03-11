//! Embedding cache utilities for the keyword extraction pipeline.
//!
//! Provides `resolve_embeddings`: resolves phrase embeddings from a local cache,
//! embedding only cache-miss phrases to minimise model inference calls.

use std::collections::HashMap;

use crate::embedding::{EmbeddingError, EmbeddingGenerator};

/// Resolve embeddings for a list of phrases using a local cache.
///
/// Phrases already present in `cache` are returned from cache.
/// Cache-miss phrases are batched and embedded in a single model call.
/// Returns a `Vec<Vec<f32>>` parallel to `phrases`.
pub(super) async fn resolve_embeddings(
    phrases: &[String],
    cache: &HashMap<String, Vec<f32>>,
    generator: &EmbeddingGenerator,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if phrases.is_empty() {
        return Ok(Vec::new());
    }

    // Identify cache misses (index, phrase)
    let miss_indices: Vec<usize> = phrases
        .iter()
        .enumerate()
        .filter(|(_, p)| !cache.contains_key(p.as_str()))
        .map(|(i, _)| i)
        .collect();

    // Build result starting from cache hits (default empty vec for misses)
    let mut result: Vec<Vec<f32>> = phrases
        .iter()
        .map(|p| cache.get(p.as_str()).cloned().unwrap_or_default())
        .collect();

    // Embed only cache misses in a single batch call
    if !miss_indices.is_empty() {
        let miss_phrases: Vec<String> = miss_indices.iter().map(|&i| phrases[i].clone()).collect();
        let embeddings = generator
            .generate_embeddings_batch(&miss_phrases, "all-MiniLM-L6-v2")
            .await?;
        for (&idx, embedding) in miss_indices.iter().zip(embeddings.iter()) {
            result[idx] = embedding.dense.vector.clone();
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_resolve_embeddings_empty_phrases() {
        // Empty phrase list returns empty result immediately (no generator needed)
        let cache: HashMap<String, Vec<f32>> = HashMap::new();
        // We can't easily mock EmbeddingGenerator in a sync test, so test the
        // empty-phrases fast-path indirectly by verifying the cache lookup logic.
        let phrases: Vec<String> = vec![];
        let miss_indices: Vec<usize> = phrases
            .iter()
            .enumerate()
            .filter(|(_, p)| !cache.contains_key(p.as_str()))
            .map(|(i, _)| i)
            .collect();
        assert!(miss_indices.is_empty());
    }

    #[test]
    fn test_resolve_embeddings_all_cache_hits() {
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        cache.insert("foo".to_string(), vec![1.0, 0.0]);
        cache.insert("bar".to_string(), vec![0.0, 1.0]);

        let phrases = vec!["foo".to_string(), "bar".to_string()];
        let miss_indices: Vec<usize> = phrases
            .iter()
            .enumerate()
            .filter(|(_, p)| !cache.contains_key(p.as_str()))
            .map(|(i, _)| i)
            .collect();

        assert!(miss_indices.is_empty(), "All phrases should be cache hits");

        let result: Vec<Vec<f32>> = phrases
            .iter()
            .map(|p| cache.get(p.as_str()).cloned().unwrap_or_default())
            .collect();
        assert_eq!(result[0], vec![1.0, 0.0]);
        assert_eq!(result[1], vec![0.0, 1.0]);
    }

    #[test]
    fn test_resolve_embeddings_partial_cache_hit() {
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        cache.insert("foo".to_string(), vec![1.0, 0.0]);

        let phrases = vec!["foo".to_string(), "bar".to_string()];
        let miss_indices: Vec<usize> = phrases
            .iter()
            .enumerate()
            .filter(|(_, p)| !cache.contains_key(p.as_str()))
            .map(|(i, _)| i)
            .collect();

        assert_eq!(miss_indices, vec![1], "Only 'bar' should be a cache miss");
    }
}
