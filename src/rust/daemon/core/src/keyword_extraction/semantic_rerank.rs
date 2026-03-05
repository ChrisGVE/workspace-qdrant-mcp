//! Semantic reranking of lexical candidates using FastEmbed cosine similarity.
//!
//! Scores each candidate phrase by cosine similarity to the parent document's
//! summary vector. Combines semantic score with lexical (TF) score for final ranking.

use super::lexical_candidates::LexicalCandidate;
use crate::embedding::{EmbeddingError, EmbeddingGenerator};

/// A candidate after semantic reranking with combined scores.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    /// The candidate phrase
    pub phrase: String,
    /// N-gram size (1, 2, or 3)
    pub ngram_size: u8,
    /// Raw term frequency
    pub raw_tf: u32,
    /// Sublinear TF score from lexical extraction
    pub lexical_score: f64,
    /// Cosine similarity to parent summary vector
    pub semantic_score: f64,
    /// Combined score: semantic_score * lexical_score
    pub combined_score: f64,
}

/// Configuration for semantic reranking.
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Minimum cosine similarity threshold to keep a candidate
    pub min_similarity: f64,
    /// Weight for semantic score in combined scoring (0.0 to 1.0)
    pub semantic_weight: f64,
    /// Batch size for embedding generation
    pub batch_size: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.15,
            semantic_weight: 0.6,
            batch_size: 32,
        }
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    dot / denom
}

/// Compute a weighted mean vector from a set of vectors with weights.
///
/// Used to compute parent summary vectors from chunk embeddings.
pub fn weighted_mean_vector(vectors: &[(Vec<f32>, f64)]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }

    let dim = vectors[0].0.len();
    if dim == 0 {
        return None;
    }

    let total_weight: f64 = vectors.iter().map(|(_, w)| w).sum();
    if total_weight < 1e-10 {
        return None;
    }

    let mut result = vec![0.0f32; dim];
    for (vec, weight) in vectors {
        if vec.len() != dim {
            continue;
        }
        for (i, v) in vec.iter().enumerate() {
            result[i] += (*v as f64 * weight) as f32;
        }
    }

    // Normalize by total weight
    for v in &mut result {
        *v /= total_weight as f32;
    }

    Some(result)
}

/// Rerank lexical candidates using semantic similarity to a parent summary vector.
///
/// # Arguments
/// * `candidates` - Lexical candidates from TF-IDF extraction
/// * `parent_vector` - Summary vector of the parent document (384-dim)
/// * `embedding_generator` - EmbeddingGenerator for embedding candidate phrases
/// * `config` - Reranking configuration
///
/// # Returns
/// Ranked candidates sorted by combined score descending.
pub async fn rerank_candidates(
    candidates: Vec<LexicalCandidate>,
    parent_vector: &[f32],
    embedding_generator: &EmbeddingGenerator,
    config: &RerankConfig,
) -> Result<Vec<RankedCandidate>, EmbeddingError> {
    if candidates.is_empty() || parent_vector.is_empty() {
        return Ok(Vec::new());
    }

    let phrases: Vec<String> = candidates.iter().map(|c| c.phrase.clone()).collect();

    // Embed all candidate phrases in batches
    let embeddings = embedding_generator
        .generate_embeddings_batch(&phrases, "all-MiniLM-L6-v2")
        .await?;

    // Score each candidate
    let sem_weight = config.semantic_weight;
    let lex_weight = 1.0 - sem_weight;

    let mut ranked: Vec<RankedCandidate> = candidates
        .into_iter()
        .zip(embeddings.iter())
        .filter_map(|(candidate, embedding)| {
            let semantic_score = cosine_similarity(&embedding.dense.vector, parent_vector);

            // Filter by minimum similarity
            if semantic_score < config.min_similarity {
                return None;
            }

            let combined = sem_weight * semantic_score + lex_weight * candidate.tf_score;

            Some(RankedCandidate {
                phrase: candidate.phrase,
                ngram_size: candidate.ngram_size,
                raw_tf: candidate.raw_tf,
                lexical_score: candidate.tf_score,
                semantic_score,
                combined_score: combined,
            })
        })
        .collect();

    // Sort by combined score descending
    ranked.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "Opposite vectors should have similarity ~-1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Different length vectors should return 0.0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should return 0.0");
    }

    #[test]
    fn test_weighted_mean_vector_basic() {
        let vectors = vec![(vec![1.0, 0.0, 0.0], 1.0), (vec![0.0, 1.0, 0.0], 1.0)];
        let mean = weighted_mean_vector(&vectors).unwrap();
        assert_eq!(mean.len(), 3);
        assert!((mean[0] - 0.5).abs() < 1e-5);
        assert!((mean[1] - 0.5).abs() < 1e-5);
        assert!((mean[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_mean_vector_weighted() {
        let vectors = vec![(vec![1.0, 0.0], 3.0), (vec![0.0, 1.0], 1.0)];
        let mean = weighted_mean_vector(&vectors).unwrap();
        assert!(
            (mean[0] - 0.75).abs() < 1e-5,
            "Expected 0.75, got {}",
            mean[0]
        );
        assert!(
            (mean[1] - 0.25).abs() < 1e-5,
            "Expected 0.25, got {}",
            mean[1]
        );
    }

    #[test]
    fn test_weighted_mean_vector_empty() {
        let vectors: Vec<(Vec<f32>, f64)> = vec![];
        assert!(weighted_mean_vector(&vectors).is_none());
    }

    #[test]
    fn test_weighted_mean_vector_zero_weights() {
        let vectors = vec![(vec![1.0, 0.0], 0.0), (vec![0.0, 1.0], 0.0)];
        assert!(weighted_mean_vector(&vectors).is_none());
    }

    #[test]
    fn test_ranked_candidate_struct() {
        let candidate = RankedCandidate {
            phrase: "vector search".to_string(),
            ngram_size: 2,
            raw_tf: 5,
            lexical_score: 2.6,
            semantic_score: 0.85,
            combined_score: 1.55,
        };
        assert_eq!(candidate.phrase, "vector search");
        assert_eq!(candidate.ngram_size, 2);
    }

    #[test]
    fn test_rerank_config_defaults() {
        let config = RerankConfig::default();
        assert!((config.min_similarity - 0.15).abs() < 1e-6);
        assert!((config.semantic_weight - 0.6).abs() < 1e-6);
        assert_eq!(config.batch_size, 32);
    }
}
