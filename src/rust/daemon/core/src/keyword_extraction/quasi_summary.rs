//! Quasi-summary generation via weighted mean of chunk embeddings.
//!
//! Produces a parent summary vector without LLM summarization.
//! Code documents use BM25-style salience weighting; prose documents
//! use centrality weighting (chunk similarity to other chunks).

use super::semantic_rerank::{cosine_similarity, weighted_mean_vector};

/// Configuration for quasi-summary generation.
#[derive(Debug, Clone)]
pub struct QuasiSummaryConfig {
    /// BM25 k1 parameter for TF saturation
    pub bm25_k1: f64,
    /// BM25 b parameter for length normalization
    pub bm25_b: f64,
    /// Number of top chunks to select as extractive gist
    pub gist_chunks: usize,
}

impl Default for QuasiSummaryConfig {
    fn default() -> Self {
        Self {
            bm25_k1: 1.2,
            bm25_b: 0.75,
            gist_chunks: 3,
        }
    }
}

/// Result of quasi-summary generation.
#[derive(Debug, Clone)]
pub struct QuasiSummary {
    /// Weighted mean vector (same dimension as chunk vectors)
    pub summary_vector: Vec<f32>,
    /// Indices of the top gist chunks (sorted by weight descending)
    pub gist_indices: Vec<usize>,
    /// Weight assigned to each chunk (parallel with input chunks)
    pub chunk_weights: Vec<f64>,
}

/// Compute BM25-style salience weight for a chunk relative to the full document.
///
/// Each chunk is treated as a "query" against the document.
/// Weight = sum of BM25 term scores for terms in the chunk.
fn bm25_chunk_weight(
    chunk_tokens: &[String],
    doc_token_counts: &std::collections::HashMap<String, u32>,
    doc_total_tokens: usize,
    avg_chunk_len: f64,
    config: &QuasiSummaryConfig,
) -> f64 {
    if chunk_tokens.is_empty() || doc_total_tokens == 0 {
        return 1.0;
    }

    // Count term frequencies in this chunk
    let mut chunk_tf: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for token in chunk_tokens {
        *chunk_tf.entry(token.as_str()).or_insert(0) += 1;
    }

    let chunk_len = chunk_tokens.len() as f64;
    let k1 = config.bm25_k1;
    let b = config.bm25_b;

    let mut weight = 0.0;
    for (term, &tf) in &chunk_tf {
        let df = doc_token_counts.get(*term).copied().unwrap_or(0) as f64;
        let n = doc_total_tokens as f64;

        // IDF component: ln((N - df + 0.5) / (df + 0.5))
        let idf = ((n - df + 0.5) / (df + 0.5)).ln().max(0.0);

        // TF saturation with length normalization
        let tf_f = tf as f64;
        let norm_len = 1.0 - b + b * (chunk_len / avg_chunk_len);
        let tf_component = (tf_f * (k1 + 1.0)) / (tf_f + k1 * norm_len);

        weight += idf * tf_component;
    }

    weight.max(0.01) // Floor to prevent zero weights
}

/// Compute centrality weight for prose chunks.
///
/// Each chunk's weight = mean cosine similarity to all other chunks.
/// More central chunks (similar to many others) get higher weight.
fn centrality_weights(chunk_vectors: &[Vec<f32>]) -> Vec<f64> {
    let n = chunk_vectors.len();
    if n <= 1 {
        return vec![1.0; n];
    }

    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut total_sim = 0.0;
        for j in 0..n {
            if i != j {
                total_sim += cosine_similarity(&chunk_vectors[i], &chunk_vectors[j]);
            }
        }
        weights[i] = (total_sim / (n - 1) as f64).max(0.01);
    }

    weights
}

/// Generate quasi-summary for code documents.
///
/// Uses BM25-style weighting: chunks with distinctive terms get higher weight.
///
/// # Arguments
/// * `chunk_vectors` - Embeddings for each chunk
/// * `chunk_tokens` - Tokenized text for each chunk (parallel with vectors)
/// * `config` - Summary configuration
pub fn summarize_code(
    chunk_vectors: &[Vec<f32>],
    chunk_tokens: &[Vec<String>],
    config: &QuasiSummaryConfig,
) -> Option<QuasiSummary> {
    if chunk_vectors.is_empty() || chunk_tokens.is_empty() {
        return None;
    }

    let n = chunk_vectors.len().min(chunk_tokens.len());

    // Build document-level token counts (how many chunks contain each term)
    let mut doc_token_counts: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    for tokens in &chunk_tokens[..n] {
        let unique: std::collections::HashSet<&str> =
            tokens.iter().map(|t| t.as_str()).collect();
        for term in unique {
            *doc_token_counts.entry(term.to_string()).or_insert(0) += 1;
        }
    }

    let avg_chunk_len: f64 =
        chunk_tokens[..n].iter().map(|t| t.len() as f64).sum::<f64>() / n as f64;

    // Compute weights
    let chunk_weights: Vec<f64> = chunk_tokens[..n]
        .iter()
        .map(|tokens| {
            bm25_chunk_weight(tokens, &doc_token_counts, n, avg_chunk_len, config)
        })
        .collect();

    build_summary(chunk_vectors, &chunk_weights, config)
}

/// Generate quasi-summary for prose documents.
///
/// Uses centrality weighting: chunks similar to many others are more central.
///
/// # Arguments
/// * `chunk_vectors` - Embeddings for each chunk
/// * `config` - Summary configuration
pub fn summarize_prose(
    chunk_vectors: &[Vec<f32>],
    config: &QuasiSummaryConfig,
) -> Option<QuasiSummary> {
    if chunk_vectors.is_empty() {
        return None;
    }

    let chunk_weights = centrality_weights(chunk_vectors);
    build_summary(chunk_vectors, &chunk_weights, config)
}

/// Build summary from chunk vectors and their weights.
fn build_summary(
    chunk_vectors: &[Vec<f32>],
    chunk_weights: &[f64],
    config: &QuasiSummaryConfig,
) -> Option<QuasiSummary> {
    let n = chunk_vectors.len().min(chunk_weights.len());
    if n == 0 {
        return None;
    }

    // Build weighted pairs for weighted_mean_vector
    let pairs: Vec<(Vec<f32>, f64)> = chunk_vectors[..n]
        .iter()
        .zip(chunk_weights[..n].iter())
        .map(|(v, &w)| (v.clone(), w))
        .collect();

    let summary_vector = weighted_mean_vector(&pairs)?;

    // Select top gist chunk indices by weight
    let mut indexed_weights: Vec<(usize, f64)> =
        chunk_weights[..n].iter().copied().enumerate().collect();
    indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let gist_indices: Vec<usize> = indexed_weights
        .iter()
        .take(config.gist_chunks.min(n))
        .map(|(i, _)| *i)
        .collect();

    Some(QuasiSummary {
        summary_vector,
        gist_indices,
        chunk_weights: chunk_weights[..n].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centrality_weights_basic() {
        // Three vectors: A and B similar, C different
        let vectors = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.85, 0.15, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let weights = centrality_weights(&vectors);
        assert_eq!(weights.len(), 3);
        // A and B should have higher centrality than C (they're similar to each other)
        assert!(
            weights[0] > weights[2],
            "A should be more central than C: {} vs {}",
            weights[0],
            weights[2]
        );
        assert!(
            weights[1] > weights[2],
            "B should be more central than C: {} vs {}",
            weights[1],
            weights[2]
        );
    }

    #[test]
    fn test_centrality_weights_single() {
        let vectors = vec![vec![1.0, 0.0]];
        let weights = centrality_weights(&vectors);
        assert_eq!(weights, vec![1.0]);
    }

    #[test]
    fn test_centrality_weights_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let weights = centrality_weights(&vectors);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_bm25_chunk_weight_distinctive_terms() {
        let mut doc_counts = std::collections::HashMap::new();
        doc_counts.insert("common".to_string(), 10);
        doc_counts.insert("rare".to_string(), 1);

        let chunk_with_rare = vec!["rare".to_string(), "common".to_string()];
        let chunk_all_common = vec!["common".to_string(), "common".to_string()];
        let config = QuasiSummaryConfig::default();

        let w_rare = bm25_chunk_weight(&chunk_with_rare, &doc_counts, 10, 2.0, &config);
        let w_common = bm25_chunk_weight(&chunk_all_common, &doc_counts, 10, 2.0, &config);

        assert!(
            w_rare > w_common,
            "Chunk with rare term should have higher weight: {} vs {}",
            w_rare,
            w_common
        );
    }

    #[test]
    fn test_bm25_chunk_weight_empty() {
        let doc_counts = std::collections::HashMap::new();
        let config = QuasiSummaryConfig::default();
        let w = bm25_chunk_weight(&[], &doc_counts, 0, 1.0, &config);
        assert_eq!(w, 1.0, "Empty chunk should return neutral weight");
    }

    #[test]
    fn test_summarize_code_basic() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let tokens = vec![
            vec!["async".to_string(), "fn".to_string()],
            vec!["let".to_string(), "mut".to_string()],
            vec!["async".to_string(), "runtime".to_string()],
        ];
        let config = QuasiSummaryConfig::default();

        let summary = summarize_code(&vectors, &tokens, &config);
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert_eq!(s.summary_vector.len(), 3);
        assert_eq!(s.chunk_weights.len(), 3);
        assert!(s.gist_indices.len() <= 3);
    }

    #[test]
    fn test_summarize_prose_basic() {
        let vectors = vec![
            vec![0.9, 0.1, 0.0],  // similar to B
            vec![0.85, 0.15, 0.0], // similar to A
            vec![0.0, 0.0, 1.0],  // outlier
        ];
        let config = QuasiSummaryConfig::default();

        let summary = summarize_prose(&vectors, &config);
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert_eq!(s.summary_vector.len(), 3);
        // Gist should prefer central chunks (A and B)
        assert!(
            s.gist_indices.contains(&0) || s.gist_indices.contains(&1),
            "Gist should include at least one central chunk"
        );
    }

    #[test]
    fn test_summarize_empty() {
        let config = QuasiSummaryConfig::default();
        assert!(summarize_code(&[], &[], &config).is_none());
        assert!(summarize_prose(&[], &config).is_none());
    }

    #[test]
    fn test_gist_indices_sorted_by_weight() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        // Tokens that make chunk 2 have highest BM25 weight (rare terms)
        let tokens = vec![
            vec!["the".to_string()],            // very common
            vec!["the".to_string()],            // very common
            vec!["quantum".to_string()],        // rare
        ];
        let config = QuasiSummaryConfig {
            gist_chunks: 2,
            ..Default::default()
        };

        let summary = summarize_code(&vectors, &tokens, &config);
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert_eq!(s.gist_indices.len(), 2);
        // First gist index should be the chunk with highest weight
        let w0 = s.chunk_weights[s.gist_indices[0]];
        let w1 = s.chunk_weights[s.gist_indices[1]];
        assert!(
            w0 >= w1,
            "Gist indices should be sorted by weight descending: {} vs {}",
            w0,
            w1
        );
    }

    #[test]
    fn test_summary_vector_dimensions() {
        let vectors = vec![
            vec![0.1; 384],
            vec![0.2; 384],
        ];
        let config = QuasiSummaryConfig::default();
        let summary = summarize_prose(&vectors, &config).unwrap();
        assert_eq!(
            summary.summary_vector.len(),
            384,
            "Summary should match input dimension (384)"
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = QuasiSummaryConfig::default();
        assert!((config.bm25_k1 - 1.2).abs() < 1e-6);
        assert!((config.bm25_b - 0.75).abs() < 1e-6);
        assert_eq!(config.gist_chunks, 3);
    }
}
