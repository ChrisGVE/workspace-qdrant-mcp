//! Tag selection with Maximal Marginal Relevance (MMR) diversity.
//!
//! Selects 5-12 tags per document ensuring topic diversity.
//! Uses MMR: each subsequent tag maximizes relevance while minimizing
//! redundancy with already-selected tags.

use super::semantic_rerank::{cosine_similarity, RankedCandidate};

/// A selected tag with scoring metadata.
#[derive(Debug, Clone)]
pub struct SelectedTag {
    /// The tag phrase
    pub phrase: String,
    /// Tag type: concept or structural
    pub tag_type: TagType,
    /// MMR score at time of selection
    pub score: f64,
    /// Diversity score: 1 - max_similarity_to_already_selected
    pub diversity_score: f64,
    /// Original semantic similarity to parent
    pub semantic_score: f64,
    /// N-gram size
    pub ngram_size: u8,
}

/// Tag classification.
#[derive(Debug, Clone, PartialEq)]
pub enum TagType {
    /// Derived from content semantics
    Concept,
    /// Derived from metadata (language, framework, layer)
    Structural,
}

impl TagType {
    pub fn as_str(&self) -> &str {
        match self {
            TagType::Concept => "concept",
            TagType::Structural => "structural",
        }
    }
}

/// Configuration for tag selection.
#[derive(Debug, Clone)]
pub struct TagSelectionConfig {
    /// Maximum tags to select
    pub max_tags: usize,
    /// MMR lambda: balance between relevance and diversity (0.0 to 1.0)
    /// Higher = more relevance, lower = more diversity
    pub lambda: f64,
    /// Maximum cosine similarity between any two selected tags
    pub max_inter_tag_similarity: f64,
    /// Minimum stability count for code documents (chunks containing the tag)
    pub min_stability_for_code: u32,
    /// Minimum number of chunks before stability filter applies
    pub stability_chunk_threshold: usize,
}

impl Default for TagSelectionConfig {
    fn default() -> Self {
        Self {
            max_tags: 8,
            lambda: 0.7,
            max_inter_tag_similarity: 0.80,
            min_stability_for_code: 2,
            stability_chunk_threshold: 5,
        }
    }
}

/// Select diverse tags from ranked candidates using MMR.
///
/// # Arguments
/// * `candidates` - Ranked candidates with embeddings
/// * `candidate_vectors` - 384-dim embeddings for each candidate (parallel with candidates)
/// * `config` - Tag selection configuration
///
/// Candidates and candidate_vectors must have the same length.
pub fn select_tags(
    candidates: &[RankedCandidate],
    candidate_vectors: &[Vec<f32>],
    config: &TagSelectionConfig,
) -> Vec<SelectedTag> {
    if candidates.is_empty() || candidate_vectors.is_empty() {
        return Vec::new();
    }

    assert_eq!(
        candidates.len(),
        candidate_vectors.len(),
        "candidates and vectors must have same length"
    );

    let mut selected: Vec<(usize, f64)> = Vec::new(); // (index, mmr_score)
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();

    while selected.len() < config.max_tags && !remaining.is_empty() {
        let mut best_idx = None;
        let mut best_mmr = f64::NEG_INFINITY;

        for &r in &remaining {
            let relevance = candidates[r].combined_score;

            // Compute max similarity to already selected tags
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|(s, _)| cosine_similarity(&candidate_vectors[r], &candidate_vectors[*s]))
                    .fold(0.0f64, |a, b| a.max(b))
            };

            // Reject if too similar to an already-selected tag
            if max_sim > config.max_inter_tag_similarity && !selected.is_empty() {
                continue;
            }

            // MMR score
            let mmr = config.lambda * relevance - (1.0 - config.lambda) * max_sim;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = Some(r);
            }
        }

        match best_idx {
            Some(idx) => {
                selected.push((idx, best_mmr));
                remaining.retain(|&r| r != idx);
            }
            None => break, // No more candidates pass the similarity filter
        }
    }

    // Build output
    selected
        .iter()
        .map(|(idx, mmr_score)| {
            let candidate = &candidates[*idx];
            let diversity = if selected.len() > 1 {
                let max_sim = selected
                    .iter()
                    .filter(|(s, _)| *s != *idx)
                    .map(|(s, _)| {
                        cosine_similarity(&candidate_vectors[*idx], &candidate_vectors[*s])
                    })
                    .fold(0.0f64, |a, b| a.max(b));
                1.0 - max_sim
            } else {
                1.0
            };

            SelectedTag {
                phrase: candidate.phrase.clone(),
                tag_type: TagType::Concept,
                score: *mmr_score,
                diversity_score: diversity,
                semantic_score: candidate.semantic_score,
                ngram_size: candidate.ngram_size,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(phrase: &str, combined: f64, semantic: f64) -> RankedCandidate {
        RankedCandidate {
            phrase: phrase.to_string(),
            ngram_size: phrase.split(' ').count() as u8,
            raw_tf: 3,
            lexical_score: 1.5,
            semantic_score: semantic,
            combined_score: combined,
        }
    }

    #[test]
    fn test_select_tags_basic() {
        let candidates = vec![
            make_candidate("vector search", 0.9, 0.85),
            make_candidate("database", 0.7, 0.65),
            make_candidate("embedding", 0.6, 0.55),
        ];
        // Use orthogonal vectors to ensure all are selected (no redundancy)
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let config = TagSelectionConfig {
            max_tags: 3,
            ..Default::default()
        };

        let tags = select_tags(&candidates, &vectors, &config);
        assert_eq!(tags.len(), 3);
        // First tag should be highest relevance
        assert_eq!(tags[0].phrase, "vector search");
    }

    #[test]
    fn test_select_tags_diversity() {
        let candidates = vec![
            make_candidate("vector search", 0.9, 0.85),
            make_candidate("vector indexing", 0.85, 0.80), // Very similar to first
            make_candidate("grpc protocol", 0.6, 0.55),    // Different topic
        ];
        // Make first two very similar, third different
        let vectors = vec![
            vec![0.95, 0.31, 0.0], // "vector search"
            vec![0.95, 0.30, 0.0], // "vector indexing" - very similar
            vec![0.0, 0.0, 1.0],   // "grpc protocol" - orthogonal
        ];
        let config = TagSelectionConfig {
            max_tags: 2,
            max_inter_tag_similarity: 0.80,
            ..Default::default()
        };

        let tags = select_tags(&candidates, &vectors, &config);
        assert_eq!(tags.len(), 2);

        // "vector search" should be first (highest relevance)
        assert_eq!(tags[0].phrase, "vector search");
        // "grpc protocol" should be second (diversity over "vector indexing")
        assert_eq!(
            tags[1].phrase, "grpc protocol",
            "Should select diverse 'grpc protocol' over similar 'vector indexing'"
        );
    }

    #[test]
    fn test_select_tags_max_limit() {
        let candidates: Vec<RankedCandidate> = (0..20)
            .map(|i| make_candidate(&format!("tag_{}", i), 1.0 - i as f64 * 0.05, 0.5))
            .collect();
        // Use random-ish orthogonal vectors
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 20];
                v[i] = 1.0;
                v
            })
            .collect();
        let config = TagSelectionConfig {
            max_tags: 5,
            ..Default::default()
        };

        let tags = select_tags(&candidates, &vectors, &config);
        assert!(tags.len() <= 5, "Should not exceed max_tags");
    }

    #[test]
    fn test_select_tags_empty_input() {
        let tags = select_tags(&[], &[], &TagSelectionConfig::default());
        assert!(tags.is_empty());
    }

    #[test]
    fn test_tag_type_as_str() {
        assert_eq!(TagType::Concept.as_str(), "concept");
        assert_eq!(TagType::Structural.as_str(), "structural");
    }

    #[test]
    fn test_select_tags_single_candidate() {
        let candidates = vec![make_candidate("only tag", 0.9, 0.85)];
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        let config = TagSelectionConfig::default();

        let tags = select_tags(&candidates, &vectors, &config);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].phrase, "only tag");
        assert_eq!(
            tags[0].diversity_score, 1.0,
            "Single tag should have max diversity"
        );
    }

    #[test]
    fn test_select_tags_all_identical_vectors() {
        let candidates = vec![
            make_candidate("tag_a", 0.9, 0.85),
            make_candidate("tag_b", 0.8, 0.75),
            make_candidate("tag_c", 0.7, 0.65),
        ];
        // All identical vectors - high inter-similarity
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        let config = TagSelectionConfig {
            max_tags: 3,
            max_inter_tag_similarity: 0.80,
            ..Default::default()
        };

        let tags = select_tags(&candidates, &vectors, &config);
        // Should only select one (all others are too similar)
        assert_eq!(
            tags.len(),
            1,
            "Only first should be selected when all are identical"
        );
        assert_eq!(tags[0].phrase, "tag_a");
    }

    #[test]
    fn test_tag_selection_config_defaults() {
        let config = TagSelectionConfig::default();
        assert_eq!(config.max_tags, 8);
        assert!((config.lambda - 0.7).abs() < 1e-6);
        assert!((config.max_inter_tag_similarity - 0.80).abs() < 1e-6);
    }
}
