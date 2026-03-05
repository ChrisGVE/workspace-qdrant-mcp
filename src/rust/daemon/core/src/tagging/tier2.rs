//! Tier 2 automated tagging -- embedding-based classification.
//!
//! Uses FastEmbed cosine similarity for zero-shot taxonomy matching:
//! 1. Load a taxonomy of ~180 concept terms from YAML
//! 2. Embed each term once (cached in memory)
//! 3. For each document, compute aggregate embedding from chunk embeddings
//! 4. Cosine similarity against all taxonomy embeddings
//! 5. Top-N matches above threshold become concept tags

use std::collections::HashMap;

use crate::embedding::{EmbeddingError, EmbeddingGenerator};
use crate::keyword_extraction::semantic_rerank::cosine_similarity;
use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

use super::taxonomy::TaxonomyEntry;

// ── Configuration ────────────────────────────────────────────────────────

/// Configuration for Tier 2 tagging.
#[derive(Debug, Clone)]
pub struct Tier2Config {
    /// Minimum cosine similarity to accept a taxonomy match.
    pub similarity_threshold: f64,
    /// Maximum number of taxonomy tags to assign per document.
    pub max_tags: usize,
    /// Minimum gap between the best and next-best category match
    /// to suppress near-ties (noisy matches).
    pub min_score_gap: f64,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.35,
            max_tags: 10,
            min_score_gap: 0.02,
        }
    }
}

// ── Tagger ───────────────────────────────────────────────────────────────

/// A taxonomy match result before final selection.
#[derive(Debug, Clone)]
pub struct TaxonomyMatch {
    /// The matched taxonomy term.
    pub term: String,
    /// The category of the matched term.
    pub category: String,
    /// Cosine similarity score.
    pub score: f64,
}

/// Tier 2 tagger: zero-shot taxonomy classification via embeddings.
///
/// Holds pre-computed taxonomy embeddings. Thread-safe via shared ref.
pub struct Tier2Tagger {
    /// Taxonomy entries parallel with `embeddings`.
    entries: Vec<TaxonomyEntry>,
    /// Pre-computed 384-dim embeddings for each taxonomy term.
    embeddings: Vec<Vec<f32>>,
    /// Configuration.
    config: Tier2Config,
}

impl Tier2Tagger {
    /// Create a new tagger by embedding all taxonomy terms.
    ///
    /// This is an expensive operation (~180 embeddings) but only runs once.
    pub async fn new(
        entries: Vec<TaxonomyEntry>,
        embedding_generator: &EmbeddingGenerator,
        config: Tier2Config,
    ) -> Result<Self, EmbeddingError> {
        let terms: Vec<String> = entries.iter().map(|e| e.term.clone()).collect();
        let results = embedding_generator
            .generate_embeddings_batch(&terms, "all-MiniLM-L6-v2")
            .await?;
        let embeddings: Vec<Vec<f32>> = results.into_iter().map(|r| r.dense.vector).collect();

        Ok(Self {
            entries,
            embeddings,
            config,
        })
    }

    /// Create a tagger with pre-computed embeddings (for testing).
    pub fn from_precomputed(
        entries: Vec<TaxonomyEntry>,
        embeddings: Vec<Vec<f32>>,
        config: Tier2Config,
    ) -> Self {
        assert_eq!(entries.len(), embeddings.len());
        Self {
            entries,
            embeddings,
            config,
        }
    }

    /// Classify a document by its aggregate embedding.
    ///
    /// Returns taxonomy matches sorted by score descending, limited to
    /// `config.max_tags` and filtered by `config.similarity_threshold`.
    pub fn classify(&self, document_embedding: &[f32]) -> Vec<TaxonomyMatch> {
        if document_embedding.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        // Score every taxonomy term
        let mut matches: Vec<TaxonomyMatch> = self
            .entries
            .iter()
            .zip(self.embeddings.iter())
            .map(|(entry, tax_emb)| TaxonomyMatch {
                term: entry.term.clone(),
                category: entry.category.clone(),
                score: cosine_similarity(tax_emb, document_embedding),
            })
            .filter(|m| m.score >= self.config.similarity_threshold)
            .collect();

        // Sort descending by score
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate by category: keep best term per category
        let mut seen_categories: HashMap<String, f64> = HashMap::new();
        matches.retain(|m| {
            match seen_categories.get(&m.category) {
                Some(&best) => {
                    // Allow second term from same category only if gap is large
                    best - m.score > self.config.min_score_gap * 3.0
                }
                None => {
                    seen_categories.insert(m.category.clone(), m.score);
                    true
                }
            }
        });

        // Limit to max_tags
        matches.truncate(self.config.max_tags);

        matches
    }

    /// Convert taxonomy matches to `SelectedTag` for integration with the
    /// existing keyword extraction pipeline.
    pub fn matches_to_tags(matches: &[TaxonomyMatch]) -> Vec<SelectedTag> {
        matches
            .iter()
            .map(|m| SelectedTag {
                phrase: format!("tax:{}", m.category),
                tag_type: TagType::Concept,
                score: m.score,
                diversity_score: 1.0,
                semantic_score: m.score,
                ngram_size: 1,
            })
            .collect()
    }

    /// Get the number of taxonomy entries.
    pub fn taxonomy_size(&self) -> usize {
        self.entries.len()
    }

    /// Get configuration.
    pub fn config(&self) -> &Tier2Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_taxonomy() -> Vec<TaxonomyEntry> {
        vec![
            TaxonomyEntry {
                term: "machine learning algorithms".into(),
                category: "machine-learning".into(),
            },
            TaxonomyEntry {
                term: "web application development".into(),
                category: "web-development".into(),
            },
            TaxonomyEntry {
                term: "database indexing".into(),
                category: "databases".into(),
            },
            TaxonomyEntry {
                term: "unit testing".into(),
                category: "testing".into(),
            },
            TaxonomyEntry {
                term: "async await programming".into(),
                category: "concurrency".into(),
            },
        ]
    }

    /// Create mock embeddings: each term gets a distinct unit vector.
    fn mock_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn test_classify_exact_match() {
        let entries = sample_taxonomy();
        let embeddings = mock_embeddings(5, 10);
        let config = Tier2Config {
            similarity_threshold: 0.5,
            max_tags: 5,
            min_score_gap: 0.02,
        };
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let mut query = vec![0.0f32; 10];
        query[0] = 1.0;

        let matches = tagger.classify(&query);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].category, "machine-learning");
        assert!((matches[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_partial_match() {
        let entries = sample_taxonomy();
        let embeddings = mock_embeddings(5, 10);
        let config = Tier2Config {
            similarity_threshold: 0.3,
            max_tags: 5,
            min_score_gap: 0.02,
        };
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let mut query = vec![0.0f32; 10];
        query[0] = 0.7;
        query[1] = 0.7;
        let norm = (0.7f32 * 0.7 + 0.7 * 0.7).sqrt();
        query[0] /= norm;
        query[1] /= norm;

        let matches = tagger.classify(&query);
        assert_eq!(matches.len(), 2);
        assert!(matches[0].score > 0.5);
        assert!(matches[1].score > 0.5);
    }

    #[test]
    fn test_classify_below_threshold() {
        let entries = sample_taxonomy();
        let embeddings = mock_embeddings(5, 10);
        let config = Tier2Config {
            similarity_threshold: 0.9,
            max_tags: 5,
            min_score_gap: 0.02,
        };
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let mut query = vec![0.1f32; 10];
        query[0] = 0.5;

        let matches = tagger.classify(&query);
        assert!(matches.is_empty() || matches[0].score >= 0.9);
    }

    #[test]
    fn test_classify_empty_embedding() {
        let entries = sample_taxonomy();
        let embeddings = mock_embeddings(5, 10);
        let config = Tier2Config::default();
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let matches = tagger.classify(&[]);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_classify_empty_taxonomy() {
        let config = Tier2Config::default();
        let tagger = Tier2Tagger::from_precomputed(Vec::new(), Vec::new(), config);

        let query = vec![1.0f32; 10];
        let matches = tagger.classify(&query);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_classify_max_tags_limit() {
        let entries: Vec<TaxonomyEntry> = (0..20)
            .map(|i| TaxonomyEntry {
                term: format!("term {}", i),
                category: format!("category-{}", i),
            })
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..20).map(|_| vec![1.0, 0.0, 0.0]).collect();
        let config = Tier2Config {
            similarity_threshold: 0.5,
            max_tags: 3,
            min_score_gap: 0.02,
        };
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let query = vec![1.0, 0.0, 0.0];
        let matches = tagger.classify(&query);
        assert!(matches.len() <= 3, "Should respect max_tags limit");
    }

    #[test]
    fn test_classify_category_dedup() {
        let entries = vec![
            TaxonomyEntry {
                term: "rust programming".into(),
                category: "programming-languages".into(),
            },
            TaxonomyEntry {
                term: "python programming".into(),
                category: "programming-languages".into(),
            },
            TaxonomyEntry {
                term: "web development".into(),
                category: "web-development".into(),
            },
        ];
        let embeddings = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.85, 0.15, 0.0],
            vec![0.8, 0.2, 0.0],
        ];
        let config = Tier2Config {
            similarity_threshold: 0.3,
            max_tags: 10,
            min_score_gap: 0.02,
        };
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, config);

        let query = vec![1.0, 0.0, 0.0];
        let matches = tagger.classify(&query);

        let lang_count = matches
            .iter()
            .filter(|m| m.category == "programming-languages")
            .count();
        assert!(lang_count <= 1, "Should deduplicate within category");
    }

    #[test]
    fn test_matches_to_tags() {
        let matches = vec![
            TaxonomyMatch {
                term: "machine learning".into(),
                category: "machine-learning".into(),
                score: 0.85,
            },
            TaxonomyMatch {
                term: "web development".into(),
                category: "web-development".into(),
                score: 0.72,
            },
        ];

        let tags = Tier2Tagger::matches_to_tags(&matches);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].phrase, "tax:machine-learning");
        assert_eq!(tags[0].tag_type, TagType::Concept);
        assert!((tags[0].score - 0.85).abs() < 1e-6);
        assert_eq!(tags[1].phrase, "tax:web-development");
    }

    #[test]
    fn test_matches_to_tags_empty() {
        let tags = Tier2Tagger::matches_to_tags(&[]);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_tier2_config_defaults() {
        let config = Tier2Config::default();
        assert!((config.similarity_threshold - 0.35).abs() < 1e-6);
        assert_eq!(config.max_tags, 10);
        assert!((config.min_score_gap - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_taxonomy_size() {
        let entries = sample_taxonomy();
        let n = entries.len();
        let embeddings = mock_embeddings(n, 10);
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, Tier2Config::default());
        assert_eq!(tagger.taxonomy_size(), 5);
    }
}
