//! Keyword selection with DF penalty.
//!
//! Selects top-K keywords from ranked candidates using:
//! - Combined semantic + lexical score from reranking
//! - Document frequency (DF) penalty to suppress generic terms
//! - Stability count (how many chunks contain the keyword)

use super::semantic_rerank::RankedCandidate;

/// A selected keyword with final scoring.
#[derive(Debug, Clone)]
pub struct SelectedKeyword {
    /// The keyword phrase
    pub phrase: String,
    /// Final score after DF penalty
    pub score: f64,
    /// Semantic similarity to parent
    pub semantic_score: f64,
    /// Lexical TF score
    pub lexical_score: f64,
    /// Number of chunks containing this keyword
    pub stability_count: u32,
    /// N-gram size
    pub ngram_size: u8,
}

/// Configuration for keyword selection.
#[derive(Debug, Clone)]
pub struct KeywordSelectionConfig {
    /// Maximum keywords to select per document
    pub max_keywords: usize,
    /// Maximum DF ratio (terms appearing in >80% of corpus are too generic)
    pub max_df_ratio: f64,
    /// Total documents in corpus (for DF calculation)
    pub corpus_size: u64,
}

impl Default for KeywordSelectionConfig {
    fn default() -> Self {
        Self {
            max_keywords: 50,
            max_df_ratio: 0.80,
            corpus_size: 0,
        }
    }
}

/// Compute BM25-style IDF weight for a term.
///
/// Formula: `ln((N - df + 0.5) / (df + 0.5))`
/// Returns 0.0 for terms that appear in all documents.
pub fn idf_weight(total_docs: u64, doc_freq: u64) -> f64 {
    if total_docs == 0 || doc_freq == 0 {
        return 1.0; // No corpus data, neutral weight
    }
    let n = total_docs as f64;
    let df = doc_freq as f64;
    let idf = ((n - df + 0.5) / (df + 0.5)).ln();
    idf.max(0.0) // Clamp to non-negative
}

/// Select top-K keywords from ranked candidates.
///
/// # Arguments
/// * `candidates` - Reranked candidates from semantic_rerank
/// * `doc_freq_lookup` - Function that returns document frequency for a term
/// * `chunk_count_lookup` - Function that returns how many chunks contain the term
/// * `config` - Selection configuration
pub fn select_keywords<F, G>(
    candidates: &[RankedCandidate],
    doc_freq_lookup: F,
    chunk_count_lookup: G,
    config: &KeywordSelectionConfig,
) -> Vec<SelectedKeyword>
where
    F: Fn(&str) -> u64,
    G: Fn(&str) -> u32,
{
    let mut selected: Vec<SelectedKeyword> = candidates
        .iter()
        .filter_map(|c| {
            let df = doc_freq_lookup(&c.phrase);

            // Filter out terms that are too common
            if config.corpus_size > 0 {
                let df_ratio = df as f64 / config.corpus_size as f64;
                if df_ratio > config.max_df_ratio {
                    return None;
                }
            }

            // Compute IDF-weighted score
            let idf = idf_weight(config.corpus_size, df);
            let score = c.combined_score * idf;

            let stability = chunk_count_lookup(&c.phrase);

            Some(SelectedKeyword {
                phrase: c.phrase.clone(),
                score,
                semantic_score: c.semantic_score,
                lexical_score: c.lexical_score,
                stability_count: stability,
                ngram_size: c.ngram_size,
            })
        })
        .collect();

    // Sort by score descending
    selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top K
    selected.truncate(config.max_keywords);

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(phrase: &str, combined: f64, semantic: f64, lexical: f64) -> RankedCandidate {
        RankedCandidate {
            phrase: phrase.to_string(),
            ngram_size: phrase.split(' ').count() as u8,
            raw_tf: 3,
            lexical_score: lexical,
            semantic_score: semantic,
            combined_score: combined,
        }
    }

    #[test]
    fn test_select_keywords_basic() {
        let candidates = vec![
            make_candidate("vector search", 0.9, 0.85, 2.0),
            make_candidate("embedding", 0.7, 0.65, 1.8),
            make_candidate("database", 0.5, 0.45, 1.5),
        ];
        let config = KeywordSelectionConfig {
            max_keywords: 10,
            ..Default::default()
        };

        let selected = select_keywords(&candidates, |_| 0, |_| 2, &config);
        assert_eq!(selected.len(), 3);
        // First should be highest scored
        assert_eq!(selected[0].phrase, "vector search");
    }

    #[test]
    fn test_select_keywords_df_penalty() {
        let candidates = vec![
            make_candidate("data", 0.9, 0.85, 2.0),   // Very common term
            make_candidate("qdrant", 0.7, 0.65, 1.8),  // Rare term
        ];
        let config = KeywordSelectionConfig {
            max_keywords: 10,
            max_df_ratio: 0.80,
            corpus_size: 100,
        };

        // "data" appears in 90% of docs, "qdrant" in 5%
        let selected = select_keywords(
            &candidates,
            |phrase| if phrase == "data" { 90 } else { 5 },
            |_| 2,
            &config,
        );

        // "data" should be filtered (90% > 80% max)
        assert!(!selected.iter().any(|k| k.phrase == "data"), "'data' should be filtered by DF ratio");
        assert!(selected.iter().any(|k| k.phrase == "qdrant"), "'qdrant' should survive");
    }

    #[test]
    fn test_select_keywords_max_limit() {
        let candidates: Vec<RankedCandidate> = (0..100)
            .map(|i| make_candidate(&format!("term_{}", i), 1.0 - i as f64 * 0.01, 0.5, 1.0))
            .collect();
        let config = KeywordSelectionConfig {
            max_keywords: 5,
            ..Default::default()
        };

        let selected = select_keywords(&candidates, |_| 0, |_| 1, &config);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_select_keywords_stability_count() {
        let candidates = vec![
            make_candidate("async runtime", 0.8, 0.7, 1.5),
        ];
        let config = KeywordSelectionConfig::default();

        let selected = select_keywords(&candidates, |_| 0, |_| 5, &config);
        assert_eq!(selected[0].stability_count, 5);
    }

    #[test]
    fn test_idf_weight_basic() {
        // Rare term (df=1 out of 1000)
        let rare = idf_weight(1000, 1);
        // Common term (df=500 out of 1000)
        let common = idf_weight(1000, 500);

        assert!(rare > common, "Rare term should have higher IDF: {} vs {}", rare, common);
    }

    #[test]
    fn test_idf_weight_zero_corpus() {
        let idf = idf_weight(0, 0);
        assert_eq!(idf, 1.0, "Zero corpus should return neutral weight");
    }

    #[test]
    fn test_idf_weight_universal_term() {
        // Term appears in all documents
        let idf = idf_weight(100, 100);
        assert_eq!(idf, 0.0, "Universal term should have IDF = 0");
    }

    #[test]
    fn test_select_keywords_empty_input() {
        let selected = select_keywords(&[], |_| 0, |_| 0, &KeywordSelectionConfig::default());
        assert!(selected.is_empty());
    }
}
