//! Keyword/tag extraction pipeline orchestrator.
//!
//! Chains the full extraction pipeline:
//! 1. Quasi-summary vector generation
//! 2. Lexical candidate extraction (TF-IDF)
//! 3. LSP candidate extraction (imports/symbols)
//! 4. Semantic reranking (cosine similarity to summary)
//! 5. Keyword selection (IDF penalty)
//! 6. Tag selection (MMR diversity)
//! 7. Keyword basket assignment
//! 8. Structural tag extraction (code only)

use std::path::Path;

use crate::embedding::{EmbeddingError, EmbeddingGenerator};

use super::basket_assignment::{self, BasketConfig, KeywordBasket};
use super::keyword_selector::{self, KeywordSelectionConfig, SelectedKeyword};
use super::lexical_candidates::{self, LexicalConfig};
use super::lsp_candidates::{self, LspCandidateConfig};
use super::quasi_summary::{self, QuasiSummaryConfig};
use super::semantic_rerank::{self, RerankConfig};
use super::structural_tags;
use super::tag_selector::{self, SelectedTag, TagSelectionConfig};

/// Configuration for the full extraction pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub lexical: LexicalConfig,
    pub rerank: RerankConfig,
    pub keyword: KeywordSelectionConfig,
    pub tag: TagSelectionConfig,
    pub basket: BasketConfig,
    pub summary: QuasiSummaryConfig,
    pub lsp: LspCandidateConfig,
    /// Weight for co-occurrence centrality boosting (0.0 = disabled).
    /// Default: 0.0 (no boosting until graph has data).
    pub cooccurrence_weight: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            lexical: LexicalConfig::default(),
            rerank: RerankConfig::default(),
            keyword: KeywordSelectionConfig::default(),
            tag: TagSelectionConfig::default(),
            basket: BasketConfig::default(),
            summary: QuasiSummaryConfig::default(),
            lsp: LspCandidateConfig::default(),
            cooccurrence_weight: 0.0,
        }
    }
}

/// Input for pipeline extraction.
pub struct PipelineInput<'a> {
    /// File path (for structural tags and language detection)
    pub file_path: &'a Path,
    /// Full text content of the document
    pub full_text: &'a str,
    /// Language identifier from tree-sitter (e.g., "rust", "python")
    pub language: Option<&'a str>,
    /// Whether this is a code file (vs prose)
    pub is_code: bool,
    /// Chunk vectors (384-dim, one per chunk)
    pub chunk_vectors: &'a [Vec<f32>],
    /// Chunk text content (parallel with chunk_vectors)
    pub chunk_texts: &'a [String],
    /// Total documents in the collection corpus
    pub corpus_size: u64,
    /// Pre-computed document frequency map: term → document count.
    /// Used for IDF penalty in keyword selection. Empty map falls back to 0.
    pub df_lookup: &'a std::collections::HashMap<String, u64>,
    /// Pre-computed co-occurrence centrality scores for symbols.
    /// Used to boost tags by cross-file symbol importance. None = skip boosting.
    pub centrality_scores: Option<&'a std::collections::HashMap<String, f64>>,
}

/// Result of the extraction pipeline.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Summary vector (weighted mean of chunk embeddings)
    pub summary_vector: Option<Vec<f32>>,
    /// Gist chunk indices (top salient/central chunks)
    pub gist_indices: Vec<usize>,
    /// Selected keywords with scores
    pub keywords: Vec<SelectedKeyword>,
    /// Selected concept tags with scores
    pub tags: Vec<SelectedTag>,
    /// Structural tags (language, framework, layer)
    pub structural_tags: Vec<SelectedTag>,
    /// Keyword baskets organized by tag
    pub baskets: Vec<KeywordBasket>,
}

impl ExtractionResult {
    /// Get keyword phrases as a string list (for Qdrant payload).
    pub fn keyword_phrases(&self) -> Vec<String> {
        self.keywords.iter().map(|k| k.phrase.clone()).collect()
    }

    /// Get concept tag phrases as a string list (for Qdrant payload).
    pub fn tag_phrases(&self) -> Vec<String> {
        self.tags.iter().map(|t| t.phrase.clone()).collect()
    }

    /// Get structural tags as a map (for Qdrant payload).
    pub fn structural_tags_map(&self) -> std::collections::HashMap<String, Vec<String>> {
        let mut map: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for tag in &self.structural_tags {
            if let Some((prefix, value)) = tag.phrase.split_once(':') {
                map.entry(prefix.to_string())
                    .or_default()
                    .push(value.to_string());
            }
        }
        map
    }

    /// Get basket map (for Qdrant payload).
    pub fn basket_map(&self) -> std::collections::HashMap<String, Vec<String>> {
        basket_assignment::baskets_to_map(&self.baskets)
    }
}

/// Run the full extraction pipeline.
///
/// This is the main entry point. It orchestrates all extraction modules
/// and returns a complete `ExtractionResult`.
///
/// If any step fails (e.g., embedding generation), the pipeline continues
/// with partial results rather than failing entirely.
pub async fn run_pipeline(
    input: &PipelineInput<'_>,
    embedding_generator: &EmbeddingGenerator,
    config: &PipelineConfig,
) -> ExtractionResult {
    // Step 1: Generate quasi-summary vector
    let summary = if input.is_code {
        let chunk_tokens: Vec<Vec<String>> = input
            .chunk_texts
            .iter()
            .map(|text| tokenize_for_summary(text))
            .collect();
        quasi_summary::summarize_code(input.chunk_vectors, &chunk_tokens, &config.summary)
    } else {
        quasi_summary::summarize_prose(input.chunk_vectors, &config.summary)
    };

    let (summary_vector, gist_indices) = match summary {
        Some(s) => (Some(s.summary_vector), s.gist_indices),
        None => (None, Vec::new()),
    };

    // If no summary vector, we can't do semantic reranking - return minimal result
    let parent_vector = match &summary_vector {
        Some(v) => v,
        None => {
            return ExtractionResult {
                summary_vector: None,
                gist_indices,
                keywords: Vec::new(),
                tags: Vec::new(),
                structural_tags: extract_structural(input),
                baskets: Vec::new(),
            };
        }
    };

    // Step 2: Extract lexical candidates
    let lexical_config = LexicalConfig {
        is_code: input.is_code,
        ..config.lexical.clone()
    };
    let mut lexical_candidates = lexical_candidates::extract_candidates(input.full_text, &lexical_config);

    // Step 3: LSP candidate extraction (code only)
    if input.is_code {
        if let Some(lang) = input.language {
            let lsp_candidates =
                lsp_candidates::extract_import_candidates(input.full_text, lang, &config.lsp);
            lexical_candidates =
                lsp_candidates::merge_candidates(lexical_candidates, &lsp_candidates, config.lsp.priority_boost);
        }
    }

    // Step 4: Semantic reranking
    let ranked = match semantic_rerank::rerank_candidates(
        lexical_candidates,
        parent_vector,
        embedding_generator,
        &config.rerank,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Semantic reranking failed, using empty candidates: {}", e);
            Vec::new()
        }
    };

    if ranked.is_empty() {
        return ExtractionResult {
            summary_vector,
            gist_indices,
            keywords: Vec::new(),
            tags: Vec::new(),
            structural_tags: extract_structural(input),
            baskets: Vec::new(),
        };
    }

    // Step 5: Select keywords with IDF penalty
    let kw_config = KeywordSelectionConfig {
        corpus_size: input.corpus_size,
        ..config.keyword.clone()
    };
    let keywords = keyword_selector::select_keywords(
        &ranked,
        |phrase| input.df_lookup.get(phrase).copied().unwrap_or(0),
        |_phrase| {
            // Stability: count how many chunks contain this term
            input
                .chunk_texts
                .iter()
                .filter(|text| text.to_lowercase().contains(&_phrase.to_lowercase()))
                .count() as u32
        },
        &kw_config,
    );

    // Step 6: Embed candidates for tag selection (need vectors for MMR)
    let candidate_phrases: Vec<String> = ranked.iter().take(50).map(|c| c.phrase.clone()).collect();
    let candidate_vectors = match embed_phrases(&candidate_phrases, embedding_generator).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Failed to embed candidates for tag selection: {}", e);
            return ExtractionResult {
                summary_vector,
                gist_indices,
                keywords,
                tags: Vec::new(),
                structural_tags: extract_structural(input),
                baskets: Vec::new(),
            };
        }
    };

    // Step 6.5: Boost candidates by co-occurrence centrality (Task 31)
    let mut ranked_subset: Vec<_> = ranked.iter().take(50).cloned().collect();
    if let Some(centrality) = input.centrality_scores {
        if config.cooccurrence_weight > 0.0 {
            for candidate in &mut ranked_subset {
                if let Some(&cent) = centrality.get(&candidate.phrase) {
                    candidate.combined_score =
                        config.cooccurrence_weight * cent
                            + (1.0 - config.cooccurrence_weight) * candidate.combined_score;
                }
            }
            ranked_subset.sort_by(|a, b| {
                b.combined_score
                    .partial_cmp(&a.combined_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    // Step 7: Select tags with MMR diversity
    let tags = tag_selector::select_tags(&ranked_subset, &candidate_vectors, &config.tag);

    // Step 8: Keyword basket assignment
    let keyword_phrases: Vec<String> = keywords.iter().map(|k| k.phrase.clone()).collect();
    let tag_phrases: Vec<String> = tags.iter().map(|t| t.phrase.clone()).collect();

    let (kw_vectors, tag_vectors) = match (
        embed_phrases(&keyword_phrases, embedding_generator).await,
        embed_phrases(&tag_phrases, embedding_generator).await,
    ) {
        (Ok(kv), Ok(tv)) => (kv, tv),
        _ => {
            tracing::warn!("Failed to embed keywords/tags for basket assignment");
            return ExtractionResult {
                summary_vector,
                gist_indices,
                keywords,
                tags,
                structural_tags: extract_structural(input),
                baskets: Vec::new(),
            };
        }
    };

    let baskets =
        basket_assignment::assign_baskets(&keywords, &kw_vectors, &tags, &tag_vectors, &config.basket);

    // Step 9: Structural tags (code only)
    let structural_tags = extract_structural(input);

    ExtractionResult {
        summary_vector,
        gist_indices,
        keywords,
        tags,
        structural_tags,
        baskets,
    }
}

/// Extract structural tags from input metadata.
fn extract_structural(input: &PipelineInput<'_>) -> Vec<SelectedTag> {
    if input.is_code {
        structural_tags::extract_structural_tags(input.file_path, input.full_text, input.language)
    } else {
        Vec::new()
    }
}

/// Tokenize text for BM25 summary weighting.
fn tokenize_for_summary(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .collect()
}

/// Embed a batch of phrases using the embedding generator.
async fn embed_phrases(
    phrases: &[String],
    generator: &EmbeddingGenerator,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if phrases.is_empty() {
        return Ok(Vec::new());
    }

    let embeddings = generator
        .generate_embeddings_batch(phrases, "all-MiniLM-L6-v2")
        .await?;

    Ok(embeddings.into_iter().map(|e| e.dense.vector).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tag_selector::TagType;

    #[test]
    fn test_tokenize_for_summary() {
        let tokens = tokenize_for_summary("Hello, world! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single char words should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.keyword.max_keywords, 50);
        assert_eq!(config.tag.max_tags, 8);
        assert!((config.basket.min_similarity - 0.40).abs() < 1e-6);
    }

    #[test]
    fn test_extraction_result_phrases() {
        let result = ExtractionResult {
            summary_vector: None,
            gist_indices: vec![],
            keywords: vec![SelectedKeyword {
                phrase: "vector search".to_string(),
                score: 0.9,
                semantic_score: 0.8,
                lexical_score: 1.0,
                stability_count: 3,
                ngram_size: 2,
            }],
            tags: vec![SelectedTag {
                phrase: "indexing".to_string(),
                tag_type: TagType::Concept,
                score: 0.8,
                diversity_score: 0.9,
                semantic_score: 0.7,
                ngram_size: 1,
            }],
            structural_tags: vec![SelectedTag {
                phrase: "language:rust".to_string(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            }],
            baskets: vec![],
        };

        assert_eq!(result.keyword_phrases(), vec!["vector search"]);
        assert_eq!(result.tag_phrases(), vec!["indexing"]);

        let struct_map = result.structural_tags_map();
        assert_eq!(
            struct_map.get("language").unwrap(),
            &vec!["rust".to_string()]
        );
    }

    #[test]
    fn test_structural_tags_map_multiple() {
        let result = ExtractionResult {
            summary_vector: None,
            gist_indices: vec![],
            keywords: vec![],
            tags: vec![],
            structural_tags: vec![
                SelectedTag {
                    phrase: "framework:tokio".to_string(),
                    tag_type: TagType::Structural,
                    score: 1.0,
                    diversity_score: 1.0,
                    semantic_score: 1.0,
                    ngram_size: 1,
                },
                SelectedTag {
                    phrase: "framework:serde".to_string(),
                    tag_type: TagType::Structural,
                    score: 1.0,
                    diversity_score: 1.0,
                    semantic_score: 1.0,
                    ngram_size: 1,
                },
                SelectedTag {
                    phrase: "language:rust".to_string(),
                    tag_type: TagType::Structural,
                    score: 1.0,
                    diversity_score: 1.0,
                    semantic_score: 1.0,
                    ngram_size: 1,
                },
            ],
            baskets: vec![],
        };

        let map = result.structural_tags_map();
        assert_eq!(map.get("framework").unwrap().len(), 2);
        assert_eq!(map.get("language").unwrap(), &vec!["rust".to_string()]);
    }
}
