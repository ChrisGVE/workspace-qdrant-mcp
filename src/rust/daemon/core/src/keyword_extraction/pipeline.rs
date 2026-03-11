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

use std::collections::HashMap;
use std::path::Path;

use crate::embedding::EmbeddingGenerator;

use super::embedding_cache::resolve_embeddings;

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
    let (summary_vector, gist_indices) = generate_summary(input, config);

    let parent_vector = match &summary_vector {
        Some(v) => v,
        None => return minimal_result(None, gist_indices, input),
    };

    // Steps 2-4: Extract and rerank candidates; collect phrase→vector cache
    let (ranked, phrase_cache) =
        extract_and_rerank(input, parent_vector, embedding_generator, config).await;
    if ranked.is_empty() {
        return minimal_result(summary_vector, gist_indices, input);
    }

    // Step 5: Select keywords with IDF penalty
    let keywords = select_keywords(input, &ranked, config);

    // Steps 6-7: Select tags with MMR diversity and centrality boosting
    let (tags, _candidate_vectors) = match select_tags_with_diversity(
        input,
        &ranked,
        &phrase_cache,
        embedding_generator,
        config,
    )
    .await
    {
        Some(result) => result,
        None => {
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

    // Step 8: Keyword basket assignment (reuses phrase_cache from step 2-4)
    let baskets =
        match build_baskets(&keywords, &tags, &phrase_cache, embedding_generator, config).await {
            Some(b) => b,
            None => Vec::new(),
        };

    ExtractionResult {
        summary_vector,
        gist_indices,
        keywords,
        tags,
        structural_tags: extract_structural(input),
        baskets,
    }
}

/// Step 1: Generate quasi-summary vector from chunk embeddings.
fn generate_summary(
    input: &PipelineInput<'_>,
    config: &PipelineConfig,
) -> (Option<Vec<f32>>, Vec<usize>) {
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

    match summary {
        Some(s) => (Some(s.summary_vector), s.gist_indices),
        None => (None, Vec::new()),
    }
}

/// Steps 2-4: Extract lexical/LSP candidates and semantically rerank them.
///
/// Returns the ranked candidates and a `phrase → dense vector` cache for all
/// surviving candidates. Downstream stages use the cache to avoid re-embedding.
async fn extract_and_rerank(
    input: &PipelineInput<'_>,
    parent_vector: &[f32],
    embedding_generator: &EmbeddingGenerator,
    config: &PipelineConfig,
) -> (Vec<semantic_rerank::RankedCandidate>, HashMap<String, Vec<f32>>) {
    let lexical_config = LexicalConfig {
        is_code: input.is_code,
        ..config.lexical.clone()
    };
    let mut candidates = lexical_candidates::extract_candidates(input.full_text, &lexical_config);

    if input.is_code {
        if let Some(lang) = input.language {
            let lsp = lsp_candidates::extract_import_candidates(input.full_text, lang, &config.lsp);
            candidates =
                lsp_candidates::merge_candidates(candidates, &lsp, config.lsp.priority_boost);
        }
    }

    match semantic_rerank::rerank_candidates(
        candidates,
        parent_vector,
        embedding_generator,
        &config.rerank,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("Semantic reranking failed, using empty candidates: {}", e);
            (Vec::new(), HashMap::new())
        }
    }
}

/// Step 5: Select keywords with IDF penalty and chunk stability.
fn select_keywords(
    input: &PipelineInput<'_>,
    ranked: &[semantic_rerank::RankedCandidate],
    config: &PipelineConfig,
) -> Vec<SelectedKeyword> {
    let kw_config = KeywordSelectionConfig {
        corpus_size: input.corpus_size,
        ..config.keyword.clone()
    };
    keyword_selector::select_keywords(
        ranked,
        |phrase| input.df_lookup.get(phrase).copied().unwrap_or(0),
        |phrase| {
            input
                .chunk_texts
                .iter()
                .filter(|text| text.to_lowercase().contains(&phrase.to_lowercase()))
                .count() as u32
        },
        &kw_config,
    )
}

/// Steps 6-7: Resolve candidate embeddings (from cache), apply centrality boosting,
/// select tags with MMR diversity. Returns None if embedding fails.
async fn select_tags_with_diversity(
    input: &PipelineInput<'_>,
    ranked: &[semantic_rerank::RankedCandidate],
    phrase_cache: &HashMap<String, Vec<f32>>,
    embedding_generator: &EmbeddingGenerator,
    config: &PipelineConfig,
) -> Option<(Vec<SelectedTag>, Vec<Vec<f32>>)> {
    let candidate_phrases: Vec<String> = ranked.iter().take(50).map(|c| c.phrase.clone()).collect();
    let candidate_vectors =
        match resolve_embeddings(&candidate_phrases, phrase_cache, embedding_generator).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to embed candidates for tag selection: {}", e);
                return None;
            }
        };

    let mut ranked_subset: Vec<_> = ranked.iter().take(50).cloned().collect();
    if let Some(centrality) = input.centrality_scores {
        if config.cooccurrence_weight > 0.0 {
            for candidate in &mut ranked_subset {
                if let Some(&cent) = centrality.get(&candidate.phrase) {
                    candidate.combined_score = config.cooccurrence_weight * cent
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

    let tags = tag_selector::select_tags(&ranked_subset, &candidate_vectors, &config.tag);
    Some((tags, candidate_vectors))
}

/// Step 8: Resolve embeddings for keywords and tags (from cache), then assign baskets.
async fn build_baskets(
    keywords: &[SelectedKeyword],
    tags: &[SelectedTag],
    phrase_cache: &HashMap<String, Vec<f32>>,
    embedding_generator: &EmbeddingGenerator,
    config: &PipelineConfig,
) -> Option<Vec<KeywordBasket>> {
    let keyword_phrases: Vec<String> = keywords.iter().map(|k| k.phrase.clone()).collect();
    let tag_phrases: Vec<String> = tags.iter().map(|t| t.phrase.clone()).collect();

    match (
        resolve_embeddings(&keyword_phrases, phrase_cache, embedding_generator).await,
        resolve_embeddings(&tag_phrases, phrase_cache, embedding_generator).await,
    ) {
        (Ok(kv), Ok(tv)) => Some(basket_assignment::assign_baskets(
            keywords,
            &kv,
            tags,
            &tv,
            &config.basket,
        )),
        _ => {
            tracing::warn!("Failed to embed keywords/tags for basket assignment");
            None
        }
    }
}

/// Build a minimal result with only structural tags (used for early returns).
fn minimal_result(
    summary_vector: Option<Vec<f32>>,
    gist_indices: Vec<usize>,
    input: &PipelineInput<'_>,
) -> ExtractionResult {
    ExtractionResult {
        summary_vector,
        gist_indices,
        keywords: Vec::new(),
        tags: Vec::new(),
        structural_tags: extract_structural(input),
        baskets: Vec::new(),
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
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::tag_selector::TagType;
    use super::*;

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
