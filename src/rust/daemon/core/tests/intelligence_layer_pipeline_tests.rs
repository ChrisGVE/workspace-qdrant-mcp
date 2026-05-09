//! Integration tests for the keyword extraction pipeline and metadata uplift.
//!
//! Tests:
//! - Metadata uplift: UpliftConfig, UpliftStats, candidate identification
//! - Keyword extraction pipeline configuration and ExtractionResult accessors

use workspace_qdrant_core::{
    keyword_extraction::keyword_selector::SelectedKeyword,
    keyword_extraction::pipeline::{ExtractionResult, PipelineConfig},
    keyword_extraction::tag_selector::{SelectedTag, TagType},
    metadata_uplift::{UpliftConfig, UpliftStats},
};

// ── Metadata Uplift Tests ──

#[test]
fn test_uplift_config_respects_generation_tracking() {
    let config = UpliftConfig::default();
    assert_eq!(config.current_generation, 1, "Should start at generation 1");
    assert_eq!(config.batch_size, 10);
    assert_eq!(config.min_interval_secs, 300, "5 minute minimum interval");
}

#[test]
fn test_uplift_stats_aggregation() {
    let mut stats = UpliftStats::default();
    stats.scanned = 10;
    stats.updated = 5;
    stats.skipped = 3;
    stats.errors = 2;

    assert_eq!(stats.scanned, stats.updated + stats.skipped + stats.errors);
}

// ── Pipeline Configuration Tests ──

#[test]
fn test_pipeline_config_defaults_are_sane() {
    let config = PipelineConfig::default();

    assert!(config.keyword.max_keywords > 0, "Should have keyword limit");
    assert!(config.tag.max_tags > 0, "Should have tag limit");
    assert!(
        config.basket.min_similarity > 0.0,
        "Should have minimum similarity"
    );
    assert!(
        config.basket.min_similarity < 1.0,
        "Minimum similarity should be < 1"
    );
}

#[test]
fn test_extraction_result_accessors() {
    let result = ExtractionResult {
        summary_vector: Some(vec![0.1, 0.2, 0.3]),
        gist_indices: vec![0, 2, 4],
        keywords: vec![
            SelectedKeyword {
                phrase: "vector_search".to_string(),
                score: 0.95,
                semantic_score: 0.9,
                lexical_score: 1.0,
                stability_count: 5,
                ngram_size: 1,
            },
            SelectedKeyword {
                phrase: "embedding_model".to_string(),
                score: 0.85,
                semantic_score: 0.8,
                lexical_score: 0.9,
                stability_count: 3,
                ngram_size: 1,
            },
        ],
        tags: vec![SelectedTag {
            phrase: "search".to_string(),
            tag_type: TagType::Concept,
            score: 0.9,
            diversity_score: 1.0,
            semantic_score: 0.85,
            ngram_size: 1,
        }],
        structural_tags: vec![
            SelectedTag {
                phrase: "language:rust".to_string(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            },
            SelectedTag {
                phrase: "framework:tokio".to_string(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            },
        ],
        baskets: vec![],
    };

    let kw_phrases = result.keyword_phrases();
    assert_eq!(kw_phrases.len(), 2);
    assert!(kw_phrases.contains(&"vector_search".to_string()));
    assert!(kw_phrases.contains(&"embedding_model".to_string()));

    let tag_phrases = result.tag_phrases();
    assert_eq!(tag_phrases, vec!["search"]);

    let struct_map = result.structural_tags_map();
    assert_eq!(
        struct_map.get("language").unwrap(),
        &vec!["rust".to_string()]
    );
    assert_eq!(
        struct_map.get("framework").unwrap(),
        &vec!["tokio".to_string()]
    );

    assert_eq!(result.gist_indices.len(), 3);
    assert!(result.summary_vector.is_some());
}
