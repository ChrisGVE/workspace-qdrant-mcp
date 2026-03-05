//! Collection-specific extraction tuning.
//!
//! Different collections benefit from different extraction parameters:
//! - **projects**: Emphasize LSP symbols, structural tags, stricter filtering
//! - **libraries**: Pure prose, more keywords, detailed hierarchies
//! - **scratchpad**: Higher diversity, lower stability, lighter extraction

use super::basket_assignment::BasketConfig;
use super::keyword_selector::KeywordSelectionConfig;
use super::lexical_candidates::LexicalConfig;
use super::lsp_candidates::LspCandidateConfig;
use super::pipeline::PipelineConfig;
use super::quasi_summary::QuasiSummaryConfig;
use super::semantic_rerank::RerankConfig;
use super::tag_selector::TagSelectionConfig;

/// Get a `PipelineConfig` tuned for a specific collection.
///
/// Falls back to default config for unknown collections.
pub fn config_for_collection(collection: &str) -> PipelineConfig {
    match collection {
        "projects" => projects_config(),
        "libraries" => libraries_config(),
        "scratchpad" => scratchpad_config(),
        _ => PipelineConfig::default(),
    }
}

/// Config for the **projects** collection.
///
/// Emphasizes code-aware extraction:
/// - LSP symbol extraction enabled with standard boost
/// - Structural tags (language, framework, layer) are primary classifiers
/// - Moderate keyword count (30-50 typical)
/// - Strict junk filtering for code boilerplate
/// - Standard 8 concept tags per document
fn projects_config() -> PipelineConfig {
    PipelineConfig {
        lexical: LexicalConfig {
            max_candidates: 200,
            ..LexicalConfig::default()
        },
        rerank: RerankConfig::default(),
        keyword: KeywordSelectionConfig {
            max_keywords: 50,
            max_df_ratio: 0.80,
            ..KeywordSelectionConfig::default()
        },
        tag: TagSelectionConfig {
            max_tags: 8,
            lambda: 0.7,
            min_stability_for_code: 2,
            ..TagSelectionConfig::default()
        },
        basket: BasketConfig::default(),
        summary: QuasiSummaryConfig::default(),
        lsp: LspCandidateConfig::default(),
        cooccurrence_weight: 0.3, // enabled for code projects
    }
}

/// Config for the **libraries** collection.
///
/// Optimized for prose/documentation:
/// - No LSP (library docs are ingested text, not source code)
/// - More keywords (50-100) for comprehensive coverage
/// - More tags (12) for detailed topic classification
/// - Higher diversity (lambda=0.6) for broader topic spread
/// - Lower stability requirement (docs have fewer chunks)
fn libraries_config() -> PipelineConfig {
    PipelineConfig {
        lexical: LexicalConfig {
            max_candidates: 300,
            is_code: false,
            ..LexicalConfig::default()
        },
        rerank: RerankConfig::default(),
        keyword: KeywordSelectionConfig {
            max_keywords: 100,
            max_df_ratio: 0.85, // more permissive for prose
            ..KeywordSelectionConfig::default()
        },
        tag: TagSelectionConfig {
            max_tags: 12,
            lambda: 0.6,               // higher diversity for broad documentation
            min_stability_for_code: 1, // lower threshold (docs are prose)
            stability_chunk_threshold: 3,
            ..TagSelectionConfig::default()
        },
        basket: BasketConfig {
            min_similarity: 0.35, // slightly more permissive basket assignment
        },
        summary: QuasiSummaryConfig {
            gist_chunks: 5, // more context chunks for longer documents
            ..QuasiSummaryConfig::default()
        },
        lsp: LspCandidateConfig {
            priority_boost: 1.0, // no LSP boost for libraries
            ..LspCandidateConfig::default()
        },
        cooccurrence_weight: 0.0, // disabled — no LSP symbols in prose
    }
}

/// Config for the **scratchpad** collection.
///
/// Lighter extraction for ephemeral notes:
/// - Fewer keywords (20-40) for quick categorization
/// - Fewer tags (5) to avoid over-classifying short notes
/// - Higher diversity threshold (ideas drift)
/// - Lower stability requirement (single-chunk notes are common)
fn scratchpad_config() -> PipelineConfig {
    PipelineConfig {
        lexical: LexicalConfig {
            max_candidates: 100,
            ..LexicalConfig::default()
        },
        rerank: RerankConfig::default(),
        keyword: KeywordSelectionConfig {
            max_keywords: 30,
            max_df_ratio: 0.90, // very permissive for ephemeral content
            ..KeywordSelectionConfig::default()
        },
        tag: TagSelectionConfig {
            max_tags: 5,
            lambda: 0.5, // highest diversity (ephemeral, varied content)
            min_stability_for_code: 1,
            stability_chunk_threshold: 1, // accept single-occurrence terms
            ..TagSelectionConfig::default()
        },
        basket: BasketConfig {
            min_similarity: 0.45, // stricter basket to avoid noise
        },
        summary: QuasiSummaryConfig {
            gist_chunks: 2, // fewer gist chunks for short notes
            ..QuasiSummaryConfig::default()
        },
        lsp: LspCandidateConfig {
            priority_boost: 1.0,
            ..LspCandidateConfig::default()
        },
        cooccurrence_weight: 0.0, // disabled for ephemeral content
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projects_config() {
        let config = config_for_collection("projects");
        assert_eq!(config.keyword.max_keywords, 50);
        assert_eq!(config.tag.max_tags, 8);
        assert!((config.tag.lambda - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_libraries_config() {
        let config = config_for_collection("libraries");
        assert_eq!(config.keyword.max_keywords, 100);
        assert_eq!(config.tag.max_tags, 12);
        assert!((config.tag.lambda - 0.6).abs() < 1e-6);
        assert!(!config.lexical.is_code);
    }

    #[test]
    fn test_scratchpad_config() {
        let config = config_for_collection("scratchpad");
        assert_eq!(config.keyword.max_keywords, 30);
        assert_eq!(config.tag.max_tags, 5);
        assert!((config.tag.lambda - 0.5).abs() < 1e-6);
        assert_eq!(config.tag.stability_chunk_threshold, 1);
    }

    #[test]
    fn test_unknown_collection_uses_default() {
        let config = config_for_collection("unknown_collection");
        let default = PipelineConfig::default();
        assert_eq!(config.keyword.max_keywords, default.keyword.max_keywords);
        assert_eq!(config.tag.max_tags, default.tag.max_tags);
    }

    #[test]
    fn test_libraries_more_keywords_than_projects() {
        let proj = config_for_collection("projects");
        let lib = config_for_collection("libraries");
        assert!(
            lib.keyword.max_keywords > proj.keyword.max_keywords,
            "Libraries should have more keywords than projects"
        );
    }

    #[test]
    fn test_scratchpad_fewer_tags_than_projects() {
        let proj = config_for_collection("projects");
        let scratch = config_for_collection("scratchpad");
        assert!(
            scratch.tag.max_tags < proj.tag.max_tags,
            "Scratchpad should have fewer tags than projects"
        );
    }

    #[test]
    fn test_libraries_higher_diversity_than_projects() {
        let proj = config_for_collection("projects");
        let lib = config_for_collection("libraries");
        // Lower lambda = higher diversity (more weight on dissimilarity)
        assert!(
            lib.tag.lambda < proj.tag.lambda,
            "Libraries should have higher diversity (lower lambda)"
        );
    }

    #[test]
    fn test_scratchpad_highest_diversity() {
        let scratch = config_for_collection("scratchpad");
        let proj = config_for_collection("projects");
        let lib = config_for_collection("libraries");
        assert!(scratch.tag.lambda <= lib.tag.lambda);
        assert!(scratch.tag.lambda < proj.tag.lambda);
    }
}
