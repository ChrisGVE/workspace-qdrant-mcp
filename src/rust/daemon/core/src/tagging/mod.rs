//! Automated tagging system with tiered classification.
//!
//! Provides three tiers of document tagging:
//! - **Tier 1** (zero-cost heuristics): path-derived, PDF metadata, dependency concepts
//! - **Tier 2** (embedding-based): taxonomy cosine similarity via FastEmbed
//! - **Tier 3** (LLM-assisted): structured tag extraction via AI providers
//!
//! Each tier progressively increases cost and accuracy.
//! All tags are concept-normalized before storage (see [`normalize`]).

pub mod aggregation;
pub mod concepts;
mod llm_aggregation;
pub mod normalize;
pub mod providers;
pub mod taxonomy;
pub mod taxonomy_cache;
pub mod tier1;
pub mod tier2;
pub mod tier3;
pub mod tier3_config;

// ── Tier 1 re-exports ────────────────────────────────────────────────────
pub use concepts::{
    extract_cargo_concepts, extract_gomod_concepts, extract_npm_concepts, extract_pip_concepts,
};
pub use tier1::{
    extract_path_tags, extract_pdf_metadata_tags, extract_tier1_tags, pdf_metadata_to_tags,
    PdfMetadataTags,
};

// ── Tier 2 re-exports ────────────────────────────────────────────────────
pub use aggregation::{aggregate_document_embedding, aggregate_document_embedding_weighted};
pub use taxonomy::{load_taxonomy, load_taxonomy_from_file, TaxonomyEntry};
pub use taxonomy_cache::{
    compute_taxonomy_hash, load_cached_embeddings, save_cached_embeddings, CacheLookup,
};
pub use tier2::{TaxonomyMatch, Tier2Config, Tier2Tagger};

// ── Tier 3 re-exports ────────────────────────────────────────────────────
pub use tier3::Tier3Tagger;
pub use tier3_config::{AccessMode, LlmProvider, ProviderConfig, Tier3Config};

// ── Normalization re-exports ─────────────────────────────────────────────
pub use normalize::{normalize_tag, normalize_tags};
