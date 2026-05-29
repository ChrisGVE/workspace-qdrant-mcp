//! Qdrant read-only client, filter construction, and result-fusion utilities.
//!
//! This module provides:
//!
//! - `client` — [`QdrantReadClient`]: thin async wrapper over `qdrant_client::Qdrant`
//!   for read-only operations (search, scroll, retrieve, collection_exists).
//!   The API key is stored as `secrecy::SecretString` and never exposed via
//!   Debug/Display output.
//!
//! - `filters` — [`build_filter`]: builds a `qdrant_client::qdrant::Filter`
//!   from typed search parameters, mirroring `search-filters.ts`.
//!
//! - `fusion` — RRF fusion, score threshold, and source-diversity re-ranking,
//!   mirroring `search-qdrant.ts` lines 164–194 and `search-diversity.ts`.
//!
//! All search/scroll/retrieve network calls carry the `#[cfg(not(feature = "integration-tests"))]`
//! guard on their unit tests so the test suite stays hermetic.

pub mod client;
pub mod filters;
pub mod fusion;

pub use client::QdrantReadClient;
pub use filters::{build_filter, extract_glob_prefix, FilterParams};
pub use fusion::{
    apply_rrf_fusion, apply_score_threshold, diversify_results, DiversityConfig,
    DEFAULT_DIVERSITY_CONFIG, DEFAULT_SCORE_THRESHOLD, RRF_K,
};
