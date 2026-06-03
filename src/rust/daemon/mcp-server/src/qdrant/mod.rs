//! Qdrant read-only client, filter construction, and result-fusion utilities.
//!
//! This module provides:
//!
//! - `client` — [`QdrantReadClient`]: thin async wrapper over `qdrant_client::Qdrant`
//!   for read-only operations (search, scroll, retrieve, collection_exists).
//!   The API key is stored as `secrecy::SecretString` and never exposed via
//!   Debug/Display output.
//!
//! - `endpoint` — [`grpc_endpoint`]: translates a REST-style Qdrant URL
//!   (`:6333`) to the gRPC port (`:6334`) the `qdrant_client` crate requires.
//!
//! - `filters` — [`build_filter`]: builds a `qdrant_client::qdrant::Filter`
//!   from typed search parameters, mirroring `search-filters.ts`.
//!
//! - `fusion` — RRF fusion, score threshold, and source-diversity re-ranking,
//!   mirroring `search-qdrant.ts` lines 164–194 and `search-diversity.ts`.
//!
//! All search/scroll/retrieve network calls carry the `#[cfg(not(feature = "integration-tests"))]`
//! guard on their unit tests so the test suite stays hermetic.

pub mod endpoint;
pub mod filters;
pub mod fusion;

// QdrantReadClient (+ value types) now live in the shared wqm-client crate
// (WI-d3, #82). Re-export the module so existing `crate::qdrant::client::{…}`
// paths and the local `SearchQdrant for QdrantReadClient` adapter keep
// resolving against the now-foreign type.
pub use wqm_client::qdrant::client;

pub use client::{QdrantPoint, QdrantReadClient, QdrantRetrievedPoint};
pub use endpoint::grpc_endpoint;
pub use filters::{build_filter, extract_glob_prefix, FilterParams};
pub use fusion::{
    apply_rrf_fusion, apply_score_threshold, diversify_results, DiversityConfig,
    DEFAULT_DIVERSITY_CONFIG, DEFAULT_SCORE_THRESHOLD, RRF_K,
};
