//! Search-result vocabulary and pure ranking primitives shared across the
//! storage crates and daemon-core.
//!
//! Location: `wqm-common/src/search/` — leaf crate, no Qdrant/daemon deps.
//! Context: F0 relocation home for `SearchResult` (the storage search-hit type)
//! and the pure Reciprocal-Rank-Fusion primitives (`rrf_merge`/`rrf_score`), so
//! the read crate (`wqm-storage`, F10/F17) can rank results without depending on
//! daemon-core (FP-2 one canonical home, DR GP-9 no duplication).
//! Neighbors: `types` (the hit struct), `rrf` (fusion), `super::error`.

pub mod rrf;
pub mod types;
