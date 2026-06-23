//! Pure Reciprocal-Rank-Fusion (RRF) primitives.
//!
//! Location: `wqm-common/src/search/rrf.rs` — leaf crate, no Qdrant/daemon deps.
//! Context: canonical home (F0) of the pure RRF merge logic, relocated from
//! `daemon-core/storage/cross_collection_search.rs` so the read crate
//! (`wqm-storage`, used by F10/F17 multi-store fan-out) can fuse ranked result
//! lists WITHOUT a daemon-core edge (dissolves ARCH-04). Only the pure pieces
//! live here — `rrf_score`, `rrf_merge`, and its result/accumulator types. The
//! `cross_collection_search` async method, `apply_collection_diversity`, and the
//! diversity/config types STAY in daemon-core (they fan over a live `Qdrant`
//! handle and would pull the qdrant async stack into the leaf — MF-4/IMPL-08).
//! Neighbors: `super::types::SearchResult` (the per-hit input type).
//!
//! Algorithm: RRF (Cormack et al., 2009) — `score = Σ 1 / (k + rank_i)` over the
//! collections a document appears in; k=60 is the standard constant.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::SearchResult;

/// A single search result with cross-collection provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCollectionResult {
    /// The underlying search result.
    pub result: SearchResult,
    /// The collection this result originated from.
    pub source_collection: String,
    /// The RRF score (fused across collections, post-diversity if applied).
    pub rrf_score: f32,
    /// Per-collection rank contributions: collection_name -> rank (1-based).
    pub rank_contributions: HashMap<String, usize>,
}

/// Intermediate accumulator for a single document during RRF fusion.
#[derive(Debug, Clone)]
struct FusionEntry {
    result: SearchResult,
    source_collection: String,
    rrf_score: f32,
    rank_contributions: HashMap<String, usize>,
}

/// Compute the RRF score for a given rank (1-based) with constant k.
///
/// Formula: `1.0 / (k + rank)`
#[inline]
pub fn rrf_score(k: f32, rank: usize) -> f32 {
    1.0 / (k + rank as f32)
}

/// Merge per-collection ranked result lists into a single fused ranking.
///
/// Each entry in `per_collection_results` maps collection name to an ordered
/// list of `SearchResult` (best-first). The function accumulates RRF scores
/// across collections for each unique document ID and returns a descending-
/// score-sorted list of `CrossCollectionResult`.
pub fn rrf_merge(
    per_collection_results: &[(String, Vec<SearchResult>)],
    k: f32,
) -> Vec<CrossCollectionResult> {
    // doc_id -> FusionEntry
    let mut fused: HashMap<String, FusionEntry> = HashMap::new();

    for (collection_name, results) in per_collection_results {
        for (rank_zero, result) in results.iter().enumerate() {
            let rank = rank_zero + 1; // 1-based
            let score = rrf_score(k, rank);

            let entry = fused
                .entry(result.id.clone())
                .or_insert_with(|| FusionEntry {
                    result: result.clone(),
                    source_collection: collection_name.clone(),
                    rrf_score: 0.0,
                    rank_contributions: HashMap::new(),
                });

            entry.rrf_score += score;
            entry
                .rank_contributions
                .insert(collection_name.clone(), rank);
        }
    }

    let mut merged: Vec<CrossCollectionResult> = fused
        .into_values()
        .map(|entry| CrossCollectionResult {
            result: entry.result,
            source_collection: entry.source_collection,
            rrf_score: entry.rrf_score,
            rank_contributions: entry.rank_contributions,
        })
        .collect();

    merged.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn make_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        }
    }

    // ── RRF score math ──────────────────────────────────────────────

    #[test]
    fn rrf_score_rank_1_k60() {
        let score = rrf_score(60.0, 1);
        // 1 / (60 + 1) = 1/61
        let expected = 1.0 / 61.0;
        assert!(
            (score - expected).abs() < 1e-7,
            "rank=1, k=60: expected {expected}, got {score}"
        );
    }

    #[test]
    fn rrf_score_rank_10_k60() {
        let score = rrf_score(60.0, 10);
        let expected = 1.0 / 70.0;
        assert!(
            (score - expected).abs() < 1e-7,
            "rank=10, k=60: expected {expected}, got {score}"
        );
    }

    #[test]
    fn rrf_score_higher_rank_yields_lower_score() {
        let s1 = rrf_score(60.0, 1);
        let s2 = rrf_score(60.0, 2);
        let s10 = rrf_score(60.0, 10);
        assert!(s1 > s2, "rank 1 > rank 2");
        assert!(s2 > s10, "rank 2 > rank 10");
    }

    #[test]
    fn rrf_score_custom_k() {
        // k=0 means pure 1/rank
        let score = rrf_score(0.0, 1);
        assert!((score - 1.0).abs() < 1e-7);

        let score = rrf_score(0.0, 5);
        assert!((score - 0.2).abs() < 1e-7);
    }

    // ── RRF merge: single collection ────────────────────────────────

    #[test]
    fn rrf_merge_single_collection() {
        let results = vec![(
            "projects".to_string(),
            vec![
                make_result("doc-a", 0.95),
                make_result("doc-b", 0.80),
                make_result("doc-c", 0.70),
            ],
        )];

        let merged = rrf_merge(&results, 60.0);
        assert_eq!(merged.len(), 3);

        // doc-a at rank 1: 1/61
        let expected_a = 1.0 / 61.0;
        assert!(
            (merged[0].rrf_score - expected_a).abs() < 1e-7,
            "doc-a score: expected {expected_a}, got {}",
            merged[0].rrf_score
        );
        assert_eq!(merged[0].result.id, "doc-a");
        assert_eq!(merged[0].source_collection, "projects");

        // Verify descending order
        assert!(merged[0].rrf_score >= merged[1].rrf_score);
        assert!(merged[1].rrf_score >= merged[2].rrf_score);
    }

    // ── RRF merge: multi-collection with shared IDs ─────────────────

    #[test]
    fn rrf_merge_multi_collection_shared_doc() {
        // doc-shared appears in both collections at different ranks
        let results = vec![
            (
                "projects".to_string(),
                vec![
                    make_result("doc-shared", 0.95), // rank 1
                    make_result("doc-proj", 0.80),   // rank 2
                ],
            ),
            (
                "libraries".to_string(),
                vec![
                    make_result("doc-lib", 0.90),    // rank 1
                    make_result("doc-shared", 0.85), // rank 2
                ],
            ),
        ];

        let merged = rrf_merge(&results, 60.0);

        // doc-shared should have the highest fused score:
        // 1/61 (from projects rank 1) + 1/62 (from libraries rank 2)
        let expected_shared = 1.0 / 61.0 + 1.0 / 62.0;
        let shared = merged.iter().find(|r| r.result.id == "doc-shared").unwrap();
        assert!(
            (shared.rrf_score - expected_shared).abs() < 1e-7,
            "doc-shared fused score: expected {expected_shared}, got {}",
            shared.rrf_score
        );

        // doc-shared should have rank contributions from both collections
        assert_eq!(shared.rank_contributions.get("projects"), Some(&1));
        assert_eq!(shared.rank_contributions.get("libraries"), Some(&2));

        // doc-shared should be first (highest score)
        assert_eq!(merged[0].result.id, "doc-shared");
    }

    #[test]
    fn rrf_merge_multi_collection_no_overlap() {
        let results = vec![
            ("projects".to_string(), vec![make_result("proj-1", 0.90)]),
            ("libraries".to_string(), vec![make_result("lib-1", 0.85)]),
            ("rules".to_string(), vec![make_result("rule-1", 0.80)]),
        ];

        let merged = rrf_merge(&results, 60.0);
        assert_eq!(merged.len(), 3);

        // All at rank 1 in their respective collection, so all get 1/61
        let expected = 1.0 / 61.0;
        for entry in &merged {
            assert!(
                (entry.rrf_score - expected).abs() < 1e-7,
                "Each doc at rank 1: expected {expected}, got {}",
                entry.rrf_score
            );
        }
    }

    // ── RRF merge: empty inputs ─────────────────────────────────────

    #[test]
    fn rrf_merge_empty_collections() {
        let results: Vec<(String, Vec<SearchResult>)> = vec![];
        let merged = rrf_merge(&results, 60.0);
        assert!(merged.is_empty());
    }

    #[test]
    fn rrf_merge_all_collections_empty() {
        let results = vec![
            ("projects".to_string(), vec![]),
            ("libraries".to_string(), vec![]),
        ];
        let merged = rrf_merge(&results, 60.0);
        assert!(merged.is_empty());
    }

    #[test]
    fn rrf_merge_one_collection_empty() {
        let results = vec![
            ("projects".to_string(), vec![make_result("doc-a", 0.9)]),
            ("libraries".to_string(), vec![]),
        ];
        let merged = rrf_merge(&results, 60.0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].result.id, "doc-a");
    }
}
