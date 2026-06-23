//! Cross-collection search with Reciprocal Rank Fusion (RRF).
//!
//! Runs parallel searches against multiple Qdrant collections and merges
//! results using RRF scoring: `score = sum(1 / (k + rank_i))` where k=60.
//! An optional diversity penalty reduces dominance of any single collection.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::debug;

use super::client::StorageClient;
use super::types::{SearchParams, SearchResult, StorageError};
use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

// ─── Constants ─────────────────────────────────────────────────────────

/// Default RRF constant (k). Higher values flatten score differences
/// between ranks; 60 is the standard from Cormack et al. (2009).
pub const RRF_K: f32 = 60.0;

/// All 4 canonical searchable collections.
pub const ALL_COLLECTIONS: &[&str] = &[
    COLLECTION_PROJECTS,
    COLLECTION_LIBRARIES,
    COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
];

// ─── Types ─────────────────────────────────────────────────────────────

/// Configuration for cross-collection search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCollectionConfig {
    /// Collections to search (defaults to all 4 canonical collections).
    pub collections: Vec<String>,
    /// RRF constant (k). Default: 60.
    pub rrf_k: f32,
    /// Optional diversity penalty configuration.
    pub diversity: Option<CollectionDiversityConfig>,
}

impl Default for CrossCollectionConfig {
    fn default() -> Self {
        Self {
            collections: ALL_COLLECTIONS.iter().map(|s| (*s).to_string()).collect(),
            rrf_k: RRF_K,
            diversity: None,
        }
    }
}

/// Diversity penalty configuration for post-fusion re-ranking.
///
/// After RRF fusion, results from over-represented collections receive a
/// multiplicative penalty to promote variety across collection sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionDiversityConfig {
    /// Maximum results from a single collection before penalties apply.
    pub max_per_collection: usize,
    /// Penalty multiplier applied to each result beyond `max_per_collection`
    /// from the same collection. Applied cumulatively: the (n+1)th result
    /// gets `score * penalty^1`, the (n+2)th gets `score * penalty^2`, etc.
    pub penalty_factor: f32,
}

impl Default for CollectionDiversityConfig {
    fn default() -> Self {
        Self {
            max_per_collection: 5,
            penalty_factor: 0.8,
        }
    }
}

/// The pure RRF primitives (`rrf_score`, `rrf_merge`, `CrossCollectionResult`)
/// were relocated to `wqm-common::search::rrf` (F0) so the read crate can fuse
/// without a daemon-core edge. Re-exported here so `crate::storage::rrf_merge`
/// and the `impl StorageClient` method below are unchanged (FP-2).
pub use wqm_common::search::rrf::{rrf_merge, rrf_score, CrossCollectionResult};

/// Aggregated response from a cross-collection search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCollectionResponse {
    /// Ranked results after RRF fusion and optional diversity penalty.
    pub results: Vec<CrossCollectionResult>,
    /// Number of results found per collection (before fusion).
    pub per_collection_counts: HashMap<String, usize>,
    /// Total number of unique results across all collections.
    pub total_unique: usize,
}

// ─── RRF scoring ───────────────────────────────────────────────────────
// `rrf_score`, `rrf_merge`, `CrossCollectionResult`, and the private
// `FusionEntry` accumulator now live in `wqm-common::search::rrf` (F0,
// re-exported above). `apply_collection_diversity` stays here — it is part of
// the cross-collection orchestration, applied after fusion.

/// Apply a collection diversity penalty to fused results.
///
/// After the initial RRF merge, results from over-represented collections
/// have their scores penalised. For collection C, the first
/// `max_per_collection` results keep their full score; subsequent results
/// get `rrf_score * penalty_factor^excess` where `excess` is how many
/// positions beyond the cap this result is.
///
/// The list is re-sorted after penalties are applied.
pub fn apply_collection_diversity(
    results: &mut Vec<CrossCollectionResult>,
    config: &CollectionDiversityConfig,
) {
    // Count how many results we have already seen from each collection.
    let mut collection_counts: HashMap<String, usize> = HashMap::new();

    // First pass: mark excess entries. We need stable iteration order so
    // we process in the current (RRF-sorted) order.
    for result in results.iter_mut() {
        let count = collection_counts
            .entry(result.source_collection.clone())
            .or_insert(0);
        *count += 1;

        if *count > config.max_per_collection {
            let excess = *count - config.max_per_collection;
            result.rrf_score *= config.penalty_factor.powi(excess as i32);
        }
    }

    // Re-sort by adjusted score.
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ─── StorageClient integration ─────────────────────────────────────────

impl StorageClient {
    /// Perform a cross-collection search with RRF fusion.
    ///
    /// Searches the configured collections in parallel and merges results
    /// using Reciprocal Rank Fusion. Optionally applies a collection
    /// diversity penalty to reduce dominance of a single collection.
    ///
    /// # Arguments
    ///
    /// * `params` - Base search parameters (vectors, mode, limit, filter).
    /// * `config` - Cross-collection configuration (collections, k, diversity).
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if all collection searches fail. Partial
    /// failures are logged and the available results are fused.
    #[tracing::instrument(
        name = "qdrant.cross_collection_search",
        skip_all,
        fields(
            collections = ?config.collections,
            limit = params.limit,
            rrf_k = config.rrf_k,
        )
    )]
    pub async fn cross_collection_search(
        &self,
        params: SearchParams,
        config: &CrossCollectionConfig,
    ) -> Result<CrossCollectionResponse, StorageError> {
        let started = std::time::Instant::now();
        let limit = params.limit;

        // Fan out searches to each collection concurrently.
        let mut handles = Vec::with_capacity(config.collections.len());
        for collection in &config.collections {
            let collection = collection.clone();
            let search_params = params.clone();
            handles.push((collection, search_params));
        }

        // Execute all searches concurrently using tokio::join on futures.
        let futures: Vec<_> = handles
            .into_iter()
            .map(|(coll, search_params)| {
                let coll_name = coll.clone();
                async move {
                    let result = self.search(&coll_name, search_params).await;
                    (coll, result)
                }
            })
            .collect();

        let search_results = futures::future::join_all(futures).await;

        // Collect successes, log failures.
        let mut per_collection: Vec<(String, Vec<SearchResult>)> = Vec::new();
        let mut per_collection_counts: HashMap<String, usize> = HashMap::new();
        let mut all_failed = true;

        for (collection, result) in search_results {
            match result {
                Ok(results) => {
                    all_failed = false;
                    let count = results.len();
                    per_collection_counts.insert(collection.clone(), count);
                    debug!(
                        collection = %collection,
                        results = count,
                        "Cross-collection search: collection returned results"
                    );
                    per_collection.push((collection, results));
                }
                Err(e) => {
                    debug!(
                        collection = %collection,
                        error = %e,
                        "Cross-collection search: collection search failed"
                    );
                    per_collection_counts.insert(collection, 0);
                }
            }
        }

        if all_failed {
            return Err(StorageError::Search(
                "All collection searches failed in cross-collection search".to_string(),
            ));
        }

        // RRF merge.
        let mut merged = rrf_merge(&per_collection, config.rrf_k);
        let total_unique = merged.len();

        // Apply diversity penalty if configured.
        if let Some(ref diversity_config) = config.diversity {
            apply_collection_diversity(&mut merged, diversity_config);
        }

        // Inject provenance metadata into each result's payload.
        for entry in &mut merged {
            entry.result.payload.insert(
                "source_collection".to_string(),
                serde_json::json!(entry.source_collection),
            );
            entry.result.score = entry.rrf_score;
        }

        // Truncate to requested limit.
        merged.truncate(limit);

        debug!(
            total_unique = total_unique,
            returned = merged.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Cross-collection search completed"
        );

        crate::monitoring::metrics_core::METRICS.record_qdrant(
            "cross_collection_search",
            started.elapsed(),
            None,
        );

        Ok(CrossCollectionResponse {
            results: merged,
            per_collection_counts,
            total_unique,
        })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

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

    // NOTE: the pure rrf_score / rrf_merge unit tests were relocated to
    // `wqm-common::search::rrf` alongside the functions (F0). The tests below
    // exercise the re-exported `rrf_merge` together with the daemon-side
    // diversity/orchestration logic that STAYS in this crate.

    // ── RRF merge: 4 collections ────────────────────────────────────

    #[test]
    fn rrf_merge_four_collections_doc_in_all() {
        // Same doc appears as rank 1 in all 4 collections
        let results = vec![
            ("projects".to_string(), vec![make_result("universal", 0.9)]),
            ("libraries".to_string(), vec![make_result("universal", 0.8)]),
            ("rules".to_string(), vec![make_result("universal", 0.7)]),
            (
                "scratchpad".to_string(),
                vec![make_result("universal", 0.6)],
            ),
        ];

        let merged = rrf_merge(&results, 60.0);
        assert_eq!(merged.len(), 1);

        // 4 * (1/61)
        let expected = 4.0 / 61.0;
        assert!(
            (merged[0].rrf_score - expected).abs() < 1e-7,
            "Universal doc in 4 collections at rank 1: expected {expected}, got {}",
            merged[0].rrf_score
        );
        assert_eq!(merged[0].rank_contributions.len(), 4);
    }

    // ── RRF merge: ordering correctness ─────────────────────────────

    #[test]
    fn rrf_merge_ordering_respects_multi_collection_boost() {
        // doc-a appears in 2 collections (rank 1 each): 2 * 1/61
        // doc-b appears in 1 collection (rank 1): 1/61
        let results = vec![
            (
                "projects".to_string(),
                vec![make_result("doc-a", 0.9), make_result("doc-b", 0.8)],
            ),
            ("libraries".to_string(), vec![make_result("doc-a", 0.7)]),
        ];

        let merged = rrf_merge(&results, 60.0);
        assert_eq!(
            merged[0].result.id, "doc-a",
            "Multi-collection doc ranks higher"
        );
    }

    // ── Diversity penalty ───────────────────────────────────────────

    #[test]
    fn diversity_no_penalty_within_cap() {
        let config = CollectionDiversityConfig {
            max_per_collection: 3,
            penalty_factor: 0.5,
        };

        let mut results = vec![
            CrossCollectionResult {
                result: make_result("a", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.05,
                rank_contributions: HashMap::new(),
            },
            CrossCollectionResult {
                result: make_result("b", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.04,
                rank_contributions: HashMap::new(),
            },
            CrossCollectionResult {
                result: make_result("c", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.03,
                rank_contributions: HashMap::new(),
            },
        ];

        let scores_before: Vec<f32> = results.iter().map(|r| r.rrf_score).collect();
        apply_collection_diversity(&mut results, &config);
        let scores_after: Vec<f32> = results.iter().map(|r| r.rrf_score).collect();

        assert_eq!(
            scores_before, scores_after,
            "No penalty when within max_per_collection"
        );
    }

    #[test]
    fn diversity_penalty_applied_beyond_cap() {
        let config = CollectionDiversityConfig {
            max_per_collection: 2,
            penalty_factor: 0.5,
        };

        let mut results = vec![
            CrossCollectionResult {
                result: make_result("a", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.050,
                rank_contributions: HashMap::new(),
            },
            CrossCollectionResult {
                result: make_result("b", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.045,
                rank_contributions: HashMap::new(),
            },
            // 3rd from projects: excess=1, penalty=0.5^1
            CrossCollectionResult {
                result: make_result("c", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.040,
                rank_contributions: HashMap::new(),
            },
            // 4th from projects: excess=2, penalty=0.5^2
            CrossCollectionResult {
                result: make_result("d", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.035,
                rank_contributions: HashMap::new(),
            },
        ];

        apply_collection_diversity(&mut results, &config);

        // First two untouched
        let a = results.iter().find(|r| r.result.id == "a").unwrap();
        assert!((a.rrf_score - 0.050).abs() < 1e-7);

        let b = results.iter().find(|r| r.result.id == "b").unwrap();
        assert!((b.rrf_score - 0.045).abs() < 1e-7);

        // 3rd: 0.040 * 0.5 = 0.020
        let c = results.iter().find(|r| r.result.id == "c").unwrap();
        assert!(
            (c.rrf_score - 0.020).abs() < 1e-7,
            "3rd result penalised: expected 0.020, got {}",
            c.rrf_score
        );

        // 4th: 0.035 * 0.5^2 = 0.035 * 0.25 = 0.00875
        let d = results.iter().find(|r| r.result.id == "d").unwrap();
        assert!(
            (d.rrf_score - 0.008_75).abs() < 1e-7,
            "4th result penalised^2: expected 0.00875, got {}",
            d.rrf_score
        );
    }

    #[test]
    fn diversity_penalty_reorders_results() {
        let config = CollectionDiversityConfig {
            max_per_collection: 1,
            penalty_factor: 0.1,
        };

        let mut results = vec![
            CrossCollectionResult {
                result: make_result("proj-1", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.050,
                rank_contributions: HashMap::new(),
            },
            // This projects result is beyond cap, will be heavily penalised
            CrossCollectionResult {
                result: make_result("proj-2", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.045,
                rank_contributions: HashMap::new(),
            },
            // This lib result should bubble up after penalty
            CrossCollectionResult {
                result: make_result("lib-1", 0.0),
                source_collection: "libraries".to_string(),
                rrf_score: 0.030,
                rank_contributions: HashMap::new(),
            },
        ];

        apply_collection_diversity(&mut results, &config);

        // After penalty: proj-1=0.050, proj-2=0.045*0.1=0.0045, lib-1=0.030
        // Order: proj-1, lib-1, proj-2
        assert_eq!(results[0].result.id, "proj-1");
        assert_eq!(results[1].result.id, "lib-1");
        assert_eq!(results[2].result.id, "proj-2");
    }

    #[test]
    fn diversity_penalty_multiple_collections() {
        let config = CollectionDiversityConfig {
            max_per_collection: 1,
            penalty_factor: 0.5,
        };

        let mut results = vec![
            CrossCollectionResult {
                result: make_result("p1", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.060,
                rank_contributions: HashMap::new(),
            },
            CrossCollectionResult {
                result: make_result("l1", 0.0),
                source_collection: "libraries".to_string(),
                rrf_score: 0.055,
                rank_contributions: HashMap::new(),
            },
            // 2nd from projects, excess=1
            CrossCollectionResult {
                result: make_result("p2", 0.0),
                source_collection: "projects".to_string(),
                rrf_score: 0.050,
                rank_contributions: HashMap::new(),
            },
            // 2nd from libraries, excess=1
            CrossCollectionResult {
                result: make_result("l2", 0.0),
                source_collection: "libraries".to_string(),
                rrf_score: 0.045,
                rank_contributions: HashMap::new(),
            },
        ];

        apply_collection_diversity(&mut results, &config);

        let p1 = results.iter().find(|r| r.result.id == "p1").unwrap();
        assert!((p1.rrf_score - 0.060).abs() < 1e-7);

        let l1 = results.iter().find(|r| r.result.id == "l1").unwrap();
        assert!((l1.rrf_score - 0.055).abs() < 1e-7);

        let p2 = results.iter().find(|r| r.result.id == "p2").unwrap();
        assert!((p2.rrf_score - 0.025).abs() < 1e-7); // 0.050 * 0.5

        let l2 = results.iter().find(|r| r.result.id == "l2").unwrap();
        assert!((l2.rrf_score - 0.022_5).abs() < 1e-7); // 0.045 * 0.5
    }

    // ── Config defaults ─────────────────────────────────────────────

    #[test]
    fn cross_collection_config_default_has_all_4_collections() {
        let config = CrossCollectionConfig::default();
        assert_eq!(config.collections.len(), 4);
        assert!(config.collections.contains(&"projects".to_string()));
        assert!(config.collections.contains(&"libraries".to_string()));
        assert!(config.collections.contains(&"rules".to_string()));
        assert!(config.collections.contains(&"scratchpad".to_string()));
        assert!((config.rrf_k - 60.0).abs() < 1e-7);
        assert!(config.diversity.is_none());
    }

    #[test]
    fn collection_diversity_config_default() {
        let config = CollectionDiversityConfig::default();
        assert_eq!(config.max_per_collection, 5);
        assert!((config.penalty_factor - 0.8).abs() < 1e-7);
    }

    // ── Result provenance metadata ──────────────────────────────────

    #[test]
    fn rrf_merge_sets_source_collection() {
        let results = vec![
            ("rules".to_string(), vec![make_result("rule-1", 0.9)]),
            ("scratchpad".to_string(), vec![make_result("note-1", 0.8)]),
        ];

        let merged = rrf_merge(&results, 60.0);

        let rule = merged.iter().find(|r| r.result.id == "rule-1").unwrap();
        assert_eq!(rule.source_collection, "rules");

        let note = merged.iter().find(|r| r.result.id == "note-1").unwrap();
        assert_eq!(note.source_collection, "scratchpad");
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn rrf_merge_large_rank_list() {
        // 100 results from a single collection
        let results = vec![(
            "projects".to_string(),
            (0..100)
                .map(|i| make_result(&format!("doc-{i}"), 1.0 - (i as f32 / 100.0)))
                .collect(),
        )];

        let merged = rrf_merge(&results, 60.0);
        assert_eq!(merged.len(), 100);

        // Verify monotonically decreasing scores
        for window in merged.windows(2) {
            assert!(
                window[0].rrf_score >= window[1].rrf_score,
                "Scores must be monotonically decreasing"
            );
        }

        // First: 1/61, Last: 1/160
        let first_expected = 1.0 / 61.0;
        let last_expected = 1.0 / 160.0;
        assert!((merged[0].rrf_score - first_expected).abs() < 1e-7);
        assert!((merged[99].rrf_score - last_expected).abs() < 1e-7);
    }

    #[test]
    fn rrf_merge_respects_k_parameter() {
        let results = vec![("projects".to_string(), vec![make_result("doc-a", 0.9)])];

        // k=0: score = 1/1 = 1.0
        let merged_k0 = rrf_merge(&results, 0.0);
        assert!((merged_k0[0].rrf_score - 1.0).abs() < 1e-7);

        // k=100: score = 1/101
        let merged_k100 = rrf_merge(&results, 100.0);
        assert!((merged_k100[0].rrf_score - 1.0 / 101.0).abs() < 1e-7);
    }

    #[test]
    fn diversity_empty_results() {
        let config = CollectionDiversityConfig::default();
        let mut results: Vec<CrossCollectionResult> = vec![];
        apply_collection_diversity(&mut results, &config);
        assert!(results.is_empty());
    }

    // ── Fusion + diversity integration ──────────────────────────────

    #[test]
    fn fusion_then_diversity_integration() {
        // Simulate a realistic scenario: projects dominate, but diversity
        // should bring libraries results up.
        let per_collection = vec![
            (
                "projects".to_string(),
                vec![
                    make_result("p1", 0.95),
                    make_result("p2", 0.90),
                    make_result("p3", 0.85),
                    make_result("p4", 0.80),
                ],
            ),
            (
                "libraries".to_string(),
                vec![make_result("l1", 0.88), make_result("l2", 0.82)],
            ),
        ];

        let mut merged = rrf_merge(&per_collection, 60.0);

        // Before diversity: p1 and l1 both at rank 1 in their collections
        // p1: 1/61, l1: 1/61, p2: 1/62, l2: 1/62, p3: 1/63, p4: 1/64

        let diversity = CollectionDiversityConfig {
            max_per_collection: 2,
            penalty_factor: 0.3,
        };

        apply_collection_diversity(&mut merged, &diversity);

        // After diversity: p3 (rank 3 in projects) gets 1/63 * 0.3
        // p4 (rank 4) gets 1/64 * 0.3^2
        // l1 and l2 untouched (only 2 from libraries, within cap)
        let _p3 = merged.iter().find(|r| r.result.id == "p3").unwrap();
        let _p4 = merged.iter().find(|r| r.result.id == "p4").unwrap();
        let _l1 = merged.iter().find(|r| r.result.id == "l1").unwrap();

        // l1 should rank above penalised p3
        let l1_pos = merged.iter().position(|r| r.result.id == "l1").unwrap();
        let p3_pos = merged.iter().position(|r| r.result.id == "p3").unwrap();
        assert!(
            l1_pos < p3_pos,
            "Library result should rank above penalised project result"
        );

        // p4 should be last (most penalised)
        let p4_pos = merged.iter().position(|r| r.result.id == "p4").unwrap();
        assert_eq!(p4_pos, merged.len() - 1, "Most penalised result is last");
    }
}
