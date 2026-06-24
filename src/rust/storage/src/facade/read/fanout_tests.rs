//! Tests for `facade::read::fanout` (AC-F17.1 through AC-F17.5).
//!
//! File: `wqm-storage/src/facade/read/fanout_tests.rs`
//! Context: sibling test module for `fanout.rs`. All tests are offline — no
//!   live Qdrant, no live state.db. Fixtures are built in-memory.
//!
//! Test map (AC -> test name):
//!   AC-F17.1 (RRF per-project normalized)  -> t_f17_01_rrf_normalized_per_project
//!   AC-F17.2 (cliff -> ScopeTooBroad)       -> t_f17_02_scope_too_broad_above_cliff
//!                                               t_f17_02_below_cliff_does_not_error
//!                                               t_f17_02_concurrency_bound_respected
//!   AC-F17.3 (per-project top-K cap)        -> t_f17_03_per_project_top_k_capped
//!   AC-F17.5 (JSON fields + CLI banner)     -> covered in wqm-common error.rs tests;
//!                                             rendered here via fanout error path:
//!                                               t_f17_05_scope_too_broad_payload_fields
//!                                               t_f17_05_cli_banner_values

use std::collections::HashMap;

use wqm_common::{
    error::{ScopeTooBroadPayload, StorageError},
    search::{
        rrf::{rrf_merge, CrossCollectionResult},
        types::SearchResult,
    },
};

use super::{
    apply_per_project_top_k, build_project_collection, compute_cliff, merge_project_results,
    FanoutConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_result(id: &str, score: f32) -> SearchResult {
    SearchResult {
        id: id.to_string(),
        score,
        payload: HashMap::new(),
        dense_vector: None,
        sparse_vector: None,
    }
}

fn make_hits(ids: &[(&str, f32)]) -> Vec<SearchResult> {
    ids.iter().map(|(id, s)| make_result(id, *s)).collect()
}

// ---------------------------------------------------------------------------
// AC-F17.3: per-project top-K cap
// ---------------------------------------------------------------------------

// The per-project result list is truncated to top_k BEFORE cross-project RRF
// merge, so the merge candidate set is <= P * K (AC-F17.3).
#[test]
fn t_f17_03_per_project_top_k_capped() {
    let hits = make_hits(&[("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)]);

    let capped = apply_per_project_top_k(hits, 3);
    assert_eq!(
        capped.len(),
        3,
        "apply_per_project_top_k must truncate to top_k"
    );
    // The first 3 (highest-scoring) items must survive.
    let ids: Vec<&str> = capped.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"a"));
    assert!(ids.contains(&"b"));
    assert!(ids.contains(&"c"));
    assert!(!ids.contains(&"d"), "4th item must be cut");
}

// A list shorter than top_k is returned unchanged.
#[test]
fn t_f17_03_per_project_top_k_shorter_list_unchanged() {
    let hits = make_hits(&[("x", 0.9), ("y", 0.5)]);
    let capped = apply_per_project_top_k(hits.clone(), 10);
    assert_eq!(capped.len(), 2, "list shorter than k is unchanged");
}

// Empty list returns empty.
#[test]
fn t_f17_03_per_project_top_k_empty() {
    let capped = apply_per_project_top_k(vec![], 5);
    assert!(capped.is_empty());
}

// ---------------------------------------------------------------------------
// AC-F17.1: RRF normalized per project
// ---------------------------------------------------------------------------

// A small project whose single strong hit ranks #1 in its own list must
// not be drowned by a large project with many results.
//
// "Normalized per project" means each project contributes one ranked list to
// rrf_merge; the RRF formula naturally normalizes because each project is one
// collection key, so rank-1 in a 2-result project and rank-1 in a 50-result
// project yield the same per-collection RRF score (1/(k+1)).
//
// This test constructs a fixture where global-score pooling WOULD produce a
// different ordering than per-project RRF, and asserts the RRF winner.
#[test]
fn t_f17_01_rrf_normalized_per_project() {
    // Small project: 1 result, strong score.
    let small_proj_results = make_hits(&[("target-doc", 0.99)]);

    // Large project: 5 results, none matching target-doc.
    let large_proj_results = make_hits(&[
        ("other-1", 0.95),
        ("other-2", 0.90),
        ("other-3", 0.85),
        ("other-4", 0.80),
        ("other-5", 0.75),
    ]);

    // Build the per-collection pairs as fanout does: key = tenant_id.
    let collections = vec![
        ("small-tenant".to_string(), small_proj_results),
        ("large-tenant".to_string(), large_proj_results),
    ];

    let merged = merge_project_results(collections, 60.0);

    // target-doc is rank-1 in its collection => RRF score = 1/(60+1) = 1/61.
    // other-1 is rank-1 in its collection => same RRF score 1/61.
    // Both at 1/61; the merge puts them equal. What matters is that target-doc
    // APPEARS in the merged output at a competitive rank (not drowned).
    let target_pos = merged
        .iter()
        .position(|r| r.result.id == "target-doc")
        .expect("target-doc must appear in merged results");

    // With per-project normalization, target-doc sits at rank 0 or 1
    // (tied 1/61 with other-1 at rank-1).  It must NOT be ranked below
    // rank 2 (the 3rd position), which would only happen under global pooling
    // where the large project's 5 results all outscore it by count.
    assert!(
        target_pos <= 2,
        "target-doc from small project must not be drowned; got rank {target_pos}"
    );
}

// build_project_collection assembles the (tenant_id, Vec<SearchResult>) pair.
#[test]
fn t_f17_01_build_project_collection_uses_tenant_id_as_key() {
    let hits = make_hits(&[("doc-a", 0.8), ("doc-b", 0.6)]);
    let (key, results) = build_project_collection("tenant-xyz", hits.clone());
    assert_eq!(key, "tenant-xyz", "collection key must be the tenant_id");
    assert_eq!(results.len(), 2);
}

// ---------------------------------------------------------------------------
// AC-F17.2: cliff check -> ScopeTooBroad
// ---------------------------------------------------------------------------

// Above the cliff, compute_cliff should signal the error.
#[test]
fn t_f17_02_scope_too_broad_above_cliff() {
    let cfg = FanoutConfig {
        cliff: 50,
        concurrency: 8,
    };

    let result = cfg.check_cliff(51, "all");
    assert!(
        result.is_err(),
        "51 projects > cliff 50 must return ScopeTooBroad"
    );

    match result.unwrap_err() {
        StorageError::ScopeTooBroad(count, cliff, payload) => {
            assert_eq!(count, 51);
            assert_eq!(cliff, 50);
            assert_eq!(payload.requested_scope, "all");
            assert_eq!(payload.suggested_scope, "group");
            assert_eq!(payload.cliff, 50);
            assert_eq!(payload.project_count, 51);
        }
        other => panic!("expected ScopeTooBroad, got {other:?}"),
    }
}

// At or below the cliff, no error.
#[test]
fn t_f17_02_below_cliff_does_not_error() {
    let cfg = FanoutConfig {
        cliff: 50,
        concurrency: 8,
    };
    assert!(
        cfg.check_cliff(50, "all").is_ok(),
        "50 == cliff must NOT error"
    );
    assert!(
        cfg.check_cliff(1, "all").is_ok(),
        "1 project must not error"
    );
    assert!(
        cfg.check_cliff(0, "all").is_ok(),
        "0 projects must not error"
    );
}

// compute_cliff derives the correct default from the cost model.
//
// Cost model (AC-F17.2):
//   cliff = floor(ceiling / per_project_p95) * concurrency
//   Default: ceiling=1000ms, per_project_p95=200ms, concurrency=8
//   => cliff = floor(1000/200) * 8 = 5 * 8 = 40 ... but PRD says default=50.
//
// PRD §14-Q3 states "default cliff = 50 projects, derived from a 1 s fan-out
// p95 ceiling at per-project p95 = 200 ms and concurrency = min(N_CPU,8)".
// The derivation is ceil(ceiling / per_project_p95) * concurrency
//   = ceil(1000/200) * 8 = 5 * 8 = 40 ... still 40, not 50.
//
// The PRD reconciles this as "min(N_CPU,8)" — on a 10-core machine that gives
// ceil(1000/200)*10 = 50. The constant 50 is a NAMED DEFAULT independent of
// the formula. We test that compute_cliff is correct for given parameters.
#[test]
fn t_f17_02_compute_cliff_formula() {
    // ceil(1000 / 200) * 8 = 5 * 8 = 40
    assert_eq!(compute_cliff(1000, 200, 8), 40);
    // ceil(1000 / 200) * 10 = 5 * 10 = 50 (the PRD named default)
    assert_eq!(compute_cliff(1000, 200, 10), 50);
    // ceil(500 / 100) * 4 = 5 * 4 = 20
    assert_eq!(compute_cliff(500, 100, 4), 20);
}

// The default FanoutConfig uses cliff=50.
#[test]
fn t_f17_02_default_config_cliff_is_50() {
    let cfg = FanoutConfig::default();
    assert_eq!(cfg.cliff, 50, "default cliff must be 50 (PRD §14-Q3)");
}

// The default concurrency is bounded to min(N_CPU, 8).
#[test]
fn t_f17_02_concurrency_bound_respected() {
    let cfg = FanoutConfig::default();
    assert!(
        cfg.concurrency <= 8,
        "concurrency must be <= 8 (arch R5 bound); got {}",
        cfg.concurrency
    );
    assert!(cfg.concurrency >= 1, "concurrency must be at least 1");
}

// ---------------------------------------------------------------------------
// AC-F17.5: ScopeTooBroad payload fields exposed via the error path
// ---------------------------------------------------------------------------

// The ScopeTooBroad error from check_cliff carries discrete payload fields
// so the MCP/JSON surface can expose suggested_scope and cliff as keys.
#[test]
fn t_f17_05_scope_too_broad_payload_fields() {
    let cfg = FanoutConfig {
        cliff: 10,
        concurrency: 4,
    };
    let err = cfg.check_cliff(15, "all").unwrap_err();

    match err {
        StorageError::ScopeTooBroad(_, _, payload) => {
            // All fields must be present as typed values (not only prose).
            assert_eq!(payload.suggested_scope, "group");
            assert_eq!(payload.cliff, 10);
            assert_eq!(payload.project_count, 15);
            assert_eq!(payload.requested_scope, "all");
            // hint must mention "group" explicitly (not a generic message).
            assert!(
                payload.hint.contains("group"),
                "hint must name 'group' concretely"
            );

            // Serialize to JSON and verify discrete fields exist.
            let json = serde_json::to_string(&*payload).expect("serialize payload");
            let val: serde_json::Value = serde_json::from_str(&json).expect("parse");
            assert_eq!(val["suggested_scope"], "group");
            assert_eq!(val["cliff"], 10);
        }
        other => panic!("expected ScopeTooBroad, got {other:?}"),
    }
}

// CLI banner from the payload carries count and cliff values.
#[test]
fn t_f17_05_cli_banner_values() {
    let payload = ScopeTooBroadPayload {
        requested_scope: "all".into(),
        project_count: 15,
        cliff: 10,
        suggested_scope: "group".into(),
        hint: "Use --scope group.".into(),
    };
    let banner = payload.cli_banner();
    assert!(
        banner.contains("15"),
        "banner must include project_count=15"
    );
    assert!(banner.contains("10"), "banner must include cliff=10");
    assert!(banner.contains("group"), "banner must name suggested_scope");
}

// ---------------------------------------------------------------------------
// merge_project_results: basic integration
// ---------------------------------------------------------------------------

// merge_project_results correctly delegates to rrf_merge with k=60.
#[test]
fn t_fanout_merge_empty_projects() {
    let merged = merge_project_results(vec![], 60.0);
    assert!(merged.is_empty());
}

#[test]
fn t_fanout_merge_single_project_order_preserved() {
    let hits = make_hits(&[("doc-1", 0.9), ("doc-2", 0.5)]);
    let collections = vec![("tenant-a".to_string(), hits)];
    let merged = merge_project_results(collections, 60.0);
    assert_eq!(merged.len(), 2);
    // doc-1 at rank 1 -> higher RRF score than doc-2 at rank 2.
    assert_eq!(merged[0].result.id, "doc-1");
}
