//! Score threshold, sparse expansion weight, field-order, and raw-score tests.
//!
//! - Score threshold applied ONLY at Qdrant query level (dense + sparse legs),
//!   NOT post-fusion. See flow.rs:15-16 and scratchpad note.
//! - Sparse expansion merged at DEFAULT_EXPANSION_WEIGHT (0.5).
//!   search-expansion.ts:37: `if (!(index in merged)) merged[index] = value * weight;`
//! - SearchResponse and SearchResult JSON key order verified via serialized JSON.
//! - Scores are raw f64, unrounded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::*;

use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use crate::tools::search::expansion::merge_sparse_vectors;
use crate::tools::search::options::DEFAULT_EXPANSION_WEIGHT;

// ---------------------------------------------------------------------------
// Score-tracking Qdrant stub
//
// Records the score_threshold each leg was called with so tests can assert
// the threshold was forwarded verbatim to Qdrant (not applied post-fusion).
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct ScoreTrackingQdrant {
    dense_results: Vec<QdrantPoint>,
    sparse_results: Vec<QdrantPoint>,
    /// Captured dense score_threshold per call (index, threshold).
    dense_thresholds: Arc<Mutex<Vec<Option<f32>>>>,
    /// Captured sparse score_threshold per call.
    sparse_thresholds: Arc<Mutex<Vec<Option<f32>>>>,
}

impl SearchQdrant for ScoreTrackingQdrant {
    async fn search_dense(
        &self,
        _collection: &str,
        _vector: Vec<f32>,
        _limit: u64,
        score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        self.dense_thresholds.lock().unwrap().push(score_threshold);
        Ok(self.dense_results.clone())
    }

    async fn search_sparse(
        &self,
        _collection: &str,
        _indices: Vec<u32>,
        _values: Vec<f32>,
        _limit: u64,
        score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        self.sparse_thresholds.lock().unwrap().push(score_threshold);
        Ok(self.sparse_results.clone())
    }

    async fn scroll_page(
        &self,
        _collection: &str,
        _filter: Option<qdrant_client::qdrant::Filter>,
        _limit: u32,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }

    async fn retrieve_by_ids(
        &self,
        _collection: &str,
        _ids: Vec<String>,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// AC: Score threshold forwarded to Qdrant legs, NOT applied post-fusion.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn score_threshold_forwarded_to_qdrant_legs_only() {
    // Use a non-default threshold to make the assertion unambiguous.
    const THRESHOLD: f64 = 0.55;

    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2, 0.3]),
        sparse: Some(HashMap::from([(1u32, 0.9f32)])),
        unavailable: false,
    };

    // Two results: one below threshold (would be filtered if we re-applied it
    // post-fusion), one above.  The stub ignores the threshold — it returns
    // both regardless.  The test asserts the threshold was PASSED to Qdrant
    // and that the pipeline does NOT additionally filter fused results.
    let qdrant = ScoreTrackingQdrant {
        dense_results: vec![
            make_qdrant_point("high", 0.9, "high score content"),
            make_qdrant_point("low", 0.1, "low score content"), // below threshold
        ],
        sparse_results: vec![make_qdrant_point("mid", 0.6, "mid score content")],
        ..Default::default()
    };

    let opts = SearchOptions {
        score_threshold: THRESHOLD,
        ..opts_with_mode("threshold test", SearchMode::Hybrid)
    };

    let resp = run_search_pipeline(&mut daemon, &qdrant, Vec::new(), &opts, None, false).await;

    // 1. Threshold was forwarded to the dense leg.
    let dense_thresholds = qdrant.dense_thresholds.lock().unwrap().clone();
    assert!(
        !dense_thresholds.is_empty(),
        "dense search must have been called"
    );
    assert_eq!(
        dense_thresholds[0],
        Some(THRESHOLD as f32),
        "score_threshold must be forwarded to dense search_dense call"
    );

    // 2. Threshold was forwarded to the sparse leg (at 0.5× per flow_collect.rs:42-43).
    let sparse_thresholds = qdrant.sparse_thresholds.lock().unwrap().clone();
    assert!(
        !sparse_thresholds.is_empty(),
        "sparse search must have been called"
    );
    let expected_sparse_threshold = (THRESHOLD * 0.5) as f32;
    assert_eq!(
        sparse_thresholds[0],
        Some(expected_sparse_threshold),
        "sparse threshold must be score_threshold * 0.5 (flow_collect.rs:43)"
    );

    // 3. Fused results are NOT additionally filtered: the stub returned "low"
    //    (score 0.1 < threshold), and after RRF fusion it should still appear in
    //    resp.results because post-fusion filtering is intentionally absent.
    let ids: Vec<&str> = resp.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&"low"),
        "post-fusion score filtering must NOT be applied — 'low' must survive RRF"
    );
}

#[tokio::test]
async fn semantic_mode_dense_threshold_forwarded_no_sparse_leg() {
    const THRESHOLD: f64 = 0.45;

    let mut daemon = StubDaemon {
        dense: Some(vec![0.5, 0.5]),
        sparse: Some(HashMap::from([(3u32, 0.7f32)])),
        unavailable: false,
    };
    let qdrant = ScoreTrackingQdrant {
        dense_results: vec![make_qdrant_point("sem1", 0.8, "semantic result")],
        ..Default::default()
    };
    let opts = SearchOptions {
        score_threshold: THRESHOLD,
        ..opts_with_mode("semantic threshold", SearchMode::Semantic)
    };

    let _ = run_search_pipeline(&mut daemon, &qdrant, Vec::new(), &opts, None, false).await;

    let dense_t = qdrant.dense_thresholds.lock().unwrap().clone();
    assert_eq!(
        dense_t[0],
        Some(THRESHOLD as f32),
        "dense threshold forwarded in semantic mode"
    );
    let sparse_t = qdrant.sparse_thresholds.lock().unwrap().clone();
    assert!(
        sparse_t.is_empty(),
        "sparse leg must not fire in semantic mode"
    );
}

// ---------------------------------------------------------------------------
// Sparse expansion merged at DEFAULT_EXPANSION_WEIGHT (0.5).
//
// search-expansion.ts:37:
//   `if (!(index in merged)) merged[index] = value * weight;`
// Only new indices are added; existing indices are NOT modified.
// ---------------------------------------------------------------------------

#[test]
fn sparse_expansion_merged_at_default_weight() {
    // DEFAULT_EXPANSION_WEIGHT = 0.5 (search-expansion.ts default, options.rs:28).
    let original: HashMap<u32, f32> = HashMap::from([(1u32, 1.0f32), (2u32, 0.8f32)]);
    // Expansion has index 2 (overlap — must NOT override) and index 3 (new).
    let expansion: HashMap<u32, f32> = HashMap::from([(2u32, 0.9f32), (3u32, 0.6f32)]);

    let merged = merge_sparse_vectors(&original, &expansion, DEFAULT_EXPANSION_WEIGHT);

    // Existing indices unchanged.
    assert_eq!(merged[&1], 1.0f32, "existing index 1 must not be modified");
    assert_eq!(
        merged[&2], 0.8f32,
        "overlapping index 2 must keep original value"
    );
    // New index added at reduced weight.
    let expected_3 = (0.6f64 * DEFAULT_EXPANSION_WEIGHT) as f32;
    assert_eq!(
        merged[&3], expected_3,
        "new index 3 must be added at value * DEFAULT_EXPANSION_WEIGHT (0.5)"
    );
}

#[test]
fn sparse_expansion_no_overlap_all_new_indices_at_half_weight() {
    // search-expansion.ts:37: new indices get value * weight.
    let original: HashMap<u32, f32> = HashMap::from([(10u32, 0.5f32)]);
    let expansion: HashMap<u32, f32> = HashMap::from([(20u32, 1.0f32), (30u32, 0.4f32)]);

    let merged = merge_sparse_vectors(&original, &expansion, 0.5);

    assert_eq!(merged[&10], 0.5f32);
    assert!((merged[&20] - 0.5f32).abs() < 1e-6, "1.0 * 0.5 = 0.5");
    assert!((merged[&30] - 0.2f32).abs() < 1e-6, "0.4 * 0.5 = 0.2");
}

#[test]
fn sparse_expansion_empty_expansion_returns_original() {
    let original: HashMap<u32, f32> = HashMap::from([(5u32, 0.7f32)]);
    let expansion: HashMap<u32, f32> = HashMap::new();

    let merged = merge_sparse_vectors(&original, &expansion, DEFAULT_EXPANSION_WEIGHT);

    assert_eq!(merged, original);
}

// ---------------------------------------------------------------------------
// JSON field order tests.
//
// SearchResponse: results, total, query, mode, scope, collections_searched,
//   status?, status_reason?, branch?, diversity_score?
// SearchResult: id, score, collection, content, title?, metadata,
//   provenance?, parent_context?, graph_context?
// Optional fields omitted when None (skip_serializing_if = "Option::is_none").
//
// serde_json::Map uses BTreeMap (alphabetical) without `preserve_order`, so
// key order is extracted from the raw JSON string at depth-1.
// ---------------------------------------------------------------------------

/// Extract top-level JSON object key names in appearance order from `json`.
fn top_level_keys(json: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let bytes = json.as_bytes();
    let mut i = 0;
    let mut depth = 0i32;
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => {
                depth += 1;
                i += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                i += 1;
            }
            b'"' if depth == 1 => {
                i += 1;
                let mut key = String::new();
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    key.push(bytes[i] as char);
                    i += 1;
                }
                while i < bytes.len() && matches!(bytes[i], b' ' | b'\t') {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b':' {
                    keys.push(key);
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    keys
}

#[test]
fn search_response_json_field_order() {
    use crate::tools::search::types::{SearchMode, SearchResponse, SearchScope};
    // Required-only (no optionals).
    let resp_req = SearchResponse {
        results: vec![],
        total: 0,
        query: "t".into(),
        mode: SearchMode::Hybrid,
        scope: SearchScope::All,
        collections_searched: vec!["projects".into()],
        status: None,
        status_reason: None,
        branch: None,
        diversity_score: None,
    };
    assert_eq!(
        top_level_keys(&serde_json::to_string(&resp_req).unwrap()),
        &[
            "results",
            "total",
            "query",
            "mode",
            "scope",
            "collections_searched"
        ],
        "SearchResponse required fields order (search-types.ts)"
    );
    // All optionals present.
    let resp_full = SearchResponse {
        results: vec![],
        total: 3,
        query: "q".into(),
        mode: SearchMode::Semantic,
        scope: SearchScope::Project,
        collections_searched: vec!["projects".into()],
        status: Some("uncertain".into()),
        status_reason: Some("r".into()),
        branch: Some("main".into()),
        diversity_score: Some(0.75),
    };
    assert_eq!(
        top_level_keys(&serde_json::to_string(&resp_full).unwrap()),
        &[
            "results",
            "total",
            "query",
            "mode",
            "scope",
            "collections_searched",
            "status",
            "status_reason",
            "branch",
            "diversity_score"
        ],
        "SearchResponse all-optional fields order (search-types.ts)"
    );
}

#[test]
fn search_result_json_field_order() {
    use crate::tools::search::types::{Provenance, SearchResult};
    // No title, no provenance.
    let r_min = SearchResult {
        id: "p1".into(),
        score: 0.85,
        collection: "projects".into(),
        content: "c".into(),
        title: None,
        metadata: HashMap::new(),
        provenance: None,
        parent_context: None,
        graph_context: None,
    };
    assert_eq!(
        top_level_keys(&serde_json::to_string(&r_min).unwrap()),
        &["id", "score", "collection", "content", "metadata"],
        "SearchResult required field order (search-types.ts)"
    );
    // With title + provenance.
    let prov = Provenance {
        source: "projects".into(),
        library_name: None,
        library_path: None,
        doc_title: None,
        source_project_id: None,
    };
    let r_full = SearchResult {
        id: "p2".into(),
        score: 0.72,
        collection: "libraries".into(),
        content: "c".into(),
        title: Some("Doc".into()),
        metadata: HashMap::new(),
        provenance: Some(prov),
        parent_context: None,
        graph_context: None,
    };
    assert_eq!(
        top_level_keys(&serde_json::to_string(&r_full).unwrap()),
        &[
            "id",
            "score",
            "collection",
            "content",
            "title",
            "metadata",
            "provenance"
        ],
        "SearchResult with title+provenance field order (search-types.ts)"
    );
}

// ---------------------------------------------------------------------------
// Scores are raw f64 — no rounding.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scores_are_raw_unrounded() {
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1]),
        sparse: None,
        unavailable: false,
    };
    // An awkward score that would be rounded if we ever applied rounding.
    let raw_score = 0.123456789f64;
    let qdrant = StubQdrant {
        dense_results: vec![make_qdrant_point("r1", raw_score, "content")],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_with_mode("raw score query", SearchMode::Semantic),
        None,
        false,
    )
    .await;

    assert!(!resp.results.is_empty());
    assert_eq!(
        resp.results[0].score, raw_score,
        "scores must be raw f64, never rounded"
    );
}

#[tokio::test]
async fn scores_raw_after_rrf_fusion() {
    // RRF scores are computed as 1/(k+rank) sums — verify they aren't rounded.
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2]),
        sparse: Some(HashMap::from([(1u32, 0.8f32)])),
        unavailable: false,
    };
    let qdrant = StubQdrant {
        dense_results: vec![make_qdrant_point("d1", 0.9, "dense")],
        sparse_results: vec![make_qdrant_point("d1", 0.7, "dense")], // same id → RRF boost
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_with_mode("rrf score", SearchMode::Hybrid),
        None,
        false,
    )
    .await;

    assert!(!resp.results.is_empty());
    let score = resp.results[0].score;
    // RRF score for rank-1 + rank-1 at k=60: 1/61 + 1/61 ≈ 0.032786…
    // The important invariant: serialised score equals the Rust f64 value (no rounding).
    let json = serde_json::to_string(&resp).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();
    let json_score = v["results"][0]["score"].as_f64().unwrap();
    assert_eq!(
        json_score, score,
        "serialised score must equal raw f64 without rounding"
    );
}
