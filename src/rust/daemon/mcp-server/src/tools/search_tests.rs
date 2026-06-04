//! Hermetic tests for the `search` tool pipeline — stubs, helpers, AC-S1/S2/S3.
//!
//! Split into sibling files for size compliance:
//! - `search_tests.rs`          — stubs + AC-S1 hybrid, AC-S2 semantic, AC-S3 keyword
//! - `search_tests_fallback.rs` — AC-S4 daemon-down, AC-S5/F-001 refusal
//! - `search_tests_score.rs`    — score-threshold, sparse expansion, field order

use std::collections::HashMap;

use serde_json::Value;

use super::exact::ExactSearchDaemon;
use super::flow::{run_search_pipeline, EmbedDaemon, SearchQdrant};
use super::flow_fallback::FALLBACK_STATUS_REASON;
use super::graph_context::GraphQueryDaemon;
use super::options::SearchOptions;
use super::types::{SearchMode, SearchScope};
use crate::proto::{
    QueryRelatedRequest, QueryRelatedResponse, TextSearchRequest, TextSearchResponse,
};
use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};

// Pull in sibling test modules.
#[path = "search_tests_fallback.rs"]
mod fallback_tests;
#[path = "search_tests_m3_m4_m5.rs"]
mod m3_m4_m5_tests;
#[path = "search_tests_score.rs"]
mod score_tests;

// ---------------------------------------------------------------------------
// Stub daemon (shared across all search test modules via pub(super))
// ---------------------------------------------------------------------------

/// Controls stub daemon behavior.
#[derive(Default)]
pub(super) struct StubDaemon {
    /// Dense embedding to return. `None` → return transport error.
    pub dense: Option<Vec<f32>>,
    /// Sparse vector to return. `None` → return transport error.
    pub sparse: Option<HashMap<u32, f32>>,
    /// If true, both embed_text and generate_sparse_vector return errors.
    pub unavailable: bool,
}

impl EmbedDaemon for StubDaemon {
    async fn embed_text(&mut self, _text: &str) -> Result<Vec<f32>, tonic::Status> {
        if self.unavailable {
            return Err(tonic::Status::unavailable("daemon down"));
        }
        Ok(self.dense.clone().unwrap_or_default())
    }

    async fn generate_sparse_vector(
        &mut self,
        _text: &str,
    ) -> Result<HashMap<u32, f32>, tonic::Status> {
        if self.unavailable {
            return Err(tonic::Status::unavailable("daemon down"));
        }
        Ok(self.sparse.clone().unwrap_or_default())
    }
}

impl GraphQueryDaemon for StubDaemon {
    async fn query_related(
        &mut self,
        _request: QueryRelatedRequest,
    ) -> Result<QueryRelatedResponse, tonic::Status> {
        Ok(QueryRelatedResponse {
            nodes: vec![],
            total: 0,
            query_time_ms: 0,
        })
    }
}

impl ExactSearchDaemon for StubDaemon {
    async fn text_search(
        &mut self,
        _request: TextSearchRequest,
    ) -> Result<TextSearchResponse, tonic::Status> {
        Ok(TextSearchResponse {
            matches: vec![],
            total_matches: 0,
            truncated: false,
            query_time_ms: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Stub Qdrant (shared)
// ---------------------------------------------------------------------------

/// Build a scored Qdrant point for use in stubs.
pub(super) fn make_qdrant_point(id: &str, score: f64, content: &str) -> QdrantPoint {
    let mut payload = HashMap::new();
    payload.insert("content".to_string(), Value::String(content.to_string()));
    payload.insert(
        "tenant_id".to_string(),
        Value::String("tenant_abc".to_string()),
    );
    QdrantPoint {
        id: id.to_string(),
        score,
        payload,
    }
}

/// Build a retrieved (no-score) Qdrant point for use in stubs.
pub(super) fn make_retrieved_point(id: &str, content: &str) -> QdrantRetrievedPoint {
    let mut payload = HashMap::new();
    payload.insert("content".to_string(), Value::String(content.to_string()));
    payload.insert(
        "tenant_id".to_string(),
        Value::String("tenant_abc".to_string()),
    );
    QdrantRetrievedPoint {
        id: id.to_string(),
        payload,
    }
}

/// Stub Qdrant with configurable responses per collection.
#[derive(Default)]
pub(super) struct StubQdrant {
    pub dense_results: Vec<QdrantPoint>,
    pub sparse_results: Vec<QdrantPoint>,
    pub scroll_results: Vec<QdrantRetrievedPoint>,
    /// Dense search returns Err for this collection name when Some.
    pub dense_error_coll: Option<String>,
    /// Sparse search returns Err for this collection name when Some.
    pub sparse_error_coll: Option<String>,
}

impl SearchQdrant for StubQdrant {
    async fn search_dense(
        &self,
        collection: &str,
        _vector: Vec<f32>,
        _limit: u64,
        _score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        if let Some(ref err_coll) = self.dense_error_coll {
            if err_coll == collection {
                return Err(anyhow::anyhow!(
                    "Qdrant collection unavailable: {collection}"
                ));
            }
        }
        Ok(self.dense_results.clone())
    }

    async fn search_sparse(
        &self,
        collection: &str,
        _indices: Vec<u32>,
        _values: Vec<f32>,
        _limit: u64,
        _score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        if let Some(ref err_coll) = self.sparse_error_coll {
            if err_coll == collection {
                return Err(anyhow::anyhow!("Qdrant sparse unavailable: {collection}"));
            }
        }
        Ok(self.sparse_results.clone())
    }

    async fn scroll_page(
        &self,
        _collection: &str,
        _filter: Option<qdrant_client::qdrant::Filter>,
        _limit: u32,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(self.scroll_results.clone())
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
// SearchOptions helpers (shared)
// ---------------------------------------------------------------------------

pub(super) fn opts_hybrid(query: &str, limit: usize) -> SearchOptions {
    SearchOptions {
        query: query.to_string(),
        collection: None,
        mode: SearchMode::Hybrid,
        limit,
        score_threshold: 0.3,
        scope: SearchScope::All,
        branch: None,
        file_type: None,
        project_id: None,
        library_name: None,
        library_path: None,
        include_libraries: false,
        tag: None,
        tags: None,
        expand_context: false,
        path_glob: None,
        component: None,
        exact: false,
        context_lines: 0,
        include_graph_context: false,
        diverse: false,
        limit_explicit: false,
    }
}

pub(super) fn opts_with_mode(query: &str, mode: SearchMode) -> SearchOptions {
    SearchOptions {
        mode,
        ..opts_hybrid(query, 10)
    }
}

pub(super) fn opts_project_scope(query: &str) -> SearchOptions {
    SearchOptions {
        scope: SearchScope::Project,
        project_id: None, // deliberately unresolved
        ..opts_hybrid(query, 10)
    }
}

pub(super) fn opts_project_with_id(query: &str, pid: &str) -> SearchOptions {
    SearchOptions {
        scope: SearchScope::Project,
        project_id: Some(pid.to_string()),
        ..opts_hybrid(query, 10)
    }
}

pub(super) fn opts_keyword(query: &str, limit: usize) -> SearchOptions {
    SearchOptions {
        mode: SearchMode::Keyword,
        ..opts_hybrid(query, limit)
    }
}

// ---------------------------------------------------------------------------
// AC-S1: Hybrid — embed + sparse legs both fire, RRF fusion produces results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s1_hybrid_mode_fires_both_legs() {
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2, 0.3]),
        sparse: Some(HashMap::from([(1u32, 0.8f32), (2u32, 0.5f32)])),
        unavailable: false,
    };
    let qdrant = StubQdrant {
        dense_results: vec![
            make_qdrant_point("d1", 0.9, "dense result one"),
            make_qdrant_point("d2", 0.8, "dense result two"),
        ],
        sparse_results: vec![
            make_qdrant_point("s1", 0.7, "sparse result one"),
            make_qdrant_point("d1", 0.6, "dense result one"), // overlap — RRF boost
        ],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_with_mode("test query", SearchMode::Hybrid),
        None,
        false,
        &Default::default(),
        &(),
    )
    .await;

    assert_eq!(resp.mode, SearchMode::Hybrid);
    assert!(
        resp.status.is_none(),
        "should be clean (no degraded status)"
    );
    assert!(!resp.results.is_empty());
    // d1 appears in both dense and sparse → highest RRF score
    assert_eq!(
        &resp.results[0].id, "d1",
        "overlap item should rank highest after RRF"
    );
    for r in &resp.results {
        assert!(r.score > 0.0, "all scores must be positive");
    }
}

// ---------------------------------------------------------------------------
// AC-S2: Semantic mode — only dense leg fires, sparse never queried
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s2_semantic_mode_dense_only() {
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2]),
        sparse: Some(HashMap::from([(1u32, 0.9f32)])),
        unavailable: false,
    };

    struct TrackingQdrant {
        dense_results: Vec<QdrantPoint>,
        sparse_call_count: std::sync::atomic::AtomicUsize,
    }
    impl SearchQdrant for TrackingQdrant {
        async fn search_dense(
            &self,
            _c: &str,
            _v: Vec<f32>,
            _l: u64,
            _t: Option<f32>,
            _f: Option<qdrant_client::qdrant::Filter>,
        ) -> anyhow::Result<Vec<QdrantPoint>> {
            Ok(self.dense_results.clone())
        }
        async fn search_sparse(
            &self,
            _c: &str,
            _i: Vec<u32>,
            _v: Vec<f32>,
            _l: u64,
            _t: Option<f32>,
            _f: Option<qdrant_client::qdrant::Filter>,
        ) -> anyhow::Result<Vec<QdrantPoint>> {
            self.sparse_call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![])
        }
        async fn scroll_page(
            &self,
            _c: &str,
            _f: Option<qdrant_client::qdrant::Filter>,
            _l: u32,
        ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
            Ok(vec![])
        }
        async fn retrieve_by_ids(
            &self,
            _c: &str,
            _ids: Vec<String>,
        ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
            Ok(vec![])
        }
    }

    let qdrant = TrackingQdrant {
        dense_results: vec![make_qdrant_point("sem1", 0.88, "semantic hit")],
        sparse_call_count: std::sync::atomic::AtomicUsize::new(0),
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_with_mode("semantic query", SearchMode::Semantic),
        None,
        false,
        &Default::default(),
        &(),
    )
    .await;

    assert_eq!(resp.mode, SearchMode::Semantic);
    assert_eq!(
        qdrant
            .sparse_call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "sparse must NOT be called in semantic mode"
    );
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].id, "sem1");
    assert_eq!(resp.results[0].score, 0.88, "raw score preserved, no RRF");
    assert!(resp.status.is_none());
}

// ---------------------------------------------------------------------------
// AC-S3: Keyword mode — only sparse leg fires, embed_text never called
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s3_keyword_mode_sparse_only() {
    struct TrackingDaemon {
        embed_count: std::sync::atomic::AtomicUsize,
        sparse: HashMap<u32, f32>,
    }
    impl EmbedDaemon for TrackingDaemon {
        async fn embed_text(&mut self, _text: &str) -> Result<Vec<f32>, tonic::Status> {
            self.embed_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![0.1, 0.2])
        }
        async fn generate_sparse_vector(
            &mut self,
            _text: &str,
        ) -> Result<HashMap<u32, f32>, tonic::Status> {
            Ok(self.sparse.clone())
        }
    }
    impl GraphQueryDaemon for TrackingDaemon {
        async fn query_related(
            &mut self,
            _r: QueryRelatedRequest,
        ) -> Result<QueryRelatedResponse, tonic::Status> {
            Ok(QueryRelatedResponse {
                nodes: vec![],
                total: 0,
                query_time_ms: 0,
            })
        }
    }

    let mut daemon = TrackingDaemon {
        embed_count: std::sync::atomic::AtomicUsize::new(0),
        sparse: HashMap::from([(5u32, 0.9f32), (10u32, 0.4f32)]),
    };
    let qdrant = StubQdrant {
        sparse_results: vec![make_qdrant_point("kw1", 0.75, "keyword match")],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_with_mode("keyword query", SearchMode::Keyword),
        None,
        false,
        &Default::default(),
        &(),
    )
    .await;

    assert_eq!(resp.mode, SearchMode::Keyword);
    assert_eq!(
        daemon.embed_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "embed_text must NOT be called in keyword mode"
    );
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].id, "kw1");
    assert!(resp.status.is_none());
}

// ---------------------------------------------------------------------------
// FALLBACK_STATUS_REASON constant matches TS exactly
// ---------------------------------------------------------------------------

#[test]
fn fallback_status_reason_matches_ts_search_qdrant_ts_405() {
    // search-qdrant.ts:405: 'Daemon unavailable - using fallback text search'
    assert_eq!(
        FALLBACK_STATUS_REASON,
        "Daemon unavailable - using fallback text search"
    );
}
