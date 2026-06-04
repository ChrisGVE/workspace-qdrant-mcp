//! Search-result PARITY tests (WI-d4 #82 task 28).
//!
//! These drive the full `run_search_pipeline` end-to-end with mocked
//! embedder / Qdrant / graph daemons (via the pipeline traits) and assert the
//! fused + ranked result set — exercising BOTH GitHub #80 (graph-expansion
//! fusion) and GitHub #81 (group/all-scope relevance decay) in a single flow,
//! so a future refactor cannot silently change the ranking the MCP server
//! returns.

use std::collections::HashMap;

use serde_json::Value;

use super::flow::{run_search_pipeline, EmbedDaemon, SearchQdrant};
use super::graph_context::compute_node_id;
use super::options::SearchOptions;
use super::scope::ScopeContext;
use crate::models::{SearchMode, SearchScope};
use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use crate::workspace_daemon::{
    QueryRelatedRequest, QueryRelatedResponse, TextSearchRequest, TextSearchResponse,
    TraversalNodeProto,
};

// ── Mocks ────────────────────────────────────────────────────────────────────

/// Daemon mock: fixed dense embedding + graph nodes keyed by requested node_id.
#[derive(Default)]
struct MockDaemon {
    dense: Vec<f32>,
    graph: HashMap<String, Vec<TraversalNodeProto>>,
}

impl EmbedDaemon for MockDaemon {
    async fn embed_text(&mut self, _text: &str) -> Result<Vec<f32>, tonic::Status> {
        Ok(self.dense.clone())
    }
    async fn generate_sparse_vector(
        &mut self,
        _text: &str,
    ) -> Result<HashMap<u32, f32>, tonic::Status> {
        Ok(HashMap::new())
    }
}

impl super::graph_context::GraphQueryDaemon for MockDaemon {
    async fn query_related(
        &mut self,
        request: QueryRelatedRequest,
    ) -> Result<QueryRelatedResponse, tonic::Status> {
        let nodes = self
            .graph
            .get(&request.node_id)
            .cloned()
            .unwrap_or_default();
        Ok(QueryRelatedResponse {
            total: nodes.len() as u32,
            nodes,
            query_time_ms: 0,
        })
    }
}

impl super::exact::ExactSearchDaemon for MockDaemon {
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

/// Qdrant mock returning fixed dense points for every collection.
struct MockQdrant {
    dense: Vec<QdrantPoint>,
}

impl SearchQdrant for MockQdrant {
    async fn search_dense(
        &self,
        _c: &str,
        _v: Vec<f32>,
        _l: u64,
        _t: Option<f32>,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        Ok(self.dense.clone())
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

fn code_point(
    id: &str,
    score: f64,
    sym: &str,
    ctype: &str,
    tenant: &str,
    file: &str,
) -> QdrantPoint {
    let mut payload: HashMap<String, Value> = HashMap::new();
    payload.insert("chunk_symbol_name".into(), Value::String(sym.into()));
    payload.insert("chunk_chunk_type".into(), Value::String(ctype.into()));
    payload.insert("tenant_id".into(), Value::String(tenant.into()));
    payload.insert("file_path".into(), Value::String(file.into()));
    payload.insert("content".into(), Value::String(format!("body of {sym}")));
    QdrantPoint {
        id: id.into(),
        score,
        payload,
    }
}

fn graph_node(node_id: &str, sym: &str, depth: u32) -> TraversalNodeProto {
    TraversalNodeProto {
        node_id: node_id.into(),
        symbol_name: sym.into(),
        symbol_type: "function".into(),
        file_path: "src/exp.rs".into(),
        edge_type: "CALLS".into(),
        depth,
        path: String::new(),
        tenant_id: String::new(),
        edge_confidence: 1.0,
    }
}

fn opts_semantic_graph() -> SearchOptions {
    SearchOptions {
        query: "q".into(),
        collection: Some("projects".into()),
        mode: SearchMode::Semantic,
        limit: 10,
        score_threshold: 0.0,
        scope: SearchScope::Group,
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
        include_graph_context: true,
        diverse: false,
        limit_explicit: false,
    }
}

// ── Parity test: #81 decay + #80 graph fusion in one pipeline run ────────────

#[tokio::test]
async fn parity_decay_then_graph_fusion_ranking_is_stable() {
    // Two semantic hits in different tenants. `a` is a code symbol (becomes a
    // graph-expansion candidate); `b` is a plain doc (not expanded). The
    // group-scope decay map downweights tenant t2, and graph fusion alpha-scales
    // originals and inserts one expanded node.
    let a = code_point("a", 0.9, "foo", "function", "t1", "src/a.rs");
    let mut b_payload: HashMap<String, Value> = HashMap::new();
    b_payload.insert("tenant_id".into(), Value::String("t2".into()));
    b_payload.insert("content".into(), Value::String("plain doc".into()));
    let b = QdrantPoint {
        id: "b".into(),
        score: 0.85,
        payload: b_payload,
    };

    let a_node = compute_node_id("t1", "src/a.rs", "foo", "function");
    let mut daemon = MockDaemon {
        dense: vec![0.1, 0.2, 0.3],
        ..Default::default()
    };
    daemon
        .graph
        .insert(a_node, vec![graph_node("exp1", "bar", 1)]);

    let qdrant = MockQdrant { dense: vec![a, b] };

    // #81: scope=group decay — t1 kept, t2 downweighted to 0.4.
    let mut decay = HashMap::new();
    decay.insert("t1".to_string(), 1.0);
    decay.insert("t2".to_string(), 0.4);
    let scope_ctx = ScopeContext {
        group_tenant_ids: Some(vec!["t1".into(), "t2".into()]),
        decay_map: Some(decay),
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_semantic_graph(),
        Some("t1"),
        false,
        &scope_ctx,
        &(),
    )
    .await;

    // Expected ranking:
    //   decay: a=0.90, b=0.34   (#81)
    //   graph fusion alpha*0.7:  a=0.63, b=0.238; expanded exp1 = (1-0.7)*0.8 = 0.24  (#80)
    //   sort desc → a(0.63) > exp1(0.24) > b(0.238)
    let ids: Vec<&str> = resp.results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["a", "exp1", "b"],
        "decay (#81) must push t2's `b` below the graph-expanded `exp1` (#80)"
    );

    // Scores reflect both passes.
    let by_id: HashMap<&str, f64> = resp
        .results
        .iter()
        .map(|r| (r.id.as_str(), r.score))
        .collect();
    assert!(
        (by_id["a"] - 0.63).abs() < 1e-9,
        "a: 0.9*1.0 decay then *0.7 alpha"
    );
    assert!(
        (by_id["exp1"] - 0.24).abs() < 1e-9,
        "exp1: (1-0.7)*0.8 proximity"
    );
    assert!(
        (by_id["b"] - 0.238).abs() < 1e-9,
        "b: 0.85*0.4 decay then *0.7 alpha"
    );

    // The graph-expanded node carries narrow metadata (no _search_type) and the
    // graph_expansion source marker — TS `nodeToSearchResult` parity.
    let exp = resp.results.iter().find(|r| r.id == "exp1").unwrap();
    assert!(
        !exp.metadata.contains_key("_search_type"),
        "narrow metadata"
    );
    assert_eq!(
        exp.metadata.get("source").and_then(|v| v.as_str()),
        Some("graph_expansion")
    );
    assert!(exp.provenance.is_none(), "graph nodes carry no provenance");
}
