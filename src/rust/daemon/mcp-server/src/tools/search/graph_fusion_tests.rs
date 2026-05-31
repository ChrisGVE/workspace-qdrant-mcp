//! Hermetic tests for the graph-expansion fusion pass (GitHub #80).

use std::collections::HashMap;

use serde_json::Value;

use super::*;
use crate::proto::{QueryRelatedResponse, TraversalNodeProto};
use crate::qdrant::fusion::{SearchType, TaggedResult};

// ── Fixtures ─────────────────────────────────────────────────────────────────

fn code_result(
    id: &str,
    score: f64,
    sym: &str,
    ctype: &str,
    tenant: &str,
    file: &str,
) -> TaggedResult {
    let mut payload: HashMap<String, Value> = HashMap::new();
    payload.insert("chunk_symbol_name".into(), Value::String(sym.into()));
    payload.insert("chunk_chunk_type".into(), Value::String(ctype.into()));
    payload.insert("tenant_id".into(), Value::String(tenant.into()));
    payload.insert("file_path".into(), Value::String(file.into()));
    payload.insert("content".into(), Value::String(format!("body of {sym}")));
    TaggedResult {
        id: id.into(),
        score,
        collection: "projects".into(),
        payload,
        search_type: SearchType::Semantic,
    }
}

fn node(node_id: &str, sym: &str, depth: u32) -> TraversalNodeProto {
    TraversalNodeProto {
        node_id: node_id.into(),
        symbol_name: sym.into(),
        symbol_type: "function".into(),
        file_path: "src/other.rs".into(),
        edge_type: "CALLS".into(),
        depth,
        path: String::new(),
    }
}

/// Daemon stub returning canned nodes keyed by requested `node_id`.
#[derive(Default)]
struct MapDaemon {
    responses: HashMap<String, Vec<TraversalNodeProto>>,
    fail: bool,
}

impl GraphQueryDaemon for MapDaemon {
    async fn query_related(
        &mut self,
        request: crate::proto::QueryRelatedRequest,
    ) -> Result<QueryRelatedResponse, tonic::Status> {
        if self.fail {
            return Err(tonic::Status::unavailable("down"));
        }
        let nodes = self
            .responses
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn no_code_candidates_is_noop() {
    // A non-code chunk type produces no candidates → results untouched.
    let mut results = vec![code_result("v1", 0.9, "readme", "text", "t", "README.md")];
    let before = results[0].score;
    let mut daemon = MapDaemon::default();
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].score, before, "no candidates → no alpha scaling");
}

#[tokio::test]
async fn candidates_but_no_expansion_still_alpha_scales_originals() {
    // A code candidate exists but the daemon errors → no expansion, yet TS still
    // applies `r.score *= ALPHA` to every original result.
    let mut results = vec![code_result("v1", 1.0, "foo", "function", "t", "src/a.rs")];
    let mut daemon = MapDaemon {
        fail: true,
        ..Default::default()
    };
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;
    assert_eq!(results.len(), 1);
    assert!(
        (results[0].score - 0.7).abs() < 1e-9,
        "original scaled by ALPHA"
    );
}

#[tokio::test]
async fn expansion_adds_nodes_and_fuses_scores() {
    let seed = code_result("v1", 1.0, "foo", "function", "t", "src/a.rs");
    let seed_node_id = compute_node_id("t", "src/a.rs", "foo", "function");
    let mut results = vec![seed];

    let mut daemon = MapDaemon::default();
    daemon.responses.insert(
        seed_node_id.clone(),
        vec![node("expanded_1", "bar", 1), node("expanded_2", "baz", 2)],
    );
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;

    // Original scaled to 0.7; two expanded nodes added at (1-0.7)*proximity.
    assert_eq!(results.len(), 3);
    let by_id: HashMap<&str, &TaggedResult> = results.iter().map(|r| (r.id.as_str(), r)).collect();
    assert!((by_id["v1"].score - 0.7).abs() < 1e-9);
    // depth 1 → proximity 0.8 → 0.3*0.8 = 0.24
    assert!((by_id["expanded_1"].score - 0.24).abs() < 1e-9);
    // depth 2 → proximity 0.5 → 0.3*0.5 = 0.15
    assert!((by_id["expanded_2"].score - 0.15).abs() < 1e-9);
    // Expanded node carries the graph_expansion marker + SearchType::Graph.
    assert_eq!(
        by_id["expanded_1"]
            .payload
            .get("source")
            .and_then(|v| v.as_str()),
        Some("graph_expansion")
    );
    assert_eq!(by_id["expanded_1"].search_type, SearchType::Graph);
    // Sorted by score desc: v1(0.70) > expanded_1(0.24) > expanded_2(0.15).
    assert_eq!(results[0].id, "v1");
    assert_eq!(results[1].id, "expanded_1");
    assert_eq!(results[2].id, "expanded_2");
}

#[tokio::test]
async fn convergence_bonus_applies_to_existing_vector_result() {
    // Two vector results; expanding v1 returns a node whose id == v2's id →
    // v2 gets +0.1 BEFORE the *0.7 scaling: (0.5 + 0.1) * 0.7 = 0.42.
    let v1 = code_result("v1", 1.0, "foo", "function", "t", "src/a.rs");
    let v2 = code_result("v2_id", 0.5, "bar", "function", "t", "src/b.rs");
    let seed_node_id = compute_node_id("t", "src/a.rs", "foo", "function");
    let mut results = vec![v1, v2];

    let mut daemon = MapDaemon::default();
    daemon
        .responses
        .insert(seed_node_id, vec![node("v2_id", "bar", 1)]);
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;

    // No new node added (v2_id already present); v2 bonused then scaled.
    assert_eq!(results.len(), 2);
    let by_id: HashMap<&str, &TaggedResult> = results.iter().map(|r| (r.id.as_str(), r)).collect();
    assert!((by_id["v2_id"].score - 0.42).abs() < 1e-9, "(0.5+0.1)*0.7");
    assert!((by_id["v1"].score - 0.7).abs() < 1e-9);
}

#[tokio::test]
async fn self_node_skipped_and_per_result_cap_enforced() {
    let seed = code_result("v1", 1.0, "foo", "function", "t", "src/a.rs");
    let seed_node_id = compute_node_id("t", "src/a.rs", "foo", "function");
    let mut results = vec![seed];

    // Return the seed itself (must be skipped) + 7 distinct nodes (cap = 5).
    let mut nodes = vec![node(&seed_node_id, "foo", 0)];
    for i in 0..7 {
        nodes.push(node(&format!("n{i}"), "sym", 1));
    }
    let mut daemon = MapDaemon::default();
    daemon.responses.insert(seed_node_id.clone(), nodes);
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;

    // The cap slices the node list to 5 FIRST (TS `slice(0, MAX_EXPANDED_PER_RESULT)`),
    // then the self node (first in the list) is skipped — so only 4 of the sliced 5
    // become new results. The self node is never added.
    let expanded = results.iter().filter(|r| r.id != "v1").count();
    assert_eq!(expanded, 4, "self node consumes one of the 5 sliced slots");
    assert!(results.iter().all(|r| r.id != seed_node_id));
}

#[tokio::test]
async fn empty_relative_path_does_not_fall_back_to_file_path() {
    // TS `relative_path ?? file_path` is nullish: a present-but-empty
    // relative_path is taken as "" and the row is then skipped by `!filePath`.
    // It must NOT fall back to file_path → no candidate → no expansion at all.
    let mut payload: HashMap<String, Value> = HashMap::new();
    payload.insert("chunk_symbol_name".into(), Value::String("foo".into()));
    payload.insert("chunk_chunk_type".into(), Value::String("function".into()));
    payload.insert("tenant_id".into(), Value::String("t".into()));
    payload.insert("relative_path".into(), Value::String(String::new()));
    payload.insert("file_path".into(), Value::String("src/a.rs".into()));
    let mut results = vec![TaggedResult {
        id: "v1".into(),
        score: 1.0,
        collection: "projects".into(),
        payload,
        search_type: SearchType::Semantic,
    }];

    // Daemon HAS a response under the node_id derived from the file_path
    // fallback — if the bug were present it would expand; with the fix it never
    // becomes a candidate, so the daemon is never queried.
    let fallback_node_id = compute_node_id("t", "src/a.rs", "foo", "function");
    let mut daemon = MapDaemon::default();
    daemon
        .responses
        .insert(fallback_node_id, vec![node("expanded_1", "bar", 1)]);
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;

    // No candidate → no expansion, and no alpha scaling (candidates empty).
    assert_eq!(results.len(), 1, "no graph-expanded node added");
    assert!((results[0].score - 1.0).abs() < 1e-9, "score untouched");
}

#[tokio::test]
async fn expanded_node_converts_to_narrow_search_result() {
    // A graph-expanded TaggedResult converts to a SearchResult mirroring TS
    // `nodeToSearchResult`: content/title set, but metadata is NARROW
    // (no content/title/_search_type keys) and provenance is None.
    let seed = code_result("v1", 1.0, "foo", "function", "t", "src/a.rs");
    let seed_node_id = compute_node_id("t", "src/a.rs", "foo", "function");
    let mut results = vec![seed];
    let mut daemon = MapDaemon::default();
    daemon
        .responses
        .insert(seed_node_id, vec![node("expanded_1", "bar", 1)]);
    expand_and_fuse_with_graph(&mut daemon, &mut results, "projects").await;

    let expanded = results
        .into_iter()
        .find(|r| r.id == "expanded_1")
        .expect("expanded node present");
    let sr = crate::tools::search::flow_collect::tagged_to_search_result(expanded);
    assert_eq!(sr.title.as_deref(), Some("bar"));
    assert_eq!(sr.content, "function bar in src/other.rs");
    assert!(sr.provenance.is_none(), "graph nodes carry no provenance");
    assert!(
        !sr.metadata.contains_key("content"),
        "narrow metadata: no content"
    );
    assert!(
        !sr.metadata.contains_key("title"),
        "narrow metadata: no title"
    );
    assert!(
        !sr.metadata.contains_key("_search_type"),
        "narrow metadata: no _search_type"
    );
    assert_eq!(
        sr.metadata.get("source").and_then(|v| v.as_str()),
        Some("graph_expansion")
    );
    assert_eq!(
        sr.metadata
            .get("chunk_symbol_name")
            .and_then(|v| v.as_str()),
        Some("bar")
    );
}
