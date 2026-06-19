//! Tests for the LadybugDB graph store.
//!
//! All tests create isolated tempdir databases. Each LadybugDB `Database`
//! reserves a large (multi-TiB) sparse virtual mmap for its max data-file size,
//! independent of `buffer_pool_size`, so running many in parallel exhausts the
//! process virtual-address space ("Mmap for size … failed"). Every kuzu-backed
//! test is therefore `#[serial]` so at most one live `Database` exists at a
//! time — the conformance suite (few stores) and the SQLite tests stay parallel.

use std::path::PathBuf;

use serial_test::serial;

use crate::graph::{EdgeType, GraphEdge, GraphNode, GraphStore, NodeType};

use super::config::LadybugConfig;
use super::store::LadybugGraphStore;

// ---- Helpers -----------------------------------------------------------------

const T: &str = "test-tenant";

/// Buffer pool for tests (64 MB). Must be large enough to hold the catalog
/// for all 6 rel tables. Avoids mmap failures during parallel test execution.
const TEST_BUFFER_POOL: u64 = 64 * 1024 * 1024;

/// Create a fresh store in a tempdir.
fn fresh_store(name: &str) -> (LadybugGraphStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join(name),
        buffer_pool_size: TEST_BUFFER_POOL,
        max_num_threads: 2,
    };
    let store = LadybugGraphStore::new(config).unwrap();
    (store, tmp)
}

// ---- Unit tests (no async) ---------------------------------------------------

#[test]
fn test_escape_cypher() {
    use super::store::escape_cypher;
    assert_eq!(escape_cypher("hello"), "hello");
    assert_eq!(escape_cypher("it's"), "it\\'s");
    assert_eq!(escape_cypher("a'b'c"), "a\\'b\\'c");
}

#[test]
fn test_default_config() {
    let config = LadybugConfig::default();
    assert!(config.db_path.to_string_lossy().contains("graph"));
    assert_eq!(config.max_num_threads, 4);
}

#[test]
fn test_custom_config() {
    let config = LadybugConfig {
        db_path: PathBuf::from("/tmp/test-graph"),
        buffer_pool_size: 512 * 1024 * 1024,
        max_num_threads: 8,
    };
    assert_eq!(config.db_path, PathBuf::from("/tmp/test-graph"));
    assert_eq!(config.buffer_pool_size, 512 * 1024 * 1024);
}

// ---- Store lifecycle ---------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_store_create() {
    let (_, _tmp) = fresh_store("graph_create");
    // If we got here, the store was created successfully
}

#[tokio::test]
#[serial]
async fn test_ladybug_store_reopen() {
    // Verify schema is idempotent across reopens
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join("reopen_test"),
        buffer_pool_size: TEST_BUFFER_POOL,
        max_num_threads: 2,
    };

    // First open: creates schema
    {
        let store = LadybugGraphStore::new(config.clone()).unwrap();
        let node = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
        store.upsert_node(&node).await.unwrap();
    }

    // Second open: schema already exists, data persists
    {
        let store = LadybugGraphStore::new(config).unwrap();
        let stats = store.stats(Some(T), None).await.unwrap();
        assert_eq!(stats.total_nodes, 1, "Data should persist across reopens");
    }
}

// ---- Node operations ---------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_upsert_and_stats() {
    let (store, _tmp) = fresh_store("graph_upsert");

    let node = GraphNode::new(T, "src/main.rs", "main", NodeType::Function);
    store.upsert_node(&node).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_nodes, 1);
    assert_eq!(*stats.nodes_by_type.get("function").unwrap(), 1);
}

#[tokio::test]
#[serial]
async fn test_ladybug_upsert_idempotent() {
    let (store, _tmp) = fresh_store("graph_idempotent");

    let mut node = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    node.language = Some("rust".to_string());
    store.upsert_node(&node).await.unwrap();

    // Upsert again with updated properties
    node.language = Some("python".to_string());
    store.upsert_node(&node).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_nodes, 1, "MERGE should not duplicate");
}

#[tokio::test]
#[serial]
async fn test_ladybug_batch_upsert() {
    let (store, _tmp) = fresh_store("graph_batch");

    let nodes = vec![
        GraphNode::new(T, "a.rs", "foo", NodeType::Function),
        GraphNode::new(T, "b.rs", "bar", NodeType::Struct),
        GraphNode::new(T, "c.rs", "baz", NodeType::Class),
    ];
    store.upsert_nodes(&nodes).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(*stats.nodes_by_type.get("function").unwrap_or(&0), 1);
    assert_eq!(*stats.nodes_by_type.get("struct").unwrap_or(&0), 1);
    assert_eq!(*stats.nodes_by_type.get("class").unwrap_or(&0), 1);
}

#[tokio::test]
#[serial]
async fn test_ladybug_upsert_empty() {
    let (store, _tmp) = fresh_store("graph_empty_upsert");
    store.upsert_nodes(&[]).await.unwrap();
    let stats = store.stats(None, None).await.unwrap();
    assert_eq!(stats.total_nodes, 0);
}

// ---- Edge operations ---------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_insert_edge() {
    let (store, _tmp) = fresh_store("graph_edge");

    let node_a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let node_b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    store
        .upsert_nodes(&[node_a.clone(), node_b.clone()])
        .await
        .unwrap();

    let edge = GraphEdge::new(T, &node_a.node_id, &node_b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edge(&edge).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_edges, 1);
    assert_eq!(*stats.edges_by_type.get("CALLS").unwrap(), 1);
}

#[tokio::test]
#[serial]
async fn test_ladybug_insert_edges_batch() {
    let (store, _tmp) = fresh_store("graph_edges_batch");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "baz", NodeType::Struct);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(T, &a.node_id, &c.node_id, EdgeType::UsesType, "a.rs"),
        GraphEdge::new(T, &b.node_id, &c.node_id, EdgeType::Imports, "b.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_edges, 3);
    assert_eq!(*stats.edges_by_type.get("CALLS").unwrap(), 1);
    assert_eq!(*stats.edges_by_type.get("USES_TYPE").unwrap(), 1);
    assert_eq!(*stats.edges_by_type.get("IMPORTS").unwrap(), 1);
}

#[tokio::test]
#[serial]
async fn test_ladybug_insert_edges_empty() {
    let (store, _tmp) = fresh_store("graph_empty_edges");
    store.insert_edges(&[]).await.unwrap();
}

// ---- Delete operations -------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_delete_edges_by_file() {
    let (store, _tmp) = fresh_store("graph_del_edges");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "baz", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    // Edges owned by a.rs and b.rs
    let edges = vec![
        GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(T, &a.node_id, &c.node_id, EdgeType::Imports, "a.rs"),
        GraphEdge::new(T, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    let stats_before = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats_before.total_edges, 3);

    // Delete edges owned by a.rs
    store.delete_edges_by_file(T, "a.rs").await.unwrap();

    let stats_after = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats_after.total_edges, 1, "Only b.rs edge should remain");
    assert_eq!(stats_after.total_nodes, 3, "Nodes should not be deleted");
}

#[tokio::test]
#[serial]
async fn test_ladybug_delete_tenant() {
    let (store, _tmp) = fresh_store("graph_del_tenant");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    let edge = GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[edge]).await.unwrap();

    // Also add data for a different tenant
    let other = GraphNode::new("other-tenant", "x.rs", "x", NodeType::Function);
    store.upsert_nodes(&[other]).await.unwrap();

    store.delete_tenant(T).await.unwrap();

    let stats_t = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats_t.total_nodes, 0, "All test-tenant data gone");
    assert_eq!(stats_t.total_edges, 0);

    let stats_other = store.stats(Some("other-tenant"), None).await.unwrap();
    assert_eq!(stats_other.total_nodes, 1, "Other tenant unaffected");
}

// ---- Query operations --------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_query_related() {
    let (store, _tmp) = fresh_store("graph_query");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "baz", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(T, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    // 1-hop from a: should find b
    let related = store
        .query_related(T, &a.node_id, 1, None, None)
        .await
        .unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].symbol_name, "bar");

    // 2-hop from a: should find both b and c
    let related = store
        .query_related(T, &a.node_id, 2, None, None)
        .await
        .unwrap();
    assert_eq!(related.len(), 2);
    let names: Vec<&str> = related.iter().map(|n| n.symbol_name.as_str()).collect();
    assert!(names.contains(&"bar"));
    assert!(names.contains(&"baz"));
}

#[tokio::test]
#[serial]
async fn test_ladybug_query_related_edge_filter() {
    let (store, _tmp) = fresh_store("graph_query_filter");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "baz", NodeType::Struct);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(T, &a.node_id, &c.node_id, EdgeType::UsesType, "a.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    // Filter to only CALLS: should find b but not c
    let related = store
        .query_related(T, &a.node_id, 1, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].symbol_name, "bar");

    // Filter to only USES_TYPE: should find c but not b
    let related = store
        .query_related(T, &a.node_id, 1, Some(&[EdgeType::UsesType]), None)
        .await
        .unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].symbol_name, "baz");
}

#[tokio::test]
#[serial]
async fn test_ladybug_impact_analysis() {
    let (store, _tmp) = fresh_store("graph_impact");

    // Build call chain: caller -> middle -> target
    let target = GraphNode::new(T, "t.rs", "target_fn", NodeType::Function);
    let middle = GraphNode::new(T, "m.rs", "middle_fn", NodeType::Function);
    let caller = GraphNode::new(T, "c.rs", "caller_fn", NodeType::Function);
    store
        .upsert_nodes(&[target.clone(), middle.clone(), caller.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(T, &caller.node_id, &middle.node_id, EdgeType::Calls, "c.rs"),
        GraphEdge::new(T, &middle.node_id, &target.node_id, EdgeType::Calls, "m.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    let report = store
        .impact_analysis(T, "target_fn", None, None)
        .await
        .unwrap();
    assert_eq!(report.symbol_name, "target_fn");
    assert!(
        report.total_impacted >= 1,
        "At least middle_fn should be impacted"
    );
}

// ---- Reingest pattern (T44) --------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_reingest_file() {
    let (store, _tmp) = fresh_store("graph_reingest");

    let a = GraphNode::new(T, "a.rs", "foo", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "bar", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "baz", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    // Initial edges from a.rs
    let old_edge = GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[old_edge]).await.unwrap();

    let stats_before = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats_before.total_edges, 1);

    // Reingest a.rs: delete old edges, insert new ones (a -> c instead of a -> b)
    store.delete_edges_by_file(T, "a.rs").await.unwrap();
    let new_edge = GraphEdge::new(T, &a.node_id, &c.node_id, EdgeType::Calls, "a.rs");
    store.upsert_nodes(&[a.clone()]).await.unwrap();
    store.insert_edges(&[new_edge]).await.unwrap();

    let stats_after = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats_after.total_edges, 1, "Old edge replaced by new");
    assert_eq!(stats_after.total_nodes, 3, "All nodes preserved");

    // Verify new edge target: query_related from a should find c, not b
    let related = store
        .query_related(T, &a.node_id, 1, None, None)
        .await
        .unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].symbol_name, "baz");
}

// ---- Stats completeness ------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_stats_global() {
    let (store, _tmp) = fresh_store("graph_stats_global");

    // Two tenants
    let a1 = GraphNode::new("t1", "a.rs", "a", NodeType::Function);
    let a2 = GraphNode::new("t2", "b.rs", "b", NodeType::Struct);
    store.upsert_nodes(&[a1.clone(), a2.clone()]).await.unwrap();

    // Global stats (None tenant filter)
    let stats = store.stats(None, None).await.unwrap();
    assert_eq!(stats.total_nodes, 2);
}

// ---- Cypher passthrough ------------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_execute_cypher() {
    let (store, _tmp) = fresh_store("graph_cypher");
    let rows = store.execute_cypher("RETURN 1 + 2 AS result").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "3");
}

// ---- find_path ---------------------------------------------------------------

/// Build a 2-node chain (A→B via CALLS) in a fresh store.
/// Returns (store, tmp, node_a, node_b).
async fn build_chain_2(
    name: &str,
) -> (
    LadybugGraphStore,
    tempfile::TempDir,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
) {
    use crate::graph::NodeType;
    let (store, tmp) = fresh_store(name);
    let a = crate::graph::GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = crate::graph::GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    store
        .insert_edges(&[GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            EdgeType::Calls,
            "a.rs",
        )])
        .await
        .unwrap();
    (store, tmp, a, b)
}

/// Build a 3-node chain (A→B→C via CALLS) in a fresh store.
async fn build_chain_3(
    name: &str,
) -> (
    LadybugGraphStore,
    tempfile::TempDir,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
) {
    use crate::graph::NodeType;
    let (store, tmp) = fresh_store(name);
    let a = crate::graph::GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = crate::graph::GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    let c = crate::graph::GraphNode::new(T, "c.rs", "gamma", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    store
        .insert_edges(&[
            GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
            GraphEdge::new(T, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
        ])
        .await
        .unwrap();
    (store, tmp, a, b, c)
}

/// Build a 4-node chain (A→B→C→D via CALLS) in a fresh store.
async fn build_chain_4(
    name: &str,
) -> (
    LadybugGraphStore,
    tempfile::TempDir,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
    crate::graph::GraphNode,
) {
    use crate::graph::NodeType;
    let (store, tmp) = fresh_store(name);
    let a = crate::graph::GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = crate::graph::GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    let c = crate::graph::GraphNode::new(T, "c.rs", "gamma", NodeType::Function);
    let d = crate::graph::GraphNode::new(T, "d.rs", "delta", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone(), d.clone()])
        .await
        .unwrap();
    store
        .insert_edges(&[
            GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
            GraphEdge::new(T, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
            GraphEdge::new(T, &c.node_id, &d.node_id, EdgeType::Calls, "c.rs"),
        ])
        .await
        .unwrap();
    (store, tmp, a, b, c, d)
}

/// Assert a TraversalNode has the sentinel fields set by find_path.
fn assert_path_node_sentinels(n: &crate::graph::TraversalNode, expected_depth: u32, tenant: &str) {
    assert_eq!(n.depth, expected_depth, "node {} wrong depth", n.node_id);
    assert_eq!(n.edge_type, "", "edge_type must be empty");
    assert_eq!(n.path, "", "path must be empty");
    assert_eq!(n.tenant_id, tenant, "tenant_id mismatch");
    assert!(
        (n.edge_confidence - 1.0).abs() < f64::EPSILON,
        "edge_confidence must be 1.0"
    );
}

/// (a) 2-hop path A→B→C: find_path(A,C) returns [A,B,C] at depths [0,1,2].
#[tokio::test]
#[serial]
async fn test_find_path_2hop() {
    let (store, _tmp, a, _b, c) = build_chain_3("fp_2hop").await;
    let path = store
        .find_path(T, &a.node_id, &c.node_id, 5, None, None)
        .await
        .unwrap();
    let path = path.expect("2-hop path must exist");
    assert_eq!(path.len(), 3, "path must have 3 nodes: A, B, C");
    assert_eq!(path[0].node_id, a.node_id);
    assert_eq!(path[2].node_id, c.node_id);
    for (i, node) in path.iter().enumerate() {
        assert_path_node_sentinels(node, i as u32, T);
    }
    // Verify actual symbol names (source→target direction)
    assert_eq!(path[0].symbol_name, "alpha");
    assert_eq!(path[1].symbol_name, "beta");
    assert_eq!(path[2].symbol_name, "gamma");
}

/// (b) 3-hop path A→B→C→D: find_path(A,D) returns [A,B,C,D] at depths [0,1,2,3].
#[tokio::test]
#[serial]
async fn test_find_path_3hop() {
    let (store, _tmp, a, _b, _c, d) = build_chain_4("fp_3hop").await;
    let path = store
        .find_path(T, &a.node_id, &d.node_id, 5, None, None)
        .await
        .unwrap();
    let path = path.expect("3-hop path must exist");
    assert_eq!(path.len(), 4);
    assert_eq!(path[0].node_id, a.node_id);
    assert_eq!(path[3].node_id, d.node_id);
    for (i, node) in path.iter().enumerate() {
        assert_path_node_sentinels(node, i as u32, T);
    }
}

/// (c) Disconnected nodes: find_path returns None.
#[tokio::test]
#[serial]
async fn test_find_path_no_path() {
    use crate::graph::NodeType;
    let (store, _tmp) = fresh_store("fp_nopath");
    let a = crate::graph::GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = crate::graph::GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    // Insert nodes but NO edge between them
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

    let result = store
        .find_path(T, &a.node_id, &b.node_id, 5, None, None)
        .await
        .unwrap();
    assert!(result.is_none(), "disconnected nodes must return None");
}

/// (d) Self-path: find_path(A, A) returns Some([A]) at depth 0.
#[tokio::test]
#[serial]
async fn test_find_path_self() {
    let (store, _tmp, a, _b) = build_chain_2("fp_self").await;
    let path = store
        .find_path(T, &a.node_id, &a.node_id, 5, None, None)
        .await
        .unwrap();
    let path = path.expect("self-path must return Some");
    assert_eq!(path.len(), 1);
    assert_eq!(path[0].node_id, a.node_id);
    assert_path_node_sentinels(&path[0], 0, T);
    assert_eq!(path[0].symbol_name, "alpha");
}

/// (e) Edge-type filter excludes the only available path.
/// Graph: A→B via USES_TYPE, but we ask for CALLS only → no path.
#[tokio::test]
#[serial]
async fn test_find_path_edge_type_filter_excludes() {
    use crate::graph::NodeType;
    let (store, _tmp) = fresh_store("fp_filter_excl");
    let a = crate::graph::GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = crate::graph::GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    store
        .insert_edges(&[GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            EdgeType::UsesType,
            "a.rs",
        )])
        .await
        .unwrap();

    // Filter to CALLS only — no CALLS edge exists, so no path
    let result = store
        .find_path(T, &a.node_id, &b.node_id, 5, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    assert!(
        result.is_none(),
        "CALLS filter must exclude the USES_TYPE path"
    );
}

/// (e continued) Edge-type filter allows the path when the type matches.
#[tokio::test]
#[serial]
async fn test_find_path_edge_type_filter_allows() {
    let (store, _tmp, a, _b, c) = build_chain_3("fp_filter_allow").await;
    // Chain uses CALLS edges; filter to CALLS — path must be found
    let path = store
        .find_path(T, &a.node_id, &c.node_id, 5, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    assert!(path.is_some(), "CALLS filter must allow a CALLS path");
    assert_eq!(path.unwrap().len(), 3);
}

/// (f) max_depth bound: path is 3 hops but max_depth=2 → None.
#[tokio::test]
#[serial]
async fn test_find_path_max_depth_too_small() {
    let (store, _tmp, a, _b, _c, d) = build_chain_4("fp_maxdepth").await;
    // Path A→D is 3 hops; max_depth=2 must return None
    let result = store
        .find_path(T, &a.node_id, &d.node_id, 2, None, None)
        .await
        .unwrap();
    assert!(
        result.is_none(),
        "path longer than max_depth must return None"
    );
}

/// (f continued) max_depth exactly equal to path length works.
#[tokio::test]
#[serial]
async fn test_find_path_max_depth_exact() {
    let (store, _tmp, a, _b, _c, d) = build_chain_4("fp_maxdepth_exact").await;
    // Path A→D is 3 hops; max_depth=3 must succeed
    let path = store
        .find_path(T, &a.node_id, &d.node_id, 3, None, None)
        .await
        .unwrap();
    assert!(
        path.is_some(),
        "max_depth exactly equal to path length must find the path"
    );
    assert_eq!(path.unwrap().len(), 4);
}

// ---- Parameterized query injection safety ------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_injection_safe() {
    let (store, _tmp) = fresh_store("graph_injection");

    // A symbol name with Cypher-injection-style payload
    let malicious_name = "foo' OR 1=1 --";
    let node = GraphNode::new(T, "a.rs", malicious_name, NodeType::Function);
    store.upsert_node(&node).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_nodes, 1, "Node created despite special chars");
}

// ---- Stub-edge resolution ----------------------------------------------------

#[tokio::test]
#[serial]
async fn test_ladybug_resolve_stub_edges_by_name() {
    let (store, _tmp) = fresh_store("graph_stub_resolve");

    // Real caller (a.rs) + real callee (b.rs) + a name-only stub "callee".
    let caller = GraphNode::new(T, "a.rs", "caller", NodeType::Function);
    let callee = GraphNode::new(T, "b.rs", "callee", NodeType::Function);
    let stub = GraphNode::stub(T, "callee", NodeType::Function);
    store
        .upsert_nodes(&[caller.clone(), callee.clone(), stub.clone()])
        .await
        .unwrap();
    // Dangling: caller -> stub("callee").
    let dangling = GraphEdge::new(T, &caller.node_id, &stub.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[dangling]).await.unwrap();

    // Unmatched external stub ("println") — no project node — stays dangling.
    let ext = GraphNode::stub(T, "println", NodeType::Function);
    store.upsert_nodes(&[ext.clone()]).await.unwrap();
    let ext_edge = GraphEdge::new(T, &caller.node_id, &ext.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[ext_edge]).await.unwrap();

    let repointed = store.resolve_stub_edges(T).await.unwrap();
    assert_eq!(repointed, 1, "only the matchable stub edge repoints");

    // Edge now reaches the REAL callee in b.rs.
    let related = store
        .query_related(T, &caller.node_id, 1, None, None)
        .await
        .unwrap();
    assert!(
        related.iter().any(|n| n.file_path == "b.rs"),
        "resolved edge should target the real callee node in b.rs"
    );
    // The matched stub no longer reachable; the external "println" stub stays.
    assert!(
        !related.iter().any(|n| n.node_id == stub.node_id),
        "matched stub should no longer be a neighbour after repoint"
    );
}

#[tokio::test]
#[serial]
async fn test_ladybug_resolve_stub_edges_skips_ambiguous() {
    let (store, _tmp) = fresh_store("graph_stub_ambiguous");

    // Two real nodes named "new" in different files → ambiguous.
    let caller = GraphNode::new(T, "a.rs", "caller", NodeType::Function);
    let new_a = GraphNode::new(T, "x.rs", "new", NodeType::Function);
    let new_b = GraphNode::new(T, "y.rs", "new", NodeType::Function);
    let stub = GraphNode::stub(T, "new", NodeType::Function);
    store
        .upsert_nodes(&[caller.clone(), new_a, new_b, stub.clone()])
        .await
        .unwrap();
    let dangling = GraphEdge::new(T, &caller.node_id, &stub.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[dangling]).await.unwrap();

    let repointed = store.resolve_stub_edges(T).await.unwrap();
    assert_eq!(
        repointed, 0,
        "ambiguous name (2 defining files) must not repoint"
    );
}

#[tokio::test]
#[serial]
async fn test_ladybug_graph_tenants() {
    let (store, _tmp) = fresh_store("graph_tenants");
    store
        .upsert_nodes(&[
            GraphNode::new("tenant-a", "a.rs", "foo", NodeType::Function),
            GraphNode::new("tenant-b", "b.rs", "bar", NodeType::Function),
        ])
        .await
        .unwrap();
    let mut tenants = store.graph_tenants().await.unwrap();
    tenants.sort();
    assert_eq!(
        tenants,
        vec!["tenant-a".to_string(), "tenant-b".to_string()]
    );
}

#[tokio::test]
#[serial]
async fn test_ladybug_resolve_stub_edges_same_file_tiebreaker() {
    let (store, _tmp) = fresh_store("graph_stub_samefile");

    // Two real "helper" nodes in different files — normally ambiguous — but the
    // caller's own file (a.rs) defines one, so that definition wins.
    let caller = GraphNode::new(T, "a.rs", "caller", NodeType::Function);
    let local = GraphNode::new(T, "a.rs", "helper", NodeType::Function);
    let other = GraphNode::new(T, "b.rs", "helper", NodeType::Function);
    let stub = GraphNode::stub(T, "helper", NodeType::Function);
    store
        .upsert_nodes(&[caller.clone(), local.clone(), other, stub.clone()])
        .await
        .unwrap();
    let dangling = GraphEdge::new(T, &caller.node_id, &stub.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[dangling]).await.unwrap();

    let repointed = store.resolve_stub_edges(T).await.unwrap();
    assert_eq!(
        repointed, 1,
        "same-file definition resolves despite a name clash"
    );

    // The repointed edge reaches the local (a.rs) helper, not b.rs.
    let related = store
        .query_related(T, &caller.node_id, 1, None, None)
        .await
        .unwrap();
    assert!(
        related.iter().any(|n| n.node_id == local.node_id),
        "same-file helper (a.rs) should be the chosen target"
    );
    assert!(
        related.iter().all(|n| n.file_path != "b.rs"),
        "the b.rs helper must NOT be chosen when the caller's own file defines one"
    );
}

// ── export_adjacency (LadybugDB) ───────────────────────────────────────────────

/// (a) Empty tenant: both node_ids and edges are empty.
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_empty_tenant() {
    let (store, _tmp) = fresh_store("ea_empty");
    let result = store
        .export_adjacency("no-such-tenant", None)
        .await
        .unwrap();
    assert!(result.node_ids.is_empty(), "empty tenant yields no nodes");
    assert!(result.edges.is_empty(), "empty tenant yields no edges");
}

/// (b) Single isolated node: node_ids has one entry, edges is empty.
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_single_isolated_node() {
    let (store, _tmp) = fresh_store("ea_isolated");
    let node = GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    store.upsert_nodes(&[node]).await.unwrap();

    let result = store.export_adjacency(T, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 1, "one node expected");
    assert!(result.edges.is_empty(), "isolated node has no edges");
}

/// (c) Two connected nodes A→B: correct (src_idx, tgt_idx, weight) returned.
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_two_connected_nodes() {
    let (store, _tmp) = fresh_store("ea_two");
    let a = GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    let edge = GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[edge]).await.unwrap();

    let result = store.export_adjacency(T, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 2, "two nodes");

    let idx_a = result
        .node_ids
        .iter()
        .position(|id| id == &a.node_id)
        .unwrap();
    let idx_b = result
        .node_ids
        .iter()
        .position(|id| id == &b.node_id)
        .unwrap();

    assert_eq!(result.edges.len(), 1, "exactly one edge");
    let (si, ti, w) = result.edges[0];
    assert_eq!(si, idx_a, "source index must be A");
    assert_eq!(ti, idx_b, "target index must be B");
    // LadybugDB rel tables carry a weight DOUBLE property (confirmed in
    // init_schema DDL). Default GraphEdge weight is 1.0.
    assert!((w - 1.0).abs() < f64::EPSILON, "default weight must be 1.0");
}

/// (d) edge_types=Some([Calls]) returns only CALLS edges; None returns all.
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_edge_type_filter() {
    let (store, _tmp) = fresh_store("ea_filter");
    let a = GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    let b = GraphNode::new(T, "b.rs", "beta", NodeType::Function);
    let c = GraphNode::new(T, "c.rs", "gamma", NodeType::Struct);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    let calls_edge = GraphEdge::new(T, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    let uses_edge = GraphEdge::new(T, &a.node_id, &c.node_id, EdgeType::UsesType, "a.rs");
    store.insert_edges(&[calls_edge, uses_edge]).await.unwrap();

    // Filter to CALLS only: one edge.
    let calls_only = store
        .export_adjacency(T, Some(&[EdgeType::Calls]))
        .await
        .unwrap();
    assert_eq!(calls_only.node_ids.len(), 3, "all nodes always returned");
    assert_eq!(calls_only.edges.len(), 1, "CALLS filter yields one edge");

    // No filter: both edges.
    let all = store.export_adjacency(T, None).await.unwrap();
    assert_eq!(all.edges.len(), 2, "None filter yields both edges");
}

/// (e) Orphan edge (endpoint absent from node table) is silently skipped.
///
/// We insert only node A (no B) but try to create an edge A→B.
/// LadybugDB MATCH-based CREATE will silently produce no edge when B is
/// absent. This test therefore validates the no-edge case; the skip logic
/// in the Rust post-processing layer is covered by the SQLite orphan test.
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_orphan_edge_skipped() {
    let (store, _tmp) = fresh_store("ea_orphan");
    let a = GraphNode::new(T, "a.rs", "alpha", NodeType::Function);
    // Insert only A — no B node exists.
    store.upsert_nodes(&[a.clone()]).await.unwrap();

    // Attempt to insert an edge whose target does not exist.
    // LadybugDB MATCH(a)-MATCH(b)-CREATE will produce no edge (MATCH fails).
    // This is idiomatic lbug behaviour — no panic, just a no-op.
    let fake_b_id = "nonexistent-node-id";
    let edge = GraphEdge {
        edge_id: "fake-edge".to_string(),
        tenant_id: T.to_string(),
        source_node_id: a.node_id.clone(),
        target_node_id: fake_b_id.to_string(),
        edge_type: EdgeType::Calls,
        source_file: "a.rs".to_string(),
        weight: 1.0,
        metadata_json: None,
        branch: None,
    };
    // insert_edge silently fails (no matching target node) — do not unwrap.
    let _ = store.insert_edge(&edge).await;

    let result = store.export_adjacency(T, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 1, "only A in the node list");
    assert!(
        result.edges.is_empty(),
        "no valid edge when target is absent"
    );
}

/// (f) node_ids are sorted deterministically (DOM-01).
#[tokio::test]
#[serial]
async fn test_lbug_export_adjacency_node_ids_sorted() {
    let (store, _tmp) = fresh_store("ea_sorted");
    // Insert nodes — LadybugDB stores them in insertion order internally;
    // our query uses ORDER BY n.node_id to guarantee deterministic output.
    let nodes = vec![
        GraphNode::new(T, "c.rs", "gamma", NodeType::Function),
        GraphNode::new(T, "a.rs", "alpha", NodeType::Function),
        GraphNode::new(T, "b.rs", "beta", NodeType::Function),
    ];
    store.upsert_nodes(&nodes).await.unwrap();

    let result = store.export_adjacency(T, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 3, "three nodes");
    // node_ids must be in ascending lexicographic order.
    let mut sorted = result.node_ids.clone();
    sorted.sort();
    assert_eq!(
        result.node_ids, sorted,
        "node_ids must be sorted by node_id"
    );
}
