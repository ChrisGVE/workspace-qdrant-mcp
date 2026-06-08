//! Tests for the LadybugDB graph store.
//!
//! All tests create isolated tempdir databases so they run independently.
//! Buffer pool is set to 4MB (small footprint for parallel test execution).

use std::path::PathBuf;

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
async fn test_ladybug_store_create() {
    let (_, _tmp) = fresh_store("graph_create");
    // If we got here, the store was created successfully
}

#[tokio::test]
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
async fn test_ladybug_upsert_and_stats() {
    let (store, _tmp) = fresh_store("graph_upsert");

    let node = GraphNode::new(T, "src/main.rs", "main", NodeType::Function);
    store.upsert_node(&node).await.unwrap();

    let stats = store.stats(Some(T), None).await.unwrap();
    assert_eq!(stats.total_nodes, 1);
    assert_eq!(*stats.nodes_by_type.get("function").unwrap(), 1);
}

#[tokio::test]
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
async fn test_ladybug_upsert_empty() {
    let (store, _tmp) = fresh_store("graph_empty_upsert");
    store.upsert_nodes(&[]).await.unwrap();
    let stats = store.stats(None, None).await.unwrap();
    assert_eq!(stats.total_nodes, 0);
}

// ---- Edge operations ---------------------------------------------------------

#[tokio::test]
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
async fn test_ladybug_insert_edges_empty() {
    let (store, _tmp) = fresh_store("graph_empty_edges");
    store.insert_edges(&[]).await.unwrap();
}

// ---- Delete operations -------------------------------------------------------

#[tokio::test]
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
async fn test_ladybug_execute_cypher() {
    let (store, _tmp) = fresh_store("graph_cypher");
    let rows = store.execute_cypher("RETURN 1 + 2 AS result").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "3");
}

// ---- Parameterized query injection safety ------------------------------------

#[tokio::test]
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
