//! Unit tests for graph module: CRUD, recursive CTE queries, impact analysis.

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

use super::*;

/// Create an in-memory graph database for testing.
async fn test_store() -> SqliteGraphStore {
    let opts = SqliteConnectOptions::new()
        .filename(":memory:")
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .unwrap();

    // Run schema migration manually (GraphDbManager::migrate_v1 logic)
    sqlx::query(
        "CREATE TABLE graph_nodes (
            node_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            symbol_name TEXT NOT NULL,
            symbol_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            signature TEXT,
            language TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("CREATE INDEX idx_nodes_tenant ON graph_nodes(tenant_id)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_nodes_file ON graph_nodes(tenant_id, file_path)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_nodes_symbol ON graph_nodes(tenant_id, symbol_name)")
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query(
        "CREATE TABLE graph_edges (
            edge_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata_json TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("CREATE INDEX idx_edges_tenant ON graph_edges(tenant_id)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_edges_source ON graph_edges(source_node_id)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_edges_target ON graph_edges(target_node_id)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_edges_source_file ON graph_edges(tenant_id, source_file)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE INDEX idx_edges_type ON graph_edges(edge_type)")
        .execute(&pool)
        .await
        .unwrap();

    SqliteGraphStore::new(pool)
}

const TENANT: &str = "test-tenant";

// ── Node ID determinism ──────────────────────────────────────────────

#[test]
fn test_node_id_deterministic() {
    let id1 = compute_node_id("t1", "src/main.rs", "main", NodeType::Function);
    let id2 = compute_node_id("t1", "src/main.rs", "main", NodeType::Function);
    assert_eq!(id1, id2);
    assert_eq!(id1.len(), 32); // 16 bytes = 32 hex chars
}

#[test]
fn test_node_id_differs_by_type() {
    let fn_id = compute_node_id("t1", "src/lib.rs", "Foo", NodeType::Function);
    let struct_id = compute_node_id("t1", "src/lib.rs", "Foo", NodeType::Struct);
    assert_ne!(fn_id, struct_id);
}

#[test]
fn test_node_id_differs_by_tenant() {
    let id1 = compute_node_id("tenant-a", "f.rs", "x", NodeType::Function);
    let id2 = compute_node_id("tenant-b", "f.rs", "x", NodeType::Function);
    assert_ne!(id1, id2);
}

#[test]
fn test_edge_id_deterministic() {
    let id1 = compute_edge_id("node-a", "node-b", EdgeType::Calls);
    let id2 = compute_edge_id("node-a", "node-b", EdgeType::Calls);
    assert_eq!(id1, id2);
    assert_eq!(id1.len(), 32);
}

#[test]
fn test_edge_id_differs_by_type() {
    let calls = compute_edge_id("a", "b", EdgeType::Calls);
    let imports = compute_edge_id("a", "b", EdgeType::Imports);
    assert_ne!(calls, imports);
}

// ── GraphNode constructors ───────────────────────────────────────────

#[test]
fn test_graph_node_new() {
    let node = GraphNode::new(TENANT, "src/main.rs", "main", NodeType::Function);
    assert!(!node.node_id.is_empty());
    assert_eq!(node.tenant_id, TENANT);
    assert_eq!(node.symbol_name, "main");
    assert_eq!(node.file_path, "src/main.rs");
    assert!(node.start_line.is_none());
}

#[test]
fn test_graph_node_stub() {
    let stub = GraphNode::stub(TENANT, "HashMap", NodeType::Struct);
    assert!(!stub.node_id.is_empty());
    assert_eq!(stub.file_path, ""); // stub has empty path
}

// ── Upsert node ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_upsert_node_insert() {
    let store = test_store().await;
    let node = GraphNode::new(TENANT, "src/lib.rs", "Config", NodeType::Struct);
    store.upsert_node(&node).await.unwrap();

    // Verify via stats
    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 1);
}

#[tokio::test]
async fn test_upsert_node_update() {
    let store = test_store().await;

    // Insert stub first
    let stub = GraphNode::stub(TENANT, "Foo", NodeType::Struct);
    let stub_id = stub.node_id.clone();
    store.upsert_node(&stub).await.unwrap();

    // Upsert with full info — same node_id because same (tenant, "", "Foo", struct)
    let mut full = GraphNode::stub(TENANT, "Foo", NodeType::Struct);
    full.file_path = "src/foo.rs".to_string();
    full.start_line = Some(10);
    full.end_line = Some(50);
    store.upsert_node(&full).await.unwrap();

    // Should still be one node
    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 1);

    // Verify file_path was updated (non-empty replaces empty)
    let row: (String,) = sqlx::query_as("SELECT file_path FROM graph_nodes WHERE node_id = ?1")
        .bind(&stub_id)
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(row.0, "src/foo.rs");
}

// ── Batch upsert nodes ──────────────────────────────────────────────

#[tokio::test]
async fn test_upsert_nodes_batch() {
    let store = test_store().await;
    let nodes = vec![
        GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function),
        GraphNode::new(TENANT, "b.rs", "beta", NodeType::Function),
        GraphNode::new(TENANT, "c.rs", "gamma", NodeType::Struct),
    ];
    store.upsert_nodes(&nodes).await.unwrap();

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 3);
}

#[tokio::test]
async fn test_upsert_nodes_empty() {
    let store = test_store().await;
    store.upsert_nodes(&[]).await.unwrap(); // should not error
}

// ── Insert edge ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_insert_edge() {
    let store = test_store().await;

    let caller = GraphNode::new(TENANT, "a.rs", "foo", NodeType::Function);
    let callee = GraphNode::new(TENANT, "b.rs", "bar", NodeType::Function);
    store
        .upsert_nodes(&[caller.clone(), callee.clone()])
        .await
        .unwrap();

    let edge = GraphEdge::new(
        TENANT,
        &caller.node_id,
        &callee.node_id,
        EdgeType::Calls,
        "a.rs",
    );
    store.insert_edge(&edge).await.unwrap();

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_edges, 1);
    assert_eq!(stats.edges_by_type.get("CALLS"), Some(&1));
}

#[tokio::test]
async fn test_insert_edge_duplicate_ignored() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "x", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "y", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

    let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edge(&edge).await.unwrap();
    store.insert_edge(&edge).await.unwrap(); // duplicate — should not error

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_edges, 1);
}

// ── Batch insert edges ──────────────────────────────────────────────

#[tokio::test]
async fn test_insert_edges_batch() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    let c = GraphNode::new(TENANT, "c.rs", "c", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(TENANT, &a.node_id, &c.node_id, EdgeType::Imports, "a.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_edges, 2);
}

// ── Delete edges by file ────────────────────────────────────────────

#[tokio::test]
async fn test_delete_edges_by_file() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    let c = GraphNode::new(TENANT, "c.rs", "c", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(TENANT, &a.node_id, &c.node_id, EdgeType::Imports, "a.rs"),
        GraphEdge::new(TENANT, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    // Delete edges from a.rs only
    let deleted = store.delete_edges_by_file(TENANT, "a.rs").await.unwrap();
    assert_eq!(deleted, 2);

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_edges, 1); // only the b->c edge remains
}

// ── Delete tenant ───────────────────────────────────────────────────

#[tokio::test]
async fn test_delete_tenant() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

    let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edge(&edge).await.unwrap();

    let deleted = store.delete_tenant(TENANT).await.unwrap();
    assert_eq!(deleted, 3); // 1 edge + 2 nodes

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.total_edges, 0);
}

// ── Query related (recursive CTE) ──────────────────────────────────

/// Build a call chain: a -> b -> c -> d
async fn build_call_chain(store: &SqliteGraphStore) -> (GraphNode, GraphNode, GraphNode, GraphNode)
{
    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    let c = GraphNode::new(TENANT, "c.rs", "c", NodeType::Function);
    let d = GraphNode::new(TENANT, "d.rs", "d", NodeType::Function);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone(), d.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(TENANT, &b.node_id, &c.node_id, EdgeType::Calls, "b.rs"),
        GraphEdge::new(TENANT, &c.node_id, &d.node_id, EdgeType::Calls, "c.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    (a, b, c, d)
}

#[tokio::test]
async fn test_query_related_1_hop() {
    let store = test_store().await;
    let (a, b, _c, _d) = build_call_chain(&store).await;

    let results = store
        .query_related(TENANT, &a.node_id, 1, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, b.node_id);
    assert_eq!(results[0].depth, 1);
}

#[tokio::test]
async fn test_query_related_2_hops() {
    let store = test_store().await;
    let (a, _b, _c, _d) = build_call_chain(&store).await;

    let results = store
        .query_related(TENANT, &a.node_id, 2, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].depth, 1);
    assert_eq!(results[1].depth, 2);
}

#[tokio::test]
async fn test_query_related_3_hops_reaches_end() {
    let store = test_store().await;
    let (a, _b, _c, _d) = build_call_chain(&store).await;

    let results = store
        .query_related(TENANT, &a.node_id, 3, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 3); // b, c, d
}

#[tokio::test]
async fn test_query_related_max_hops_boundary() {
    let store = test_store().await;
    let (a, _b, _c, _d) = build_call_chain(&store).await;

    // max_hops=0 should return nothing
    let results = store
        .query_related(TENANT, &a.node_id, 0, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_query_related_edge_type_filter() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    let c = GraphNode::new(TENANT, "c.rs", "c", NodeType::Struct);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
        GraphEdge::new(TENANT, &a.node_id, &c.node_id, EdgeType::UsesType, "a.rs"),
    ];
    store.insert_edges(&edges).await.unwrap();

    // Filter to CALLS only
    let results = store
        .query_related(TENANT, &a.node_id, 1, Some(&[EdgeType::Calls]))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, b.node_id);

    // Filter to USES_TYPE only
    let results = store
        .query_related(TENANT, &a.node_id, 1, Some(&[EdgeType::UsesType]))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, c.node_id);
}

// ── Impact analysis ─────────────────────────────────────────────────

#[tokio::test]
async fn test_impact_analysis_direct_callers() {
    let store = test_store().await;

    let caller1 = GraphNode::new(TENANT, "a.rs", "caller1", NodeType::Function);
    let caller2 = GraphNode::new(TENANT, "b.rs", "caller2", NodeType::Function);
    let target = GraphNode::new(TENANT, "lib.rs", "target_fn", NodeType::Function);
    store
        .upsert_nodes(&[caller1.clone(), caller2.clone(), target.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(
            TENANT,
            &caller1.node_id,
            &target.node_id,
            EdgeType::Calls,
            "a.rs",
        ),
        GraphEdge::new(
            TENANT,
            &caller2.node_id,
            &target.node_id,
            EdgeType::Calls,
            "b.rs",
        ),
    ];
    store.insert_edges(&edges).await.unwrap();

    let report = store
        .impact_analysis(TENANT, "target_fn", Some("lib.rs"))
        .await
        .unwrap();

    assert_eq!(report.symbol_name, "target_fn");
    assert_eq!(report.total_impacted, 2);
    assert!(report
        .impacted_nodes
        .iter()
        .all(|n| n.impact_type == "direct_caller"));
}

#[tokio::test]
async fn test_impact_analysis_transitive() {
    let store = test_store().await;

    // indirect_caller -> direct_caller -> target
    let indirect = GraphNode::new(TENANT, "a.rs", "indirect", NodeType::Function);
    let direct = GraphNode::new(TENANT, "b.rs", "direct", NodeType::Function);
    let target = GraphNode::new(TENANT, "c.rs", "target", NodeType::Function);
    store
        .upsert_nodes(&[indirect.clone(), direct.clone(), target.clone()])
        .await
        .unwrap();

    let edges = vec![
        GraphEdge::new(
            TENANT,
            &indirect.node_id,
            &direct.node_id,
            EdgeType::Calls,
            "a.rs",
        ),
        GraphEdge::new(
            TENANT,
            &direct.node_id,
            &target.node_id,
            EdgeType::Calls,
            "b.rs",
        ),
    ];
    store.insert_edges(&edges).await.unwrap();

    let report = store
        .impact_analysis(TENANT, "target", Some("c.rs"))
        .await
        .unwrap();

    assert_eq!(report.total_impacted, 2);
    // direct at distance 1, indirect at distance 2
    let direct_node = report
        .impacted_nodes
        .iter()
        .find(|n| n.symbol_name == "direct")
        .unwrap();
    assert_eq!(direct_node.distance, 1);
    let indirect_node = report
        .impacted_nodes
        .iter()
        .find(|n| n.symbol_name == "indirect")
        .unwrap();
    assert_eq!(indirect_node.distance, 2);
}

#[tokio::test]
async fn test_impact_analysis_symbol_not_found() {
    let store = test_store().await;

    let report = store
        .impact_analysis(TENANT, "nonexistent", None)
        .await
        .unwrap();

    assert_eq!(report.total_impacted, 0);
    assert!(report.impacted_nodes.is_empty());
}

// ── Stats ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_empty() {
    let store = test_store().await;
    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.total_edges, 0);
}

#[tokio::test]
async fn test_stats_by_type() {
    let store = test_store().await;

    let nodes = vec![
        GraphNode::new(TENANT, "a.rs", "a", NodeType::Function),
        GraphNode::new(TENANT, "b.rs", "b", NodeType::Function),
        GraphNode::new(TENANT, "c.rs", "C", NodeType::Struct),
    ];
    store.upsert_nodes(&nodes).await.unwrap();

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.nodes_by_type.get("function"), Some(&2));
    assert_eq!(stats.nodes_by_type.get("struct"), Some(&1));
}

#[tokio::test]
async fn test_stats_all_tenants() {
    let store = test_store().await;

    let node_a = GraphNode::new("tenant-a", "a.rs", "x", NodeType::Function);
    let node_b = GraphNode::new("tenant-b", "b.rs", "y", NodeType::Function);
    store
        .upsert_nodes(&[node_a, node_b])
        .await
        .unwrap();

    let stats = store.stats(None).await.unwrap();
    assert_eq!(stats.total_nodes, 2);
}

// ── Prune orphans ───────────────────────────────────────────────────

#[tokio::test]
async fn test_prune_orphans() {
    let store = test_store().await;

    let connected_a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let connected_b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    let orphan = GraphNode::new(TENANT, "c.rs", "orphan", NodeType::Function);
    store
        .upsert_nodes(&[connected_a.clone(), connected_b.clone(), orphan.clone()])
        .await
        .unwrap();

    // Only a->b has an edge; orphan has none
    let edge = GraphEdge::new(
        TENANT,
        &connected_a.node_id,
        &connected_b.node_id,
        EdgeType::Calls,
        "a.rs",
    );
    store.insert_edge(&edge).await.unwrap();

    let pruned = store.prune_orphans(TENANT).await.unwrap();
    assert_eq!(pruned, 1);

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert_eq!(stats.total_nodes, 2); // only connected nodes remain
}

#[tokio::test]
async fn test_prune_orphans_none_to_prune() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

    let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edge(&edge).await.unwrap();

    let pruned = store.prune_orphans(TENANT).await.unwrap();
    assert_eq!(pruned, 0);
}

// ── Edge type enum round-trip ───────────────────────────────────────

#[test]
fn test_edge_type_round_trip() {
    for et in [
        EdgeType::Calls,
        EdgeType::Contains,
        EdgeType::Imports,
        EdgeType::UsesType,
        EdgeType::Extends,
        EdgeType::Implements,
    ] {
        let s = et.as_str();
        let parsed = EdgeType::from_str(s).unwrap();
        assert_eq!(parsed, et);
    }
}

#[test]
fn test_node_type_round_trip() {
    for nt in [
        NodeType::File,
        NodeType::Function,
        NodeType::AsyncFunction,
        NodeType::Class,
        NodeType::Method,
        NodeType::Struct,
        NodeType::Trait,
        NodeType::Interface,
        NodeType::Enum,
        NodeType::Impl,
        NodeType::Module,
        NodeType::Constant,
        NodeType::TypeAlias,
        NodeType::Macro,
    ] {
        let s = nt.as_str();
        let parsed = NodeType::from_str(s).unwrap();
        assert_eq!(parsed, nt);
    }
}
