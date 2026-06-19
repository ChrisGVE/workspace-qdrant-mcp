//! Tests for graph traversal queries, impact analysis, stats, and orphan pruning.

use super::*;

// -- Helper: build a call chain a -> b -> c -> d --

async fn build_call_chain(
    store: &SqliteGraphStore,
) -> (GraphNode, GraphNode, GraphNode, GraphNode) {
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

// -- Query related (recursive CTE) --

#[tokio::test]
async fn test_query_related_1_hop() {
    let store = test_store().await;
    let (a, b, _c, _d) = build_call_chain(&store).await;

    let results = store
        .query_related(TENANT, &a.node_id, 1, None, None)
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
        .query_related(TENANT, &a.node_id, 2, None, None)
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
        .query_related(TENANT, &a.node_id, 3, None, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 3); // b, c, d
}

#[tokio::test]
async fn test_query_related_max_hops_boundary() {
    let store = test_store().await;
    let (a, _b, _c, _d) = build_call_chain(&store).await;

    let results = store
        .query_related(TENANT, &a.node_id, 0, None, None)
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

    let results = store
        .query_related(TENANT, &a.node_id, 1, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, b.node_id);

    let results = store
        .query_related(TENANT, &a.node_id, 1, Some(&[EdgeType::UsesType]), None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, c.node_id);
}

// -- Impact analysis --

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
        .impact_analysis(TENANT, "target_fn", Some("lib.rs"), None)
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
        .impact_analysis(TENANT, "target", Some("c.rs"), None)
        .await
        .unwrap();

    assert_eq!(report.total_impacted, 2);
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
        .impact_analysis(TENANT, "nonexistent", None, None)
        .await
        .unwrap();

    assert_eq!(report.total_impacted, 0);
    assert!(report.impacted_nodes.is_empty());
}

// -- Stats --

#[tokio::test]
async fn test_stats_empty() {
    let store = test_store().await;
    let stats = store.stats(Some(TENANT), None).await.unwrap();
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

    let stats = store.stats(Some(TENANT), None).await.unwrap();
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.nodes_by_type.get("function"), Some(&2));
    assert_eq!(stats.nodes_by_type.get("struct"), Some(&1));
}

#[tokio::test]
async fn test_stats_all_tenants() {
    let store = test_store().await;

    let node_a = GraphNode::new("tenant-a", "a.rs", "x", NodeType::Function);
    let node_b = GraphNode::new("tenant-b", "b.rs", "y", NodeType::Function);
    store.upsert_nodes(&[node_a, node_b]).await.unwrap();

    let stats = store.stats(None, None).await.unwrap();
    assert_eq!(stats.total_nodes, 2);
}

// -- Prune orphans --

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

    let stats = store.stats(Some(TENANT), None).await.unwrap();
    assert_eq!(stats.total_nodes, 2);
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

// -- Branch filtering --

#[tokio::test]
async fn test_query_related_branch_filter() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function).with_branch("main");
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function).with_branch("main");
    let c = GraphNode::new(TENANT, "c.rs", "c", NodeType::Function).with_branch("feature");
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();

    let edge_ab =
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs").with_branch("main");
    let edge_ac = GraphEdge::new(TENANT, &a.node_id, &c.node_id, EdgeType::Calls, "a.rs")
        .with_branch("feature");
    store.insert_edges(&[edge_ab, edge_ac]).await.unwrap();

    // No branch filter: see both
    let all = store
        .query_related(TENANT, &a.node_id, 1, None, None)
        .await
        .unwrap();
    assert_eq!(all.len(), 2);

    // Branch "main": only b
    let main_only = store
        .query_related(TENANT, &a.node_id, 1, None, Some("main"))
        .await
        .unwrap();
    assert_eq!(main_only.len(), 1);
    assert_eq!(main_only[0].node_id, b.node_id);

    // Branch "feature": only c
    let feat = store
        .query_related(TENANT, &a.node_id, 1, None, Some("feature"))
        .await
        .unwrap();
    assert_eq!(feat.len(), 1);
    assert_eq!(feat[0].node_id, c.node_id);

    // Wildcard "*": cross-branch
    let wild = store
        .query_related(TENANT, &a.node_id, 1, None, Some("*"))
        .await
        .unwrap();
    assert_eq!(wild.len(), 2);
}

#[tokio::test]
async fn test_stats_branch_filter() {
    let store = test_store().await;

    let a = GraphNode::new(TENANT, "a.rs", "a", NodeType::Function).with_branch("main");
    let b = GraphNode::new(TENANT, "b.rs", "b", NodeType::Function).with_branch("dev");
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

    let edge =
        GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs").with_branch("main");
    store.insert_edge(&edge).await.unwrap();

    let all = store.stats(Some(TENANT), None).await.unwrap();
    assert_eq!(all.total_nodes, 2);
    assert_eq!(all.total_edges, 1);

    let main = store.stats(Some(TENANT), Some("main")).await.unwrap();
    assert_eq!(main.total_nodes, 1);
    assert_eq!(main.total_edges, 1);

    let dev = store.stats(Some(TENANT), Some("dev")).await.unwrap();
    assert_eq!(dev.total_nodes, 1);
    assert_eq!(dev.total_edges, 0);
}

#[tokio::test]
async fn test_impact_analysis_branch_filter() {
    let store = test_store().await;

    let caller = GraphNode::new(TENANT, "a.rs", "caller", NodeType::Function).with_branch("main");
    let target = GraphNode::new(TENANT, "b.rs", "target", NodeType::Function).with_branch("main");
    let other = GraphNode::new(TENANT, "c.rs", "other", NodeType::Function).with_branch("dev");
    store
        .upsert_nodes(&[caller.clone(), target.clone(), other.clone()])
        .await
        .unwrap();

    let e1 = GraphEdge::new(
        TENANT,
        &caller.node_id,
        &target.node_id,
        EdgeType::Calls,
        "a.rs",
    )
    .with_branch("main");
    let e2 = GraphEdge::new(
        TENANT,
        &other.node_id,
        &target.node_id,
        EdgeType::Calls,
        "c.rs",
    )
    .with_branch("dev");
    store.insert_edges(&[e1, e2]).await.unwrap();

    let all = store
        .impact_analysis(TENANT, "target", Some("b.rs"), None)
        .await
        .unwrap();
    assert_eq!(all.total_impacted, 2);

    let main = store
        .impact_analysis(TENANT, "target", Some("b.rs"), Some("main"))
        .await
        .unwrap();
    assert_eq!(main.total_impacted, 1);
    assert_eq!(main.impacted_nodes[0].symbol_name, "caller");
}

// ── export_adjacency (SQLite) ──────────────────────────────────────────────────

/// (a) Empty tenant: both node_ids and edges are empty.
#[tokio::test]
async fn test_export_adjacency_empty_tenant() {
    let store = test_store().await;
    let result = store
        .export_adjacency("no-such-tenant", None)
        .await
        .unwrap();
    assert!(result.node_ids.is_empty(), "empty tenant yields no nodes");
    assert!(result.edges.is_empty(), "empty tenant yields no edges");
}

/// (b) Single isolated node: node_ids has one entry, edges is empty.
#[tokio::test]
async fn test_export_adjacency_single_isolated_node() {
    let store = test_store().await;
    let node = GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function);
    store.upsert_nodes(&[node]).await.unwrap();

    let result = store.export_adjacency(TENANT, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 1, "one node expected");
    assert!(result.edges.is_empty(), "isolated node has no edges");
}

/// (c) Two connected nodes A→B: correct (0, 1, weight) returned.
#[tokio::test]
async fn test_export_adjacency_two_connected_nodes() {
    let store = test_store().await;
    let a = GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "beta", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[edge]).await.unwrap();

    let result = store.export_adjacency(TENANT, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 2);

    // node_ids are sorted by node_id string, so find indices by value.
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
    assert!((w - 1.0).abs() < f64::EPSILON, "default weight is 1.0");
}

/// (d) edge_types=Some([Calls]) returns only CALLS edges; None returns all.
#[tokio::test]
async fn test_export_adjacency_edge_type_filter() {
    let store = test_store().await;
    let a = GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "beta", NodeType::Function);
    let c = GraphNode::new(TENANT, "c.rs", "gamma", NodeType::Struct);
    store
        .upsert_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    let calls_edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    let uses_edge = GraphEdge::new(TENANT, &a.node_id, &c.node_id, EdgeType::UsesType, "a.rs");
    store.insert_edges(&[calls_edge, uses_edge]).await.unwrap();

    // Filter to CALLS only: one edge.
    let calls_only = store
        .export_adjacency(TENANT, Some(&[EdgeType::Calls]))
        .await
        .unwrap();
    assert_eq!(calls_only.node_ids.len(), 3, "all nodes always returned");
    assert_eq!(calls_only.edges.len(), 1, "CALLS filter yields one edge");

    // No filter: both edges.
    let all = store.export_adjacency(TENANT, None).await.unwrap();
    assert_eq!(all.edges.len(), 2, "None filter yields both edges");
}

/// (e) Orphan edge (endpoint absent from nodes) is silently skipped.
///
/// We insert a node pair + edge, then delete one of the nodes directly to
/// create a dangling foreign-key condition (SQLite FK enforcement off for
/// legacy compatibility). The orphan edge must not appear in the export.
#[tokio::test]
async fn test_export_adjacency_orphan_edge_skipped() {
    let store = test_store().await;
    let a = GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function);
    let b = GraphNode::new(TENANT, "b.rs", "beta", NodeType::Function);
    store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
    let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
    store.insert_edges(&[edge]).await.unwrap();

    // Delete node B directly, leaving the edge dangling. The graph_edges→
    // graph_nodes FK is RESTRICT, so a node with a referencing edge cannot be
    // deleted while FK enforcement is on. Disable FKs on a single dedicated
    // connection to synthesise the orphan state the export must defend against.
    let mut conn = store.pool().acquire().await.unwrap();
    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(&mut *conn)
        .await
        .unwrap();
    sqlx::query("DELETE FROM graph_nodes WHERE node_id = ?1")
        .bind(&b.node_id)
        .execute(&mut *conn)
        .await
        .unwrap();
    drop(conn);

    let result = store.export_adjacency(TENANT, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 1, "only A remains");
    assert!(
        result.edges.is_empty(),
        "orphan edge (B missing) must be skipped"
    );
}

/// (f) node_ids are sorted deterministically (DOM-01).
#[tokio::test]
async fn test_export_adjacency_node_ids_sorted() {
    let store = test_store().await;
    // Insert nodes in reverse alphabetical order to confirm sorting.
    let nodes = vec![
        GraphNode::new(TENANT, "c.rs", "gamma", NodeType::Function),
        GraphNode::new(TENANT, "a.rs", "alpha", NodeType::Function),
        GraphNode::new(TENANT, "b.rs", "beta", NodeType::Function),
    ];
    store.upsert_nodes(&nodes).await.unwrap();

    let result = store.export_adjacency(TENANT, None).await.unwrap();
    assert_eq!(result.node_ids.len(), 3);
    // node_ids must be in ascending lexicographic order.
    let mut sorted = result.node_ids.clone();
    sorted.sort();
    assert_eq!(
        result.node_ids, sorted,
        "node_ids must be sorted by node_id"
    );
}
