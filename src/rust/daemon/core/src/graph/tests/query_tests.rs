//! Tests for graph traversal queries, impact analysis, stats, and orphan pruning.

use super::*;

// -- Helper: build a call chain a -> b -> c -> d --

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

// -- Query related (recursive CTE) --

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

// -- Stats --

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
