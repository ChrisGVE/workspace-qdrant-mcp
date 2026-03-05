use super::*;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

/// Create an in-memory SQLite pool with graph schema.
async fn setup_graph_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();

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
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("CREATE INDEX idx_nodes_tenant ON graph_nodes(tenant_id)")
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
            created_at TEXT NOT NULL DEFAULT ''
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query("CREATE INDEX idx_edges_tenant ON graph_edges(tenant_id)")
        .execute(&pool)
        .await
        .unwrap();

    pool
}

async fn insert_node(pool: &SqlitePool, tenant: &str, id: &str, name: &str, stype: &str) {
    sqlx::query(
        "INSERT OR IGNORE INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type, file_path)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(id)
    .bind(tenant)
    .bind(name)
    .bind(stype)
    .bind(format!("{}.rs", name))
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_edge(pool: &SqlitePool, tenant: &str, src: &str, tgt: &str, etype: &str) {
    let edge_id = format!("{}_{}_{}_{}", tenant, src, tgt, etype);
    sqlx::query(
        "INSERT OR IGNORE INTO graph_edges (edge_id, tenant_id, source_node_id, target_node_id, edge_type, source_file)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&edge_id)
    .bind(tenant)
    .bind(src)
    .bind(tgt)
    .bind(etype)
    .bind("src.rs")
    .execute(pool)
    .await
    .unwrap();
}

/// Build a diamond graph: A -> B, A -> C, B -> D, C -> D
async fn build_diamond(pool: &SqlitePool) {
    for (id, name) in &[
        ("a", "alpha"),
        ("b", "beta"),
        ("c", "gamma"),
        ("d", "delta"),
    ] {
        insert_node(pool, "t1", id, name, "function").await;
    }
    insert_edge(pool, "t1", "a", "b", "CALLS").await;
    insert_edge(pool, "t1", "a", "c", "CALLS").await;
    insert_edge(pool, "t1", "b", "d", "CALLS").await;
    insert_edge(pool, "t1", "c", "d", "CALLS").await;
}

/// Build a chain: A -> B -> C -> D -> E
async fn build_chain(pool: &SqlitePool) {
    for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e")] {
        insert_node(pool, "t1", id, name, "function").await;
    }
    insert_edge(pool, "t1", "a", "b", "CALLS").await;
    insert_edge(pool, "t1", "b", "c", "CALLS").await;
    insert_edge(pool, "t1", "c", "d", "CALLS").await;
    insert_edge(pool, "t1", "d", "e", "CALLS").await;
}

/// Build two clusters: {A,B,C} densely connected, {D,E,F} densely connected,
/// with one bridge B->D.
async fn build_two_clusters(pool: &SqlitePool) {
    for (id, name) in &[
        ("a", "a"),
        ("b", "b"),
        ("c", "c"),
        ("d", "d"),
        ("e", "e"),
        ("f", "f"),
    ] {
        insert_node(pool, "t1", id, name, "function").await;
    }
    // Cluster 1: a-b-c fully connected
    insert_edge(pool, "t1", "a", "b", "CALLS").await;
    insert_edge(pool, "t1", "b", "a", "CALLS").await;
    insert_edge(pool, "t1", "a", "c", "CALLS").await;
    insert_edge(pool, "t1", "c", "a", "CALLS").await;
    insert_edge(pool, "t1", "b", "c", "CALLS").await;
    insert_edge(pool, "t1", "c", "b", "CALLS").await;
    // Cluster 2: d-e-f fully connected
    insert_edge(pool, "t1", "d", "e", "CALLS").await;
    insert_edge(pool, "t1", "e", "d", "CALLS").await;
    insert_edge(pool, "t1", "d", "f", "CALLS").await;
    insert_edge(pool, "t1", "f", "d", "CALLS").await;
    insert_edge(pool, "t1", "e", "f", "CALLS").await;
    insert_edge(pool, "t1", "f", "e", "CALLS").await;
    // Bridge: b -> d
    insert_edge(pool, "t1", "b", "d", "CALLS").await;
}

// ─── PageRank tests ──────────────────────────────────────────────

#[tokio::test]
async fn test_pagerank_empty_graph() {
    let pool = setup_graph_pool().await;
    let config = PageRankConfig::default();
    let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_pagerank_single_node() {
    let pool = setup_graph_pool().await;
    insert_node(&pool, "t1", "a", "alpha", "function").await;

    let config = PageRankConfig::default();
    let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!((results[0].score - 1.0).abs() < 0.01); // single node gets all rank
}

#[tokio::test]
async fn test_pagerank_diamond() {
    let pool = setup_graph_pool().await;
    build_diamond(&pool).await;

    let config = PageRankConfig::default();
    let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
    assert_eq!(results.len(), 4);

    // Node D should have highest PageRank (two incoming edges)
    let d_score = results.iter().find(|r| r.node_id == "d").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;
    assert!(
        d_score > a_score,
        "D (sink with 2 inputs) should rank higher than A (source): d={}, a={}",
        d_score,
        a_score
    );
}

#[tokio::test]
async fn test_pagerank_chain() {
    let pool = setup_graph_pool().await;
    build_chain(&pool).await;

    let config = PageRankConfig::default();
    let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
    assert_eq!(results.len(), 5);

    // All scores should sum to approximately 1.0
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "PageRank scores should sum to ~1.0, got {}",
        total
    );
}

#[tokio::test]
async fn test_pagerank_convergence() {
    let pool = setup_graph_pool().await;
    build_diamond(&pool).await;

    let config = PageRankConfig {
        damping: 0.85,
        max_iterations: 1000,
        tolerance: 1e-10,
        ..Default::default()
    };
    let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();

    // Should converge to stable values
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!((total - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_pagerank_edge_type_filter() {
    let pool = setup_graph_pool().await;
    // A -CALLS-> B, A -IMPORTS-> C
    insert_node(&pool, "t1", "a", "a", "function").await;
    insert_node(&pool, "t1", "b", "b", "function").await;
    insert_node(&pool, "t1", "c", "c", "function").await;
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;
    insert_edge(&pool, "t1", "a", "c", "IMPORTS").await;

    let config = PageRankConfig::default();
    let results = compute_pagerank(&pool, "t1", &config, Some(&["CALLS"]))
        .await
        .unwrap();

    // C should have low PageRank since IMPORTS edges are excluded
    let b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
    let c_score = results.iter().find(|r| r.node_id == "c").unwrap().score;
    assert!(
        b_score > c_score,
        "B should rank higher when only CALLS are considered"
    );
}

// ─── Community detection tests ───────────────────────────────────

#[tokio::test]
async fn test_communities_empty() {
    let pool = setup_graph_pool().await;
    let config = CommunityConfig::default();
    let communities = detect_communities(&pool, "t1", &config, None)
        .await
        .unwrap();
    assert!(communities.is_empty());
}

#[tokio::test]
async fn test_communities_two_disconnected_clusters() {
    let pool = setup_graph_pool().await;

    // Two disconnected clusters: {a,b,c} and {d,e,f}
    for (id, name) in &[
        ("a", "a"),
        ("b", "b"),
        ("c", "c"),
        ("d", "d"),
        ("e", "e"),
        ("f", "f"),
    ] {
        insert_node(&pool, "t1", id, name, "function").await;
    }
    // Cluster 1
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;
    insert_edge(&pool, "t1", "b", "c", "CALLS").await;
    insert_edge(&pool, "t1", "c", "a", "CALLS").await;
    // Cluster 2
    insert_edge(&pool, "t1", "d", "e", "CALLS").await;
    insert_edge(&pool, "t1", "e", "f", "CALLS").await;
    insert_edge(&pool, "t1", "f", "d", "CALLS").await;

    let config = CommunityConfig {
        max_iterations: 100,
        min_community_size: 2,
    };
    let communities = detect_communities(&pool, "t1", &config, None)
        .await
        .unwrap();

    // Should detect exactly 2 communities
    assert_eq!(
        communities.len(),
        2,
        "Expected 2 disconnected communities, got {}",
        communities.len()
    );

    // Each community should have 3 members
    assert_eq!(communities[0].members.len(), 3);
    assert_eq!(communities[1].members.len(), 3);
}

#[tokio::test]
async fn test_communities_fully_connected() {
    let pool = setup_graph_pool().await;

    // All nodes connected → one community
    for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c")] {
        insert_node(&pool, "t1", id, name, "function").await;
    }
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;
    insert_edge(&pool, "t1", "b", "c", "CALLS").await;
    insert_edge(&pool, "t1", "c", "a", "CALLS").await;

    let config = CommunityConfig::default();
    let communities = detect_communities(&pool, "t1", &config, None)
        .await
        .unwrap();

    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].members.len(), 3);
}

#[tokio::test]
async fn test_communities_min_size_filter() {
    let pool = setup_graph_pool().await;

    // Two nodes connected, one isolated
    insert_node(&pool, "t1", "a", "a", "function").await;
    insert_node(&pool, "t1", "b", "b", "function").await;
    insert_node(&pool, "t1", "c", "c", "function").await;
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;

    let config = CommunityConfig {
        min_community_size: 2,
        ..Default::default()
    };
    let communities = detect_communities(&pool, "t1", &config, None)
        .await
        .unwrap();

    // Only the {a, b} community should pass the filter
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].members.len(), 2);
}

#[tokio::test]
async fn test_communities_sorted_by_size() {
    let pool = setup_graph_pool().await;
    build_two_clusters(&pool).await;

    // Add extra node to cluster 1 to make it bigger
    insert_node(&pool, "t1", "g", "g", "function").await;
    insert_edge(&pool, "t1", "a", "g", "CALLS").await;
    insert_edge(&pool, "t1", "g", "a", "CALLS").await;

    let config = CommunityConfig::default();
    let communities = detect_communities(&pool, "t1", &config, None)
        .await
        .unwrap();

    if communities.len() >= 2 {
        assert!(
            communities[0].members.len() >= communities[1].members.len(),
            "Communities should be sorted by size descending"
        );
    }
}

// ─── Betweenness centrality tests ────────────────────────────────

#[tokio::test]
async fn test_betweenness_empty() {
    let pool = setup_graph_pool().await;
    let results = compute_betweenness_centrality(&pool, "t1", None, None)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_betweenness_chain() {
    let pool = setup_graph_pool().await;
    build_chain(&pool).await;

    let results = compute_betweenness_centrality(&pool, "t1", None, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 5);

    // Middle nodes (b, c, d) should have higher betweenness than endpoints
    let _b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
    let c_score = results.iter().find(|r| r.node_id == "c").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;
    let e_score = results.iter().find(|r| r.node_id == "e").unwrap().score;

    assert!(
        c_score >= a_score,
        "Center node c should have >= betweenness than endpoint a: c={}, a={}",
        c_score,
        a_score
    );
    assert!(
        c_score >= e_score,
        "Center node c should have >= betweenness than endpoint e: c={}, e={}",
        c_score,
        e_score
    );
}

#[tokio::test]
async fn test_betweenness_bridge_node() {
    let pool = setup_graph_pool().await;
    build_two_clusters(&pool).await;

    let results = compute_betweenness_centrality(&pool, "t1", None, None)
        .await
        .unwrap();

    // Bridge nodes (b and d) should have highest betweenness
    let b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
    let d_score = results.iter().find(|r| r.node_id == "d").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;

    // b connects the two clusters, so it should have high betweenness
    assert!(
        b_score > a_score || d_score > a_score,
        "Bridge nodes should have higher betweenness: b={}, d={}, a={}",
        b_score,
        d_score,
        a_score
    );
}

#[tokio::test]
async fn test_betweenness_small_graph() {
    let pool = setup_graph_pool().await;

    // Two nodes, one edge
    insert_node(&pool, "t1", "a", "a", "function").await;
    insert_node(&pool, "t1", "b", "b", "function").await;
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;

    let results = compute_betweenness_centrality(&pool, "t1", None, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    // With only 2 nodes, betweenness is 0 for both
    assert!(results.iter().all(|r| r.score == 0.0));
}

#[tokio::test]
async fn test_betweenness_with_sampling() {
    let pool = setup_graph_pool().await;
    build_chain(&pool).await;

    // Sample only 2 source nodes
    let results = compute_betweenness_centrality(&pool, "t1", None, Some(2))
        .await
        .unwrap();
    assert_eq!(results.len(), 5);
}

// ─── Load adjacency ──────────────────────────────────────────────

#[tokio::test]
async fn test_load_adjacency() {
    let pool = setup_graph_pool().await;
    build_diamond(&pool).await;

    let graph = load_adjacency_graph(&pool, "t1", None).await.unwrap();
    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.outgoing.get("a").unwrap().len(), 2); // a -> b, a -> c
    assert_eq!(graph.incoming.get("d").unwrap().len(), 2); // b -> d, c -> d
}

#[tokio::test]
async fn test_load_adjacency_filtered() {
    let pool = setup_graph_pool().await;

    insert_node(&pool, "t1", "a", "a", "function").await;
    insert_node(&pool, "t1", "b", "b", "function").await;
    insert_edge(&pool, "t1", "a", "b", "CALLS").await;
    insert_edge(&pool, "t1", "a", "b", "IMPORTS").await;

    // Filter to CALLS only
    let graph = load_adjacency_graph(&pool, "t1", Some(&["CALLS"]))
        .await
        .unwrap();
    let out = graph.outgoing.get("a").unwrap();
    assert_eq!(out.len(), 1); // only the CALLS edge
}
