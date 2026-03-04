//! Integration tests for graph algorithms: PageRank, community detection,
//! betweenness centrality, and edge type filtering in algorithms.

#[allow(dead_code)]
#[path = "common/graph_helpers.rs"]
mod graph_helpers;

use graph_helpers::{
    build_rust_file_chunks, build_rust_main_chunks, build_typescript_chunks,
    create_factory_store, ingest_file_chunks, TENANT,
};
use tempfile::tempdir;
use workspace_qdrant_core::graph::algorithms::{self, CommunityConfig, PageRankConfig};

// ────────────────────────────────────────────────────────────────────────────
// PageRank
// ────────────────────────────────────────────────────────────────────────────

/// PageRank on a realistic extracted graph.
#[tokio::test]
async fn test_pagerank_on_extracted_graph() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;
    ingest_file_chunks(&store, &build_rust_main_chunks(), TENANT, "src/main.rs").await;

    let guard = store.read().await;
    let pool = guard.pool();

    let config = PageRankConfig::default();
    let results = algorithms::compute_pagerank(pool, TENANT, &config, None)
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "PageRank should produce results for non-empty graph"
    );

    // All scores should be positive and sum to ~1.0
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "PageRank scores should sum to ~1.0, got {}",
        total
    );

    // Each entry should have valid metadata
    for entry in &results {
        assert!(!entry.node_id.is_empty());
        assert!(!entry.symbol_name.is_empty());
        assert!(!entry.symbol_type.is_empty());
    }
}

/// PageRank with edge type filtering.
#[tokio::test]
async fn test_pagerank_with_edge_filter() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    let guard = store.read().await;
    let pool = guard.pool();

    let config = PageRankConfig::default();

    // Only consider CALLS edges
    let calls_only = algorithms::compute_pagerank(pool, TENANT, &config, Some(&["CALLS"]))
        .await
        .unwrap();

    // Only consider IMPORTS edges
    let imports_only = algorithms::compute_pagerank(pool, TENANT, &config, Some(&["IMPORTS"]))
        .await
        .unwrap();

    // Both should succeed; results may differ in size/scores
    assert!(!calls_only.is_empty() || !imports_only.is_empty());
}

// ────────────────────────────────────────────────────────────────────────────
// Community detection
// ────────────────────────────────────────────────────────────────────────────

/// Community detection on an extracted graph.
#[tokio::test]
async fn test_community_detection_on_extracted_graph() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;
    ingest_file_chunks(
        &store,
        &build_typescript_chunks(),
        TENANT,
        "src/App.tsx",
    )
    .await;

    let guard = store.read().await;
    let pool = guard.pool();

    let config = CommunityConfig {
        max_iterations: 100,
        min_community_size: 1,
    };
    let communities = algorithms::detect_communities(pool, TENANT, &config, None)
        .await
        .unwrap();

    assert!(
        !communities.is_empty(),
        "community detection should find at least one community"
    );

    for community in &communities {
        assert!(!community.members.is_empty());
        for member in &community.members {
            assert!(!member.node_id.is_empty());
            assert!(!member.symbol_name.is_empty());
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Betweenness centrality
// ────────────────────────────────────────────────────────────────────────────

/// Betweenness centrality on an extracted graph.
#[tokio::test]
async fn test_betweenness_on_extracted_graph() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;
    ingest_file_chunks(&store, &build_rust_main_chunks(), TENANT, "src/main.rs").await;

    let guard = store.read().await;
    let pool = guard.pool();

    let results = algorithms::compute_betweenness_centrality(pool, TENANT, None, None)
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "betweenness should produce results for non-empty graph"
    );

    for entry in &results {
        assert!(
            entry.score >= 0.0,
            "betweenness score should be non-negative: {}",
            entry.score
        );
        assert!(!entry.node_id.is_empty());
    }
}
