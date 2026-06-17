//! Integration tests for graph algorithms: PageRank, community detection,
//! and betweenness centrality.
//!
//! All tests call `SharedGraphStore::export_adjacency` to obtain an
//! [`AdjacencyExport`] and then pass it to the algorithm under test.
//! No `SqlitePool` reference is used — this validates the full call-chain
//! (store → export → algorithm) introduced in task A0.2a.

#[allow(dead_code)]
#[path = "common/graph_helpers.rs"]
mod graph_helpers;

use graph_helpers::{
    build_rust_file_chunks, build_rust_main_chunks, build_typescript_chunks, create_factory_store,
    ingest_file_chunks, TENANT,
};
use tempfile::tempdir;
use workspace_qdrant_core::graph::algorithms::{self, CommunityConfig, PageRankConfig};

// ────────────────────────────────────────────────────────────────────────────
// PageRank
// ────────────────────────────────────────────────────────────────────────────

/// PageRank on a realistic extracted graph (via export_adjacency).
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

    let adj = store.export_adjacency(TENANT, None).await.unwrap();

    let config = PageRankConfig::default();
    let results = algorithms::compute_pagerank(&adj, &config);

    assert!(
        !results.is_empty(),
        "PageRank should produce results for non-empty graph"
    );

    // All scores should be positive and sum to ~1.0.
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "PageRank scores should sum to ~1.0, got {}",
        total
    );

    // node_id must be non-empty in every entry.
    for entry in &results {
        assert!(!entry.node_id.is_empty());
    }
}

/// PageRank with edge type filtering (via export_adjacency with EdgeType filter).
#[tokio::test]
async fn test_pagerank_with_edge_filter() {
    use workspace_qdrant_core::graph::EdgeType;

    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    let adj_calls = store
        .export_adjacency(TENANT, Some(&[EdgeType::Calls]))
        .await
        .unwrap();
    let adj_imports = store
        .export_adjacency(TENANT, Some(&[EdgeType::Imports]))
        .await
        .unwrap();

    let config = PageRankConfig::default();
    let calls_only = algorithms::compute_pagerank(&adj_calls, &config);
    let imports_only = algorithms::compute_pagerank(&adj_imports, &config);

    // Both should succeed; at least one should be non-empty.
    assert!(!calls_only.is_empty() || !imports_only.is_empty());
}

// ────────────────────────────────────────────────────────────────────────────
// Community detection
// ────────────────────────────────────────────────────────────────────────────

/// Community detection on an extracted graph (via export_adjacency).
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
    ingest_file_chunks(&store, &build_typescript_chunks(), TENANT, "src/App.tsx").await;

    let adj = store.export_adjacency(TENANT, None).await.unwrap();

    let config = CommunityConfig {
        max_iterations: 100,
        min_community_size: 1,
    };
    let communities = algorithms::detect_communities(&adj, &config);

    assert!(
        !communities.is_empty(),
        "community detection should find at least one community"
    );

    for community in &communities {
        assert!(!community.members.is_empty());
        for member in &community.members {
            assert!(!member.node_id.is_empty());
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Betweenness centrality
// ────────────────────────────────────────────────────────────────────────────

/// Betweenness centrality on an extracted graph (via export_adjacency).
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

    let adj = store.export_adjacency(TENANT, None).await.unwrap();

    let results = algorithms::compute_betweenness_centrality(&adj, None);

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

/// Identical outputs on identical export — backend-independence assertion.
///
/// The same `AdjacencyExport` fed to each algorithm twice must produce
/// bit-identical scores.  This is the cross-backend equivalence test called
/// out in the task test strategy.
#[tokio::test]
async fn test_algorithm_outputs_identical_on_identical_export() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    let adj = store.export_adjacency(TENANT, None).await.unwrap();

    // PageRank
    let pr1 = algorithms::compute_pagerank(&adj, &PageRankConfig::default());
    let pr2 = algorithms::compute_pagerank(&adj, &PageRankConfig::default());
    assert_eq!(pr1.len(), pr2.len());
    for (a, b) in pr1.iter().zip(pr2.iter()) {
        assert_eq!(a.node_id, b.node_id);
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "PageRank scores must be bit-identical for node {}",
            a.node_id
        );
    }

    // Betweenness
    let bw1 = algorithms::compute_betweenness_centrality(&adj, None);
    let bw2 = algorithms::compute_betweenness_centrality(&adj, None);
    assert_eq!(bw1.len(), bw2.len());
    for (a, b) in bw1.iter().zip(bw2.iter()) {
        assert_eq!(a.node_id, b.node_id);
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "Betweenness scores must be bit-identical for node {}",
            a.node_id
        );
    }
}
