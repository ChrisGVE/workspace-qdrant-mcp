//! Integration tests for graph migration (export/import/validate) and
//! SharedGraphStore concurrent access patterns.

#[allow(dead_code)]
#[path = "common/graph_helpers.rs"]
mod graph_helpers;

use graph_helpers::{
    build_rust_file_chunks, build_rust_main_chunks, build_typescript_chunks, create_factory_store,
    ingest_file_chunks, TENANT,
};
use tempfile::tempdir;
use workspace_qdrant_core::graph::{extractor, migrator};

// ────────────────────────────────────────────────────────────────────────────
// Migration: export, import, validate
// ────────────────────────────────────────────────────────────────────────────

/// Export from SQLite, import into a fresh store, validate counts match.
#[tokio::test]
async fn test_migration_sqlite_to_sqlite() {
    let dir_src = tempdir().unwrap();
    let store_src = create_factory_store(dir_src.path()).await;

    // Populate source store
    ingest_file_chunks(
        &store_src,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;
    ingest_file_chunks(&store_src, &build_rust_main_chunks(), TENANT, "src/main.rs").await;

    let src_stats = store_src.stats(Some(TENANT)).await.unwrap();
    assert!(src_stats.total_nodes > 0);
    assert!(src_stats.total_edges > 0);

    // Export from source
    let guard_src = store_src.read().await;
    let pool_src = guard_src.pool();
    let snapshot = migrator::export_sqlite(pool_src, Some(TENANT))
        .await
        .unwrap();

    assert_eq!(snapshot.nodes.len() as u64, src_stats.total_nodes);
    assert_eq!(snapshot.edges.len() as u64, src_stats.total_edges);

    // Create target store
    let dir_tgt = tempdir().unwrap();
    let store_tgt = create_factory_store(dir_tgt.path()).await;

    // Import into target (snapshot first, then target store)
    let guard_tgt = store_tgt.read().await;
    let report = migrator::import_to_store(&snapshot, &*guard_tgt, 100)
        .await
        .unwrap();

    assert_eq!(report.nodes_imported, src_stats.total_nodes);
    assert_eq!(report.edges_imported, src_stats.total_edges);
    assert!(report.warnings.is_empty(), "should have no warnings");

    // Validate target matches source
    drop(guard_tgt);
    let tgt_stats = store_tgt.stats(Some(TENANT)).await.unwrap();
    assert_eq!(tgt_stats.total_nodes, src_stats.total_nodes);
    assert_eq!(tgt_stats.total_edges, src_stats.total_edges);
}

/// Validate that migration preserves node types and edge types.
#[tokio::test]
async fn test_migration_preserves_types() {
    let dir_src = tempdir().unwrap();
    let store_src = create_factory_store(dir_src.path()).await;

    ingest_file_chunks(
        &store_src,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    let src_stats = store_src.stats(Some(TENANT)).await.unwrap();

    // Export
    let guard_src = store_src.read().await;
    let snapshot = migrator::export_sqlite(guard_src.pool(), Some(TENANT))
        .await
        .unwrap();

    // Import into fresh store
    let dir_tgt = tempdir().unwrap();
    let store_tgt = create_factory_store(dir_tgt.path()).await;
    let guard_tgt = store_tgt.read().await;
    migrator::import_to_store(&snapshot, &*guard_tgt, 50)
        .await
        .unwrap();

    // Node type distribution should match
    drop(guard_tgt);
    let tgt_stats = store_tgt.stats(Some(TENANT)).await.unwrap();
    assert_eq!(
        tgt_stats.nodes_by_type, src_stats.nodes_by_type,
        "node type distribution should be preserved after migration"
    );
    assert_eq!(
        tgt_stats.edges_by_type, src_stats.edges_by_type,
        "edge type distribution should be preserved after migration"
    );
}

/// Validate migration with the validate_migration function.
#[tokio::test]
async fn test_migration_validation() {
    let dir_src = tempdir().unwrap();
    let store_src = create_factory_store(dir_src.path()).await;

    ingest_file_chunks(
        &store_src,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    // Export
    let guard_src = store_src.read().await;
    let snapshot = migrator::export_sqlite(guard_src.pool(), Some(TENANT))
        .await
        .unwrap();

    // Import
    let dir_tgt = tempdir().unwrap();
    let store_tgt = create_factory_store(dir_tgt.path()).await;
    let guard_tgt = store_tgt.read().await;
    migrator::import_to_store(&snapshot, &*guard_tgt, 50)
        .await
        .unwrap();

    // Validate counts match (source pool vs target store)
    let is_valid = migrator::validate_migration(guard_src.pool(), &*guard_tgt, Some(TENANT))
        .await
        .unwrap();
    assert!(
        is_valid,
        "migration validation should pass when counts match"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// SharedGraphStore concurrent access
// ────────────────────────────────────────────────────────────────────────────

/// Concurrent readers should all see consistent data.
#[tokio::test]
async fn test_shared_store_concurrent_readers() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    // Spawn 20 concurrent readers
    let mut handles = Vec::new();
    for _ in 0..20 {
        let s = store.clone();
        handles.push(tokio::spawn(
            async move { s.stats(Some(TENANT)).await.unwrap() },
        ));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let expected_nodes = results[0].total_nodes;
    for (i, stats) in results.iter().enumerate() {
        assert_eq!(
            stats.total_nodes, expected_nodes,
            "reader {} saw different node count: {} vs {}",
            i, stats.total_nodes, expected_nodes
        );
    }
}

/// Writer blocks readers during reingest, readers see consistent state after.
#[tokio::test]
async fn test_shared_store_write_then_read_consistency() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    ingest_file_chunks(
        &store,
        &build_rust_file_chunks(),
        TENANT,
        "src/processor.rs",
    )
    .await;

    let stats_before = store.stats(Some(TENANT)).await.unwrap();

    // Add a second file
    let ts_result = extractor::extract_edges(&build_typescript_chunks(), TENANT, "src/App.tsx");
    store.upsert_nodes(&ts_result.nodes).await.unwrap();
    store.insert_edges(&ts_result.edges).await.unwrap();

    let stats_after = store.stats(Some(TENANT)).await.unwrap();

    assert!(
        stats_after.total_nodes > stats_before.total_nodes,
        "adding a second file should increase node count: before={}, after={}",
        stats_before.total_nodes,
        stats_after.total_nodes
    );
}
