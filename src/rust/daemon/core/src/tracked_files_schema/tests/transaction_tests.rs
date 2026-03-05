use super::super::*;
use super::{create_test_pool, setup_tables};

#[tokio::test]
async fn test_insert_tracked_file_tx_commit() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx,
        "w1",
        "src/tx_test.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "txhash1",
        2,
        Some("tree_sitter"),
        ProcessingStatus::Done,
        ProcessingStatus::Done,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .expect("Tx insert failed");
    tx.commit().await.unwrap();

    assert!(file_id > 0);
    let found = lookup_tracked_file(&pool, "w1", "src/tx_test.rs", Some("main"))
        .await
        .unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().file_hash, "txhash1");
}

#[tokio::test]
async fn test_insert_tracked_file_tx_rollback() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    {
        let mut tx = pool.begin().await.unwrap();
        let _file_id = insert_tracked_file_tx(
            &mut tx,
            "w1",
            "src/rollback.rs",
            Some("main"),
            Some("code"),
            Some("rust"),
            "2025-01-01T00:00:00Z",
            "rollback_hash",
            1,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            None,
            None,
            false,
            None,
            None,
            None,
        )
        .await
        .expect("Tx insert failed");
        // Drop tx without committing = implicit rollback
    }

    let found = lookup_tracked_file(&pool, "w1", "src/rollback.rs", Some("main"))
        .await
        .unwrap();
    assert!(found.is_none(), "Rolled-back insert should not be visible");
}

#[tokio::test]
async fn test_transaction_atomicity_insert_and_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx,
        "w1",
        "src/atomic.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "atomic_hash",
        2,
        Some("tree_sitter"),
        ProcessingStatus::Done,
        ProcessingStatus::Done,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let chunks = vec![
        (
            "pt-1".to_string(),
            0,
            "ch1".to_string(),
            Some(ChunkType::Function),
            Some("main".to_string()),
            Some(1),
            Some(20),
        ),
        (
            "pt-2".to_string(),
            1,
            "ch2".to_string(),
            Some(ChunkType::Struct),
            Some("Config".to_string()),
            Some(22),
            Some(40),
        ),
    ];
    insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/atomic.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(found.chunk_count, 2);
    let point_ids = get_chunk_point_ids(&pool, found.file_id).await.unwrap();
    assert_eq!(point_ids.len(), 2);
}

#[tokio::test]
async fn test_transaction_atomicity_rollback_both() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool,
        "w1",
        "src/base.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "base_hash",
        0,
        None,
        ProcessingStatus::None,
        ProcessingStatus::None,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    {
        let mut tx = pool.begin().await.unwrap();
        update_tracked_file_tx(
            &mut tx,
            file_id,
            "2025-02-01T00:00:00Z",
            "new_hash",
            3,
            Some("tree_sitter"),
            ProcessingStatus::Done,
            ProcessingStatus::Done,
            None,
            None,
        )
        .await
        .unwrap();

        let chunks = vec![(
            "p1".to_string(),
            0,
            "c1".to_string(),
            None,
            None,
            None,
            None,
        )];
        insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
            .await
            .unwrap();
        // Drop tx = rollback
    }

    let found = lookup_tracked_file(&pool, "w1", "src/base.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        found.file_hash, "base_hash",
        "Hash should not have changed after rollback"
    );
    assert_eq!(
        found.chunk_count, 0,
        "Chunk count should not have changed after rollback"
    );

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(point_ids.len(), 0, "No chunks should exist after rollback");
}

#[tokio::test]
async fn test_delete_tracked_file_tx() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool,
        "w1",
        "src/delete_tx.rs",
        Some("main"),
        None,
        None,
        "2025-01-01T00:00:00Z",
        "h1",
        1,
        None,
        ProcessingStatus::None,
        ProcessingStatus::None,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let chunks = vec![(
        "p1".to_string(),
        0,
        "c1".to_string(),
        None,
        None,
        None,
        None,
    )];
    insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

    let mut tx = pool.begin().await.unwrap();
    delete_tracked_file_tx(&mut tx, file_id).await.unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/delete_tx.rs", Some("main"))
        .await
        .unwrap();
    assert!(found.is_none(), "File should be deleted");

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(point_ids.len(), 0, "Chunks should be deleted via CASCADE");
}

#[tokio::test]
async fn test_mark_and_query_needs_reconcile() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool,
        "w1",
        "src/reconcile.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "hash1",
        3,
        Some("tree_sitter"),
        ProcessingStatus::Done,
        ProcessingStatus::Done,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 0);

    mark_needs_reconcile(&pool, file_id, "test_reason: sqlite_commit_failed")
        .await
        .unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 1);
    assert_eq!(reconcile_files[0].file_id, file_id);
    assert!(reconcile_files[0].needs_reconcile);
    assert_eq!(
        reconcile_files[0].reconcile_reason.as_deref(),
        Some("test_reason: sqlite_commit_failed")
    );

    let mut tx = pool.begin().await.unwrap();
    clear_reconcile_flag_tx(&mut tx, file_id).await.unwrap();
    tx.commit().await.unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 0, "Flag should be cleared");

    let found = lookup_tracked_file(&pool, "w1", "src/reconcile.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert!(!found.needs_reconcile);
    assert!(found.reconcile_reason.is_none());
}

#[tokio::test]
async fn test_update_tracked_file_tx_clears_reconcile_flag() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool,
        "w1",
        "src/reconcile_clear.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "hash1",
        1,
        None,
        ProcessingStatus::None,
        ProcessingStatus::None,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    mark_needs_reconcile(&pool, file_id, "test_failure")
        .await
        .unwrap();

    let mut tx = pool.begin().await.unwrap();
    update_tracked_file_tx(
        &mut tx,
        file_id,
        "2025-02-01T00:00:00Z",
        "hash2",
        5,
        Some("tree_sitter"),
        ProcessingStatus::Done,
        ProcessingStatus::Done,
        None,
        None,
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/reconcile_clear.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert!(
        !found.needs_reconcile,
        "Update should clear needs_reconcile"
    );
    assert!(
        found.reconcile_reason.is_none(),
        "Update should clear reconcile_reason"
    );
}
