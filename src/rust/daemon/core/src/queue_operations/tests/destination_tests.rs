//! Reference counting, decision, and per-destination status tests (Task 5/6).

use super::*;

#[tokio::test]
async fn test_has_other_references_single_watch() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_refcount.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Set up watch_folders and tracked_files schemas
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    use crate::tracked_files_schema::CREATE_TRACKED_FILES_SQL;
    sqlx::query(CREATE_TRACKED_FILES_SQL)
        .execute(&pool)
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());

    // Insert a watch folder and a tracked file with base_point
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/project1', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_abc123', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // Single watch folder -- no other references
    let has_refs = manager
        .has_other_references("bp_abc123", "w1")
        .await
        .unwrap();
    assert!(
        !has_refs,
        "Single watch folder should have no other references"
    );
}

#[tokio::test]
async fn test_has_other_references_two_watches() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_refcount2.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    use crate::tracked_files_schema::CREATE_TRACKED_FILES_SQL;
    sqlx::query(CREATE_TRACKED_FILES_SQL)
        .execute(&pool)
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());

    // Insert two watch folders
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/clone1', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w2', '/tmp/clone2', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // Both reference the same base_point (same file version in two clones)
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_shared', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w2', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_shared', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // w1 should see w2 as another reference
    let has_refs = manager
        .has_other_references("bp_shared", "w1")
        .await
        .unwrap();
    assert!(has_refs, "Should detect w2 as another reference");

    // w2 should see w1 as another reference
    let has_refs2 = manager
        .has_other_references("bp_shared", "w2")
        .await
        .unwrap();
    assert!(has_refs2, "Should detect w1 as another reference");

    // Now delete w2's tracked file -- w1 should have no more references
    sqlx::query("DELETE FROM tracked_files WHERE watch_folder_id = 'w2'")
        .execute(&pool)
        .await
        .unwrap();

    let has_refs3 = manager
        .has_other_references("bp_shared", "w1")
        .await
        .unwrap();
    assert!(
        !has_refs3,
        "After removing w2's file, w1 should have no other references"
    );
}

#[tokio::test]
async fn test_store_queue_decision() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_decision.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue an item
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Update,
            "t1",
            "projects",
            r#"{"file_path":"/test/file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Store a decision
    let decision = wqm_common::queue_types::QueueDecision {
        delete_old: true,
        old_base_point: Some("bp_old_123".to_string()),
        new_base_point: "bp_new_456".to_string(),
        old_file_hash: Some("hash_old".to_string()),
        new_file_hash: "hash_new".to_string(),
    };

    manager
        .store_queue_decision(&queue_id, &decision)
        .await
        .unwrap();

    // Verify stored correctly
    let stored_json: Option<String> =
        sqlx::query_scalar("SELECT decision_json FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(&pool)
            .await
            .unwrap();

    assert!(stored_json.is_some(), "decision_json should be stored");
    let stored: wqm_common::queue_types::QueueDecision =
        serde_json::from_str(stored_json.as_ref().unwrap()).unwrap();
    assert!(stored.delete_old);
    assert_eq!(stored.old_base_point.as_deref(), Some("bp_old_123"));
    assert_eq!(stored.new_base_point, "bp_new_456");

    // Verify statuses set to pending
    let qdrant_status: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(qdrant_status, "pending");
}

#[tokio::test]
async fn test_update_destination_status() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_dest_status.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "t1",
            "projects",
            r#"{"file_path":"/test/file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Update qdrant status to done
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();

    let qdrant_status: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(qdrant_status, "done");

    // Update search status to failed
    manager
        .update_destination_status(&queue_id, "search", DestinationStatus::Failed)
        .await
        .unwrap();

    let search_status: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(search_status, "failed");

    // Invalid destination should error
    let err = manager
        .update_destination_status(&queue_id, "invalid", DestinationStatus::Done)
        .await;
    assert!(err.is_err(), "Invalid destination should return error");
}

#[tokio::test]
async fn test_check_and_finalize_both_done() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_both.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "fin-tenant",
            "projects",
            r#"{"file_path":"/test/fin.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Initially both pending -- should be InProgress
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::InProgress);

    // Mark qdrant done
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::InProgress);

    // Mark search done -- both done -> overall Done
    manager
        .update_destination_status(&queue_id, "search", DestinationStatus::Done)
        .await
        .unwrap();
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Done);

    // Verify overall status was updated in DB
    let row: (String,) = sqlx::query_as("SELECT status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row.0, "done");
}

#[tokio::test]
async fn test_check_and_finalize_partial_failure() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_fail.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "fin-tenant2",
            "projects",
            r#"{"file_path":"/test/fail.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Qdrant succeeds, search fails
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();
    manager
        .update_destination_status(&queue_id, "search", DestinationStatus::Failed)
        .await
        .unwrap();

    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Failed);

    // Verify overall status was updated in DB
    let row: (String,) = sqlx::query_as("SELECT status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row.0, "failed");
}

#[tokio::test]
async fn test_check_and_finalize_nonexistent_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_missing.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let err = manager.check_and_finalize("nonexistent-id").await;
    assert!(err.is_err());
}

#[tokio::test]
async fn test_mark_explicit_destination_results_orchestration_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_pending.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Scan,
            "resolve-tenant",
            "projects",
            r#"{"project_root":"/test"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Both statuses start as pending -- ensure_destinations_resolved should set both to done
    manager
        .mark_explicit_destination_results(&queue_id)
        .await
        .unwrap();

    let qs: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    let ss: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(qs, "done");
    assert_eq!(ss, "done");

    // check_and_finalize should now return Done
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Done);
}

#[tokio::test]
async fn test_mark_explicit_destination_results_preserves_failed_state_machine() {
    // F-010/F-056: items that registered a QueueDecision (state-machine
    // semantics) must NOT have pending sinks flipped to done. Only explicitly
    // marked sinks count; pending sinks remain pending so the queue processor
    // keeps the item for the next lease cycle.
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_preserve.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "resolve-tenant2",
            "projects",
            r#"{"file_path":"/test/file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Mark this item as state-machine driven by storing a decision.
    let decision = wqm_common::queue_types::QueueDecision {
        delete_old: false,
        old_base_point: None,
        new_base_point: "bp_new".to_string(),
        old_file_hash: None,
        new_file_hash: "h_new".to_string(),
    };
    manager
        .store_queue_decision(&queue_id, &decision)
        .await
        .unwrap();

    // Explicitly set qdrant to failed, leave search as pending
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Failed)
        .await
        .unwrap();

    // mark_explicit_destination_results must preserve both: failed qdrant AND
    // pending search (state-machine items don't auto-resolve pending).
    manager
        .mark_explicit_destination_results(&queue_id)
        .await
        .unwrap();

    let qs: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    let ss: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(qs, "failed", "Failed status must be preserved");
    assert_eq!(
        ss, "pending",
        "State-machine pending sink must stay pending (F-010)"
    );

    // check_and_finalize should now return Failed (qdrant is failed)
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Failed);
}

#[tokio::test]
async fn test_mark_explicit_destination_results_state_machine_pending_stays_pending() {
    // F-010 regression: a state-machine handler that only wrote to qdrant
    // (search left pending) must NOT have search flipped to done.
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_pending_state.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Update,
            "tenant-sm",
            "projects",
            r#"{"file_path":"/test/sm.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    let decision = wqm_common::queue_types::QueueDecision {
        delete_old: false,
        old_base_point: None,
        new_base_point: "bp_sm".to_string(),
        old_file_hash: None,
        new_file_hash: "h_sm".to_string(),
    };
    manager
        .store_queue_decision(&queue_id, &decision)
        .await
        .unwrap();

    // Handler only wrote qdrant; never touched search.
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();

    manager
        .mark_explicit_destination_results(&queue_id)
        .await
        .unwrap();

    let ss: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(
        ss, "pending",
        "Untouched search sink must stay pending; helper must not flip it to done"
    );

    // check_and_finalize must NOT mark the row done — overall stays in_progress.
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::InProgress);
}

#[tokio::test]
async fn test_mark_explicit_destination_results_preserves_done() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_done.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "resolve-tenant3",
            "projects",
            r#"{"file_path":"/test/done.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Explicitly set both to done
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();
    manager
        .update_destination_status(&queue_id, "search", DestinationStatus::Done)
        .await
        .unwrap();

    // ensure_destinations_resolved should be a no-op
    manager
        .mark_explicit_destination_results(&queue_id)
        .await
        .unwrap();

    let qs: String =
        sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    let ss: String =
        sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(&queue_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(qs, "done");
    assert_eq!(ss, "done");
}

/// F-033/F-034 regression: a handler that returns Ok but reported a sink
/// failure must end up with retry metadata (error_message, retry_count,
/// last_error_at, lease_until/backoff) — not silently stuck on `failed`
/// status with no schedule for retry.
#[tokio::test]
async fn test_destination_failure_on_success_path_records_retry_metadata() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("dest_failure_metadata.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "fts-fail-tenant",
            "projects",
            r#"{"file_path":"/test/fts_fail.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Simulate the file-ingest pipeline: Qdrant upsert succeeded, FTS5 failed.
    manager
        .update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();
    manager
        .update_destination_status(&queue_id, "search", DestinationStatus::Failed)
        .await
        .unwrap();

    // handler returned Ok → batch_processing runs:
    //   mark_explicit_destination_results (no-op here, both explicit)
    //   check_and_finalize (Failed because search_status = failed)
    //   mark_unified_failed (must populate retry metadata)
    manager
        .mark_explicit_destination_results(&queue_id)
        .await
        .unwrap();
    let overall = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(overall, QueueStatus::Failed);

    let err_msg = "fts5 index write failed: disk full";
    manager
        .mark_unified_failed(&queue_id, err_msg, false, 3)
        .await
        .unwrap();

    // Verify retry metadata is populated.
    let row: (Option<String>, i32, Option<String>, Option<String>, String) = sqlx::query_as(
        r#"SELECT error_message, retry_count, last_error_at, lease_until, status
           FROM unified_queue WHERE queue_id = ?1"#,
    )
    .bind(&queue_id)
    .fetch_one(manager.pool())
    .await
    .unwrap();
    let (got_err, retry_count, last_error_at, lease_until, status) = row;
    assert_eq!(
        got_err.as_deref(),
        Some(err_msg),
        "error_message must be populated (F-033/F-034)"
    );
    assert_eq!(retry_count, 1, "retry_count must increment to 1");
    assert!(
        last_error_at.is_some(),
        "last_error_at must be populated for the retry schedule"
    );
    assert!(
        lease_until.is_some(),
        "lease_until must be set so the next lease cycle picks up the retry"
    );
    assert_eq!(
        status, "pending",
        "Item must be re-set to pending for retry (not stuck on failed)"
    );
}
