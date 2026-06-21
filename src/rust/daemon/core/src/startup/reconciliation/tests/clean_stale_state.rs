//! Tests for `clean_stale_state`.

use sqlx::Row;

use crate::queue_operations::QueueManager;

use super::super::clean_stale_state;
use super::{create_test_pool, setup_schema};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a QueueManager backed by the given pool.
fn make_queue_manager(pool: &sqlx::SqlitePool) -> QueueManager {
    QueueManager::new(pool.clone())
}

// ---------------------------------------------------------------------------
// Existing tests — updated to pass QueueManager
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_clean_stale_state_resets_in_progress() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    // Insert a watch folder for tenant t1 (so items aren't purged as orphan tenant items)
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/test-reset-in-progress', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert two queue items with status='in_progress'
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, worker_id, lease_until) \
         VALUES ('q1', 'file', 'add', 't1', 'projects', 'in_progress', \
         'key1', 'worker-old', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, worker_id, lease_until) \
         VALUES ('q2', 'file', 'add', 't1', 'projects', 'in_progress', \
         'key2', 'worker-old', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert one pending item that should NOT be affected
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q3', 'file', 'add', 't1', 'projects', 'pending', 'key3')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(stats.items_reset, 2, "Should reset 2 in_progress items");

    // Verify the items are now pending with cleared lease fields
    let row = sqlx::query(
        "SELECT status, worker_id, lease_until FROM unified_queue WHERE queue_id = 'q1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    let status: String = row.get("status");
    let worker_id: Option<String> = row.get("worker_id");
    let lease_until: Option<String> = row.get("lease_until");

    assert_eq!(status, "pending");
    assert!(worker_id.is_none(), "worker_id should be cleared");
    assert!(lease_until.is_none(), "lease_until should be cleared");

    // Verify the pending item was not touched
    let pending_row = sqlx::query("SELECT status FROM unified_queue WHERE queue_id = 'q3'")
        .fetch_one(&pool)
        .await
        .unwrap();
    let pending_status: String = pending_row.get("status");
    assert_eq!(pending_status, "pending");
}

#[tokio::test]
async fn test_clean_stale_state_removes_old_done() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    // Insert an old done item (updated_at well in the past)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_old_done', 'file', 'add', 't1', 'projects', 'done', \
         'key_old_done', '2020-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert an old failed item (no matching tracked_files needs_reconcile=1, so purged)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_old_fail', 'file', 'add', 't1', 'projects', 'failed', \
         'key_old_fail', '2020-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a recent done item that should NOT be removed
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_recent_done', 'file', 'add', 't1', 'projects', 'done', 'key_recent')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.items_cleaned, 2,
        "Should clean 2 old done/failed items"
    );

    // Verify old items are gone
    let old_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_old_done', 'q_old_fail')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_count, 0, "Old items should be deleted");

    // Verify recent done item still exists
    let recent_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_recent_done'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(recent_count, 1, "Recent done item should not be deleted");
}

#[tokio::test]
async fn test_clean_stale_state_removes_orphan_chunks() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/nonexistent-test-dir-512', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('w1', 't1', 'main', 'fid-w1', 'ck-w1', 'file.rs', '2025-01-01T00:00:00Z', 'hash1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE relative_path = 'file.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();

    sqlx::query(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at) \
         VALUES (?1, 'point-valid', 0, 'chash1', '2025-01-01T00:00:00Z')",
    )
    .bind(file_id)
    .execute(&pool)
    .await
    .unwrap();

    // Delete tracked file without CASCADE to simulate orphan chunk.
    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&pool)
        .await
        .unwrap();

    let chunk_count_before: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        chunk_count_before, 1,
        "Orphan chunk should exist before cleanup"
    );

    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.orphan_chunks_removed, 1,
        "Should remove 1 orphan chunk"
    );

    let chunk_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(chunk_count_after, 0, "Orphan chunk should be removed");
}

#[tokio::test]
async fn test_clean_stale_state_empty_tables() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    assert!(!stats.has_changes(), "No changes expected on empty tables");
    assert_eq!(stats.items_reset, 0);
    assert_eq!(stats.items_cleaned, 0);
    assert_eq!(stats.orphan_tenant_items_removed, 0);
    assert_eq!(stats.tracked_files_removed, 0);
    assert_eq!(stats.orphan_chunks_removed, 0);
}

#[tokio::test]
async fn test_clean_stale_state_purges_orphan_tenant_items() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    // Insert a watch folder for tenant t1 (valid tenant)
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/test-orphan-tenant-512', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a failed queue item for t1 (has watch_folder → should NOT be removed)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_valid', 'file', 'add', 't1', 'projects', 'failed', 'key_valid')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert failed queue items for t_gone (NO watch_folder → should be removed)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_orphan1', 'file', 'add', 't_gone', 'projects', 'failed', \
         'key_orphan1', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_orphan2', 'file', 'add', 't_gone', 'projects', 'pending', \
         'key_orphan2', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert an in_progress item for t_gone (step 1 resets to pending, step 3 purges it)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_inprog', 'file', 'add', 't_gone', 'projects', 'in_progress', \
         'key_inprog', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a doc item for t_gone (should NOT be touched — only file items purged)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_doc', 'doc', 'add', 't_gone', 'projects', 'failed', 'key_doc')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.orphan_tenant_items_removed, 2,
        "Should remove 2 orphan tenant items (failed + pending file items for t_gone)"
    );

    let orphan_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_orphan1', 'q_orphan2')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(orphan_count, 0, "Orphan tenant items should be deleted");

    let valid_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_valid'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(valid_count, 1, "Valid tenant item should remain");

    // Step 1 resets q_inprog to pending; step 3 then purges it (orphan tenant).
    let inprog_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_inprog'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        inprog_count, 0,
        "in_progress item for orphan tenant should be reset then purged"
    );

    let doc_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_doc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        doc_count, 1,
        "Doc items should not be purged by orphan tenant cleanup"
    );
}

// ---------------------------------------------------------------------------
// F-036: enqueue Delete for missing tracked files, keep row while in flight
// ---------------------------------------------------------------------------

/// F-036: missing tracked file gets a Delete op enqueued; the tracked_files
/// row is kept while the Delete is in flight, and removed when the Delete
/// completes via `delete_unified_item` (which removes the tracking row as
/// part of the F-036 post-completion cleanup).
#[tokio::test]
async fn test_f036_enqueue_delete_keeps_row_while_in_flight() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         created_at, updated_at) \
         VALUES ('wf1', ?1, 'projects', 'tenant1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    // File that does NOT exist on disk.
    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('wf1', 't1', 'main', 'fid-wf1', 'ck-wf1', 'gone.rs', '2025-01-01T00:00:00Z', 'hash1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // --- Pass 1: run clean_stale_state ---
    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state pass 1 failed");

    // A Delete op must have been enqueued.
    assert_eq!(
        stats.deletes_enqueued, 1,
        "Should enqueue 1 Delete for the missing file"
    );

    // The tracked_files row must still exist (Delete is still pending).
    let tf_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE relative_path = 'gone.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        tf_count, 1,
        "tracked_files row must be kept while Delete is in flight"
    );

    // The Delete queue item must be pending.
    let queue_id: String = sqlx::query_scalar(
        "SELECT queue_id FROM unified_queue \
         WHERE op = 'delete' AND item_type = 'file' AND status = 'pending'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    // --- Simulate Delete completing via delete_unified_item (F-036). ---
    // This removes the queue row AND the tracked_files row.
    qm.delete_unified_item(&queue_id)
        .await
        .expect("delete_unified_item failed");

    // tracked_files row must now be gone (F-036 post-completion cleanup).
    let tf_count2: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE relative_path = 'gone.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        tf_count2, 0,
        "tracked_files row must be removed when Delete completes (F-036)"
    );
}

/// F-036: idempotent enqueue — a second pass does not double-enqueue when a
/// Delete is already pending for the same file.
#[tokio::test]
async fn test_f036_enqueue_delete_is_idempotent() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         created_at, updated_at) \
         VALUES ('wf1', ?1, 'projects', 'tenant1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('wf1', 't1', 'main', 'fid-wf1', 'ck-wf1', 'gone.rs', '2025-01-01T00:00:00Z', 'hash1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // First pass enqueues the Delete.
    let stats1 = clean_stale_state(&pool, &qm).await.expect("pass 1 failed");
    assert_eq!(stats1.deletes_enqueued, 1, "First pass should enqueue 1");

    let q_count_after_first: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE op = 'delete'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(q_count_after_first, 1, "Exactly one Delete item in queue");

    // Second pass: Delete is still pending, composite key dedup fires.
    let stats2 = clean_stale_state(&pool, &qm).await.expect("pass 2 failed");
    assert_eq!(
        stats2.deletes_enqueued, 0,
        "Second pass must not re-enqueue (idempotent)"
    );

    let q_count_after_second: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE op = 'delete'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(q_count_after_second, 1, "Still exactly one Delete item");
}

// ---------------------------------------------------------------------------
// F-020: needs_reconcile cleared only after queue item completes
// ---------------------------------------------------------------------------

/// F-020: needs_reconcile flag is NOT cleared immediately when reconcile
/// enqueues a file op — it stays set until the queue item is deleted
/// (simulating successful processing via delete_unified_item).
#[tokio::test]
async fn test_f020_needs_reconcile_cleared_on_queue_completion() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let base_path = tmp_dir.path().to_str().unwrap();
    let rel_path = "main.rs";
    std::fs::write(tmp_dir.path().join(rel_path), "fn main() {}").unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, \
         is_archived, created_at, updated_at) \
         VALUES ('wf-rc', ?1, 'projects', 'tenant-rc', 1, 0, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(base_path)
    .execute(&pool)
    .await
    .unwrap();

    let file_id: i64 = sqlx::query_scalar(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, chunk_count, collection, \
          needs_reconcile, reconcile_reason, created_at, updated_at) \
         VALUES ('wf-rc', 'tenant-rc', 'main', 'fid-rc', 'ck-rc', ?1, '2025-01-01T00:00:00Z', 'abc123', 3, 'projects', 1, \
         'test_reason', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z') \
         RETURNING file_id",
    )
    .bind(rel_path)
    .fetch_one(&pool)
    .await
    .unwrap();

    let qm = QueueManager::new(pool.clone());

    // Run reconcile_flagged_files via the recovery path.
    use crate::startup::recovery::FullRecoveryStats;
    // We call the internal function indirectly via run_startup_recovery, but
    // to keep the test focused we call through the module's public surface.
    // Use the queue manager to enqueue directly and verify the flag stays set.

    // Verify flag is set.
    let flag_before: i32 =
        sqlx::query_scalar("SELECT needs_reconcile FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(flag_before, 1, "needs_reconcile must be 1 before reconcile");

    // Trigger reconcile_flagged_files through the recovery public API.
    let mut stats = FullRecoveryStats::default();
    crate::startup::recovery::reconcile_flagged_files_for_test(&pool, &qm, &mut stats).await;

    assert_eq!(stats.reconciled, 1, "File should be reconciled");
    assert_eq!(stats.reconcile_errors, 0);

    // F-020: flag must still be 1 immediately after enqueue — not cleared yet.
    let flag_after_enqueue: i32 =
        sqlx::query_scalar("SELECT needs_reconcile FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        flag_after_enqueue, 1,
        "needs_reconcile must remain 1 immediately after enqueue (F-020)"
    );

    // Simulate queue item completing: find the queue_id and call delete_unified_item.
    let queue_id: String =
        sqlx::query_scalar("SELECT queue_id FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();

    qm.delete_unified_item(&queue_id)
        .await
        .expect("delete_unified_item failed");

    // F-020: NOW the flag must be 0.
    let flag_after_completion: i32 =
        sqlx::query_scalar("SELECT needs_reconcile FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        flag_after_completion, 0,
        "needs_reconcile must be cleared after queue item completes (F-020)"
    );
}

/// F-020 dedup: calling reconcile_flagged_files twice does NOT enqueue a
/// duplicate (composite key dedup) AND does NOT clear the flag.
#[tokio::test]
async fn test_f020_no_duplicate_enqueue_flag_stays_set() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let base_path = tmp_dir.path().to_str().unwrap();
    let rel_path = "lib.rs";
    std::fs::write(tmp_dir.path().join(rel_path), "pub fn lib() {}").unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, \
         is_archived, created_at, updated_at) \
         VALUES ('wf-dedup', ?1, 'projects', 'tenant-dedup', 1, 0, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(base_path)
    .execute(&pool)
    .await
    .unwrap();

    let file_id: i64 = sqlx::query_scalar(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, chunk_count, collection, \
          needs_reconcile, reconcile_reason, created_at, updated_at) \
         VALUES ('wf-dedup', 'tenant-dedup', 'main', 'fid-dedup', 'ck-dedup', ?1, '2025-01-01T00:00:00Z', 'abc123', 3, 'projects', 1, \
         'test_reason', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z') \
         RETURNING file_id",
    )
    .bind(rel_path)
    .fetch_one(&pool)
    .await
    .unwrap();

    let qm = QueueManager::new(pool.clone());
    use crate::startup::recovery::FullRecoveryStats;

    // First reconcile call — enqueues one item.
    let mut stats1 = FullRecoveryStats::default();
    crate::startup::recovery::reconcile_flagged_files_for_test(&pool, &qm, &mut stats1).await;
    assert_eq!(stats1.reconciled, 1);

    let q_count_after_first: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-dedup'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(q_count_after_first, 1, "One queue item after first call");

    // F-020: flag must still be set after first reconcile.
    let flag_after_first: i32 =
        sqlx::query_scalar("SELECT needs_reconcile FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        flag_after_first, 1,
        "Flag must remain set after first reconcile (F-020)"
    );

    // Second reconcile call — composite key dedup: no new item inserted.
    let mut stats2 = FullRecoveryStats::default();
    crate::startup::recovery::reconcile_flagged_files_for_test(&pool, &qm, &mut stats2).await;
    assert_eq!(stats2.reconciled, 1, "Item still counted as reconciled");

    let q_count_after_second: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-dedup'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        q_count_after_second, 1,
        "No duplicate queue item inserted (F-020 dedup)"
    );

    // F-020: flag must still be set — not cleared on dedup path.
    let flag_after_second: i32 =
        sqlx::query_scalar("SELECT needs_reconcile FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        flag_after_second, 1,
        "Flag must remain set after deduped second reconcile (F-020)"
    );
}

// ---------------------------------------------------------------------------
// F-043: failed rows preserved when they hold the sole repair intent
// ---------------------------------------------------------------------------

/// F-043: a failed row older than 7 days is NOT purged when the corresponding
/// tracked_files row has needs_reconcile=1 and no other pending/in-progress
/// queue items exist for the same file path.
#[tokio::test]
async fn test_f043_failed_row_preserved_as_sole_repair_intent() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();
    // Post-T7: unified_queue.file_path stores the relative path anchored
    // to watch_folders.path.
    let rel_file = "broken.rs";

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         created_at, updated_at) \
         VALUES ('wf-f043', ?1, 'projects', 'tenant-f043', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    // tracked_files row with needs_reconcile=1 (the file still "exists" for this test
    // — the exemption is about the queue row, not disk state).
    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, needs_reconcile, \
          reconcile_reason, created_at, updated_at) \
         VALUES ('wf-f043', 't-f043', 'main', 'fid-f043', 'ck-f043', 'broken.rs', '2025-01-01T00:00:00Z', 'hash1', 1, \
         'ingest_tx_failed', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Old failed queue item (>7 days) that is the sole repair record.
    // Note: max_retries column was dropped in migration v27; omit it here.
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, item_type, op, tenant_id, collection, status, idempotency_key, \
          file_path, retry_count, updated_at) \
         VALUES ('q-sole-fail', 'file', 'update', 'tenant-f043', 'projects', 'failed', \
         'key-sole-fail', ?1, 3, '2020-01-01T00:00:00Z')",
    )
    .bind(rel_file)
    .execute(&pool)
    .await
    .unwrap();

    // Run clean_stale_state — the failed row should be PRESERVED (F-043).
    let stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    // items_cleaned counts rows actually deleted; must be 0 for the exempt row.
    let preserved: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q-sole-fail'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        preserved, 1,
        "Failed row must be preserved when it is the sole repair intent (F-043)"
    );
    let _ = stats; // stats.items_cleaned may be 0 or more depending on other rows
}

/// F-043: the exemption is lifted once a newer pending queue item exists for
/// the same file path. The old failed row is then eligible for purge.
#[tokio::test]
async fn test_f043_failed_row_purged_when_newer_pending_exists() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;
    let qm = make_queue_manager(&pool);

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();
    // Post-T7: unified_queue.file_path stores the relative path anchored
    // to watch_folders.path.
    let rel_file = "broken.rs";

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         created_at, updated_at) \
         VALUES ('wf-f043b', ?1, 'projects', 'tenant-f043b', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
          relative_path, file_mtime, file_hash, needs_reconcile, \
          reconcile_reason, created_at, updated_at) \
         VALUES ('wf-f043b', 't-f043b', 'main', 'fid-f043b', 'ck-f043b', 'broken.rs', '2025-01-01T00:00:00Z', 'hash1', 1, \
         'ingest_tx_failed', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Old failed queue item (>7 days). max_retries dropped in migration v27.
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, item_type, op, tenant_id, collection, status, idempotency_key, \
          file_path, retry_count, updated_at) \
         VALUES ('q-old-fail', 'file', 'update', 'tenant-f043b', 'projects', 'failed', \
         'key-old-fail', ?1, 3, '2020-01-01T00:00:00Z')",
    )
    .bind(rel_file)
    .execute(&pool)
    .await
    .unwrap();

    // A newer pending queue item for the same file path — exemption does not apply.
    // Use op='add' (different from old row's 'update') to avoid composite-key conflict.
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, item_type, op, tenant_id, collection, status, idempotency_key, \
          file_path, retry_count) \
         VALUES ('q-new-pending', 'file', 'add', 'tenant-f043b', 'projects', 'pending', \
         'key-new-pending', ?1, 0)",
    )
    .bind(rel_file)
    .execute(&pool)
    .await
    .unwrap();

    let _stats = clean_stale_state(&pool, &qm)
        .await
        .expect("clean_stale_state failed");

    // Old failed row should now be purged (newer pending item covers the repair).
    let old_fail_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q-old-fail'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        old_fail_count, 0,
        "Old failed row should be purged when a newer pending item exists (F-043)"
    );

    // Newer pending item should still be present.
    let new_pending_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q-new-pending'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(new_pending_count, 1, "Newer pending item must remain");
}
