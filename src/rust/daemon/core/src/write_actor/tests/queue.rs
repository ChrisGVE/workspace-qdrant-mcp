//! Tests for queue-related WriteActor commands:
//! EnqueueItem, RetryAll, RetryItem, CleanQueue, CancelItems,
//! RemoveItem, CleanQueueByCollection.

use crate::write_actor::commands::*;

use super::common::setup_test_db;

// ── EnqueueItem tests ────────────────────────────────────────────────

#[tokio::test]
async fn enqueue_item_creates_new_entry() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .enqueue_item(EnqueueItemData {
            item_type: "file".into(),
            op: "add".into(),
            tenant_id: "tenant-1".into(),
            collection: "projects".into(),
            payload_json: r#"{"path":"/tmp/foo.rs"}"#.into(),
            branch: "main".into(),
            metadata_json: None,
        })
        .await
        .expect("enqueue should succeed");

    assert!(result.is_new, "first enqueue should be new");
    assert!(!result.queue_id.is_empty());
    assert!(!result.idempotency_key.is_empty());
}

#[tokio::test]
async fn enqueue_item_idempotency_returns_existing() {
    let (_pool, handle) = setup_test_db().await;

    let data = || EnqueueItemData {
        item_type: "file".into(),
        op: "add".into(),
        tenant_id: "tenant-1".into(),
        collection: "projects".into(),
        payload_json: r#"{"path":"/tmp/foo.rs"}"#.into(),
        branch: "main".into(),
        metadata_json: None,
    };

    let first = handle.enqueue_item(data()).await.unwrap();
    let second = handle.enqueue_item(data()).await.unwrap();

    assert!(first.is_new);
    assert!(!second.is_new, "duplicate enqueue must not be new");
    assert_eq!(first.queue_id, second.queue_id);
    assert_eq!(first.idempotency_key, second.idempotency_key);
}

#[tokio::test]
async fn enqueue_item_rejects_empty_tenant() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .enqueue_item(EnqueueItemData {
            item_type: "file".into(),
            op: "add".into(),
            tenant_id: "".into(),
            collection: "projects".into(),
            payload_json: "{}".into(),
            branch: "main".into(),
            metadata_json: None,
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("tenant_id"));
}

// ── RetryAll tests ───────────────────────────────────────────────────

#[tokio::test]
async fn retry_all_resets_failed_items() {
    let (pool, handle) = setup_test_db().await;

    // Insert two failed items directly
    for i in 0..2 {
        let now = wqm_common::timestamps::now_utc();
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'add', 'tenant-1', 'projects', 'failed', '{}', ?3, ?3)",
        )
        .bind(format!("failed-{}", i))
        .bind(format!("idem-{}", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Insert one done item that should NOT be reset
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('done-1', 'idem-done', 'file', 'add', 'tenant-1', 'projects', 'done', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle.retry_all().await.unwrap();
    assert_eq!(result.reset_count, 2);
}

// ── RetryItem tests ──────────────────────────────────────────────────

#[tokio::test]
async fn retry_item_resets_failed_item_by_prefix() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, retry_count, payload_json, created_at, updated_at) \
         VALUES ('abc-123-def', 'idem-abc', 'file', 'add', 't1', 'projects', 'failed', 2, '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .retry_item(RetryItemData {
            queue_id: "abc".into(),
        })
        .await
        .unwrap();

    assert!(result.found);
    assert_eq!(result.resolved_id, "abc-123-def");
    assert_eq!(result.previous_status, "failed");
    assert_eq!(result.previous_retry_count, 2);
    assert!(result.reset);
}

#[tokio::test]
async fn retry_item_skips_non_failed() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('pending-1', 'idem-p', 'file', 'add', 't1', 'projects', 'pending', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .retry_item(RetryItemData {
            queue_id: "pending-1".into(),
        })
        .await
        .unwrap();

    assert!(result.found);
    assert!(!result.reset, "pending item should not be reset");
    assert_eq!(result.previous_status, "pending");
}

#[tokio::test]
async fn retry_item_not_found() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .retry_item(RetryItemData {
            queue_id: "nonexistent".into(),
        })
        .await
        .unwrap();

    assert!(!result.found);
}

// ── CleanQueue tests ─────────────────────────────────────────────────

#[tokio::test]
async fn clean_queue_deletes_old_items() {
    let (pool, handle) = setup_test_db().await;

    // Insert an item with a very old timestamp
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('old-1', 'idem-old', 'file', 'add', 't1', 'projects', 'done', '{}', \
                 '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a recent item that should NOT be cleaned
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('new-1', 'idem-new', 'file', 'add', 't1', 'projects', 'done', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let deleted = handle
        .clean_queue(CleanQueueData {
            older_than_days: 1,
            statuses: vec!["done".into()],
        })
        .await
        .unwrap();

    assert_eq!(deleted, 1);
}

#[tokio::test]
async fn clean_queue_rejects_invalid_status() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .clean_queue(CleanQueueData {
            older_than_days: 1,
            statuses: vec!["pending".into()],
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid status"));
}

// ── CancelItems tests ────────────────────────────────────────────────

#[tokio::test]
async fn cancel_items_deletes_pending_for_tenant() {
    let (pool, handle) = setup_test_db().await;

    // Register a watch folder so resolve_tenant can find the tenant
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/proj', 'projects', 'tenant-a', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Insert pending items for tenant-a
    for i in 0..3 {
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'add', 'tenant-a', 'projects', 'pending', '{}', ?3, ?3)",
        )
        .bind(format!("cancel-{}", i))
        .bind(format!("idem-cancel-{}", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Insert an in_progress item that must NOT be cancelled
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('ip-1', 'idem-ip', 'file', 'add', 'tenant-a', 'projects', 'in_progress', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .cancel_items(CancelItemsData {
            tenant_id: "tenant-a".into(),
            statuses: vec!["pending".into()],
            dry_run: false,
        })
        .await
        .unwrap();

    assert_eq!(result.count, 3);
    assert!(!result.is_dry_run);

    // Verify in_progress item still exists
    let remaining = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-a'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(remaining, 1);
}

#[tokio::test]
async fn cancel_items_dry_run_does_not_delete() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/proj', 'projects', 'tenant-b', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('dry-1', 'idem-dry', 'file', 'add', 'tenant-b', 'projects', 'pending', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .cancel_items(CancelItemsData {
            tenant_id: "tenant-b".into(),
            statuses: vec![],
            dry_run: true,
        })
        .await
        .unwrap();

    assert!(result.is_dry_run);
    assert_eq!(result.count, 1);

    // Verify item still exists
    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'dry-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);
}

// ── RemoveItem tests ─────────────────────────────────────────────────

#[tokio::test]
async fn remove_item_deletes_by_exact_id() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('rm-exact-1', 'idem-rm', 'file', 'add', 't1', 'projects', 'pending', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .remove_item(RemoveItemData {
            queue_id: "rm-exact-1".into(),
        })
        .await
        .unwrap();

    assert!(result.found);
    assert_eq!(result.resolved_id, "rm-exact-1");
    assert_eq!(result.item_type, "file");
    assert_eq!(result.status, "pending");

    // Verify deletion
    let count = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'rm-exact-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn remove_item_not_found() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .remove_item(RemoveItemData {
            queue_id: "nonexistent".into(),
        })
        .await
        .unwrap();

    assert!(!result.found);
}

// ── CleanQueueByCollection tests ─────────────────────────────────────

#[tokio::test]
async fn clean_queue_by_collection_deletes_matching() {
    let (pool, handle) = setup_test_db().await;

    use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_SCRATCHPAD};
    let now = wqm_common::timestamps::now_utc();
    // Insert items in different collections
    for (i, col) in [COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_SCRATCHPAD]
        .iter()
        .enumerate()
    {
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'add', 't1', ?3, 'pending', '{}', ?4, ?4)",
        )
        .bind(format!("col-{}", i))
        .bind(format!("idem-col-{}", i))
        .bind(col)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    let deleted = handle
        .clean_queue_by_collection(CleanQueueByCollectionData {
            collections: vec!["libraries".into()],
            statuses: vec!["pending".into()],
        })
        .await
        .unwrap();

    assert_eq!(deleted, 1);

    // Verify remaining
    let remaining = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM unified_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(remaining, 2);
}
