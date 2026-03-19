//! Integration tests for WriteActor commands.
//!
//! Each test creates an in-memory SQLite database, spawns a WriteActor,
//! and exercises the command through the WriteActorHandle API.

use sqlx::SqlitePool;

use super::actor::{WriteActor, WriteActorHandle};
use super::commands::*;

/// Create an in-memory SQLite pool with the minimal schema needed by the
/// WriteActor exec methods, then spawn the actor and return both.
async fn setup_test_db() -> (SqlitePool, WriteActorHandle) {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");

    sqlx::query(
        "CREATE TABLE unified_queue (
            queue_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            branch TEXT DEFAULT 'main',
            payload_json TEXT NOT NULL DEFAULT '{}',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            error_message TEXT,
            last_error_at TEXT,
            lease_until TEXT,
            worker_id TEXT,
            qdrant_status TEXT,
            search_status TEXT,
            decision_json TEXT,
            file_path TEXT
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_paused INTEGER NOT NULL DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            pause_start_time TEXT,
            library_mode TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            cleanup_on_disable INTEGER DEFAULT 0,
            parent_watch_id TEXT,
            remote_hash TEXT,
            git_remote_url TEXT,
            last_activity_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            tenant_id TEXT,
            branch TEXT DEFAULT 'main',
            incremental INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE project_components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            name TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE search_events (
            id TEXT PRIMARY KEY,
            ts TEXT,
            session_id TEXT,
            project_id TEXT,
            actor TEXT,
            tool TEXT,
            op TEXT,
            query_text TEXT,
            filters TEXT,
            top_k INTEGER,
            result_count INTEGER,
            latency_ms INTEGER,
            top_result_refs TEXT,
            outcome TEXT,
            parent_event_id TEXT,
            created_at TEXT
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE rules_mirror (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            scope TEXT,
            tenant_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE corpus_statistics (
            collection TEXT PRIMARY KEY,
            last_corrected_n INTEGER DEFAULT 0
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE watch_folder_submodules (
            parent_watch_id TEXT NOT NULL,
            child_watch_id TEXT NOT NULL,
            PRIMARY KEY (parent_watch_id, child_watch_id)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    let handle = WriteActor::spawn(pool.clone());
    (pool, handle)
}

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

    let now = wqm_common::timestamps::now_utc();
    // Insert items in different collections
    for (i, col) in ["projects", "libraries", "scratchpad"].iter().enumerate() {
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

// ── PauseWatchers / ResumeWatchers tests ─────────────────────────────

#[tokio::test]
async fn pause_and_resume_watchers() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    // Insert two enabled, unpaused watchers
    for i in 0..2 {
        sqlx::query(
            "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, enabled, is_paused, created_at, updated_at) \
             VALUES (?1, ?2, 'projects', ?3, 1, 0, ?4, ?4)",
        )
        .bind(format!("w-{}", i))
        .bind(format!("/tmp/proj-{}", i))
        .bind(format!("t-{}", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Insert one disabled watcher that should NOT be paused
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, is_paused, created_at, updated_at) \
         VALUES ('w-disabled', '/tmp/disabled', 'projects', 't-d', 0, 0, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Pause
    let paused = handle.pause_watchers().await.unwrap();
    assert_eq!(paused, 2);

    // Verify paused state
    let paused_count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(paused_count, 2);

    // Resume
    let resumed = handle.resume_watchers().await.unwrap();
    assert_eq!(resumed, 2);

    // Verify resumed
    let paused_count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(paused_count, 0);
}

// ── EnableWatch / DisableWatch tests ─────────────────────────────────

#[tokio::test]
async fn enable_and_disable_watch() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, created_at, updated_at) \
         VALUES ('watch-toggle', '/tmp/toggle', 'projects', 't1', 1, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Disable
    let affected = handle
        .disable_watch(WatchIdData {
            watch_id: "watch-toggle".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let enabled = sqlx::query_scalar::<_, i64>(
        "SELECT enabled FROM watch_folders WHERE watch_id = 'watch-toggle'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(enabled, 0);

    // Enable
    let affected = handle
        .enable_watch(WatchIdData {
            watch_id: "watch-toggle".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let enabled = sqlx::query_scalar::<_, i64>(
        "SELECT enabled FROM watch_folders WHERE watch_id = 'watch-toggle'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(enabled, 1);
}

// ── AddLibrary / RemoveLibrary tests ─────────────────────────────────

#[tokio::test]
async fn add_and_remove_library() {
    let (pool, handle) = setup_test_db().await;

    // Add
    let add_result = handle
        .add_library(AddLibraryData {
            tag: "my-lib".into(),
            path: "/tmp/my-lib".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();

    assert!(add_result.success);
    assert_eq!(add_result.watch_id, "lib-my-lib");

    // Verify it exists
    let exists = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE watch_id = 'lib-my-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(exists, 1);

    // Adding same tag again should fail
    let dup = handle
        .add_library(AddLibraryData {
            tag: "my-lib".into(),
            path: "/tmp/my-lib-2".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();
    assert!(!dup.success);

    // Remove
    let rm_result = handle
        .remove_library(RemoveLibraryData {
            tag: "my-lib".into(),
        })
        .await
        .unwrap();

    assert!(rm_result.success);

    // Verify gone
    let exists = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE watch_id = 'lib-my-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(exists, 0);
}

#[tokio::test]
async fn remove_nonexistent_library_errors() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .remove_library(RemoveLibraryData {
            tag: "ghost".into(),
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ── LogSearchEvent tests ─────────────────────────────────────────────

#[tokio::test]
async fn log_search_event_inserts_record() {
    let (pool, handle) = setup_test_db().await;

    handle
        .log_search_event(LogSearchEventData {
            id: "evt-1".into(),
            session_id: Some("sess-1".into()),
            project_id: Some("proj-1".into()),
            actor: "claude".into(),
            tool: "search".into(),
            op: "semantic".into(),
            query_text: Some("find auth module".into()),
            filters: None,
            top_k: Some(10),
            result_count: Some(5),
            latency_ms: Some(42),
            top_result_refs: None,
            outcome: Some("success".into()),
            parent_event_id: None,
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM search_events WHERE id = 'evt-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);

    let actor =
        sqlx::query_scalar::<_, String>("SELECT actor FROM search_events WHERE id = 'evt-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(actor, "claude");
}

// ── RenameTenantAdmin tests ──────────────────────────────────────────

#[tokio::test]
async fn rename_tenant_updates_all_tables() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();

    // Insert data across tables referencing old tenant
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w-rename', '/tmp/rename', 'projects', 'old-tenant', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('q-rename', 'idem-rename', 'file', 'add', 'old-tenant', 'projects', 'pending', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, file_path, tenant_id, created_at, updated_at) \
         VALUES ('w-rename', '/tmp/rename/foo.rs', 'old-tenant', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .rename_tenant_admin(RenameTenantAdminData {
            old_tenant_id: "old-tenant".into(),
            new_tenant_id: "new-tenant".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.total_rows_updated >= 3);

    // Verify old tenant is gone from all tables
    let old_watch = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_watch, 0);

    let old_queue = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_queue, 0);

    let old_tracked = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM tracked_files WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_tracked, 0);

    // Verify new tenant exists
    let new_watch = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'new-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(new_watch, 1);
}

#[tokio::test]
async fn rename_tenant_rejects_empty_ids() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .rename_tenant_admin(RenameTenantAdminData {
            old_tenant_id: "".into(),
            new_tenant_id: "new".into(),
        })
        .await;

    assert!(result.is_err());
}

// ── RebalanceIdf tests ───────────────────────────────────────────────

#[tokio::test]
async fn rebalance_idf_updates_corpus_statistics() {
    let (pool, handle) = setup_test_db().await;

    // Insert initial stats row
    sqlx::query(
        "INSERT INTO corpus_statistics (collection, last_corrected_n) VALUES ('projects', 0)",
    )
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .rebalance_idf(RebalanceIdfData {
            collection: "projects".into(),
            last_corrected_n: 42,
        })
        .await
        .unwrap();

    assert!(result.success);

    let n = sqlx::query_scalar::<_, i64>(
        "SELECT last_corrected_n FROM corpus_statistics WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(n, 42);
}

// ── ArchiveWatch tests ───────────────────────────────────────────────

#[tokio::test]
async fn archive_and_unarchive_watch() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at) \
         VALUES ('w-arch', '/tmp/arch', 'projects', 't1', 1, 0, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Archive
    let result = handle
        .archive_watch(ArchiveWatchData {
            watch_id: "w-arch".into(),
            cascade_submodules: false,
        })
        .await
        .unwrap();
    assert_eq!(result.affected_count, 1);

    let archived = sqlx::query_scalar::<_, i64>(
        "SELECT is_archived FROM watch_folders WHERE watch_id = 'w-arch'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(archived, 1);

    // Unarchive
    let affected = handle
        .unarchive_watch(WatchIdData {
            watch_id: "w-arch".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let archived = sqlx::query_scalar::<_, i64>(
        "SELECT is_archived FROM watch_folders WHERE watch_id = 'w-arch'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(archived, 0);
}

// ── WatchLibrary tests ───────────────────────────────────────────────

#[tokio::test]
async fn watch_library_creates_new_and_reactivates() {
    let (pool, handle) = setup_test_db().await;

    // First call creates a new library watch
    let result = handle
        .watch_library(WatchLibraryData {
            tag: "new-lib".into(),
            path: "/tmp/new-lib".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.is_new);
    assert_eq!(result.watch_id, "lib-new-lib");

    // Second call reactivates existing
    let result = handle
        .watch_library(WatchLibraryData {
            tag: "new-lib".into(),
            path: "/tmp/new-lib".into(),
            mode: "incremental".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(!result.is_new);

    // Verify mode was updated
    let mode = sqlx::query_scalar::<_, String>(
        "SELECT library_mode FROM watch_folders WHERE watch_id = 'lib-new-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(mode, "incremental");
}

// ── UpsertRuleMirror / DeleteRuleMirror tests ────────────────────────

#[tokio::test]
async fn upsert_and_delete_rule_mirror() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();

    // Upsert a new rule
    handle
        .upsert_rule_mirror(UpsertRuleMirrorData {
            rule_id: "rule-1".into(),
            rule_text: "Always greet politely".into(),
            scope: "global".into(),
            tenant_id: "t1".into(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM rules_mirror WHERE rule_id = 'rule-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);

    // Upsert same rule with updated text (ON CONFLICT update)
    handle
        .upsert_rule_mirror(UpsertRuleMirrorData {
            rule_id: "rule-1".into(),
            rule_text: "Updated rule text".into(),
            scope: "global".into(),
            tenant_id: "t1".into(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .unwrap();

    let text = sqlx::query_scalar::<_, String>(
        "SELECT rule_text FROM rules_mirror WHERE rule_id = 'rule-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(text, "Updated rule text");

    // Delete
    handle
        .delete_rule_mirror(DeleteRuleMirrorData {
            rule_id: "rule-1".into(),
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM rules_mirror WHERE rule_id = 'rule-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 0);
}
