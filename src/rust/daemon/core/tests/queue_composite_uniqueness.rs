//! F-009 regression — composite partial UNIQUE on
//! `(tenant_id, branch, collection, item_type, op, file_path)` allows the
//! same file path under different scopes while still deduplicating
//! same-scope enqueues.
//!
//! The composite uniqueness is created by `CREATE_UNIFIED_QUEUE_INDEXES_SQL`
//! after `CREATE_UNIFIED_QUEUE_SQL`, which now declares `file_path TEXT`
//! (no column-level UNIQUE). The v36 migration rebuilds older databases
//! into the same shape.

use serde_json::json;
use sqlx::SqlitePool;
use workspace_qdrant_core::{
    unified_queue_schema::{ItemType, QueueOperation},
    QueueManager, CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
};

/// Build an in-memory SQLite pool with the unified_queue schema +
/// indexes (the partial composite UNIQUE is in the indexes set).
async fn build_pool() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("connect in-memory sqlite");

    // Watch folders table (required by some QueueManager codepaths).
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            is_active INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )"#,
    )
    .execute(&pool)
    .await
    .expect("create watch_folders");

    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("create unified_queue");
    for stmt in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(stmt)
            .execute(&pool)
            .await
            .expect("create index");
    }

    pool
}

fn file_payload(path: &str) -> String {
    json!({"file_path": path}).to_string()
}

/// The fresh schema must declare `file_path TEXT` (no column UNIQUE) and
/// expose the composite partial UNIQUE index. Without this, the file_path
/// collision regressions below cannot trigger correctly.
#[tokio::test]
async fn schema_has_composite_partial_unique_and_no_column_unique() {
    let pool = build_pool().await;

    // The unified_queue DDL must not declare file_path UNIQUE.
    let table_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !table_sql.contains("file_path TEXT UNIQUE"),
        "unified_queue must not declare file_path TEXT UNIQUE; got: {}",
        table_sql
    );

    // The composite partial unique index must be present.
    let composite_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type='index' AND name='idx_unified_queue_file_path_composite')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        composite_exists,
        "idx_unified_queue_file_path_composite must exist after schema init"
    );
}

/// Same file path enqueued for two distinct tenants must produce two rows
/// (no collision under the new composite uniqueness).
#[tokio::test]
async fn same_file_path_different_tenants_inserts_twice() {
    let pool = build_pool().await;
    let qm = QueueManager::new(pool.clone());

    let payload = file_payload("/data/foo.md");

    let (id_a, new_a) = qm
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "tenant-A",
            "projects",
            &payload,
            Some("main"),
            None,
        )
        .await
        .expect("enqueue tenant A");
    assert!(new_a, "tenant A first enqueue must be new");

    let (id_b, new_b) = qm
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "tenant-B",
            "projects",
            &payload,
            Some("main"),
            None,
        )
        .await
        .expect("enqueue tenant B");
    assert!(new_b, "tenant B first enqueue must be new");
    assert_ne!(id_a, id_b, "different tenants must produce distinct rows");

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE file_path = '/data/foo.md'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 2);
}

/// Same file path enqueued for the same tenant + branch + collection +
/// item_type + op must dedupe to a single row.
#[tokio::test]
async fn same_file_path_same_scope_dedupes() {
    let pool = build_pool().await;
    let qm = QueueManager::new(pool.clone());

    let payload = file_payload("/data/foo.md");

    let (id_a, new_a) = qm
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "tenant-A",
            "projects",
            &payload,
            Some("main"),
            None,
        )
        .await
        .expect("enqueue 1");
    let (id_b, new_b) = qm
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "tenant-A",
            "projects",
            &payload,
            Some("main"),
            None,
        )
        .await
        .expect("enqueue 2");

    assert!(new_a);
    assert!(!new_b, "second enqueue must be a dedup hit");
    assert_eq!(id_a, id_b, "dedupe must return the same queue_id");

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-A' AND file_path = '/data/foo.md'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count, 1);
}

/// Same file path under different collections must produce two rows.
///
/// Cross-collection separation was previously blocked by the column-level
/// UNIQUE on `file_path`.
#[tokio::test]
async fn same_file_path_different_collection_inserts_twice() {
    let pool = build_pool().await;
    let qm = QueueManager::new(pool.clone());

    let payload = file_payload("/data/foo.md");

    qm.enqueue_unified(
        ItemType::File,
        QueueOperation::Add,
        "tenant-A",
        "projects",
        &payload,
        Some("main"),
        None,
    )
    .await
    .expect("enqueue projects");
    qm.enqueue_unified(
        ItemType::File,
        QueueOperation::Add,
        "tenant-A",
        "libraries",
        &payload,
        Some("main"),
        None,
    )
    .await
    .expect("enqueue libraries");

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE file_path = '/data/foo.md'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 2);
}

/// Same path enqueued for Add then Delete (different ops) must produce two
/// rows — the new scope key includes `op`.
#[tokio::test]
async fn same_file_path_different_ops_inserts_twice() {
    let pool = build_pool().await;
    let qm = QueueManager::new(pool.clone());

    let payload = file_payload("/data/foo.md");

    qm.enqueue_unified(
        ItemType::File,
        QueueOperation::Add,
        "tenant-A",
        "projects",
        &payload,
        Some("main"),
        None,
    )
    .await
    .expect("enqueue add");
    qm.enqueue_unified(
        ItemType::File,
        QueueOperation::Delete,
        "tenant-A",
        "projects",
        &payload,
        Some("main"),
        None,
    )
    .await
    .expect("enqueue delete");

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE file_path = '/data/foo.md'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 2);
}
