//! Tests for the branch switch protocol.

use std::collections::HashSet;
use std::time::Duration;

use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;

use crate::queue_operations::QueueManager;
use crate::tracked_files_schema::{CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL};
use crate::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL, QueueOperation};
use crate::watch_folders_schema;

use super::db::{batch_update_branch, update_last_commit_hash};
use super::queue::enqueue_file_op;
use super::types::BranchSwitchStats;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_tables(pool: &SqlitePool) {
    sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await.unwrap();
    sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool).await.unwrap();
    sqlx::query(CREATE_TRACKED_FILES_SQL).execute(pool).await.unwrap();
    for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await.unwrap();
    for idx in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
}

async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, tenant_id: &str, path: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
         VALUES (?1, ?2, 'projects', ?3, 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(watch_id)
    .bind(path)
    .bind(tenant_id)
    .execute(pool).await.unwrap();
}

async fn insert_tracked_file(
    pool: &SqlitePool,
    watch_id: &str,
    file_path: &str,
    branch: &str,
    file_hash: &str,
    relative_path: &str,
    base_point: &str,
) {
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash,
         collection, base_point, relative_path, created_at, updated_at)
         VALUES (?1, ?2, ?3, '2025-01-01T00:00:00Z', ?4, 'projects', ?5, ?6, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(watch_id)
    .bind(file_path)
    .bind(branch)
    .bind(file_hash)
    .bind(base_point)
    .bind(relative_path)
    .execute(pool).await.unwrap();
}

#[test]
fn test_branch_switch_stats_default() {
    let stats = BranchSwitchStats::default();
    assert_eq!(stats.batch_updated, 0);
    assert_eq!(stats.enqueued_changed, 0);
    assert_eq!(stats.enqueued_added, 0);
    assert_eq!(stats.enqueued_deleted, 0);
    assert_eq!(stats.errors, 0);
}

#[tokio::test]
async fn test_batch_update_branch_basic() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let tenant = "t1";
    let watch_id = "w1";
    insert_watch_folder(&pool, watch_id, tenant, "/tmp/project").await;

    // Insert 3 files on branch "main"
    let bp1 = wqm_common::hashing::compute_base_point(tenant, "main", "src/a.rs", "hash_a");
    let bp2 = wqm_common::hashing::compute_base_point(tenant, "main", "src/b.rs", "hash_b");
    let bp3 = wqm_common::hashing::compute_base_point(tenant, "main", "src/c.rs", "hash_c");

    insert_tracked_file(&pool, watch_id, "src/a.rs", "main", "hash_a", "src/a.rs", &bp1).await;
    insert_tracked_file(&pool, watch_id, "src/b.rs", "main", "hash_b", "src/b.rs", &bp2).await;
    insert_tracked_file(&pool, watch_id, "src/c.rs", "main", "hash_c", "src/c.rs", &bp3).await;

    // File b.rs changed on the target branch
    let mut changed = HashSet::new();
    changed.insert("src/b.rs".to_string());

    // Batch update from "main" to "feature"
    let count = batch_update_branch(&pool, watch_id, "main", "feature", &changed)
        .await.unwrap();

    // Only 2 files should be updated (a.rs, c.rs — b.rs is in changed set)
    assert_eq!(count, 2);

    // Verify branch was updated for unchanged files
    let branch_a: String = sqlx::query_scalar(
        "SELECT branch FROM tracked_files WHERE file_path = 'src/a.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(branch_a, "feature");

    let branch_c: String = sqlx::query_scalar(
        "SELECT branch FROM tracked_files WHERE file_path = 'src/c.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(branch_c, "feature");

    // b.rs should still be on "main"
    let branch_b: String = sqlx::query_scalar(
        "SELECT branch FROM tracked_files WHERE file_path = 'src/b.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(branch_b, "main");

    // Verify base_point was recomputed for updated files
    let new_bp_a = wqm_common::hashing::compute_base_point(tenant, "feature", "src/a.rs", "hash_a");
    let stored_bp: String = sqlx::query_scalar(
        "SELECT base_point FROM tracked_files WHERE file_path = 'src/a.rs'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(stored_bp, new_bp_a);
}

#[tokio::test]
async fn test_batch_update_branch_empty_changed_set() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let tenant = "t1";
    let watch_id = "w1";
    insert_watch_folder(&pool, watch_id, tenant, "/tmp/project").await;

    let bp = wqm_common::hashing::compute_base_point(tenant, "main", "src/a.rs", "hash_a");
    insert_tracked_file(&pool, watch_id, "src/a.rs", "main", "hash_a", "src/a.rs", &bp).await;

    let changed = HashSet::new();
    let count = batch_update_branch(&pool, watch_id, "main", "dev", &changed)
        .await.unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_batch_update_branch_no_files() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    insert_watch_folder(&pool, "w1", "t1", "/tmp/empty").await;

    let changed = HashSet::new();
    let count = batch_update_branch(&pool, "w1", "main", "dev", &changed)
        .await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_update_last_commit_hash() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    insert_watch_folder(&pool, "w1", "t1", "/tmp/project").await;

    update_last_commit_hash(&pool, "w1", "abc123def456").await.unwrap();

    let hash: Option<String> = sqlx::query_scalar(
        "SELECT last_commit_hash FROM watch_folders WHERE watch_id = 'w1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(hash.as_deref(), Some("abc123def456"));
}

#[tokio::test]
async fn test_enqueue_file_op() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let qm = QueueManager::new(pool.clone());

    enqueue_file_op(&qm, "t1", "projects", "/tmp/project/src/main.rs", QueueOperation::Update, "main")
        .await.unwrap();

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 't1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(count, 1);

    let op: String = sqlx::query_scalar(
        "SELECT op FROM unified_queue WHERE tenant_id = 't1'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(op, "update");
}
