//! Tests for branch cleanup module.

use std::time::Duration;

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

use crate::tracked_files_schema::{
    CREATE_TRACKED_FILES_V40_INDEXES_SQL, CREATE_TRACKED_FILES_V40_SQL,
};
use crate::watch_folders_schema;

use super::db;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_tables(pool: &SqlitePool) {
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(CREATE_TRACKED_FILES_V40_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_V40_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
}

async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, tenant_id: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
         VALUES (?1, '/tmp/test', 'projects', ?2, 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(watch_id)
    .bind(tenant_id)
    .execute(pool).await.unwrap();
}

async fn insert_file(
    pool: &SqlitePool,
    watch_id: &str,
    relative_path: &str,
    branches: &[&str],
    base_point: &str,
) {
    let branches_json = serde_json::json!(branches).to_string();
    let primary = branches.first().copied().unwrap_or("main");
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, relative_path, primary_branch, branches,
         file_mtime, file_hash, collection, base_point, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, '2025-01-01T00:00:00Z', 'hash', 'projects', ?5,
                 '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_id)
    .bind(relative_path)
    .bind(primary)
    .bind(&branches_json)
    .bind(base_point)
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn test_fetch_affected_files_finds_branch() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;
    insert_watch_folder(&pool, "w1", "t1").await;
    insert_file(&pool, "w1", "src/a.rs", &["main", "feature"], "bp1").await;
    insert_file(&pool, "w1", "src/b.rs", &["main"], "bp2").await;

    let affected = db::fetch_affected_files(&pool, "w1", "feature")
        .await
        .unwrap();
    assert_eq!(affected.len(), 1);
    assert_eq!(affected[0].remaining_branches, 1); // only "main" remains
}

#[tokio::test]
async fn test_fetch_affected_files_empty_when_no_match() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;
    insert_watch_folder(&pool, "w1", "t1").await;
    insert_file(&pool, "w1", "src/a.rs", &["main"], "bp1").await;

    let affected = db::fetch_affected_files(&pool, "w1", "feature")
        .await
        .unwrap();
    assert!(affected.is_empty());
}

#[tokio::test]
async fn test_remove_branch_from_tracked_files() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;
    insert_watch_folder(&pool, "w1", "t1").await;
    insert_file(&pool, "w1", "src/a.rs", &["main", "feature", "dev"], "bp1").await;

    let affected = db::fetch_affected_files(&pool, "w1", "feature")
        .await
        .unwrap();
    let refs: Vec<&db::AffectedFile> = affected.iter().collect();

    let count = db::remove_branch_from_tracked_files(&pool, &refs, "feature")
        .await
        .unwrap();
    assert_eq!(count, 1);

    // Verify branches no longer contains "feature"
    let branches_json: String =
        sqlx::query_scalar("SELECT branches FROM tracked_files WHERE relative_path = 'src/a.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();
    let branches: Vec<String> = serde_json::from_str(&branches_json).unwrap();
    assert!(branches.contains(&"main".to_string()));
    assert!(branches.contains(&"dev".to_string()));
    assert!(!branches.contains(&"feature".to_string()));
}

#[tokio::test]
async fn test_delete_orphaned_files() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;
    insert_watch_folder(&pool, "w1", "t1").await;
    insert_file(&pool, "w1", "src/a.rs", &["feature"], "bp1").await;

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE relative_path = 'src/a.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();

    let deleted = db::delete_orphaned_files(&pool, &[file_id]).await.unwrap();
    assert_eq!(deleted, 1);

    // Verify file is gone
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_affected_files_orphan_detection() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;
    insert_watch_folder(&pool, "w1", "t1").await;

    // File only on "feature" — will be orphaned after deletion
    insert_file(&pool, "w1", "src/orphan.rs", &["feature"], "bp1").await;
    // File on both — will just lose "feature"
    insert_file(&pool, "w1", "src/shared.rs", &["main", "feature"], "bp2").await;

    let affected = db::fetch_affected_files(&pool, "w1", "feature")
        .await
        .unwrap();
    assert_eq!(affected.len(), 2);

    let orphan = affected.iter().find(|f| f.remaining_branches == 0).unwrap();
    assert!(orphan.base_point.as_deref() == Some("bp1"));

    let shared = affected.iter().find(|f| f.remaining_branches == 1).unwrap();
    assert!(shared.base_point.as_deref() == Some("bp2"));
}

#[test]
fn test_cleanup_result_default() {
    let result = super::BranchCleanupResult::default();
    assert_eq!(result.updated, 0);
    assert_eq!(result.deleted, 0);
    assert!(!result.skipped);
    assert_eq!(result.errors, 0);
}

#[test]
fn test_branch_existence_enum() {
    // Just verify the enum variants exist
    let _ = super::BranchExistence::ExistsLocally;
    let _ = super::BranchExistence::ExistsRemotely;
    let _ = super::BranchExistence::Gone;
    let _ = super::BranchExistence::CheckFailed("test".to_string());
}
