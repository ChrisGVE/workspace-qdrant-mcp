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

// ── search.db: file_metadata + orphaned code_lines pruning (#102) ──

async fn setup_search_tables(pool: &SqlitePool) {
    sqlx::query(crate::code_lines_schema::CREATE_FILE_METADATA_V7_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(crate::code_lines_schema::CREATE_CODE_LINES_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(crate::code_lines_schema::CREATE_CODE_LINES_FTS_SQL)
        .execute(pool)
        .await
        .unwrap();
}

async fn insert_metadata(pool: &SqlitePool, file_id: i64, tenant: &str, branch: &str) {
    sqlx::query(
        "INSERT INTO file_metadata (file_id, tenant_id, branch, file_path)
         VALUES (?1, ?2, ?3, '/p/f.rs')",
    )
    .bind(file_id)
    .bind(tenant)
    .bind(branch)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_line(pool: &SqlitePool, file_id: i64, seq: f64, content: &str) {
    let line_id: i64 = sqlx::query_scalar(
        "INSERT INTO code_lines (file_id, seq, content, line_number)
         VALUES (?1, ?2, ?3, 1) RETURNING line_id",
    )
    .bind(file_id)
    .bind(seq)
    .bind(content)
    .fetch_one(pool)
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::FTS5_INSERT_ROW_SQL)
        .bind(line_id)
        .bind(content)
        .execute(pool)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_delete_file_metadata_for_branch_prunes_orphaned_code_lines() {
    let pool = create_test_pool().await;
    setup_search_tables(&pool).await;

    // file 1: only on the deleted branch → fully orphaned
    insert_metadata(&pool, 1, "t1", "feature").await;
    insert_line(&pool, 1, 1000.0, "orphan line").await;
    // file 2: on both branches → code_lines must survive
    insert_metadata(&pool, 2, "t1", "feature").await;
    insert_metadata(&pool, 2, "t1", "main").await;
    insert_line(&pool, 2, 1000.0, "shared line").await;

    db::delete_file_metadata_for_branch(&pool, "t1", "feature").await;

    let branch_rows: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE branch = 'feature'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(branch_rows, 0, "feature rows pruned");

    let orphan_lines: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(orphan_lines, 0, "orphaned code_lines pruned");

    let shared_lines: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 2")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(shared_lines, 1, "shared file's code_lines survive");

    // FTS5 index no longer matches the orphaned content.
    let fts_hits: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH '\"orphan line\"'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(fts_hits, 0, "FTS5 entry removed with the orphaned line");

    let fts_shared: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH '\"shared line\"'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(fts_shared, 1, "shared FTS5 entry survives");
}

#[tokio::test]
async fn test_delete_file_metadata_for_branch_scoped_to_tenant() {
    let pool = create_test_pool().await;
    setup_search_tables(&pool).await;

    insert_metadata(&pool, 1, "t1", "feature").await;
    insert_metadata(&pool, 2, "t2", "feature").await;

    db::delete_file_metadata_for_branch(&pool, "t1", "feature").await;

    let t2_rows: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 't2'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(t2_rows, 1, "other tenant's rows untouched");
}

// ── #127: failed Qdrant deletions must keep local rows for retry ──

fn affected(file_id: i64, base_point: Option<&str>) -> db::AffectedFile {
    db::AffectedFile {
        file_id,
        base_point: base_point.map(str::to_string),
        branches: vec!["feature".to_string()],
        remaining_branches: 0,
    }
}

#[test]
fn test_deletable_file_ids_keeps_rows_for_failed_base_points() {
    let f1 = affected(1, Some("bp_ok"));
    let f2 = affected(2, Some("bp_failed"));
    let f3 = affected(3, None); // no base_point → no Qdrant points → deletable
    let to_delete = vec![&f1, &f2, &f3];

    let failed: std::collections::HashSet<&str> = ["bp_failed"].into_iter().collect();

    let ids = super::deletable_file_ids(&to_delete, &failed);
    assert_eq!(ids, vec![1, 3], "row for failed base_point must survive");
}

#[test]
fn test_deletable_file_ids_all_deletable_when_no_failures() {
    let f1 = affected(1, Some("bp1"));
    let f2 = affected(2, Some("bp2"));
    let to_delete = vec![&f1, &f2];

    let ids = super::deletable_file_ids(&to_delete, &std::collections::HashSet::new());
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn test_deletable_file_ids_shared_base_point_failure_keeps_all_sharers() {
    // Two files share the same base_point; a single failed Qdrant delete
    // must keep both local rows.
    let f1 = affected(1, Some("bp_shared"));
    let f2 = affected(2, Some("bp_shared"));
    let f3 = affected(3, Some("bp_other"));
    let to_delete = vec![&f1, &f2, &f3];

    let failed: std::collections::HashSet<&str> = ["bp_shared"].into_iter().collect();

    let ids = super::deletable_file_ids(&to_delete, &failed);
    assert_eq!(ids, vec![3]);
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
