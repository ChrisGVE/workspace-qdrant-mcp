//! Tests for `facade::read::list` (AC-F10.1, list_branch).
//!
//! File: `wqm-storage/src/facade/read/list_tests.rs`
//! Context: sibling test module for `list.rs`.

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use tempfile::NamedTempFile;

use super::list_branch;

async fn open_writable(path: &std::path::Path) -> SqlitePool {
    let url = format!("sqlite://{}", path.display());
    let opts = SqliteConnectOptions::from_str(&url)
        .unwrap()
        .create_if_missing(true)
        .pragma("foreign_keys", "ON")
        .pragma("journal_mode", "WAL");
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .expect("writable pool")
}

async fn open_readonly(path: &std::path::Path) -> SqlitePool {
    let url = format!("sqlite://{}", path.display());
    let opts = SqliteConnectOptions::from_str(&url)
        .unwrap()
        .read_only(true)
        .pragma("query_only", "ON")
        .pragma("journal_mode", "WAL")
        .pragma("busy_timeout", "5000");
    SqlitePoolOptions::new()
        .max_connections(2)
        .connect_with(opts)
        .await
        .expect("readonly pool")
}

async fn create_branch_file_tables(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE branches (
            branch_id TEXT PRIMARY KEY, branch_name TEXT NOT NULL,
            location TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
            sync_state TEXT NOT NULL DEFAULT 'current'
                CHECK (sync_state IN ('pending','indexing','current','error')),
            sync_metadata TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            relative_path TEXT NOT NULL, file_type TEXT, language TEXT,
            extension TEXT, is_test INTEGER NOT NULL DEFAULT 0,
            collection TEXT NOT NULL DEFAULT 'projects',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (branch_id, relative_path)
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

async fn create_blob_tables(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE blobs (
            blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_key TEXT NOT NULL UNIQUE, chunk_content_hash TEXT NOT NULL,
            point_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL,
            raw_text TEXT NOT NULL, dense_vec BLOB NOT NULL, sparse_vec BLOB NOT NULL,
            chunk_type TEXT, symbol_name TEXT, start_line INTEGER, end_line INTEGER,
            created_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE blob_refs (
            ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            blob_id INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE RESTRICT,
            UNIQUE (branch_id, file_id, chunk_index)
        )",
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE concrete (
            concrete_id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            file_mtime TEXT NOT NULL, file_hash TEXT NOT NULL,
            lsp_status TEXT NOT NULL DEFAULT 'none'
                CHECK (lsp_status IN ('none','done','failed','skipped')),
            treesitter_status TEXT NOT NULL DEFAULT 'none'
                CHECK (treesitter_status IN ('none','done','failed','skipped')),
            component TEXT, routing_reason TEXT, last_error TEXT,
            needs_reconcile INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (branch_id, file_id)
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_schema(pool: &SqlitePool) {
    create_branch_file_tables(pool).await;
    create_blob_tables(pool).await;
}

async fn insert_branch(pool: &SqlitePool, branch_id: &str) {
    sqlx::query(
        "INSERT INTO branches (branch_id, branch_name, location, active, sync_state, created_at, updated_at)
         VALUES (?1, 'main', '/proj', 1, 'current', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_file(pool: &SqlitePool, branch_id: &str, path: &str) -> i64 {
    sqlx::query_scalar(
        "INSERT INTO files (branch_id, relative_path, created_at, updated_at)
         VALUES (?1, ?2, '2026-01-01', '2026-01-01')
         RETURNING file_id",
    )
    .bind(branch_id)
    .bind(path)
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn insert_blob(pool: &SqlitePool, blob_id: i64) {
    sqlx::query(
        "INSERT INTO blobs (blob_id, content_key, chunk_content_hash, point_id, tenant_id,
          raw_text, dense_vec, sparse_vec, created_at)
         VALUES (?1, ?2, ?3, ?4, 't1', 'text', X'', X'', '2026-01-01')",
    )
    .bind(blob_id)
    .bind(format!("ck-{blob_id}"))
    .bind(format!("h-{blob_id}"))
    .bind(format!("pid-{blob_id}"))
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_blob_ref(
    pool: &SqlitePool,
    branch_id: &str,
    file_id: i64,
    blob_id: i64,
    chunk: i64,
) {
    sqlx::query(
        "INSERT INTO blob_refs (branch_id, file_id, chunk_index, blob_id)
         VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(branch_id)
    .bind(file_id)
    .bind(chunk)
    .bind(blob_id)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_concrete(pool: &SqlitePool, branch_id: &str, file_id: i64, file_hash: &str) {
    sqlx::query(
        "INSERT INTO concrete (branch_id, file_id, file_mtime, file_hash, created_at, updated_at)
         VALUES (?1, ?2, '2026-01-01T00:00:00Z', ?3, '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(file_id)
    .bind(file_hash)
    .execute(pool)
    .await
    .unwrap();
}

// list_branch returns all files with correct chunk_count and path.
#[tokio::test]
async fn t_list_branch_returns_files() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let wp = open_writable(tmp.path()).await;
        seed_schema(&wp).await;
        insert_branch(&wp, "b1").await;
        let fid = insert_file(&wp, "b1", "src/main.rs").await;
        insert_blob(&wp, 1).await;
        insert_blob(&wp, 2).await;
        insert_blob_ref(&wp, "b1", fid, 1, 0).await;
        insert_blob_ref(&wp, "b1", fid, 2, 1).await;
        insert_concrete(&wp, "b1", fid, "abc123").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;
    let entries = list_branch(&rp, "b1").await.unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "src/main.rs");
    assert_eq!(entries[0].chunk_count, 2);
    assert_eq!(entries[0].content_hash, "abc123");
}

// list_branch on empty branch returns empty vec.
#[tokio::test]
async fn t_list_branch_empty_branch() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let wp = open_writable(tmp.path()).await;
        seed_schema(&wp).await;
        insert_branch(&wp, "b1").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;
    let entries = list_branch(&rp, "b1").await.unwrap();
    assert!(entries.is_empty());
}

// list_branch is branch-isolated — branch-b files invisible from branch-a query.
#[tokio::test]
async fn t_list_branch_branch_isolated() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let wp = open_writable(tmp.path()).await;
        seed_schema(&wp).await;
        insert_branch(&wp, "b1").await;
        insert_branch(&wp, "b2").await;
        insert_file(&wp, "b1", "only_b1.rs").await;
        insert_file(&wp, "b2", "only_b2.rs").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;
    let b1_entries = list_branch(&rp, "b1").await.unwrap();
    assert_eq!(b1_entries.len(), 1);
    assert_eq!(b1_entries[0].path, "only_b1.rs");
}
