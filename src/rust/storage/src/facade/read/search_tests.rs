//! Tests for `facade::read::search` (AC-F10.1, AC-F10.2).
//!
//! File: `wqm-storage/src/facade/read/search_tests.rs`
//! Context: sibling test module for `search.rs`.
//!
//! The Qdrant path requires a live Qdrant instance and is therefore omitted
//! from unit tests (integration tests cover that layer). What we test here:
//!   - AC-F10.2: empty tenant_id / branch_id returns Validation immediately.
//!   - RRF fusion correctness (pure logic, no I/O).
//!   - SQLite enrichment (in-memory DB, no Qdrant).

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use tempfile::NamedTempFile;

use super::{build_branch_tenant_filter, enrich_from_sqlite, rrf_fuse};
use crate::qdrant::QdrantReadClient;
use wqm_common::error::StorageError;

// ---------------------------------------------------------------------------
// AC-F10.2: missing tenant/branch returns error, never all-tenant fall-through
// ---------------------------------------------------------------------------

// Helper that calls branch_search with empty tenant — must return Validation.
#[tokio::test]
async fn t_f10_02_empty_tenant_returns_error() {
    // We don't need a real Qdrant for this — the guard fires before any I/O.
    // Construct a minimal QdrantReadClient pointing nowhere (will not be called).
    let qdrant_client = qdrant_client::Qdrant::from_url("http://127.0.0.1:9999")
        .build()
        .expect("build noop client");
    let qdrant = QdrantReadClient::new(qdrant_client);

    let tmp = NamedTempFile::new().unwrap();
    let pool = open_writable(tmp.path()).await;

    let result = super::branch_search(
        &qdrant,
        &pool,
        "", // empty tenant_id
        "branch-1",
        vec![0.0f32; 3],
        vec![],
        vec![],
        5,
    )
    .await;

    assert!(
        matches!(result, Err(StorageError::Validation(_))),
        "empty tenant_id must return Validation (SEC-3), got: {result:?}"
    );
}

// AC-F10.2: empty branch_id also blocked immediately.
#[tokio::test]
async fn t_f10_02_empty_branch_returns_error() {
    let qdrant_client = qdrant_client::Qdrant::from_url("http://127.0.0.1:9999")
        .build()
        .expect("build noop client");
    let qdrant = QdrantReadClient::new(qdrant_client);

    let tmp = NamedTempFile::new().unwrap();
    let pool = open_writable(tmp.path()).await;

    let result = super::branch_search(
        &qdrant,
        &pool,
        "tenant-1",
        "", // empty branch_id
        vec![0.0f32; 3],
        vec![],
        vec![],
        5,
    )
    .await;

    assert!(
        matches!(result, Err(StorageError::Validation(_))),
        "empty branch_id must return Validation, got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// RRF fusion pure-logic tests
// ---------------------------------------------------------------------------

// Both lists agree on rank 1 → highest RRF score.
#[test]
fn t_rrf_fuse_agreement_gives_highest_score() {
    let dense = vec![("a".into(), 0.9f32), ("b".into(), 0.5f32)];
    let sparse = vec![("a".into(), 0.8f32), ("c".into(), 0.4f32)];
    let fused = rrf_fuse(dense, sparse);

    assert!(!fused.is_empty());
    // "a" appears in both lists at rank 1 → best RRF score.
    assert_eq!(fused[0].0, "a", "agreed rank-1 item must lead");
}

// Item only in one list still appears in fused output.
#[test]
fn t_rrf_fuse_single_list_item_included() {
    let dense = vec![("a".into(), 0.9f32)];
    let sparse = vec![("b".into(), 0.8f32)];
    let fused = rrf_fuse(dense, sparse);
    assert_eq!(fused.len(), 2, "both items should appear");
}

// Empty inputs → empty output.
#[test]
fn t_rrf_fuse_empty_inputs() {
    let fused = rrf_fuse(vec![], vec![]);
    assert!(fused.is_empty());
}

// Output is sorted descending by score.
#[test]
fn t_rrf_fuse_output_sorted_descending() {
    let dense = vec![
        ("a".into(), 0.1f32),
        ("b".into(), 0.9f32),
        ("c".into(), 0.5f32),
    ];
    let fused = rrf_fuse(dense, vec![]);
    for w in fused.windows(2) {
        assert!(w[0].1 >= w[1].1, "output must be sorted descending");
    }
}

// ---------------------------------------------------------------------------
// build_branch_tenant_filter: verify must conditions are populated
// ---------------------------------------------------------------------------

#[test]
fn t_filter_has_two_must_conditions() {
    let f = build_branch_tenant_filter("t1", "b1");
    assert_eq!(
        f.must.len(),
        2,
        "filter must have branch_id AND tenant_id conditions"
    );
}

// ---------------------------------------------------------------------------
// enrich_from_sqlite: in-process DB test
// ---------------------------------------------------------------------------

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

async fn seed_enrich_schema(pool: &SqlitePool) {
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

    sqlx::query(
        "CREATE TABLE blobs (
            blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_key TEXT NOT NULL UNIQUE, chunk_content_hash TEXT NOT NULL,
            point_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL,
            raw_text TEXT NOT NULL, dense_vec BLOB NOT NULL,
            sparse_vec BLOB NOT NULL, chunk_type TEXT, symbol_name TEXT,
            start_line INTEGER, end_line INTEGER, created_at TEXT NOT NULL
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

    sqlx::query("CREATE INDEX idx_blob_refs_covering ON blob_refs(blob_id, branch_id, file_id)")
        .execute(pool)
        .await
        .unwrap();
}

// enrich_from_sqlite retrieves path and blob_id for a known point_id.
#[tokio::test]
async fn t_enrich_from_sqlite_retrieves_path() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = open_writable(tmp.path()).await;
    seed_enrich_schema(&pool).await;

    // Seed minimal rows.
    sqlx::query(
        "INSERT INTO branches (branch_id, branch_name, location, active, sync_state, created_at, updated_at)
         VALUES ('b1', 'main', '/p', 1, 'current', '2026-01-01', '2026-01-01')",
    )
    .execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO files (branch_id, relative_path, created_at, updated_at)
         VALUES ('b1', 'src/lib.rs', '2026-01-01', '2026-01-01')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO blobs (blob_id, content_key, chunk_content_hash, point_id,
          tenant_id, raw_text, dense_vec, sparse_vec, symbol_name, start_line, end_line, created_at)
         VALUES (1, 'ck1', 'hash1', 'uuid-point-1', 't1',
                 'fn foo() {}', X'', X'', 'foo', 10, 15, '2026-01-01')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO blob_refs (branch_id, file_id, chunk_index, blob_id)
         VALUES ('b1', 1, 0, 1)",
    )
    .execute(&pool)
    .await
    .unwrap();

    let hits = vec![("uuid-point-1".to_string(), 0.9f32)];
    let enriched = enrich_from_sqlite(&pool, "b1", hits).await.unwrap();

    assert_eq!(enriched.len(), 1);
    assert_eq!(enriched[0].path, "src/lib.rs");
    assert_eq!(enriched[0].blob_id, 1);
    assert_eq!(enriched[0].symbol_name.as_deref(), Some("foo"));
    assert_eq!(enriched[0].start_line, Some(10));
}

// enrich_from_sqlite returns empty for unknown point_ids.
#[tokio::test]
async fn t_enrich_from_sqlite_unknown_point_id_returns_empty() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = open_writable(tmp.path()).await;
    seed_enrich_schema(&pool).await;

    sqlx::query(
        "INSERT INTO branches (branch_id, branch_name, location, active, sync_state, created_at, updated_at)
         VALUES ('b1', 'main', '/p', 1, 'current', '2026-01-01', '2026-01-01')",
    )
    .execute(&pool).await.unwrap();

    let hits = vec![("nonexistent-uuid".to_string(), 0.5f32)];
    let enriched = enrich_from_sqlite(&pool, "b1", hits).await.unwrap();
    assert!(enriched.is_empty());
}
