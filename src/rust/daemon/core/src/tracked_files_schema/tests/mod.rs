mod batch_tests;
mod crud_tests;
mod transaction_tests;
mod unit_tests;

use super::*;
use crate::schema_version::SchemaManager;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

/// Build the real v48 schema (the whole migration chain) in an in-memory DB and
/// seed one watch_folder so `tracked_files` FK constraints are satisfied.
///
/// Replaces the historic hand-built v40 fixture (`primary_branch` + `branches`)
/// so the CRUD/transaction/batch tests run against the same schema the daemon
/// runs in production (branch-lineage F6).
async fn setup_tables(pool: &SqlitePool) {
    SchemaManager::new(pool.clone())
        .run_migrations()
        .await
        .expect("v48 migration chain must apply");

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/home/user/project', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// The branch stored for a row whose caller passed `None` (no git branch).
///
/// v48 `tracked_files.branch` is NOT NULL, so the legacy "null primary_branch"
/// case maps to this sentinel — distinct from any real branch like `"main"`, so
/// a `lookup_tracked_file(..., Some("main"))` still misses it while a branch-less
/// `lookup_tracked_file(..., None)` finds it.
const NO_BRANCH_SENTINEL: &str = "(none)";

/// Insert a v48 `tracked_files` row from the simplified arguments the CRUD tests
/// use, routing to [`insert_tracked_file_v48`].
///
/// Mirrors the retired v40 `insert_tracked_file` signature so the test bodies
/// stay readable: it allocates a `file_identity_id` (mint-or-inherit) and the
/// derived `content_key`, writing a real (`is_virtual=false`, `state='present'`)
/// row. `branch = None` is stored as [`NO_BRANCH_SENTINEL`].
#[allow(clippy::too_many_arguments)]
async fn insert_test_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    branch: Option<&str>,
    file_type: Option<&str>,
    language: Option<&str>,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    collection: Option<&str>,
    extension: Option<&str>,
    is_test: bool,
    base_point: Option<&str>,
    component: Option<&str>,
) -> Result<i64, sqlx::Error> {
    let branch = branch.unwrap_or(NO_BRANCH_SENTINEL);
    let identity = allocate_file_identity(pool, "t1", branch, relative_path).await?;
    let file_identity_id = identity.id().to_string();
    let content_key = wqm_common::hashing::content_key("t1", &file_identity_id, file_hash);

    insert_tracked_file_v48(
        pool,
        watch_folder_id,
        "t1",
        branch,
        &file_identity_id,
        &content_key,
        false,     // is_virtual
        "present", // state
        file_type,
        language,
        file_mtime,
        file_hash,
        chunk_count,
        chunking_method,
        lsp_status,
        treesitter_status,
        collection.unwrap_or("projects"),
        extension,
        is_test,
        base_point,
        component,
        relative_path,
        false, // needs_reconcile
        None,  // reconcile_reason
    )
    .await
}
