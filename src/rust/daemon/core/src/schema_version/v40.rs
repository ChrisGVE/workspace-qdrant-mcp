//! Migration v40: rebuild `tracked_files` for branch management.
//!
//! Replaces the scalar `branch TEXT` column with `primary_branch TEXT` and
//! `branches TEXT NOT NULL DEFAULT '[]'` (JSON array of branch names a file
//! belongs to). The UNIQUE constraint changes from
//! `(watch_folder_id, relative_path, branch)` to
//! `(watch_folder_id, relative_path, file_hash)`, enabling content-hash
//! deduplication across branches.
//!
//! Since this is a pre-release project with no users to migrate, the
//! migration drops old data rather than copying it. The initial walk
//! re-populates the table.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info, warn};

use super::migration::Migration;
use super::{ForeignKeysGuard, SchemaError};

pub struct V40Migration;

/// Post-v40 DDL for the `tracked_files` table. The scalar `branch` column
/// is replaced by `primary_branch` + `branches` (JSON array). The UNIQUE
/// constraint moves to `(watch_folder_id, relative_path, file_hash)` for
/// content-hash deduplication.
pub const CREATE_TRACKED_FILES_V40_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    primary_branch TEXT,
    branches TEXT NOT NULL DEFAULT '[]',
    file_type TEXT,
    language TEXT,
    file_mtime TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status TEXT DEFAULT 'none' CHECK (lsp_status IN ('none', 'done', 'failed', 'skipped')),
    treesitter_status TEXT DEFAULT 'none' CHECK (treesitter_status IN ('none', 'done', 'failed', 'skipped')),
    last_error TEXT,
    needs_reconcile INTEGER DEFAULT 0,
    reconcile_reason TEXT,
    extension TEXT,
    is_test INTEGER DEFAULT 0,
    collection TEXT NOT NULL DEFAULT 'projects',
    base_point TEXT,
    relative_path TEXT NOT NULL,
    incremental INTEGER DEFAULT 0,
    component TEXT,
    routing_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, relative_path, file_hash)
)
"#;

/// Post-v40 indexes for `tracked_files`. The legacy `idx_tracked_files_branch`
/// is replaced by `idx_tracked_files_branches` over the JSON array column.
pub const CREATE_TRACKED_FILES_V40_INDEXES_SQL: &[&str] = &[
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_watch
       ON tracked_files(watch_folder_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_relative
       ON tracked_files(watch_folder_id, relative_path)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_branches
       ON tracked_files(watch_folder_id, branches)"#,
];

#[async_trait]
impl Migration for V40Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!(
            "Migration v40: replace scalar branch with branches JSON array \
             in tracked_files, change UNIQUE to content-hash"
        );

        let conn = pool.acquire().await?;

        // v35+ convention (#128): disable FK checks via the guard BEFORE
        // opening the transaction (SQLite requires the PRAGMA outside an
        // active transaction). The guard restores FK checks on every path,
        // so a pooled connection never escapes with checks disabled.
        let mut guard = ForeignKeysGuard::disable(conn).await?;

        // IMMEDIATE mode acquires the write lock upfront and prevents
        // SQLITE_BUSY during the rename/recreate sequence.
        guard.conn_mut().execute("BEGIN IMMEDIATE").await?;

        let result = rebuild_tracked_files(guard.conn_mut()).await;

        match result {
            Ok(()) => {
                guard.conn_mut().execute("COMMIT").await?;
                let _conn = guard.restore().await?;
                debug!("Migration v40: tracked_files rebuild complete");
                info!("Migration v40 complete");
                Ok(())
            }
            Err(e) => {
                // Best-effort rollback; ignore secondary errors. Also reset
                // legacy_alter_table in case the failure hit mid-rename.
                let _ = guard.conn_mut().execute("ROLLBACK").await;
                let _ = guard
                    .conn_mut()
                    .execute("PRAGMA legacy_alter_table = OFF")
                    .await;
                let _conn = guard.restore().await?;
                Err(e)
            }
        }
    }

    fn version(&self) -> i32 {
        40
    }

    fn description(&self) -> &'static str {
        "Replace scalar branch with branches JSON array in tracked_files, \
         change UNIQUE to content-hash"
    }
}

/// Rebuild `tracked_files` with the v40 schema.
///
/// Handles three cases:
/// 1. `tracked_files` exists with old schema (has `branch` column) -- full rebuild.
/// 2. `tracked_files` missing but `tracked_files_old` exists -- crash recovery.
/// 3. `tracked_files` already has `primary_branch` -- already migrated (idempotent).
/// 4. `tracked_files` does not exist at all -- skip.
async fn rebuild_tracked_files(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&mut **conn)
    .await?;

    let old_table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files_old')",
    )
    .fetch_one(&mut **conn)
    .await?;

    // Case 2: crash recovery -- tracked_files_old exists without tracked_files.
    if !table_exists && old_table_exists {
        warn!(
            "Migration v40: tracked_files missing but tracked_files_old exists -- \
             recovering from interrupted rebuild"
        );
        conn.execute(CREATE_TRACKED_FILES_V40_SQL).await?;
        conn.execute("DROP TABLE tracked_files_old").await?;
        create_all_indexes(conn).await?;
        return Ok(());
    }

    // Case 4: tracked_files does not exist at all -- nothing to rebuild.
    if !table_exists {
        debug!("Migration v40: tracked_files does not exist; skipping rebuild");
        return Ok(());
    }

    // Case 3: idempotency -- already has `primary_branch` column.
    let tracked_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
    )
    .fetch_one(&mut **conn)
    .await?;

    if tracked_sql.contains("primary_branch") {
        debug!("Migration v40: tracked_files already has primary_branch column");
        if old_table_exists {
            debug!("Migration v40: cleaning up leftover tracked_files_old");
            conn.execute("DROP TABLE tracked_files_old").await?;
        }
        return Ok(());
    }

    // Case 1: full rebuild -- rename, recreate, drop old.
    conn.execute("DROP TABLE IF EXISTS tracked_files_old")
        .await?;
    conn.execute("PRAGMA legacy_alter_table = ON").await?;
    conn.execute("ALTER TABLE tracked_files RENAME TO tracked_files_old")
        .await?;
    conn.execute("PRAGMA legacy_alter_table = OFF").await?;

    // Test fail-point: inject a failure after the rename but before the new
    // table exists — the worst possible interruption. The caller must roll
    // back, restoring the original schema and re-enabling FK checks (#128).
    #[cfg(test)]
    if INJECT_FAILURE_AFTER_RENAME.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(SchemaError::MigrationError(
            "injected test failure after tracked_files rename".into(),
        ));
    }

    conn.execute(CREATE_TRACKED_FILES_V40_SQL).await?;
    // Pre-release: discard legacy rows.
    conn.execute("DROP TABLE tracked_files_old").await?;

    create_all_indexes(conn).await?;

    debug!("Migration v40: tracked_files rebuilt with branches JSON array and content-hash UNIQUE");
    Ok(())
}

/// Create all indexes for the v40 tracked_files schema, including the
/// auxiliary indexes added by earlier migrations.
async fn create_all_indexes(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    // Core v40 indexes.
    for stmt in CREATE_TRACKED_FILES_V40_INDEXES_SQL {
        conn.execute(*stmt).await?;
    }

    // Auxiliary indexes from earlier migrations that must survive the rebuild.
    conn.execute(crate::tracked_files_schema::CREATE_RECONCILE_INDEX_SQL)
        .await?;
    conn.execute(crate::tracked_files_schema::CREATE_BASE_POINT_INDEX_SQL)
        .await?;
    conn.execute(crate::tracked_files_schema::CREATE_REFCOUNT_INDEX_SQL)
        .await?;

    Ok(())
}

/// When set to `true` in tests, `rebuild_tracked_files` injects a failure
/// after the rename-to-old step, letting tests verify that the migration
/// rolls back atomically and restores FK checks (#128).
#[cfg(test)]
pub(crate) static INJECT_FAILURE_AFTER_RENAME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Run all prior migrations so the schema is up to date through v40.
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    /// Insert a test watch_folder so FK constraints are satisfied when
    /// inserting tracked_files rows.
    async fn insert_test_watch_folder(pool: &SqlitePool, watch_id: &str) {
        let now = "2025-01-01T00:00:00.000Z";
        sqlx::query(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES (?1, '/tmp/test', 'projects', 'tenant1', ?2, ?2)",
        )
        .bind(watch_id)
        .bind(now)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn v40_rebuilds_tracked_files() {
        let pool = setup_pool().await;

        // Verify tracked_files exists.
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "tracked_files table should exist after v40");

        // Verify the DDL contains the new columns.
        let ddl: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            ddl.contains("primary_branch"),
            "DDL should contain primary_branch column"
        );
        assert!(
            ddl.contains("branches TEXT"),
            "DDL should contain branches column"
        );

        // The old scalar column was `branch TEXT` (no trailing 'es'). After
        // v40 the only column names containing "branch" are `primary_branch`
        // and `branches`. Verify the old standalone column definition is gone
        // by checking neither a comma-delimited `branch TEXT` nor a
        // newline-delimited one survives (while `branches TEXT` does).
        let has_old_scalar_branch = ddl.lines().any(|line| {
            let trimmed = line.trim().trim_start_matches(',').trim();
            trimmed.starts_with("branch TEXT") || trimmed.starts_with("branch,")
        });
        assert!(
            !has_old_scalar_branch,
            "DDL should not contain old scalar `branch TEXT` column"
        );

        // Verify the UNIQUE constraint uses file_hash, not branch.
        assert!(
            ddl.contains("UNIQUE(watch_folder_id, relative_path, file_hash)"),
            "DDL should have content-hash UNIQUE constraint"
        );
    }

    /// #128: a mid-rebuild failure must (a) leave FK checks enabled on the
    /// connection and (b) roll the schema back to its pre-v40 state.
    #[tokio::test]
    async fn v40_rollback_on_mid_rebuild_failure() {
        // Fresh pool with a pre-v40 tracked_files (scalar `branch` column).
        // Single connection so the PRAGMA assertions see the same connection
        // the migration used.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE TABLE watch_folders (watch_id TEXT PRIMARY KEY, path TEXT)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_folder_id TEXT NOT NULL,
                branch TEXT,
                relative_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
                UNIQUE(watch_folder_id, relative_path, branch)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        INJECT_FAILURE_AFTER_RENAME.store(true, std::sync::atomic::Ordering::SeqCst);
        let result = V40Migration.up(&pool).await;
        INJECT_FAILURE_AFTER_RENAME.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(result.is_err(), "injected failure must propagate");

        // (a) FK checks re-enabled on the pooled connection.
        let fk: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(fk, 1, "FK checks must be restored after a failed rebuild");

        // (b) Schema rolled back: old scalar `branch` column still present,
        // no leftover tracked_files_old.
        let ddl: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            ddl.contains("branch TEXT"),
            "pre-v40 schema must survive a failed rebuild"
        );
        assert!(
            !ddl.contains("primary_branch"),
            "no partial v40 schema after rollback"
        );
        let old_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files_old')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(!old_exists, "no leftover tracked_files_old after rollback");

        // Re-running without the fail-point completes the migration.
        V40Migration.up(&pool).await.unwrap();
        let ddl: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            ddl.contains("primary_branch"),
            "retry completes the rebuild"
        );
    }

    #[tokio::test]
    async fn v40_is_idempotent() {
        let pool = setup_pool().await;

        // Run v40 again -- should not error.
        let migration = V40Migration;
        let result = migration.up(&pool).await;
        assert!(result.is_ok(), "v40 should be idempotent");
    }

    #[tokio::test]
    async fn v40_new_unique_constraint() {
        let pool = setup_pool().await;
        insert_test_watch_folder(&pool, "wf1").await;

        let now = "2025-01-01T00:00:00.000Z";

        // Insert a row.
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, relative_path, file_hash, file_mtime, collection, \
              created_at, updated_at) \
             VALUES ('wf1', 'src/main.rs', 'hash_aaa', ?1, 'projects', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        // Same (watch_folder_id, relative_path, file_hash) should be rejected.
        let dup = sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, relative_path, file_hash, file_mtime, collection, \
              created_at, updated_at) \
             VALUES ('wf1', 'src/main.rs', 'hash_aaa', ?1, 'projects', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            dup.is_err(),
            "Duplicate (watch_folder_id, relative_path, file_hash) should be rejected"
        );

        // Same path but different hash should be allowed (content-hash dedup).
        let different_hash = sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, relative_path, file_hash, file_mtime, collection, \
              created_at, updated_at) \
             VALUES ('wf1', 'src/main.rs', 'hash_bbb', ?1, 'projects', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            different_hash.is_ok(),
            "Same path with different file_hash should be allowed"
        );
    }

    #[tokio::test]
    async fn v40_branches_default() {
        let pool = setup_pool().await;
        insert_test_watch_folder(&pool, "wf1").await;

        let now = "2025-01-01T00:00:00.000Z";

        // Insert a row without specifying branches.
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, relative_path, file_hash, file_mtime, collection, \
              created_at, updated_at) \
             VALUES ('wf1', 'src/lib.rs', 'hash_ccc', ?1, 'projects', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        // Verify branches defaults to '[]'.
        let branches: String = sqlx::query_scalar(
            "SELECT branches FROM tracked_files WHERE relative_path = 'src/lib.rs'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            branches, "[]",
            "branches column should default to empty JSON array"
        );
    }
}
