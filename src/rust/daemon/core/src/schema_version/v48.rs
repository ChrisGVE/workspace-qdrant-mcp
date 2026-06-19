//! Migration v48: rebuild `tracked_files` for the branch-lineage virtual model,
//! add the `branch_lineage` table, and add a multi-use `maintenance_meta` JSON
//! column to `db_maintenance` (branch-lineage feature F2).
//!
//! File: src/rust/daemon/core/src/schema_version/v48.rs
//! Context: one of the per-version migrations dispatched by `schema_version/mod.rs`
//! (registered in `build_registry`, applied by the `run_migrations` loop). Sits
//! alongside the other `vNN.rs` migration units; follows the v40 rebuild pattern
//! for FK-safe DROP/CREATE.
//!
//! ## What changes and why
//!
//! The v40 `tracked_files` keyed deduplication on
//! `(watch_folder_id, relative_path, file_hash)` and modeled branch membership
//! as a JSON `branches` array. The branch-lineage model is fundamentally
//! different: every (branch, path) pair gets its own row, rows can be *virtual*
//! (shadow points inherited down a lineage), and a row carries an explicit
//! lifecycle `state` ('present' / 'deleted') so deletes are logical tombstones
//! recoverable by the indexing walk. The old v40 UNIQUE constraint collides
//! head-on with this per-(branch, path) model, so the table is rebuilt rather
//! than altered.
//!
//! This migration is **DDL-only** (D1 = Replace, pre-release convention, no
//! users to migrate). It does NOT touch Qdrant, does NOT transform or
//! repopulate rows, and does NOT mint `file_identity_id`. The later indexing
//! walk drops-and-repopulates the table.
//!
//! The `db_maintenance.maintenance_meta` column is a general-purpose nullable
//! JSON store (e.g. migration bookkeeping). It is added idempotently — SQLite's
//! `ALTER TABLE ADD COLUMN` has no `IF NOT EXISTS`, so a `pragma_table_info`
//! probe guards the ALTER, making a second application of v48 a no-op.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::{ForeignKeysGuard, SchemaError};

pub struct V48Migration;

/// Drop the v40 `tracked_files` so the branch-lineage table can be created
/// fresh. Verified safe: no other state.db table FK-references `tracked_files`.
const DROP_TRACKED_FILES_SQL: &str = "DROP TABLE IF EXISTS tracked_files";

/// v48 `tracked_files`: one row per (branch, path), with virtual-shadow and
/// lifecycle-state columns for the branch-lineage model. The FK to
/// `watch_folders(watch_id)` is preserved. The UNIQUE constraint keys on
/// `(tenant_id, content_key, branch, relative_path)`; a separate partial
/// unique index enforces a single live row per `(tenant_id, content_key,
/// branch)` (deleted rows are exempt).
const CREATE_TRACKED_FILES_SQL: &str = r#"
CREATE TABLE tracked_files (
    file_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id   TEXT NOT NULL,
    tenant_id         TEXT NOT NULL,
    branch            TEXT NOT NULL,
    file_identity_id  TEXT NOT NULL,
    content_key       TEXT NOT NULL,
    is_virtual        INTEGER NOT NULL DEFAULT 0,
    state             TEXT NOT NULL DEFAULT 'present'
                          CHECK (state IN ('present','deleted')),
    file_type         TEXT,
    language          TEXT,
    file_mtime        TEXT NOT NULL,
    file_hash         TEXT NOT NULL,
    chunk_count       INTEGER DEFAULT 0,
    chunking_method   TEXT,
    lsp_status        TEXT DEFAULT 'none' CHECK (lsp_status IN ('none','done','failed','skipped')),
    treesitter_status TEXT DEFAULT 'none' CHECK (treesitter_status IN ('none','done','failed','skipped')),
    last_error        TEXT,
    needs_reconcile   INTEGER DEFAULT 0,
    reconcile_reason  TEXT,
    extension         TEXT,
    is_test           INTEGER DEFAULT 0,
    collection        TEXT NOT NULL DEFAULT 'projects',
    base_point        TEXT,
    relative_path     TEXT NOT NULL,
    incremental       INTEGER DEFAULT 0,
    component         TEXT,
    routing_reason    TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE (tenant_id, content_key, branch, relative_path)
)
"#;

/// All seven indexes for the v48 `tracked_files`. The partial unique
/// `idx_tracked_files_live_view` enforces one live (non-deleted) row per
/// `(tenant_id, content_key, branch)`.
const CREATE_TRACKED_FILES_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX idx_tracked_files_content_key   ON tracked_files(tenant_id, content_key)",
    "CREATE INDEX idx_tracked_files_file_identity ON tracked_files(tenant_id, file_identity_id)",
    "CREATE INDEX idx_tracked_files_file_hash     ON tracked_files(tenant_id, file_hash)",
    "CREATE INDEX idx_tracked_files_branch        ON tracked_files(tenant_id, branch)",
    "CREATE INDEX idx_tracked_files_state         ON tracked_files(tenant_id, branch, state)",
    "CREATE UNIQUE INDEX idx_tracked_files_live_view \
        ON tracked_files(tenant_id, content_key, branch) WHERE state != 'deleted'",
    "CREATE INDEX idx_tracked_files_path_window ON tracked_files(tenant_id, relative_path, branch, state)",
];

/// `branch_lineage`: the parent/child relationship between branches per tenant.
/// `origin` records how the edge was learned ('event' from a git hook,
/// 'inferred' from a heuristic, or 'root' for a lineage root with no parent).
/// Created with IF NOT EXISTS so a second `up()` is a no-op (the
/// `tracked_files` rebuild is naturally idempotent via DROP-then-CREATE,
/// but `branch_lineage` is never dropped).
const CREATE_BRANCH_LINEAGE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS branch_lineage (
    tenant_id      TEXT NOT NULL,
    branch         TEXT NOT NULL,
    parent_branch  TEXT,
    origin         TEXT NOT NULL CHECK (origin IN ('event','inferred','root')),
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (tenant_id, branch)
)
"#;

/// Index for walking a tenant's lineage by parent (children-of-a-branch lookups).
const CREATE_BRANCH_LINEAGE_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS idx_branch_lineage_parent ON branch_lineage(tenant_id, parent_branch)";

/// Probe for the `maintenance_meta` column so the ALTER runs at most once
/// (SQLite `ALTER TABLE ADD COLUMN` has no `IF NOT EXISTS`).
const HAS_MAINTENANCE_META_COLUMN_SQL: &str =
    "SELECT COUNT(*) > 0 FROM pragma_table_info('db_maintenance') WHERE name = 'maintenance_meta'";

/// Add the nullable JSON `maintenance_meta` column. Guarded by the probe above.
const ALTER_DB_MAINTENANCE_SQL: &str =
    "ALTER TABLE db_maintenance ADD COLUMN maintenance_meta TEXT";

#[async_trait]
impl Migration for V48Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!(
            "Migration v48: rebuild tracked_files for branch-lineage model, \
             add branch_lineage table and db_maintenance.maintenance_meta column"
        );

        // The DROP must run with FK enforcement out of the way. v35+ convention
        // (#128): disable FK checks via the guard BEFORE opening the
        // transaction (SQLite requires the PRAGMA outside an active
        // transaction). The guard restores FK checks on every path, so a
        // pooled connection never escapes with checks disabled.
        let conn = pool.acquire().await?;
        let mut guard = ForeignKeysGuard::disable(conn).await?;

        // IMMEDIATE mode acquires the write lock upfront, avoiding SQLITE_BUSY
        // during the multi-statement rebuild.
        guard.conn_mut().execute("BEGIN IMMEDIATE").await?;

        let result = apply_v48(guard.conn_mut()).await;

        match result {
            Ok(()) => {
                guard.conn_mut().execute("COMMIT").await?;
                let _conn = guard.restore().await?;
                debug!("Migration v48 complete");
                Ok(())
            }
            Err(e) => {
                let _ = guard.conn_mut().execute("ROLLBACK").await;
                let _conn = guard.restore().await?;
                Err(e)
            }
        }
    }

    fn version(&self) -> i32 {
        48
    }

    fn description(&self) -> &'static str {
        "Rebuild tracked_files for branch-lineage model, add branch_lineage \
         table and db_maintenance.maintenance_meta column"
    }
}

/// Run the v48 DDL on an open transaction: rebuild `tracked_files`, create
/// `branch_lineage`, and add `db_maintenance.maintenance_meta`.
///
/// Idempotent end to end: `tracked_files` is dropped then recreated,
/// `branch_lineage` and its index use `IF NOT EXISTS`, and the
/// `maintenance_meta` ALTER is guarded by a column probe — so a second
/// application is a clean no-op.
async fn apply_v48(conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>) -> Result<(), SchemaError> {
    // 1. Rebuild tracked_files (drop the v40 table, create the v48 one).
    conn.execute(DROP_TRACKED_FILES_SQL).await?;
    conn.execute(CREATE_TRACKED_FILES_SQL).await?;
    for stmt in CREATE_TRACKED_FILES_INDEXES_SQL {
        conn.execute(*stmt).await?;
    }

    // 2. Create the branch_lineage table and its parent-lookup index
    //    (IF NOT EXISTS — branch_lineage is never dropped, so a second
    //    up() must not re-create it).
    conn.execute(CREATE_BRANCH_LINEAGE_SQL).await?;
    conn.execute(CREATE_BRANCH_LINEAGE_INDEX_SQL).await?;

    // 3. Idempotently add db_maintenance.maintenance_meta. The id=1 row was
    //    seeded by v43 and is left untouched.
    let has_column: bool = sqlx::query_scalar(HAS_MAINTENANCE_META_COLUMN_SQL)
        .fetch_one(&mut **conn)
        .await?;
    if !has_column {
        conn.execute(ALTER_DB_MAINTENANCE_SQL).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    /// In-memory pool migrated through the current schema version (includes v48).
    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    /// Insert a watch_folder so FK constraints are satisfied for tracked_files.
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

    /// True if `column` appears in `pragma_table_info(table)`.
    async fn has_column(pool: &SqlitePool, table: &str, column: &str) -> bool {
        let probe =
            format!("SELECT COUNT(*) > 0 FROM pragma_table_info('{table}') WHERE name = ?1");
        sqlx::query_scalar(&probe)
            .bind(column)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    async fn table_exists(pool: &SqlitePool, name: &str) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?1)",
        )
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    /// T-F2-migrate-v47-v48: a fresh DB migrates cleanly to v48; both tables
    /// exist and tracked_files carries the new branch-lineage columns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f2_migrate_v47_v48() {
        let pool = setup_pool().await;

        assert!(
            table_exists(&pool, "tracked_files").await,
            "tracked_files must exist after v48"
        );
        assert!(
            table_exists(&pool, "branch_lineage").await,
            "branch_lineage must exist after v48"
        );

        for column in [
            "tenant_id",
            "branch",
            "file_identity_id",
            "content_key",
            "is_virtual",
            "state",
        ] {
            assert!(
                has_column(&pool, "tracked_files", column).await,
                "tracked_files must have column `{column}` after v48"
            );
        }
    }

    /// T-F2-lineage-consistency: every distinct (tenant_id, branch) in
    /// tracked_files has a matching branch_lineage row — zero orphans.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f2_lineage_consistency() {
        let pool = setup_pool().await;
        insert_test_watch_folder(&pool, "wf1").await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO branch_lineage \
             (tenant_id, branch, parent_branch, origin, created_at, updated_at) \
             VALUES ('tenant1', 'main', NULL, 'root', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              file_mtime, file_hash, relative_path, created_at, updated_at) \
             VALUES ('wf1', 'tenant1', 'main', 'fid1', 'ck1', ?1, 'hash1', \
                     'src/main.rs', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        // Consistency invariant: tracked_files (tenant, branch) pairs with no
        // matching branch_lineage row.
        let orphans: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ( \
                SELECT DISTINCT tf.tenant_id, tf.branch \
                FROM tracked_files tf \
                LEFT JOIN branch_lineage bl \
                  ON bl.tenant_id = tf.tenant_id AND bl.branch = tf.branch \
                WHERE bl.branch IS NULL \
             )",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            orphans, 0,
            "no tracked_files branch should lack a lineage row"
        );
    }

    /// T-F2-maintenance-meta-column: the column is present and the v43-seeded
    /// id=1 singleton row survives the migration.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f2_maintenance_meta_column() {
        let pool = setup_pool().await;

        assert!(
            has_column(&pool, "db_maintenance", "maintenance_meta").await,
            "db_maintenance must have maintenance_meta column after v48"
        );

        let singleton: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM db_maintenance WHERE id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(singleton, 1, "the seeded id=1 row must still exist");
    }

    /// T-F2-maintenance-meta-idempotent: applying v48 a second time is a no-op
    /// — no `duplicate column name` error, column still present exactly once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f2_maintenance_meta_idempotent() {
        let pool = setup_pool().await;

        let result = V48Migration.up(&pool).await;
        assert!(
            result.is_ok(),
            "re-running v48 must not error (got {result:?})"
        );

        let occurrences: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('db_maintenance') \
             WHERE name = 'maintenance_meta'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(occurrences, 1, "maintenance_meta must exist exactly once");
    }

    /// T-F2-live-view-rejects-second-path: the partial unique
    /// `idx_tracked_files_live_view` permits only one live (non-deleted) row
    /// per (tenant_id, content_key, branch); deleted rows are exempt.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f2_live_view_rejects_second_path() {
        let pool = setup_pool().await;
        insert_test_watch_folder(&pool, "wf1").await;
        let now = "2025-01-01T00:00:00.000Z";

        // First live row for (tenant1, ck1, main).
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              state, file_mtime, file_hash, relative_path, created_at, updated_at) \
             VALUES ('wf1', 'tenant1', 'main', 'fid1', 'ck1', 'present', ?1, \
                     'hash1', 'src/a.rs', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        // A second live row, same (tenant, content_key, branch) but a different
        // path, must be rejected by the partial unique live-view index.
        let second_live = sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              state, file_mtime, file_hash, relative_path, created_at, updated_at) \
             VALUES ('wf1', 'tenant1', 'main', 'fid2', 'ck1', 'present', ?1, \
                     'hash2', 'src/b.rs', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            second_live.is_err(),
            "a second live row for the same (tenant, content_key, branch) must be rejected"
        );

        // The same second path inserted as a deleted tombstone is exempt and
        // must succeed.
        let deleted_row = sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              state, file_mtime, file_hash, relative_path, created_at, updated_at) \
             VALUES ('wf1', 'tenant1', 'main', 'fid2', 'ck1', 'deleted', ?1, \
                     'hash2', 'src/b.rs', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            deleted_row.is_ok(),
            "a deleted-state row is exempt from the live-view unique index (got {deleted_row:?})"
        );
    }
}
