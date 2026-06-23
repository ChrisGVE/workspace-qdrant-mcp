//! Migration v49: add `projects` and `project_locations` tables to the central
//! state.db registry (branch-storage feature F4).
//!
//! File: src/rust/daemon/core/src/schema_version/v49.rs
//! Context: one of the per-version migrations dispatched by `schema_version/mod.rs`
//! (registered in `build_registry`, applied by the `run_migrations` loop).
//!
//! ## What changes and why
//!
//! The branch-storage model (`docs/architecture/branch-storage-model.md` §5.1)
//! introduces a lean central registry in state.db. The two new tables record every
//! known project and every (project, checkout-path, git-branch) triple:
//!
//! - `projects` — one row per registered project; `tenant_id` is UNIQUE and stable
//!   across renames; `content_key_version` starts at 3 (legacy three-slot format)
//!   and is flipped to 4 per-tenant by the F13 migration at cutover.
//!
//! - `project_locations` — one row per (project, location, branch_name) triple;
//!   `branch_id` is the pre-computed SHA256 key `branch_id(tenant_id, location,
//!   branch_name)` derived by the single canonical producer
//!   `wqm_common::hashing::branch_id` (GP-5 / DR GP-1).
//!
//! Existing tables (`watch_folders`, `unified_queue`, `db_maintenance`, etc.)
//! are UNCHANGED.
//!
//! Both tables use `CREATE TABLE IF NOT EXISTS` so a second invocation of `up()`
//! (e.g. crash-after-DDL-but-before-record) is a no-op. The indexes similarly use
//! `IF NOT EXISTS`. The migration runs inside a single `BEGIN IMMEDIATE` /
//! `COMMIT` / `ROLLBACK` transaction (v35+ convention) without a
//! `ForeignKeysGuard` (no FK-referenced table is being dropped).

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V49Migration;

/// Central project registry: one row per registered project.
///
/// `tenant_id` — UUID string, UNIQUE, stable across project renames.
/// `db_path`   — absolute path to the per-project store.db file.
/// `content_key_version` — producer-version gate introduced by F4 (Phase 1,
///     AC-F4.5 / DATA-NN-02 / MF-R4-1): records which content_key slot-shape
///     the ingest path uses for this tenant. Default 3 = legacy three-slot;
///     F13 flips this to 4 at per-tenant cutover. Created HERE by F4 (F4 is
///     the SOLE OWNER of this column's DDL — F13 only updates the value).
const CREATE_PROJECTS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS projects (
    project_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT NOT NULL,
    tenant_id             TEXT NOT NULL UNIQUE,
    db_path               TEXT NOT NULL,
    content_key_version   INTEGER NOT NULL DEFAULT 3,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
)
"#;

/// One row per (project, checkout-location, branch) triple.
///
/// `location`    — absolute checkout root path (forward-slash normalized).
/// `branch_name` — git ref name, e.g. "main" or "feat/x". Must be validated
///     against git ref-name rules (AC-F4.6 / SEC-N04) BEFORE insertion.
/// `branch_id`   — pre-computed canonical key =
///     hex(SHA256(lp(tenant_id)||lp(location)||lp(branch_name))).
///     Derived by `wqm_common::hashing::branch_id` — no second formula.
/// `sync_state`  — lifecycle state: pending (indexed queued) → indexing →
///     current (fully indexed) → error.
const CREATE_PROJECT_LOCATIONS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS project_locations (
    location_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(project_id),
    location      TEXT NOT NULL,
    branch_name   TEXT NOT NULL,
    branch_id     TEXT NOT NULL UNIQUE,
    active        INTEGER NOT NULL DEFAULT 1,
    sync_state    TEXT NOT NULL DEFAULT 'pending'
                      CHECK (sync_state IN ('pending','indexing','current','error')),
    last_synced   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)
"#;

/// Index supporting project-scoped active-location lookups (e.g. "all active
/// locations for project P").
const CREATE_IDX_TENANT_SQL: &str = "CREATE INDEX IF NOT EXISTS idx_project_locations_tenant \
     ON project_locations(project_id, active)";

/// Index supporting branch_id-keyed resolution (the primary search key for
/// `ProjectRegistry::resolve_project` — CWD → branch_id → per-project DB path).
const CREATE_IDX_BRANCH_ID_SQL: &str =
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_project_locations_branch_id \
     ON project_locations(branch_id)";

#[async_trait]
impl Migration for V49Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!(
            "Migration v49: add `projects` and `project_locations` tables \
             to central state.db registry (branch-storage F4)"
        );

        let mut conn = pool.acquire().await?;

        // IMMEDIATE mode acquires the write lock upfront, preventing SQLITE_BUSY
        // during the two-table DDL sequence.
        conn.execute("BEGIN IMMEDIATE").await?;

        let result = apply_v49(&mut conn).await;

        match result {
            Ok(()) => {
                conn.execute("COMMIT").await?;
                debug!("Migration v49 complete");
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK").await;
                Err(e)
            }
        }
    }

    fn version(&self) -> i32 {
        49
    }

    fn description(&self) -> &'static str {
        "Add `projects` and `project_locations` tables to state.db \
         (branch-storage central registry, F4)"
    }
}

/// Apply the v49 DDL on an open connection inside an active transaction.
///
/// All statements use `IF NOT EXISTS` so a second invocation (e.g. crash-after-
/// DDL-before-record) is idempotent.
async fn apply_v49(conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>) -> Result<(), SchemaError> {
    conn.execute(CREATE_PROJECTS_SQL).await?;
    conn.execute(CREATE_PROJECT_LOCATIONS_SQL).await?;
    conn.execute(CREATE_IDX_TENANT_SQL).await?;
    conn.execute(CREATE_IDX_BRANCH_ID_SQL).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use serial_test::serial;
    use sqlx::sqlite::SqlitePoolOptions;

    /// In-memory pool migrated through v49 (the full migration chain).
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

    /// Return true if `table` exists in `sqlite_master`.
    async fn table_exists(pool: &SqlitePool, name: &str) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?1)",
        )
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    /// Return true if `column` appears in `pragma_table_info(table)`.
    async fn has_column(pool: &SqlitePool, table: &str, column: &str) -> bool {
        let probe =
            format!("SELECT COUNT(*) > 0 FROM pragma_table_info('{table}') WHERE name = ?1");
        sqlx::query_scalar(&probe)
            .bind(column)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    /// Return the integer column default for `column` in `table`, or None if the
    /// column has no default or if pragma_table_info does not report one.
    async fn column_dflt_int(pool: &SqlitePool, table: &str, col: &str) -> Option<i64> {
        // pragma_table_info returns (cid, name, type, notnull, dflt_value, pk).
        let dflt: Option<Option<String>> = sqlx::query_scalar(&format!(
            "SELECT dflt_value FROM pragma_table_info('{table}') WHERE name = '{col}'"
        ))
        .fetch_optional(pool)
        .await
        .unwrap();
        dflt.flatten().and_then(|s| s.trim().parse::<i64>().ok())
    }

    // ---- T-F4-migrate: structural presence tests --------------------------------

    /// T-F4-migrate-tables-exist: after v49 both new tables are present and the
    /// pre-existing `watch_folders` table is untouched.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_migrate_tables_exist() {
        let pool = setup_pool().await;

        assert!(
            table_exists(&pool, "projects").await,
            "projects table must exist after v49"
        );
        assert!(
            table_exists(&pool, "project_locations").await,
            "project_locations table must exist after v49"
        );
        // Pre-existing tables must be untouched.
        assert!(
            table_exists(&pool, "watch_folders").await,
            "watch_folders must survive v49 unchanged"
        );
        assert!(
            table_exists(&pool, "db_maintenance").await,
            "db_maintenance must survive v49 unchanged"
        );
    }

    /// T-F4-projects-columns: `projects` carries the columns specified in arch §5.1,
    /// including `content_key_version` (AC-F4.5 / MF-R4-1 — F4 is the SOLE OWNER).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_projects_columns() {
        let pool = setup_pool().await;

        for col in [
            "project_id",
            "name",
            "tenant_id",
            "db_path",
            "content_key_version",
            "created_at",
            "updated_at",
        ] {
            assert!(
                has_column(&pool, "projects", col).await,
                "projects must have column `{col}` after v49"
            );
        }
    }

    /// T-F4-project-locations-columns: `project_locations` carries the columns from
    /// arch §5.1 including `sync_state` with its CHECK constraint.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_project_locations_columns() {
        let pool = setup_pool().await;

        for col in [
            "location_id",
            "project_id",
            "location",
            "branch_name",
            "branch_id",
            "active",
            "sync_state",
            "last_synced",
            "created_at",
            "updated_at",
        ] {
            assert!(
                has_column(&pool, "project_locations", col).await,
                "project_locations must have column `{col}` after v49"
            );
        }
    }

    // ---- T-F4-content-key-version: AC-F4.5 default-3-vs-4 tests ---------------

    /// T-F4-ckv-default-is-3: a project row inserted without specifying
    /// `content_key_version` starts at 3 (pre-F13 default, AC-F4.5).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_ckv_default_is_3() {
        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('proj-a', 'tenant-aaa', '/data/projects/tenant-aaa/store.db', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let ckv: i64 =
            sqlx::query_scalar("SELECT content_key_version FROM projects WHERE tenant_id = ?1")
                .bind("tenant-aaa")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(ckv, 3, "content_key_version must default to 3 (pre-F13)");
    }

    /// T-F4-ckv-post-f13-is-4: a project row explicitly set to version 4 (as F13
    /// will do for new projects after the cutover schema change) stores and retrieves
    /// correctly — proving the column can hold both states.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_ckv_post_f13_is_4() {
        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO projects \
             (name, tenant_id, db_path, content_key_version, created_at, updated_at) \
             VALUES ('proj-b', 'tenant-bbb', '/data/projects/tenant-bbb/store.db', 4, ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let ckv: i64 =
            sqlx::query_scalar("SELECT content_key_version FROM projects WHERE tenant_id = ?1")
                .bind("tenant-bbb")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(ckv, 4, "a post-F13 project must store version 4");
    }

    /// T-F4-ckv-ddl-in-v49: the `content_key_version` column's DEFAULT is 3 as
    /// declared in the DDL (pragma confirms F4 owns the DDL, not F13).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_ckv_ddl_in_v49() {
        let pool = setup_pool().await;
        let dflt = column_dflt_int(&pool, "projects", "content_key_version")
            .await
            .expect("content_key_version must have a DEFAULT");
        assert_eq!(
            dflt, 3,
            "DDL DEFAULT for content_key_version must be 3 (F4 owns this)"
        );
    }

    // ---- T-F4-constraints: AC-F4.1 UNIQUE / CHECK / FK constraints -------------

    /// T-F4-tenant-unique: inserting a second project with the same `tenant_id`
    /// must be rejected (UNIQUE constraint, AC-F4.1).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_tenant_unique() {
        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('proj-1', 'same-tenant', '/data/p1.db', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let dup = sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('proj-2', 'same-tenant', '/data/p2.db', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            dup.is_err(),
            "duplicate tenant_id must be rejected (UNIQUE)"
        );
    }

    /// T-F4-branch-id-unique: inserting a second location row with the same
    /// `branch_id` must be rejected (UNIQUE constraint, AC-F4.1).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_branch_id_unique() {
        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('p', 'ten-c', '/db.db', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let pid: i64 =
            sqlx::query_scalar("SELECT project_id FROM projects WHERE tenant_id = 'ten-c'")
                .fetch_one(&pool)
                .await
                .unwrap();

        let insert_loc = |bid: &str| {
            let pool = pool.clone();
            let bid = bid.to_string();
            async move {
                sqlx::query(
                    "INSERT INTO project_locations \
                     (project_id, location, branch_name, branch_id, created_at, updated_at) \
                     VALUES (?1, '/loc', 'main', ?2, '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')",
                )
                .bind(pid)
                .bind(bid)
                .execute(&pool)
                .await
            }
        };

        insert_loc("deadbeef01").await.unwrap();
        let dup = insert_loc("deadbeef01").await;
        assert!(
            dup.is_err(),
            "duplicate branch_id must be rejected (UNIQUE)"
        );
    }

    /// T-F4-sync-state-check: a sync_state outside the valid set is rejected by the
    /// CHECK constraint (AC-F4.1).
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_sync_state_check() {
        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";

        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('p', 'ten-d', '/db.db', ?1, ?1)",
        )
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let pid: i64 =
            sqlx::query_scalar("SELECT project_id FROM projects WHERE tenant_id = 'ten-d'")
                .fetch_one(&pool)
                .await
                .unwrap();

        // Valid sync_state must succeed.
        for valid in ["pending", "indexing", "current", "error"] {
            sqlx::query(
                "INSERT INTO project_locations \
                 (project_id, location, branch_name, branch_id, sync_state, \
                  created_at, updated_at) \
                 VALUES (?1, '/loc', 'main', ?2, ?3, ?4, ?4)",
            )
            .bind(pid)
            .bind(format!("bid-{valid}"))
            .bind(valid)
            .bind(now)
            .execute(&pool)
            .await
            .unwrap_or_else(|e| panic!("valid sync_state '{valid}' rejected: {e}"));
        }

        // Invalid sync_state must be rejected.
        let bad = sqlx::query(
            "INSERT INTO project_locations \
             (project_id, location, branch_name, branch_id, sync_state, \
              created_at, updated_at) \
             VALUES (?1, '/loc', 'main', 'bad-bid', 'unknown', ?2, ?2)",
        )
        .bind(pid)
        .bind(now)
        .execute(&pool)
        .await;
        assert!(
            bad.is_err(),
            "invalid sync_state 'unknown' must be rejected (CHECK)"
        );
    }

    /// T-F4-fk-project-id: inserting a location row with a non-existent
    /// project_id is rejected by the FK constraint.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_fk_project_id() {
        let pool = setup_pool().await;
        // Enable FK enforcement — SQLite defaults to OFF per connection.
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await
            .unwrap();

        let fk_err = sqlx::query(
            "INSERT INTO project_locations \
             (project_id, location, branch_name, branch_id, created_at, updated_at) \
             VALUES (9999, '/loc', 'main', 'orphan-bid', \
                     '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')",
        )
        .execute(&pool)
        .await;
        assert!(fk_err.is_err(), "orphan project_id must be rejected (FK)");
    }

    // ---- T-F4-seed-f09: AC-F4.2 two-clones-distinct ----------------------------------------

    /// T-F4-seed-f09-two-clones-distinct: two checkouts of `main` at different paths
    /// produce two `project_locations` rows with DISTINCT `branch_id` values (SEED F09).
    ///
    /// The `branch_id` values are pre-computed by the caller using
    /// `wqm_common::hashing::branch_id`; this test stores them and asserts they differ.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_seed_f09_two_clones_distinct() {
        use wqm_common::hashing::branch_id as compute_branch_id;

        let pool = setup_pool().await;
        let now = "2025-01-01T00:00:00.000Z";
        let tenant = "tenant-seed-f09";

        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at) \
             VALUES ('seed-proj', ?1, '/db.db', ?2, ?2)",
        )
        .bind(tenant)
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let pid: i64 = sqlx::query_scalar("SELECT project_id FROM projects WHERE tenant_id = ?1")
            .bind(tenant)
            .fetch_one(&pool)
            .await
            .unwrap();

        // Two clones of `main` at different checkout paths.
        let loc_alice = "/home/alice/myproject";
        let loc_bob = "/home/bob/myproject";
        let bid_alice = compute_branch_id(tenant, loc_alice, "main");
        let bid_bob = compute_branch_id(tenant, loc_bob, "main");

        // The producer guarantee: distinct paths yield distinct branch_ids.
        assert_ne!(
            bid_alice, bid_bob,
            "two clones of main at different paths must produce distinct branch_ids"
        );

        sqlx::query(
            "INSERT INTO project_locations \
             (project_id, location, branch_name, branch_id, created_at, updated_at) \
             VALUES (?1, ?2, 'main', ?3, ?4, ?4)",
        )
        .bind(pid)
        .bind(loc_alice)
        .bind(&bid_alice)
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO project_locations \
             (project_id, location, branch_name, branch_id, created_at, updated_at) \
             VALUES (?1, ?2, 'main', ?3, ?4, ?4)",
        )
        .bind(pid)
        .bind(loc_bob)
        .bind(&bid_bob)
        .bind(now)
        .execute(&pool)
        .await
        .unwrap();

        let row_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM project_locations WHERE project_id = ?1")
                .bind(pid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row_count, 2, "both clone rows must be stored");

        // Retrieve and confirm distinct branch_ids.
        let stored_bids: Vec<String> = sqlx::query_scalar(
            "SELECT branch_id FROM project_locations WHERE project_id = ?1 ORDER BY location",
        )
        .bind(pid)
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(stored_bids.len(), 2);
        assert_ne!(
            stored_bids[0], stored_bids[1],
            "stored branch_ids must differ"
        );
    }

    // ---- T-F4-idempotent: second up() is a no-op --------------------------------

    /// T-F4-idempotent: applying v49 a second time (via the public `up()`) must not
    /// error — all `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` are
    /// safe to re-run.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_idempotent() {
        let pool = setup_pool().await;
        let result = V49Migration.up(&pool).await;
        assert!(
            result.is_ok(),
            "re-running v49 must not error (got {result:?})"
        );
        assert!(table_exists(&pool, "projects").await);
        assert!(table_exists(&pool, "project_locations").await);
    }

    /// T-F4-pre-existing-tables-intact: running the full migration chain (which
    /// includes v49) must leave `watch_folders` and `unified_queue` intact.
    #[serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_pre_existing_tables_intact() {
        let pool = setup_pool().await;
        // These tables must exist (created by earlier migrations).
        assert!(
            table_exists(&pool, "watch_folders").await,
            "watch_folders missing"
        );
        assert!(
            table_exists(&pool, "unified_queue").await,
            "unified_queue missing"
        );
        assert!(
            table_exists(&pool, "db_maintenance").await,
            "db_maintenance missing"
        );
    }
}
