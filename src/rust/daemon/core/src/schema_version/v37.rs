//! Migration v37: relative-path migration — drop denormalized absolute
//! `file_path` columns and re-ingest content under the root/relative
//! discipline.
//!
//! Per spec `docs/specs/16-path-abstraction.md` §6.2 and §6.2.1, this
//! migration:
//!
//! 1. Creates the `relative_path_migration_in_progress` marker table
//!    (Phase 1).
//! 2. Truncates ingest-derived tables and rebuilds `tracked_files` with the
//!    `file_path` column dropped; rebuilds `file_metadata` in `search.db`
//!    similarly (Phase 2 — SQLite side).
//! 3. Records the v37 row in `schema_version` so subsequent startups treat
//!    the migration as run-once. The Qdrant truncation, watch walk, queue
//!    drain detection, and final marker deletion (Phase 4) happen
//!    post-startup, driven by a daemon hook outside the migration
//!    framework.
//!
//! Crash safety: the `relative_path_migration_in_progress` marker row is
//! the authoritative "migration unfinished" signal. The startup hook keeps
//! the marker until Qdrant truncation and the first queue drain post-walk
//! both succeed; only then is the row deleted (Phase 4). A crash between
//! phases leaves the marker in place; recovery on next startup re-issues
//! truncation (idempotent) and re-enqueues (idempotent via dedup key).
//!
//! Spec naming: relative-path migration / relative-path migration protocol.
//! The marker table is `relative_path_migration_in_progress`.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V37Migration;

/// DDL for the marker table that tracks an in-progress relative-path
/// migration. The marker is the authoritative signal that the daemon
/// must re-truncate Qdrant and re-walk on next startup.
pub const CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS relative_path_migration_in_progress (
    target_version INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    initial_walk_complete INTEGER NOT NULL DEFAULT 0,
    initial_pending_count INTEGER,
    PRIMARY KEY (target_version)
)
"#;

#[async_trait]
impl Migration for V37Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!(
            "Migration v37: relative-path migration — drop denormalized \
             absolute file_path columns and truncate ingest tables"
        );

        let mut conn = pool.acquire().await?;

        // ---- Phase 1: marker row ---------------------------------------
        // Crash safety: insert the marker first. If we crash before phase
        // 2 finishes, the next startup re-enters truncation (idempotent).
        conn.execute(CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL)
            .await?;
        let now_unix: i64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        sqlx::query(
            "INSERT OR REPLACE INTO relative_path_migration_in_progress \
             (target_version, started_at, initial_walk_complete, initial_pending_count) \
             VALUES (?1, ?2, 0, NULL)",
        )
        .bind(37_i32)
        .bind(now_unix)
        .execute(&mut *conn)
        .await?;
        debug!("Migration v37: phase 1 (marker inserted)");

        // ---- Phase 2 (SQLite side): truncate ingest-derived tables ----
        // Watcher-configured tables (watch_folders, rules, scratchpad) are
        // retained. The path discipline is purely about ingest content.
        //
        // Per spec §6.2 the following are truncated:
        //   tracked_files, qdrant_chunks, file_metadata (search.db is a
        //   separate file — handled by its own migration; here we only
        //   touch state.db), graph_nodes/graph_edges (graph.db — same
        //   note), unified_queue, ignore_file_mtimes.
        //
        // For tables that live in this state.db file:
        conn.execute("PRAGMA foreign_keys = OFF").await?;

        // Drop and rebuild `tracked_files` so the legacy `file_path`
        // column is gone. The replacement schema lives in
        // `tracked_files_schema/schema.rs` (no `file_path`; UNIQUE is
        // `(watch_folder_id, relative_path, branch)`).
        rebuild_tracked_files(&mut conn).await?;

        // Truncate the rest. Idempotent — ok to re-run on recovery.
        for table in ["qdrant_chunks", "unified_queue", "ignore_file_mtimes"] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
            )
            .bind(table)
            .fetch_one(&mut *conn)
            .await?;
            if exists {
                let stmt = format!("DELETE FROM {table}");
                conn.execute(stmt.as_str()).await?;
                debug!("Migration v37: truncated {}", table);
            }
        }

        conn.execute("PRAGMA foreign_keys = ON").await?;

        debug!("Migration v37: phase 2 (SQLite truncation) complete");
        info!(
            "Migration v37 complete (SQLite side). Qdrant truncation, \
             initial walk, and finalize run post-startup."
        );
        Ok(())
    }

    fn version(&self) -> i32 {
        37
    }

    fn description(&self) -> &'static str {
        "Relative-path migration: drop denormalized absolute file_path columns; \
         truncate ingest tables; insert relative_path_migration_in_progress marker"
    }
}

/// Rebuild `tracked_files` without the `file_path` column.
///
/// SQLite cannot drop a column in place for tables created before
/// `DROP COLUMN` support, and our `tracked_files` predates that. We
/// rename, recreate from the canonical DDL, and (because pre-release
/// per-CLAUDE.md "NO MIGRATION EFFORT") discard the legacy rows. The
/// initial walk re-populates the table.
async fn rebuild_tracked_files(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&mut **conn)
    .await?;

    if !table_exists {
        debug!("Migration v37: tracked_files does not exist; skipping rebuild");
        return Ok(());
    }

    // Detect whether the rebuild is already in place (idempotent reruns).
    let tracked_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
    )
    .fetch_one(&mut **conn)
    .await?;

    if !tracked_sql.contains("file_path TEXT NOT NULL") {
        debug!("Migration v37: tracked_files already lacks file_path column");
        return Ok(());
    }

    conn.execute("DROP TABLE IF EXISTS tracked_files_old")
        .await?;
    // Enable legacy_alter_table so the RENAME does NOT rewrite the FK
    // references in dependent tables (qdrant_chunks, indexed_content,
    // etc.) — we want those FKs to keep referencing the symbolic name
    // `tracked_files`, which we re-create immediately after the rename.
    conn.execute("PRAGMA legacy_alter_table = ON").await?;
    conn.execute("ALTER TABLE tracked_files RENAME TO tracked_files_old")
        .await?;
    conn.execute("PRAGMA legacy_alter_table = OFF").await?;
    conn.execute(crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL)
        .await?;
    // Pre-release: discard legacy rows.
    conn.execute("DROP TABLE tracked_files_old").await?;

    // Recreate indexes from the canonical DDL.
    for stmt in crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_INDEXES_SQL {
        conn.execute(*stmt).await?;
    }

    // Recreate the auxiliary indexes that earlier migrations added on top
    // of the base tracked_files schema (reconcile flag index, base_point
    // index, refcount index). These survive the rebuild because they are
    // re-created from the same DDL constants the original migrations used.
    conn.execute(crate::tracked_files_schema::CREATE_RECONCILE_INDEX_SQL)
        .await?;
    conn.execute(crate::tracked_files_schema::CREATE_BASE_POINT_INDEX_SQL)
        .await?;
    conn.execute(crate::tracked_files_schema::CREATE_REFCOUNT_INDEX_SQL)
        .await?;

    debug!("Migration v37: tracked_files rebuilt without file_path column");
    Ok(())
}

/// Check whether a relative-path migration is in progress.
///
/// Returns `true` if the `relative_path_migration_in_progress` marker
/// table exists AND has at least one row. Returns `false` for fresh
/// databases (no table yet) and for completed migrations (row deleted in
/// Phase 4).
pub async fn is_relative_path_migration_in_progress(
    pool: &SqlitePool,
) -> Result<bool, SchemaError> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' \
         AND name='relative_path_migration_in_progress')",
    )
    .fetch_one(pool)
    .await?;
    if !table_exists {
        return Ok(false);
    }
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM relative_path_migration_in_progress")
        .fetch_one(pool)
        .await?;
    Ok(count > 0)
}

/// Snapshot of the marker row used by the post-startup hook and the
/// status banner. None if the migration is not in progress.
#[derive(Debug, Clone)]
pub struct RelativePathMigrationStatus {
    pub target_version: i32,
    pub started_at: i64,
    pub initial_walk_complete: bool,
    pub initial_pending_count: Option<i64>,
}

/// Fetch the current marker row, if any.
pub async fn get_relative_path_migration_status(
    pool: &SqlitePool,
) -> Result<Option<RelativePathMigrationStatus>, SchemaError> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' \
         AND name='relative_path_migration_in_progress')",
    )
    .fetch_one(pool)
    .await?;
    if !table_exists {
        return Ok(None);
    }
    let row: Option<(i32, i64, i64, Option<i64>)> = sqlx::query_as(
        "SELECT target_version, started_at, initial_walk_complete, initial_pending_count \
         FROM relative_path_migration_in_progress LIMIT 1",
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(
        |(target_version, started_at, initial_walk_complete, initial_pending_count)| {
            RelativePathMigrationStatus {
                target_version,
                started_at,
                initial_walk_complete: initial_walk_complete != 0,
                initial_pending_count,
            }
        },
    ))
}

/// Mark the initial walk as complete and capture the initial pending count.
///
/// Called by the post-startup hook once `watch_dir` finishes the initial
/// walk and queue enqueue. The pending count is used by the status banner
/// to compute progress percentage.
pub async fn mark_initial_walk_complete(
    pool: &SqlitePool,
    initial_pending_count: i64,
) -> Result<(), SchemaError> {
    sqlx::query(
        "UPDATE relative_path_migration_in_progress \
         SET initial_walk_complete = 1, initial_pending_count = ?1",
    )
    .bind(initial_pending_count)
    .execute(pool)
    .await?;
    info!(
        "relative-path migration: initial walk complete; {} files to process",
        initial_pending_count
    );
    Ok(())
}

/// Finalize the relative-path migration (Phase 4).
///
/// Deletes all rows from the marker table inside a transaction. The
/// schema-version row is already present (inserted by the migration
/// runner when v37 succeeded); deleting the marker is the authoritative
/// "migration done" signal.
pub async fn finalize_relative_path_migration(pool: &SqlitePool) -> Result<(), SchemaError> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM relative_path_migration_in_progress")
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    info!("relative-path migration finalized (Phase 4 complete)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn fresh_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn marker_table_lifecycle() {
        let pool = fresh_pool().await;

        // No table → not in progress.
        assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());

        // Create the table + a marker row by hand.
        sqlx::query(CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO relative_path_migration_in_progress \
             (target_version, started_at, initial_walk_complete) VALUES (37, 0, 0)",
        )
        .execute(&pool)
        .await
        .unwrap();

        assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());

        // Status reflects the row.
        let status = get_relative_path_migration_status(&pool)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status.target_version, 37);
        assert!(!status.initial_walk_complete);
        assert_eq!(status.initial_pending_count, None);

        // Mark initial walk complete.
        mark_initial_walk_complete(&pool, 1234).await.unwrap();
        let status = get_relative_path_migration_status(&pool)
            .await
            .unwrap()
            .unwrap();
        assert!(status.initial_walk_complete);
        assert_eq!(status.initial_pending_count, Some(1234));

        // Finalize.
        finalize_relative_path_migration(&pool).await.unwrap();
        assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
    }

    #[tokio::test]
    async fn fresh_pool_reports_not_in_progress() {
        let pool = fresh_pool().await;
        let status = get_relative_path_migration_status(&pool).await.unwrap();
        assert!(status.is_none());
    }
}
