//! Migration v36: replace `unified_queue.file_path` global UNIQUE with a
//! composite partial unique index.
//!
//! SQLite cannot drop a column-level UNIQUE constraint in place, so the
//! migration rebuilds the table:
//!     1. Rename `unified_queue` → `unified_queue_old`.
//!     2. Recreate `unified_queue` from the updated DDL (file_path TEXT,
//!        no UNIQUE; the new composite partial unique index is recreated
//!        with the other indexes below).
//!     3. Skip the copy step: the project is pre-release with no users
//!        to preserve, and existing dev/test queue rows may violate the
//!        new composite constraint or rely on the old column UNIQUE
//!        behavior. Per PRD §T3, drop existing rows.
//!     4. Drop `unified_queue_old`.
//!     5. Recreate all indexes from `CREATE_UNIFIED_QUEUE_INDEXES_SQL`.
//!
//! The composite uniqueness is
//! `(tenant_id, branch, collection, item_type, op, file_path)` so the same
//! file path can legally re-appear under a different tenant, branch,
//! collection, item type, or operation. Closes F-009.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V36Migration;

#[async_trait]
impl Migration for V36Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v36: composite partial UNIQUE on unified_queue file_path");

        let mut conn = pool.acquire().await?;

        // Detect whether the rebuild is already in place — idempotent reruns
        // must be safe.
        let queue_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
        )
        .fetch_one(&mut *conn)
        .await?;

        // The legacy DDL embeds `file_path TEXT UNIQUE`. The new DDL has
        // `file_path TEXT` (no UNIQUE) and relies on the partial index.
        // We use the substring `file_path TEXT UNIQUE` as the
        // "needs-rebuild" marker.
        if !queue_sql.contains("file_path TEXT UNIQUE") {
            info!("Migration v36: unified_queue already uses composite uniqueness");
            // Still ensure the composite partial index exists.
            ensure_composite_index(&mut conn).await?;
            return Ok(());
        }

        // PRAGMA foreign_keys is per-connection. SQLite forbids re-binding
        // table rows referencing a renamed table within the same
        // transaction, so we disable FK checks for the rebuild window.
        conn.execute("PRAGMA foreign_keys = OFF").await?;
        conn.execute("DROP TABLE IF EXISTS unified_queue_old")
            .await?;

        debug!("Renaming unified_queue → unified_queue_old");
        conn.execute("ALTER TABLE unified_queue RENAME TO unified_queue_old")
            .await?;

        debug!("Recreating unified_queue without column-level UNIQUE on file_path");
        conn.execute(crate::CREATE_UNIFIED_QUEUE_SQL).await?;

        // Pre-release: drop existing dev queue rows. No users to migrate.
        // Old rows would be re-enqueued through normal traffic if still
        // relevant.
        debug!("Dropping legacy unified_queue rows (pre-release; no users)");
        conn.execute("DROP TABLE unified_queue_old").await?;

        debug!("Recreating unified_queue indexes (incl. composite partial UNIQUE)");
        for stmt in crate::CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            conn.execute(*stmt).await?;
        }

        conn.execute("PRAGMA foreign_keys = ON").await?;

        info!("Migration v36 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        36
    }

    fn description(&self) -> &'static str {
        "Replace unified_queue.file_path column UNIQUE with composite partial UNIQUE \
         (tenant_id, branch, collection, item_type, op, file_path)"
    }
}

/// Idempotency: when v36 has already rebuilt the table but the composite
/// partial index is missing (e.g. partial rerun), create it explicitly.
async fn ensure_composite_index(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    let composite_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type='index' AND name='idx_unified_queue_file_path_composite')",
    )
    .fetch_one(&mut **conn)
    .await?;
    if !composite_exists {
        debug!("Creating composite partial UNIQUE index idx_unified_queue_file_path_composite");
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_queue_file_path_composite \
             ON unified_queue(tenant_id, branch, collection, item_type, op, file_path) \
             WHERE file_path IS NOT NULL",
        )
        .await?;
    }
    Ok(())
}
