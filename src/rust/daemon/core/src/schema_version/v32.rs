//! Migration v32: Change is_active from boolean to session counter.
//!
//! The lifecycle module uses `is_active` as a reference counter (increment on
//! session register, decrement on unregister). The original `CHECK (is_active
//! IN (0, 1))` constraint blocks values > 1, breaking multi-session support.
//!
//! SQLite cannot ALTER a CHECK constraint, so this migration recreates the
//! table with `CHECK (is_active >= 0)` and fixes the partial index to use
//! `WHERE is_active > 0`.
//!
//! On fresh databases where v01 already uses the correct constraint, this
//! migration detects it and skips the table recreation.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V32Migration;

#[async_trait]
impl Migration for V32Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v32: Changing is_active from boolean to session counter");

        // Check if the constraint is already correct (fresh databases have it right)
        let table_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='watch_folders'",
        )
        .fetch_one(pool)
        .await?;

        if !table_sql.contains("is_active IN (0, 1)") {
            info!("is_active constraint already correct, checking index only");
            fix_index_if_needed(pool).await?;
            info!("Migration v32 complete (no table recreation needed)");
            return Ok(());
        }

        // Acquire a dedicated connection so PRAGMA foreign_keys = OFF persists
        // across all statements (PRAGMAs are per-connection in SQLite).
        let mut conn = pool.acquire().await?;

        // Step 1: Disable FK checks for table recreation
        conn.execute("PRAGMA foreign_keys = OFF").await?;

        // Step 2: Clean up any leftover temp table from a previous failed attempt
        conn.execute("DROP TABLE IF EXISTS watch_folders_new")
            .await?;

        // Step 3: Create the new table with corrected constraint
        debug!("Creating watch_folders_new with CHECK (is_active >= 0)");
        conn.execute(
            r#"
            CREATE TABLE watch_folders_new (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
                tenant_id TEXT NOT NULL,
                parent_watch_id TEXT,
                submodule_path TEXT,
                git_remote_url TEXT,
                remote_hash TEXT,
                disambiguation_path TEXT,
                is_active INTEGER DEFAULT 0 CHECK (is_active >= 0),
                last_activity_at TEXT,
                library_mode TEXT CHECK (library_mode IS NULL OR library_mode IN ('sync', 'incremental')),
                follow_symlinks INTEGER DEFAULT 0 CHECK (follow_symlinks IN (0, 1)),
                enabled INTEGER DEFAULT 1 CHECK (enabled IN (0, 1)),
                cleanup_on_disable INTEGER DEFAULT 0 CHECK (cleanup_on_disable IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_scan TEXT,
                is_paused INTEGER DEFAULT 0 CHECK (is_paused IN (0, 1)),
                pause_start_time TEXT,
                is_archived INTEGER DEFAULT 0 CHECK (is_archived IN (0, 1)),
                last_commit_hash TEXT,
                is_git_tracked INTEGER DEFAULT 0 CHECK (is_git_tracked IN (0, 1)),
                is_worktree INTEGER DEFAULT 0 CHECK (is_worktree IN (0, 1)),
                main_worktree_watch_id TEXT REFERENCES watch_folders_new(watch_id) ON DELETE SET NULL,
                FOREIGN KEY (parent_watch_id) REFERENCES watch_folders_new(watch_id) ON DELETE CASCADE
            )
            "#,
        )
        .await?;

        // Step 4: Copy all data
        debug!("Copying data from watch_folders to watch_folders_new");
        conn.execute(
            r#"
            INSERT INTO watch_folders_new
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path,
                   is_active, last_activity_at,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan,
                   is_paused, pause_start_time, is_archived,
                   last_commit_hash, is_git_tracked,
                   is_worktree, main_worktree_watch_id
            FROM watch_folders
            "#,
        )
        .await?;

        // Step 5: Drop old table and rename
        debug!("Replacing watch_folders with watch_folders_new");
        conn.execute("DROP TABLE watch_folders").await?;
        conn.execute("ALTER TABLE watch_folders_new RENAME TO watch_folders")
            .await?;

        // Step 6: Recreate all indexes
        debug!("Recreating indexes on watch_folders");
        conn.execute("CREATE INDEX idx_watch_remote_hash ON watch_folders(remote_hash)")
            .await?;
        conn.execute(
            "CREATE INDEX idx_watch_active ON watch_folders(is_active) WHERE is_active > 0",
        )
        .await?;
        conn.execute("CREATE INDEX idx_watch_updated ON watch_folders(updated_at)")
            .await?;
        conn.execute("CREATE INDEX idx_watch_enabled ON watch_folders(enabled) WHERE enabled = 1")
            .await?;
        conn.execute("CREATE INDEX idx_watch_parent ON watch_folders(parent_watch_id)")
            .await?;
        conn.execute(
            "CREATE INDEX idx_watch_collection_tenant ON watch_folders(collection, tenant_id)",
        )
        .await?;
        conn.execute("CREATE INDEX idx_watch_path ON watch_folders(path)")
            .await?;
        conn.execute(
            "CREATE INDEX idx_watch_main_worktree \
             ON watch_folders(main_worktree_watch_id) \
             WHERE main_worktree_watch_id IS NOT NULL",
        )
        .await?;

        // Step 7: Re-enable FK checks
        conn.execute("PRAGMA foreign_keys = ON").await?;

        info!("Migration v32 complete: is_active now supports session counting");
        Ok(())
    }

    fn version(&self) -> i32 {
        32
    }

    fn description(&self) -> &'static str {
        "Change is_active from boolean CHECK to counter CHECK (>= 0)"
    }
}

/// Fix the partial index if it uses `= 1` instead of `> 0`.
async fn fix_index_if_needed(pool: &SqlitePool) -> Result<(), SchemaError> {
    let index_sql: Option<String> = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_watch_active'",
    )
    .fetch_optional(pool)
    .await?;

    if let Some(sql) = index_sql {
        if sql.contains("is_active = 1") {
            debug!("Fixing idx_watch_active: WHERE is_active = 1 -> WHERE is_active > 0");
            sqlx::query("DROP INDEX idx_watch_active")
                .execute(pool)
                .await?;
            sqlx::query(
                "CREATE INDEX idx_watch_active \
                 ON watch_folders(is_active) WHERE is_active > 0",
            )
            .execute(pool)
            .await?;
        }
    }
    Ok(())
}
