//! Migration v31: Add git worktree columns to watch_folders table.
//!
//! Adds `is_worktree` flag and `main_worktree_watch_id` foreign key to
//! distinguish worktree watch roots from main working tree watch roots.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V31Migration;

#[async_trait]
impl Migration for V31Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v31: Adding git worktree columns to watch_folders");

        // Add is_worktree column
        let has_is_worktree: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
             WHERE name = 'is_worktree'",
        )
        .fetch_one(pool)
        .await?;

        if !has_is_worktree {
            debug!("Adding is_worktree column to watch_folders");
            sqlx::query(
                "ALTER TABLE watch_folders \
                 ADD COLUMN is_worktree INTEGER DEFAULT 0 \
                 CHECK (is_worktree IN (0, 1))",
            )
            .execute(pool)
            .await?;
        } else {
            debug!("is_worktree column already exists, skipping");
        }

        // Add main_worktree_watch_id column
        let has_main_worktree: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
             WHERE name = 'main_worktree_watch_id'",
        )
        .fetch_one(pool)
        .await?;

        if !has_main_worktree {
            debug!("Adding main_worktree_watch_id column to watch_folders");
            sqlx::query(
                "ALTER TABLE watch_folders \
                 ADD COLUMN main_worktree_watch_id TEXT \
                 REFERENCES watch_folders(watch_id) ON DELETE SET NULL",
            )
            .execute(pool)
            .await?;
        } else {
            debug!("main_worktree_watch_id column already exists, skipping");
        }

        // Create partial index for efficient worktree lookups
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_watch_main_worktree \
             ON watch_folders(main_worktree_watch_id) \
             WHERE main_worktree_watch_id IS NOT NULL",
        )
        .execute(pool)
        .await?;

        info!("Migration v31 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        31
    }

    fn description(&self) -> &'static str {
        "Add git worktree columns to watch_folders"
    }
}
