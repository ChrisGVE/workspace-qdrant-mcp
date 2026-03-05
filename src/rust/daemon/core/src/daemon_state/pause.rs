//! Pause/resume operations for watch folders (Task 543).

use sqlx::Row;

use super::{DaemonStateManager, DaemonStateResult};

impl DaemonStateManager {
    /// Pause all enabled watch folders
    /// Returns the number of watch folders paused
    pub async fn pause_all_watchers(&self) -> DaemonStateResult<u64> {
        use crate::watch_folders_schema::PAUSE_ALL_WATCHERS_SQL;
        let result = sqlx::query(PAUSE_ALL_WATCHERS_SQL)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    /// Resume all enabled watch folders
    /// Returns the number of watch folders resumed
    pub async fn resume_all_watchers(&self) -> DaemonStateResult<u64> {
        use crate::watch_folders_schema::RESUME_ALL_WATCHERS_SQL;
        let result = sqlx::query(RESUME_ALL_WATCHERS_SQL)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    /// Get list of watch folder IDs that are currently paused
    pub async fn get_paused_watch_ids(&self) -> DaemonStateResult<Vec<String>> {
        let rows: Vec<sqlx::sqlite::SqliteRow> =
            sqlx::query("SELECT watch_id FROM watch_folders WHERE is_paused = 1 AND enabled = 1")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .iter()
            .map(|r| r.try_get::<String, _>("watch_id").unwrap_or_default())
            .collect())
    }

    /// Check if any enabled watch folder is paused.
    /// Returns true if at least one enabled watch has is_paused=1.
    pub async fn any_watchers_paused(&self) -> DaemonStateResult<bool> {
        let count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1 AND enabled = 1",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(count > 0)
    }
}
