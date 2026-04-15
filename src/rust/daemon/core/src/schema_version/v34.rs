//! Migration v34: Create ignore_file_mtimes table.
//!
//! Tracks the last-seen mtime of `.gitignore` and `.wqmignore` files per
//! project. The file watcher compares against stored mtimes to avoid
//! spurious reconciliation triggers on unrelated inotify events.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V34Migration;

#[async_trait]
impl Migration for V34Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v34: Creating ignore_file_mtimes table");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS ignore_file_mtimes (
                project_root TEXT NOT NULL,
                file_path TEXT NOT NULL,
                mtime_unix INTEGER NOT NULL,
                PRIMARY KEY (project_root, file_path)
            )
            "#,
        )
        .execute(pool)
        .await?;

        info!("Migration v34 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        34
    }

    fn description(&self) -> &'static str {
        "Create ignore_file_mtimes table for .gitignore/.wqmignore mtime tracking"
    }
}
