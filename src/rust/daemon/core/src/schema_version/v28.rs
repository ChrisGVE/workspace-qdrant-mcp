//! Migration v28: Add component column to tracked_files, create project_components table.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V28Migration;

#[async_trait]
impl Migration for V28Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v28: Adding component column to tracked_files and creating project_components table");

        use crate::tracked_files_schema::MIGRATE_V28_ADD_COMPONENT_SQL;

        // Add component column to tracked_files
        let has_component: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'component'",
        )
        .fetch_one(pool)
        .await?;

        if !has_component {
            debug!("Adding component column to tracked_files");
            sqlx::query(MIGRATE_V28_ADD_COMPONENT_SQL)
                .execute(pool)
                .await?;
        } else {
            debug!("component column already exists, skipping ALTER TABLE");
        }

        // Create project_components table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS project_components (
                component_id TEXT PRIMARY KEY,
                watch_folder_id TEXT NOT NULL,
                component_name TEXT NOT NULL,
                base_path TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'auto',
                patterns TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
                UNIQUE(watch_folder_id, component_name)
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_project_components_watch_folder
             ON project_components(watch_folder_id)",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tracked_files_component
             ON tracked_files(component)",
        )
        .execute(pool)
        .await?;

        info!("Migration v28 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        28
    }
    fn description(&self) -> &'static str {
        "Add component column and project_components table"
    }
}
