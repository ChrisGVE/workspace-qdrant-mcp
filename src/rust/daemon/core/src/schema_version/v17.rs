//! Migration v17: Create operational_state table for cross-component state tracking.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V17Migration;

#[async_trait]
impl Migration for V17Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v17: Creating operational_state table");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS operational_state (
                key TEXT NOT NULL,
                component TEXT NOT NULL CHECK(component IN ('daemon', 'server', 'cli')),
                value TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (key, component, project_id)
            )
            "#,
        )
        .execute(pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_operational_state_project ON operational_state (project_id) WHERE project_id IS NOT NULL",
        )
        .execute(pool).await?;

        info!("Migration v17 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 17 }
    fn description(&self) -> &'static str { "Create operational_state table" }
}
