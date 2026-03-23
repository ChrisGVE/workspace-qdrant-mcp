//! Migration v33: Create scratchpad_mirror table.
//!
//! Provides a SQLite-side mirror of text-based scratchpad entries so they
//! can be rebuilt into Qdrant after data loss, matching the `rules_mirror`
//! pattern.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V33Migration;

#[async_trait]
impl Migration for V33Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v33: Creating scratchpad_mirror table");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS scratchpad_mirror (
                scratchpad_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                tenant_id TEXT NOT NULL DEFAULT 'global',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scratchpad_mirror_tenant \
             ON scratchpad_mirror(tenant_id)",
        )
        .execute(pool)
        .await?;

        info!("Migration v33 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        33
    }

    fn description(&self) -> &'static str {
        "Create scratchpad_mirror table for scratchpad persistence"
    }
}
