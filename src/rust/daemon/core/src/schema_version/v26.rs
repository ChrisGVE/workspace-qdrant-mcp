//! Migration v26: Create processing_timings table for queue instrumentation.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V26Migration;

#[async_trait]
impl Migration for V26Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v26: Creating processing_timings table for queue instrumentation");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS processing_timings (
                timing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id TEXT,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                phase TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_op_phase ON processing_timings (op, phase)",
        )
        .execute(pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_created ON processing_timings (created_at)",
        )
        .execute(pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_tenant ON processing_timings (tenant_id)",
        )
        .execute(pool).await?;

        info!("Migration v26 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 26 }
    fn description(&self) -> &'static str { "Create processing_timings table" }
}
