//! Migration v14: Create search_behavior view for bypass/success/fallback classification.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V14Migration;

#[async_trait]
impl Migration for V14Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v14: Creating search_behavior view");

        sqlx::query(
            r#"
            CREATE VIEW IF NOT EXISTS search_behavior AS
            WITH windowed_events AS (
                SELECT
                    session_id,
                    tool,
                    op,
                    ts,
                    LAG(tool) OVER (PARTITION BY session_id ORDER BY ts) AS prev_tool,
                    LAG(ts) OVER (PARTITION BY session_id ORDER BY ts) AS prev_ts,
                    LEAD(op) OVER (PARTITION BY session_id ORDER BY ts) AS next_op,
                    (julianday(ts) - julianday(LAG(ts) OVER (PARTITION BY session_id ORDER BY ts))) AS time_since_prev
                FROM search_events
                WHERE session_id IS NOT NULL
            )
            SELECT
                session_id,
                tool,
                op,
                ts,
                prev_tool,
                next_op,
                CASE
                    WHEN tool IN ('rg', 'grep') AND prev_tool IS NULL THEN 'bypass'
                    WHEN tool = 'mcp_qdrant' AND (next_op = 'open' OR next_op = 'expand') THEN 'success'
                    WHEN tool = 'mcp_qdrant' AND time_since_prev < 0.00139
                         AND prev_tool IN ('rg', 'grep', 'mcp_qdrant') THEN 'fallback'
                    ELSE 'unknown'
                END AS behavior
            FROM windowed_events
            "#
        )
        .execute(pool).await?;

        info!("Migration v14 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        14
    }
    fn description(&self) -> &'static str {
        "Create search_behavior view"
    }
}
