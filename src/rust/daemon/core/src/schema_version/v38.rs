//! Migration v38: Token-economy instrumentation on search_events.
//!
//! Per `docs/specs/20-token-economy-instrumentation.md`, this adds five
//! per-event columns capturing the payload-shaping economics, plus the
//! index and view used by `wqm admin token-savings` aggregations.
//!
//! Added columns (all NULLable so rows written before this migration
//! remain valid and are excluded from token-economy aggregates):
//!   - `bytes_in`        sum of pre-shape content bytes
//!   - `bytes_out`       sum of post-shape content bytes
//!   - `hits_truncated`  count of hits whose body was truncated
//!   - `shape_mode`      'truncate' | 'summary' | 'none'
//!   - `tool_version`    MCP server version, for trend attribution
//!
//! Also creates:
//!   - `idx_search_events_session_tool_ts` — powers the followup probe
//!     in the `token_savings` view.
//!   - `token_savings` view — derived `savings_bytes`, `savings_ratio`,
//!     `had_followup`, and `had_escalation` per event row.
//!
//! The view's followup probe uses a 60s window
//! (julianday delta ≤ 0.000694) and the escalation probe uses a 120s
//! window (≤ 0.001389), matching the spec's `FOLLOWUP_WINDOW` and
//! `ESCALATION_WINDOW` constants.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V38Migration;

pub const MIGRATE_V38_ADD_COLUMNS_SQL: &[&str] = &[
    "ALTER TABLE search_events ADD COLUMN bytes_in INTEGER",
    "ALTER TABLE search_events ADD COLUMN bytes_out INTEGER",
    "ALTER TABLE search_events ADD COLUMN hits_truncated INTEGER",
    "ALTER TABLE search_events ADD COLUMN shape_mode TEXT",
    "ALTER TABLE search_events ADD COLUMN tool_version TEXT",
];

pub const CREATE_SESSION_TOOL_TS_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS idx_search_events_session_tool_ts \
     ON search_events(session_id, tool, ts)";

pub const CREATE_TOKEN_SAVINGS_VIEW_SQL: &str = r#"
CREATE VIEW IF NOT EXISTS token_savings AS
SELECT
    se.id,
    se.session_id,
    se.project_id,
    se.tool,
    se.op,
    se.shape_mode,
    se.tool_version,
    se.ts,
    se.bytes_in,
    se.bytes_out,
    se.hits_truncated,
    CASE
        WHEN se.bytes_in IS NOT NULL AND se.bytes_out IS NOT NULL
            THEN se.bytes_in - se.bytes_out
        ELSE NULL
    END AS savings_bytes,
    CASE
        WHEN se.bytes_in IS NOT NULL AND se.bytes_in > 0 AND se.bytes_out IS NOT NULL
            THEN 1.0 * (se.bytes_in - se.bytes_out) / se.bytes_in
        ELSE NULL
    END AS savings_ratio,
    EXISTS (
        SELECT 1 FROM search_events nxt
        WHERE nxt.session_id = se.session_id
          AND nxt.tool       = se.tool
          AND nxt.op         = 'followup'
          AND julianday(nxt.ts) - julianday(se.ts) BETWEEN 0 AND 0.000694
    ) AS had_followup,
    EXISTS (
        SELECT 1 FROM search_events nxt
        WHERE nxt.parent_event_id = se.id
          AND nxt.op IN ('open', 'expand')
          AND julianday(nxt.ts) - julianday(se.ts) BETWEEN 0 AND 0.001389
    ) AS had_escalation
FROM search_events se
WHERE se.bytes_in IS NOT NULL
"#;

#[async_trait]
impl Migration for V38Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v38: Adding token-economy columns to search_events");

        let has_bytes_in: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('search_events') WHERE name = 'bytes_in'",
        )
        .fetch_one(pool)
        .await?;

        if !has_bytes_in {
            for alter_sql in MIGRATE_V38_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("bytes_in column already exists, skipping ALTER TABLE");
        }

        sqlx::query(CREATE_SESSION_TOOL_TS_INDEX_SQL)
            .execute(pool)
            .await?;
        sqlx::query(CREATE_TOKEN_SAVINGS_VIEW_SQL)
            .execute(pool)
            .await?;

        info!("Migration v38 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        38
    }

    fn description(&self) -> &'static str {
        "Add token-economy columns + index + token_savings view to search_events"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn fresh_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    async fn setup_search_events(pool: &SqlitePool) {
        use crate::search_events_schema::{
            CREATE_SEARCH_EVENTS_INDEXES_SQL, CREATE_SEARCH_EVENTS_SQL,
        };
        sqlx::query(CREATE_SEARCH_EVENTS_SQL)
            .execute(pool)
            .await
            .unwrap();
        for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await.unwrap();
        }
    }

    #[tokio::test]
    async fn migration_adds_columns_idempotently() {
        let pool = fresh_pool().await;
        setup_search_events(&pool).await;

        V38Migration.up(&pool).await.unwrap();

        // Columns exist
        let cols: Vec<String> =
            sqlx::query_scalar("SELECT name FROM pragma_table_info('search_events')")
                .fetch_all(&pool)
                .await
                .unwrap();
        for expected in [
            "bytes_in",
            "bytes_out",
            "hits_truncated",
            "shape_mode",
            "tool_version",
        ] {
            assert!(cols.iter().any(|c| c == expected), "missing column {expected}");
        }

        // Re-running is a no-op (does not error)
        V38Migration.up(&pool).await.unwrap();
    }

    #[tokio::test]
    async fn token_savings_view_computes_derived_columns() {
        let pool = fresh_pool().await;
        setup_search_events(&pool).await;
        V38Migration.up(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, ts, actor, tool, op, bytes_in, bytes_out, hits_truncated, shape_mode) \
             VALUES ('e1', '2026-05-26T12:00:00.000Z', 'claude', 'mcp_qdrant', 'search', 10000, 1500, 3, 'truncate')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let (savings_bytes, savings_ratio): (i64, f64) = sqlx::query_as(
            "SELECT savings_bytes, savings_ratio FROM token_savings WHERE id = 'e1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(savings_bytes, 8500);
        assert!((savings_ratio - 0.85).abs() < 1e-9);
    }

    #[tokio::test]
    async fn token_savings_view_excludes_rows_without_bytes_in() {
        let pool = fresh_pool().await;
        setup_search_events(&pool).await;
        V38Migration.up(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, ts, actor, tool, op) \
             VALUES ('e1', '2026-05-26T12:00:00.000Z', 'claude', 'mcp_qdrant', 'search')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM token_savings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn token_savings_view_detects_followup_in_window() {
        let pool = fresh_pool().await;
        setup_search_events(&pool).await;
        V38Migration.up(&pool).await.unwrap();

        // Original search and a followup 30s later — within the 60s window
        sqlx::query(
            "INSERT INTO search_events (id, session_id, ts, actor, tool, op, bytes_in, bytes_out, shape_mode) \
             VALUES ('e1', 'sess-1', '2026-05-26T12:00:00.000Z', 'claude', 'mcp_qdrant', 'search', 5000, 1000, 'truncate')",
        )
        .execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, ts, actor, tool, op) \
             VALUES ('e2', 'sess-1', '2026-05-26T12:00:30.000Z', 'claude', 'mcp_qdrant', 'followup')",
        )
        .execute(&pool).await.unwrap();

        let had_followup: bool =
            sqlx::query_scalar("SELECT had_followup FROM token_savings WHERE id = 'e1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(had_followup);
    }

    #[tokio::test]
    async fn token_savings_view_detects_escalation_via_parent_event_id() {
        let pool = fresh_pool().await;
        setup_search_events(&pool).await;
        V38Migration.up(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, session_id, ts, actor, tool, op, bytes_in, bytes_out, shape_mode) \
             VALUES ('e1', 'sess-1', '2026-05-26T12:00:00.000Z', 'claude', 'mcp_qdrant', 'search', 5000, 1000, 'truncate')",
        )
        .execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, ts, actor, tool, op, parent_event_id) \
             VALUES ('e2', 'sess-1', '2026-05-26T12:01:00.000Z', 'claude', 'mcp_qdrant', 'open', 'e1')",
        )
        .execute(&pool).await.unwrap();

        let had_escalation: bool =
            sqlx::query_scalar("SELECT had_escalation FROM token_savings WHERE id = 'e1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(had_escalation);
    }
}
