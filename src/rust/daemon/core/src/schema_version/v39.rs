//! Migration v39: Relax `op` CHECK on search_events.
//!
//! Pre-v39 the `op` column was constrained to
//! `('search', 'expand', 'open', 'followup')`. That set was incomplete:
//!
//!   - `search-exact.ts` was already writing `op = 'search_exact'`,
//!     silently failing the fire-and-forget INSERT (the daemon side
//!     swallows the constraint violation, so the events never landed).
//!   - The token-economy spec extension (`docs/specs/20-token-economy-instrumentation.md`)
//!     prescribes new ops `'grep'`, `'retrieve'`, and `'list'` for the
//!     remaining MCP tools.
//!
//! SQLite cannot `ALTER` a CHECK constraint in place, so this migration
//! rebuilds the table: rename old → create new with the expanded CHECK
//! → backfill → drop old → rename → recreate indexes and the
//! `token_savings` view.
//!
//! Idempotent: skips the rebuild when the new CHECK is already present
//! (e.g. on a fresh DB where v12+v38 created the table with the
//! up-to-date constant from the start).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V39Migration;

/// The expanded CHECK clause. Must match `search_events_schema::CREATE_SEARCH_EVENTS_SQL`.
const NEW_OP_CHECK: &str =
    "op IN ('search', 'expand', 'open', 'followup', 'grep', 'retrieve', 'list', 'search_exact')";

#[async_trait]
impl Migration for V39Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v39: Relaxing op CHECK on search_events");

        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_events')",
        )
        .fetch_one(pool)
        .await?;
        if !table_exists {
            // Pre-v12 — nothing to relax. v12 will create the table with
            // the up-to-date constant when it runs.
            debug!("Migration v39: search_events does not exist; nothing to do");
            return Ok(());
        }

        let current_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='search_events'",
        )
        .fetch_one(pool)
        .await?;

        // Idempotency check: we look for `'search_exact'` because that
        // literal appears ONLY in the relaxed `op` CHECK. The bare token
        // `'grep'` would false-positive — `'grep'` is already part of
        // the pre-v39 `tool` CHECK (`tool IN ('mcp_qdrant', 'rg', 'grep',
        // ...)`). `'search_exact'` is unique to the new op CHECK.
        if current_sql.contains("'search_exact'") {
            debug!("Migration v39: op CHECK already relaxed; skipping rebuild");
            return Ok(());
        }

        // Execute the rebuild as a single batch on the pool. Using
        // pool.execute / sqlx::query(...).execute(pool) per statement
        // avoids a subtle in-memory-DB quirk where a `PoolConnection`
        // acquired separately can see stale schema state across DDL
        // boundaries on `sqlite::memory:` test databases.
        sqlx::query("DROP VIEW IF EXISTS token_savings")
            .execute(pool)
            .await?;
        sqlx::query("PRAGMA foreign_keys = OFF")
            .execute(pool)
            .await?;
        sqlx::query("PRAGMA legacy_alter_table = ON")
            .execute(pool)
            .await?;
        sqlx::query("DROP TABLE IF EXISTS search_events_v39_old")
            .execute(pool)
            .await?;
        sqlx::query("ALTER TABLE search_events RENAME TO search_events_v39_old")
            .execute(pool)
            .await?;
        sqlx::query("PRAGMA legacy_alter_table = OFF")
            .execute(pool)
            .await?;

        // Recreate with the up-to-date schema (includes v38 columns).
        // Inline the CREATE here rather than reusing
        // search_events_schema::CREATE_SEARCH_EVENTS_SQL — that constant
        // uses `IF NOT EXISTS`, which is fragile against a still-cached
        // schema view after the rename. Spelling out a plain CREATE
        // forces SQLite to evaluate the new DDL fresh.
        sqlx::query(
            r#"CREATE TABLE search_events (
                id TEXT PRIMARY KEY NOT NULL,
                ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                session_id TEXT,
                project_id TEXT,
                actor TEXT NOT NULL CHECK (actor IN ('claude', 'user', 'daemon')),
                tool TEXT NOT NULL CHECK (tool IN ('mcp_qdrant', 'rg', 'grep', 'ctags', 'lsp', 'filesearch')),
                op TEXT NOT NULL CHECK (op IN ('search', 'expand', 'open', 'followup', 'grep', 'retrieve', 'list', 'search_exact')),
                query_text TEXT,
                filters TEXT,
                top_k INTEGER,
                result_count INTEGER,
                latency_ms INTEGER,
                top_result_refs TEXT,
                outcome TEXT,
                parent_event_id TEXT,
                bytes_in INTEGER,
                bytes_out INTEGER,
                hits_truncated INTEGER,
                shape_mode TEXT,
                tool_version TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )"#,
        )
        .execute(pool)
        .await?;

        // Backfill. The column list is explicit so a future column
        // addition that hasn't been wired into this migration can't
        // silently scramble positions.
        sqlx::query(
            "INSERT INTO search_events ( \
                id, ts, session_id, project_id, actor, tool, op, \
                query_text, filters, top_k, result_count, latency_ms, \
                top_result_refs, outcome, parent_event_id, \
                bytes_in, bytes_out, hits_truncated, shape_mode, tool_version, \
                created_at \
            ) SELECT \
                id, ts, session_id, project_id, actor, tool, op, \
                query_text, filters, top_k, result_count, latency_ms, \
                top_result_refs, outcome, parent_event_id, \
                bytes_in, bytes_out, hits_truncated, shape_mode, tool_version, \
                created_at \
            FROM search_events_v39_old",
        )
        .execute(pool)
        .await?;

        sqlx::query("DROP TABLE search_events_v39_old")
            .execute(pool)
            .await?;

        // Recreate v12 indexes.
        use crate::search_events_schema::CREATE_SEARCH_EVENTS_INDEXES_SQL;
        for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }
        // Recreate v38 index + view.
        use crate::schema_version::v38::{
            CREATE_SESSION_TOOL_TS_INDEX_SQL, CREATE_TOKEN_SAVINGS_VIEW_SQL,
        };
        sqlx::query(CREATE_SESSION_TOOL_TS_INDEX_SQL)
            .execute(pool)
            .await?;
        sqlx::query(CREATE_TOKEN_SAVINGS_VIEW_SQL)
            .execute(pool)
            .await?;

        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool)
            .await?;

        // Sanity: the new table must include 'grep' in its CHECK clause.
        let new_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='search_events'",
        )
        .fetch_one(pool)
        .await?;
        debug_assert!(
            new_sql.contains("'grep'"),
            "v39 rebuild left CHECK unchanged"
        );
        debug_assert!(new_sql.contains(NEW_OP_CHECK) || new_sql.contains("'search_exact'"));

        info!("Migration v39 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        39
    }

    fn description(&self) -> &'static str {
        "Relax op CHECK on search_events to include grep/retrieve/list/search_exact"
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

    /// Build a DB at the pre-v39 schema state: v12 table with the old
    /// CHECK clause, plus v38 columns + index + view applied on top.
    /// Mirrors a real upgrade path from v37 → v39.
    async fn setup_pre_v39(pool: &SqlitePool) {
        // Manually CREATE with the OLD CHECK so we can verify v39 rebuilds it.
        sqlx::query(
            r#"CREATE TABLE search_events (
                id TEXT PRIMARY KEY NOT NULL,
                ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                session_id TEXT,
                project_id TEXT,
                actor TEXT NOT NULL CHECK (actor IN ('claude', 'user', 'daemon')),
                tool TEXT NOT NULL CHECK (tool IN ('mcp_qdrant', 'rg', 'grep', 'ctags', 'lsp', 'filesearch')),
                op TEXT NOT NULL CHECK (op IN ('search', 'expand', 'open', 'followup')),
                query_text TEXT,
                filters TEXT,
                top_k INTEGER,
                result_count INTEGER,
                latency_ms INTEGER,
                top_result_refs TEXT,
                outcome TEXT,
                parent_event_id TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )"#,
        )
        .execute(pool)
        .await
        .unwrap();
        // v38 ALTERs
        for alter in [
            "ALTER TABLE search_events ADD COLUMN bytes_in INTEGER",
            "ALTER TABLE search_events ADD COLUMN bytes_out INTEGER",
            "ALTER TABLE search_events ADD COLUMN hits_truncated INTEGER",
            "ALTER TABLE search_events ADD COLUMN shape_mode TEXT",
            "ALTER TABLE search_events ADD COLUMN tool_version TEXT",
        ] {
            sqlx::query(alter).execute(pool).await.unwrap();
        }
    }

    #[tokio::test]
    async fn v39_rebuilds_table_to_accept_new_op_values() {
        let pool = fresh_pool().await;
        setup_pre_v39(&pool).await;

        // Pre-v39: 'grep' is rejected.
        let pre = sqlx::query(
            "INSERT INTO search_events (id, actor, tool, op, ts) \
             VALUES ('x', 'claude', 'mcp_qdrant', 'grep', '2026-05-26T00:00:00.000Z')",
        )
        .execute(&pool)
        .await;
        assert!(pre.is_err(), "pre-v39 CHECK should reject 'grep'");

        // Apply v39.
        V39Migration.up(&pool).await.unwrap();

        // Post-v39: 'grep', 'retrieve', 'list', 'search_exact' all accepted.
        for op in ["grep", "retrieve", "list", "search_exact"] {
            sqlx::query(
                "INSERT INTO search_events (id, actor, tool, op, ts) \
                 VALUES (?1, 'claude', 'mcp_qdrant', ?2, '2026-05-26T00:00:00.000Z')",
            )
            .bind(format!("evt-{}", op))
            .bind(op)
            .execute(&pool)
            .await
            .unwrap_or_else(|e| panic!("post-v39 should accept op='{}': {}", op, e));
        }
    }

    #[tokio::test]
    async fn v39_preserves_existing_rows() {
        let pool = fresh_pool().await;
        setup_pre_v39(&pool).await;

        // Seed with rows that pass the OLD CHECK.
        sqlx::query(
            "INSERT INTO search_events \
              (id, session_id, project_id, actor, tool, op, query_text, result_count, latency_ms, bytes_in, bytes_out, shape_mode, ts) \
             VALUES \
              ('e1', 'sess', 'proj-a', 'claude', 'mcp_qdrant', 'search', 'q', 5, 42, 10000, 1500, 'truncate', '2026-05-26T00:00:00.000Z')",
        )
        .execute(&pool).await.unwrap();

        V39Migration.up(&pool).await.unwrap();

        let row: (String, String, i64, i64, i64) = sqlx::query_as(
            "SELECT id, op, result_count, bytes_in, bytes_out FROM search_events WHERE id = 'e1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.0, "e1");
        assert_eq!(row.1, "search");
        assert_eq!(row.2, 5);
        assert_eq!(row.3, 10_000);
        assert_eq!(row.4, 1_500);
    }

    #[tokio::test]
    async fn v39_recreates_token_savings_view() {
        let pool = fresh_pool().await;
        setup_pre_v39(&pool).await;

        // Create the v38 view so we can verify v39 recreates it (it must
        // be dropped during rebuild to free the dependency on the table).
        use crate::schema_version::v38::CREATE_TOKEN_SAVINGS_VIEW_SQL;
        sqlx::query(CREATE_TOKEN_SAVINGS_VIEW_SQL)
            .execute(&pool)
            .await
            .unwrap();

        V39Migration.up(&pool).await.unwrap();

        let view_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='view' AND name='token_savings')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(view_exists, "v39 must recreate the token_savings view");
    }

    #[tokio::test]
    async fn v39_is_idempotent_on_fresh_db_with_new_check() {
        let pool = fresh_pool().await;
        // Fresh-DB path: v12 (with the updated constant) creates the
        // table already at the new CHECK. v39's idempotency check should
        // short-circuit without rebuilding.
        sqlx::query(crate::search_events_schema::CREATE_SEARCH_EVENTS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for index_sql in crate::search_events_schema::CREATE_SEARCH_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql).execute(&pool).await.unwrap();
        }

        V39Migration.up(&pool).await.unwrap();

        // Re-running is also a no-op.
        V39Migration.up(&pool).await.unwrap();

        // Sanity: still accepts the new ops.
        sqlx::query(
            "INSERT INTO search_events (id, actor, tool, op, ts) \
             VALUES ('y', 'claude', 'mcp_qdrant', 'grep', '2026-05-26T00:00:00.000Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
    }
}
