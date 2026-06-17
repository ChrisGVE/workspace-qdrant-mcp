//! Migration v47: relax the `search_events.actor` CHECK to admit `'benchmark'`
//! (#135 search-quality eval).
//!
//! The search-quality eval harness (`wqm benchmark search-quality`) issues real
//! searches through the live pipeline and tags each one with `actor =
//! 'benchmark'`, so that organic-query mining can exclude its own traffic:
//!
//! ```sql
//! SELECT query_text, COUNT(*) FROM search_events
//! WHERE project_id = ? AND op = 'search' AND actor != 'benchmark'
//! GROUP BY query_text ORDER BY MAX(ts) DESC
//! ```
//!
//! The original `actor` CHECK (migration v12) only listed
//! `('claude', 'user', 'daemon')`, so a `'benchmark'` insert would be rejected.
//! SQLite cannot ALTER a CHECK constraint in place, so the relaxation requires a
//! table rebuild: rename the old table aside, recreate it from the canonical
//! `CREATE_SEARCH_EVENTS_SQL` (which now carries the relaxed CHECK), copy every
//! row across, and drop the old table. See the SQLite "Making Other Kinds Of
//! Table Schema Changes" procedure: <https://www.sqlite.org/lang_altertable.html>.
//!
//! `search_events` is a standalone instrumentation log — no other table holds a
//! foreign key into it — so no `ForeignKeysGuard` / `legacy_alter_table` dance is
//! needed (contrast v40, which rebuilt an FK-referenced table). The rebuild runs
//! inside an explicit `BEGIN IMMEDIATE` / `COMMIT` (v35+ convention) so it is
//! atomic, and an idempotency probe (does the live CHECK already admit
//! `'benchmark'`?) makes a re-run after an up()-ran-but-unrecorded crash a no-op.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info, warn};

use super::migration::Migration;
use super::SchemaError;
use crate::search_events_schema::{CREATE_SEARCH_EVENTS_INDEXES_SQL, CREATE_SEARCH_EVENTS_SQL};

pub struct V47Migration;

#[async_trait]
impl Migration for V47Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v47: relax search_events.actor CHECK to admit 'benchmark'");

        let mut conn = pool.acquire().await?;

        // IMMEDIATE mode takes the write lock upfront so the rename/recreate/copy
        // sequence never races a concurrent writer with SQLITE_BUSY.
        conn.execute("BEGIN IMMEDIATE").await?;
        let result = rebuild_search_events(&mut conn).await;
        match result {
            Ok(()) => {
                conn.execute("COMMIT").await?;
                debug!("Migration v47: search_events.actor CHECK now admits 'benchmark'");
                Ok(())
            }
            Err(e) => {
                // Best-effort rollback; surface the original error.
                let _ = conn.execute("ROLLBACK").await;
                Err(e)
            }
        }
    }

    fn version(&self) -> i32 {
        47
    }

    fn description(&self) -> &'static str {
        "Relax search_events.actor CHECK to admit 'benchmark' (#135 eval harness)"
    }
}

/// Rebuild `search_events` so its `actor` CHECK admits `'benchmark'`.
///
/// Cases handled (mirrors the idempotency/crash-recovery discipline of v40):
/// 1. `search_events` already admits `'benchmark'` — already migrated, no-op.
/// 2. `search_events` missing but `search_events_old` present — crash recovery
///    from an interrupted rebuild: recreate, copy, drop old.
/// 3. `search_events` missing entirely (DB never reached v12) — nothing to do.
/// 4. Normal case — rename aside, recreate with the relaxed CHECK, copy, drop.
async fn rebuild_search_events(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    let table_exists = object_exists(conn, "table", "search_events").await?;
    let old_exists = object_exists(conn, "table", "search_events_old").await?;

    // Case 2: crash recovery — old table survived but the new one is gone.
    if !table_exists && old_exists {
        warn!(
            "Migration v47: search_events missing but search_events_old exists — \
             recovering from an interrupted rebuild"
        );
        recreate_and_copy(conn).await?;
        return Ok(());
    }

    // Case 3: search_events never existed (DB predates v12) — nothing to rebuild.
    if !table_exists {
        debug!("Migration v47: search_events does not exist; skipping rebuild");
        return Ok(());
    }

    // Case 1: idempotency — the live CHECK already lists 'benchmark'.
    let live_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='search_events'",
    )
    .fetch_one(&mut **conn)
    .await?;
    if live_sql.contains("'benchmark'") {
        debug!("Migration v47: search_events.actor CHECK already admits 'benchmark'");
        if old_exists {
            conn.execute("DROP TABLE search_events_old").await?;
        }
        return Ok(());
    }

    // Case 4: normal rebuild — rename, recreate with the relaxed CHECK, copy, drop.
    conn.execute("DROP TABLE IF EXISTS search_events_old")
        .await?;
    conn.execute("ALTER TABLE search_events RENAME TO search_events_old")
        .await?;
    recreate_and_copy(conn).await?;

    debug!("Migration v47: search_events rebuilt with relaxed actor CHECK");
    Ok(())
}

/// Recreate `search_events` from the canonical schema, copy every column across
/// from `search_events_old`, rebuild the indexes, and drop the old table.
///
/// The explicit column list keeps the copy stable against any future column
/// reordering and is exhaustive over the v12 schema (the only columns present
/// on any database that needs this migration).
async fn recreate_and_copy(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
) -> Result<(), SchemaError> {
    conn.execute(CREATE_SEARCH_EVENTS_SQL).await?;

    const COLUMNS: &str = "id, ts, session_id, project_id, actor, tool, op, \
         query_text, filters, top_k, result_count, latency_ms, \
         top_result_refs, outcome, parent_event_id, created_at";
    let copy_sql =
        format!("INSERT INTO search_events ({COLUMNS}) SELECT {COLUMNS} FROM search_events_old");
    conn.execute(copy_sql.as_str()).await?;

    for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
        conn.execute(*index_sql).await?;
    }

    conn.execute("DROP TABLE search_events_old").await?;
    Ok(())
}

/// Does a `sqlite_master` object of the given type and name exist?
async fn object_exists(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
    object_type: &str,
    name: &str,
) -> Result<bool, SchemaError> {
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?)",
    )
    .bind(object_type)
    .bind(name)
    .fetch_one(&mut **conn)
    .await?;
    Ok(exists)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    /// A pool migrated all the way to HEAD (includes v47).
    async fn migrated_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    /// The live `search_events` table SQL from `sqlite_master`.
    async fn table_sql(pool: &SqlitePool) -> String {
        sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='search_events'",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    async fn insert_actor(pool: &SqlitePool, id: &str, actor: &str) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO search_events (id, actor, tool, op) VALUES (?, ?, 'mcp_qdrant', 'search')",
        )
        .bind(id)
        .bind(actor)
        .execute(pool)
        .await
        .map(|_| ())
    }

    #[tokio::test]
    async fn test_v47_check_admits_benchmark() {
        let pool = migrated_pool().await;
        assert!(
            table_sql(&pool).await.contains("'benchmark'"),
            "actor CHECK must list 'benchmark' after v47"
        );
        insert_actor(&pool, "evt-bench", "benchmark")
            .await
            .expect("actor='benchmark' must be accepted after v47");
    }

    #[tokio::test]
    async fn test_v47_still_accepts_organic_actors() {
        let pool = migrated_pool().await;
        for (id, actor) in [
            ("evt-claude", "claude"),
            ("evt-user", "user"),
            ("evt-daemon", "daemon"),
        ] {
            insert_actor(&pool, id, actor)
                .await
                .unwrap_or_else(|_| panic!("organic actor {actor} must still be accepted"));
        }
    }

    #[tokio::test]
    async fn test_v47_rejects_unknown_actor() {
        let pool = migrated_pool().await;
        let bad = insert_actor(&pool, "evt-bad", "robot").await;
        assert!(bad.is_err(), "an unlisted actor must still be rejected");
    }

    #[tokio::test]
    async fn test_v47_preserves_existing_rows() {
        // Migrate only to v46 (state before v47), seed a row with the old CHECK,
        // then apply v47 and confirm the row survives the rebuild.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let mgr = SchemaManager::new(pool.clone());
        for v in 1..=46 {
            mgr.run_migration(v).await.unwrap();
        }
        insert_actor(&pool, "pre-v47", "claude").await.unwrap();

        mgr.run_migration(47).await.unwrap();

        let actor: String =
            sqlx::query_scalar("SELECT actor FROM search_events WHERE id = 'pre-v47'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(actor, "claude", "pre-v47 rows survive the table rebuild");
        // And the relaxed CHECK is now in effect.
        insert_actor(&pool, "post-v47", "benchmark").await.unwrap();
    }

    #[tokio::test]
    async fn test_v47_is_idempotent_on_rerun() {
        // The up()-ran-but-unrecorded crash window: re-running v47 when the CHECK
        // already admits 'benchmark' must be a no-op, not an error, and must not
        // drop any rows.
        let pool = migrated_pool().await;
        insert_actor(&pool, "keep-me", "benchmark").await.unwrap();

        let mgr = SchemaManager::new(pool.clone());
        mgr.run_migration(47)
            .await
            .expect("v47 re-run must be idempotent");

        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM search_events WHERE id = 'keep-me'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 1, "idempotent re-run must not drop rows");
        assert!(table_sql(&pool).await.contains("'benchmark'"));
    }

    #[tokio::test]
    async fn test_v47_indexes_present_after_rebuild() {
        let pool = migrated_pool().await;
        for index in [
            "idx_search_events_session",
            "idx_search_events_tool",
            "idx_search_events_project",
        ] {
            let present: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?)",
            )
            .bind(index)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(present, "{index} must be recreated after the v47 rebuild");
        }
    }
}
