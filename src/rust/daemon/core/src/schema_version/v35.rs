//! Migration v35: per-tenant `active_provider` column and `reembed` op value.
//!
//! Two changes (PRD §5.10):
//! 1. Add `active_provider TEXT DEFAULT 'openai_compatible'` to `watch_folders`
//!    so each tenant records the provider that produced its current vectors.
//! 2. Extend the `unified_queue.op` CHECK constraint to include `'reembed'`.
//!    SQLite cannot ALTER an existing CHECK; we rename → recreate → copy → drop.
//!
//! The recreated table DDL must mirror `CREATE_UNIFIED_QUEUE_SQL` in
//! `unified_queue_schema/sql.rs` — which is also updated to include `'reembed'`
//! so fresh installs and migrated databases agree.
//!
//! Atomicity: this migration wraps all DDL in `BEGIN IMMEDIATE` / `COMMIT`.
//! On any error the transaction is rolled back explicitly, leaving the schema
//! at the pre-v35 state. The `ForeignKeysGuard` from `super` ensures
//! `PRAGMA foreign_keys` is always restored to its original value even if a
//! panic occurs mid-migration.

use async_trait::async_trait;
use sqlx::{pool::PoolConnection, Executor, Sqlite, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::{ForeignKeysGuard, SchemaError};

pub struct V35Migration;

#[async_trait]
impl Migration for V35Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v35: adding watch_folders.active_provider and unified_queue.op='reembed'");

        // Acquire a dedicated connection so PRAGMA scope is isolated.
        let conn = pool.acquire().await?;

        // Disable FK checks before opening the transaction (SQLite requires
        // the PRAGMA to be set outside an active transaction).
        let mut guard = ForeignKeysGuard::disable(conn).await?;

        // Open the transaction in IMMEDIATE mode to acquire a write lock
        // upfront and prevent SQLITE_BUSY during the rename/recreate sequence.
        guard.conn_mut().execute("BEGIN IMMEDIATE").await?;

        let result = run_v35_body(guard.conn_mut()).await;

        match result {
            Ok(()) => {
                guard.conn_mut().execute("COMMIT").await?;
                // Restore FK checks on the success path.
                let _conn = guard.restore().await?;
                info!("Migration v35 complete");
                Ok(())
            }
            Err(e) => {
                // Best-effort rollback; ignore secondary errors.
                let _ = guard.conn_mut().execute("ROLLBACK").await;
                // Always restore FK checks, even on error.
                let _conn = guard.restore().await?;
                Err(e)
            }
        }
    }

    fn version(&self) -> i32 {
        35
    }

    fn description(&self) -> &'static str {
        "Add watch_folders.active_provider; extend unified_queue.op to include 'reembed'"
    }
}

/// Execute all v35 DDL within the already-open IMMEDIATE transaction.
/// Returns Ok(()) on success; the caller commits or rolls back.
async fn run_v35_body(conn: &mut PoolConnection<Sqlite>) -> Result<(), SchemaError> {
    migrate_watch_folders(conn).await?;

    // Test fail-point: inject a failure after watch_folders is altered but
    // before unified_queue is rebuilt.  The transaction will be rolled back
    // by the caller, verifying that partial DDL does not persist.
    #[cfg(test)]
    if INJECT_FAILURE_AFTER_WATCH_FOLDERS.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(SchemaError::MigrationError(
            "injected test failure after watch_folders ALTER".into(),
        ));
    }

    migrate_unified_queue(conn).await?;
    Ok(())
}

/// Part 1: add `active_provider` column to `watch_folders` if missing.
async fn migrate_watch_folders(conn: &mut PoolConnection<Sqlite>) -> Result<(), SchemaError> {
    let watch_folders_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='watch_folders'",
    )
    .fetch_one(&mut **conn)
    .await?;

    if !watch_folders_sql.contains("active_provider") {
        debug!("Adding active_provider column to watch_folders");
        conn.execute(
            "ALTER TABLE watch_folders ADD COLUMN active_provider TEXT \
             DEFAULT 'openai_compatible'",
        )
        .await?;
    } else {
        debug!("active_provider column already present, skipping ALTER");
    }
    Ok(())
}

/// Part 2: rebuild `unified_queue` with the extended `op` CHECK that includes
/// `'reembed'`. Uses rename → recreate → copy → drop because SQLite cannot
/// ALTER an existing CHECK constraint.
///
/// `PRAGMA foreign_keys = OFF` must already be set before calling this (done
/// by the `ForeignKeysGuard` in `up`).
async fn migrate_unified_queue(conn: &mut PoolConnection<Sqlite>) -> Result<(), SchemaError> {
    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&mut **conn)
    .await?;

    if queue_sql.contains("'reembed'") {
        info!("Migration v35: unified_queue already accepts 'reembed', no rebuild needed");
        return Ok(());
    }

    conn.execute("DROP TABLE IF EXISTS unified_queue_old")
        .await?;

    debug!("Renaming unified_queue → unified_queue_old");
    conn.execute("ALTER TABLE unified_queue RENAME TO unified_queue_old")
        .await?;

    debug!("Recreating unified_queue with extended op CHECK including 'reembed'");
    conn.execute(crate::CREATE_UNIFIED_QUEUE_SQL).await?;

    debug!("Copying rows from unified_queue_old");
    // Note: max_retries was dropped in v27 so is not present in either the
    // source (unified_queue_old) or the destination (unified_queue).
    conn.execute(
        "INSERT INTO unified_queue (\
             queue_id, item_type, op, tenant_id, collection, status, \
             created_at, updated_at, lease_until, worker_id, idempotency_key, \
             payload_json, retry_count, error_message, \
             last_error_at, branch, metadata, file_path, qdrant_status, \
             search_status, decision_json\
         ) SELECT \
             queue_id, item_type, op, tenant_id, collection, status, \
             created_at, updated_at, lease_until, worker_id, idempotency_key, \
             payload_json, retry_count, error_message, \
             last_error_at, branch, metadata, file_path, qdrant_status, \
             search_status, decision_json \
         FROM unified_queue_old",
    )
    .await?;

    conn.execute("DROP TABLE unified_queue_old").await?;

    debug!("Recreating unified_queue indexes");
    for stmt in crate::CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        conn.execute(*stmt).await?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cfg(test) fail-point for rollback testing
// ---------------------------------------------------------------------------

/// When set to `true` in tests, `migrate_unified_queue` injects a failure
/// after the `watch_folders` ALTER but before the queue rebuild completes.
/// This allows `test_v35_migration_rollback_on_error` to verify atomicity.
#[cfg(test)]
pub(crate) static INJECT_FAILURE_AFTER_WATCH_FOLDERS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
