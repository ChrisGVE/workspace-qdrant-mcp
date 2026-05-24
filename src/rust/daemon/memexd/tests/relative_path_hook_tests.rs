//! Integration tests for the relative-path migration post-startup hook
//! behavior (phases 2b-4).
//!
//! The hook lives in `memexd::relative_path_hook` (a private module in
//! `main.rs`). These tests exercise the same SQL queries and core library
//! functions that the hook depends on, validating:
//!
//! - `active_queue_depth` behavior on DB errors (unwrap_or(0) coercion).
//! - Walk-settle detection false positives (stable non-zero depth).
//! - Livelock scenario (queue never drains).
//! - Normal drain-and-finalize lifecycle.

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Executor, SqlitePool};
use std::time::Duration;

use workspace_qdrant_core::schema_version::v37::{
    finalize_relative_path_migration, is_relative_path_migration_in_progress,
    mark_initial_walk_complete,
};
use workspace_qdrant_core::SchemaManager;

/// Create a test pool with the full schema migrated through v37.
async fn create_migrated_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("create in-memory pool");

    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("full migration");

    pool
}

/// Create a minimal pool with just a unified_queue table (no full migration).
async fn create_minimal_pool_with_queue() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("create in-memory pool");

    pool.execute(
        "CREATE TABLE unified_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            branch TEXT,
            file_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
        )",
    )
    .await
    .expect("create unified_queue");

    pool
}

/// Replicates the `active_queue_depth` function from the hook module.
/// This is an exact copy of the private function so we can test it.
async fn active_queue_depth(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE status IN ('pending', 'in_progress')",
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0)
}

/// Replicates the walk-settle detection: two consecutive reads returning
/// the same value means "settled". No real sleeps needed for the unit
/// test since the queue is static.
async fn walk_settle_immediate(pool: &SqlitePool) -> i64 {
    let mut prev: i64 = -1;
    loop {
        let depth = active_queue_depth(pool).await;
        if depth == prev {
            return depth;
        }
        prev = depth;
    }
}

/// Replicates the poll-and-finalize logic without real sleeps.
/// Returns true if finalization was reached (queue drained to 0).
async fn try_poll_and_finalize(
    pool: &SqlitePool,
    max_iterations: usize,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    for _ in 0..max_iterations {
        let depth = active_queue_depth(pool).await;
        if depth == 0 {
            finalize_relative_path_migration(pool).await?;
            return Ok(true);
        }
    }
    Ok(false)
}

// ============================================================================
// active_queue_depth tests
// ============================================================================

/// `active_queue_depth` returns 0 on a DB error (e.g. the table does not
/// exist). This documents the current `unwrap_or(0)` behavior.
///
/// AUDIT NOTE: this is a known design trade-off. If the unified_queue
/// table is missing (catastrophic schema corruption), coercing to 0 causes
/// `poll_and_finalize` to immediately finalize the migration. The
/// alternative (propagating the error) would block migration completion
/// permanently. The current behavior is acceptable because:
///   - The migration marker check happens first (it would also fail).
///   - A missing unified_queue means the DB is deeply corrupt.
#[tokio::test]
async fn hook_active_queue_depth_returns_zero_on_db_error() {
    // Pool with NO unified_queue table -- queries against it will fail.
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("create in-memory pool");

    let depth = active_queue_depth(&pool).await;
    assert_eq!(
        depth, 0,
        "active_queue_depth must return 0 when the query fails (unwrap_or(0))"
    );
}

/// `active_queue_depth` correctly counts pending and in_progress items
/// and excludes other statuses.
#[tokio::test]
async fn hook_active_queue_depth_counts_only_active_statuses() {
    let pool = create_minimal_pool_with_queue().await;

    for (key, status) in [
        ("k1", "pending"),
        ("k2", "pending"),
        ("k3", "in_progress"),
        ("k4", "done"),
        ("k5", "failed"),
    ] {
        sqlx::query(
            "INSERT INTO unified_queue \
                 (item_type, op, tenant_id, collection, idempotency_key, status) \
             VALUES ('file', 'add', 't1', 'projects', ?1, ?2)",
        )
        .bind(key)
        .bind(status)
        .execute(&pool)
        .await
        .unwrap();
    }

    let depth = active_queue_depth(&pool).await;
    assert_eq!(depth, 3, "must count 2 pending + 1 in_progress = 3");
}

/// `active_queue_depth` returns 0 on an empty queue table.
#[tokio::test]
async fn hook_active_queue_depth_returns_zero_on_empty_queue() {
    let pool = create_minimal_pool_with_queue().await;
    let depth = active_queue_depth(&pool).await;
    assert_eq!(depth, 0);
}

// ============================================================================
// walk_settle detection tests
// ============================================================================

/// Walk-settle returns 0 when the queue is empty from the start.
#[tokio::test]
async fn hook_walk_settle_returns_zero_on_empty_queue() {
    let pool = create_migrated_pool().await;
    let result = walk_settle_immediate(&pool).await;
    assert_eq!(result, 0, "empty queue must settle at 0");
}

/// Walk-settle detects a stable non-zero queue depth as settled.
///
/// AUDIT NOTE: This documents the known false-positive behavior. If the
/// queue depth happens to be the same on two consecutive polls (walk is
/// still running but momentarily paused), the function declares settlement.
/// In production, the WALK_SETTLE_DELAY (15s) makes this unlikely.
#[tokio::test]
async fn hook_walk_settle_false_positive_on_stable_nonzero_depth() {
    let pool = create_migrated_pool().await;

    // Pre-populate queue with 5 items with unique file_paths to avoid
    // UNIQUE constraint violations on the real schema.
    for i in 0..5 {
        sqlx::query(
            "INSERT INTO unified_queue \
                 (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
             VALUES ('file', 'add', 't1', 'projects', ?1, 'main', ?2)",
        )
        .bind(format!("settle-{i}"))
        .bind(format!("src/file_{i}.rs"))
        .execute(&pool)
        .await
        .unwrap();
    }

    let result = walk_settle_immediate(&pool).await;
    assert_eq!(result, 5, "must settle at 5 (two consecutive reads match)");
}

// ============================================================================
// poll_and_finalize tests
// ============================================================================

/// Finalization completes when the queue is empty.
#[tokio::test]
async fn hook_poll_and_finalize_completes_on_empty_queue() {
    let pool = create_migrated_pool().await;

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    mark_initial_walk_complete(&pool, 0).await.unwrap();

    let finalized = try_poll_and_finalize(&pool, 1)
        .await
        .expect("must not error on empty queue");

    assert!(finalized, "must finalize when queue is empty");
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// When the queue is non-empty and never drains, finalization is never
/// reached within the iteration budget.
///
/// AUDIT NOTE: In production, unrelated traffic can keep the queue
/// non-empty indefinitely. The migration marker remaining in place is
/// safe (idempotent retry). A true livelock requires ingest rate >
/// processing rate indefinitely, which is unlikely.
#[tokio::test]
async fn hook_poll_and_finalize_livelock_does_not_finalize() {
    let pool = create_migrated_pool().await;
    mark_initial_walk_complete(&pool, 1).await.unwrap();

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't1', 'projects', 'livelock-1', 'main', 'src/stuck.rs')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Give the poll loop 10 iterations -- the queue never drains.
    let finalized = try_poll_and_finalize(&pool, 10)
        .await
        .expect("must not error");

    assert!(
        !finalized,
        "poll_and_finalize must NOT complete when queue never drains"
    );
    assert!(
        is_relative_path_migration_in_progress(&pool).await.unwrap(),
        "migration marker must remain when finalize was not reached"
    );
}
