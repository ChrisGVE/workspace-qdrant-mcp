//! Post-startup hook for the relative-path migration (spec §6.2, phases 2b–4).
//!
//! Phases 1–2 (SQLite side) run inside the schema migration system.
//! This hook runs **after** file watchers have started (Phase 6b) so the
//! initial walk is already enqueuing files by the time we get here.
//!
//! Responsibilities:
//!   - Phase 2b: truncate Qdrant ingest collections (retry on unreachable).
//!   - Phase 3: implicit — `start_all_watches()` triggers the initial walk.
//!   - Phase 3→4 bridge: wait for queue to stabilise, snapshot
//!     `initial_pending_count`, mark walk complete.
//!   - Phase 4: poll until queue drains, then finalize (delete marker row).

use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use workspace_qdrant_core::schema_version::v37::{
    finalize_relative_path_migration, is_relative_path_migration_in_progress,
    mark_initial_walk_complete,
};
use workspace_qdrant_core::StorageClient;
use wqm_common::constants::{COLLECTION_IMAGES, COLLECTION_LIBRARIES, COLLECTION_PROJECTS};

const QDRANT_RETRY_INTERVAL: Duration = Duration::from_secs(60);
const WALK_SETTLE_DELAY: Duration = Duration::from_secs(15);
const WALK_POLL_INTERVAL: Duration = Duration::from_secs(3);
const DRAIN_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub fn spawn_if_needed(pool: SqlitePool, storage: Arc<StorageClient>) -> Option<JoinHandle<()>> {
    Some(tokio::spawn(async move {
        if let Err(e) = run(pool, storage).await {
            error!("relative-path migration hook failed: {e}");
        }
    }))
}

async fn run(
    pool: SqlitePool,
    storage: Arc<StorageClient>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match is_relative_path_migration_in_progress(&pool).await {
        Ok(false) => return Ok(()),
        Err(e) => {
            warn!("Could not check migration marker (non-fatal): {e}");
            return Ok(());
        }
        Ok(true) => {}
    }

    info!("relative-path migration: post-startup hook starting (phases 2b → 4)");

    // Phase 2b: truncate Qdrant ingest collections.
    // Rules and scratchpad are user data — only truncate ingest-derived
    // collections (projects, libraries, images).
    truncate_qdrant_collections(&storage).await;

    // Phase 3 is implicit — watchers already started their initial walk.
    // Wait for the walk to settle so the pending count is meaningful.
    let initial_pending = wait_for_walk_settle(&pool, WALK_SETTLE_DELAY, WALK_POLL_INTERVAL).await;

    mark_initial_walk_complete(&pool, initial_pending).await?;

    // Phase 4: poll until queue drains, then finalize.
    poll_and_finalize(&pool, DRAIN_POLL_INTERVAL).await?;

    Ok(())
}

async fn truncate_qdrant_collections(storage: &StorageClient) {
    let collections = [COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_IMAGES];

    for name in &collections {
        loop {
            match storage.collection_exists(name).await {
                Ok(true) => match storage.delete_collection(name).await {
                    Ok(()) => {
                        info!("relative-path migration: truncated Qdrant collection '{name}'");
                        break;
                    }
                    Err(e) => {
                        warn!(
                            "relative-path migration: failed to delete '{name}', \
                             retrying in {}s: {e}",
                            QDRANT_RETRY_INTERVAL.as_secs()
                        );
                        tokio::time::sleep(QDRANT_RETRY_INTERVAL).await;
                    }
                },
                Ok(false) => {
                    info!("relative-path migration: collection '{name}' already absent");
                    break;
                }
                Err(e) => {
                    warn!(
                        "relative-path migration: Qdrant unreachable, \
                         retrying in {}s: {e}",
                        QDRANT_RETRY_INTERVAL.as_secs()
                    );
                    tokio::time::sleep(QDRANT_RETRY_INTERVAL).await;
                }
            }
        }
    }

    // Recreate the collections so the queue processor can write into them.
    loop {
        match storage.initialize_multi_tenant_collections(None).await {
            Ok(result) => {
                info!(
                    "relative-path migration: Qdrant collections recreated ({:?})",
                    result
                );
                break;
            }
            Err(e) => {
                warn!(
                    "relative-path migration: failed to recreate collections, \
                     retrying in {}s: {e}",
                    QDRANT_RETRY_INTERVAL.as_secs()
                );
                tokio::time::sleep(QDRANT_RETRY_INTERVAL).await;
            }
        }
    }
}

/// Wait for the initial walk to finish enqueuing files.
///
/// Strategy: let the walk settle for `settle_delay`, then poll every
/// `poll_interval` until the queue depth stabilises (two consecutive reads
/// return the same value). Durations are parameterised so tests can run
/// without paying the production cadence.
async fn wait_for_walk_settle(
    pool: &SqlitePool,
    settle_delay: Duration,
    poll_interval: Duration,
) -> i64 {
    tokio::time::sleep(settle_delay).await;

    let mut prev: i64 = -1;
    loop {
        let depth = active_queue_depth(pool).await;
        if depth == prev {
            info!("relative-path migration: initial walk settled, {depth} items queued");
            return depth;
        }
        prev = depth;
        tokio::time::sleep(poll_interval).await;
    }
}

/// Poll until the queue is fully drained, then finalize.
///
/// `drain_poll_interval` is parameterised so tests can run quickly without
/// changing production timing.
async fn poll_and_finalize(
    pool: &SqlitePool,
    drain_poll_interval: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("relative-path migration: waiting for queue to drain");

    let mut last_logged: i64 = -1;
    loop {
        let depth = active_queue_depth(pool).await;
        if depth == 0 {
            break;
        }
        if depth != last_logged {
            info!("relative-path migration: {depth} items remaining");
            last_logged = depth;
        }
        tokio::time::sleep(drain_poll_interval).await;
    }

    finalize_relative_path_migration(pool).await?;
    info!("relative-path migration: complete — marker deleted");
    Ok(())
}

/// Count of pending + in_progress items in the unified queue.
async fn active_queue_depth(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE status IN ('pending', 'in_progress')",
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use workspace_qdrant_core::schema_version::v37::{
        get_relative_path_migration_status, CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL,
    };
    use workspace_qdrant_core::CREATE_UNIFIED_QUEUE_SQL;

    /// Build an in-memory pool with the schemas the hook touches.
    async fn fresh_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        pool
    }

    /// Insert a marker row claiming a v37 migration is in flight.
    async fn seed_marker(pool: &SqlitePool) {
        sqlx::query(
            "INSERT INTO relative_path_migration_in_progress \
             (target_version, started_at, initial_walk_complete) VALUES (37, 0, 0)",
        )
        .execute(pool)
        .await
        .unwrap();
    }

    /// Insert a queue row with the given status. `key` makes idempotency_key
    /// unique across calls.
    async fn enqueue(pool: &SqlitePool, key: &str, status: &str) {
        sqlx::query(
            "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, status, idempotency_key) \
             VALUES ('file', 'add', 't1', 'projects', ?1, ?2)",
        )
        .bind(status)
        .bind(key)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn count_marker_rows(pool: &SqlitePool) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM relative_path_migration_in_progress")
            .fetch_one(pool)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn active_queue_depth_empty_returns_zero() {
        let pool = fresh_pool().await;
        assert_eq!(active_queue_depth(&pool).await, 0);
    }

    #[tokio::test]
    async fn active_queue_depth_counts_pending_and_in_progress() {
        let pool = fresh_pool().await;
        enqueue(&pool, "k1", "pending").await;
        enqueue(&pool, "k2", "pending").await;
        enqueue(&pool, "k3", "in_progress").await;

        assert_eq!(active_queue_depth(&pool).await, 3);
    }

    #[tokio::test]
    async fn active_queue_depth_excludes_terminal_statuses() {
        let pool = fresh_pool().await;
        enqueue(&pool, "k_pending", "pending").await;
        enqueue(&pool, "k_in_progress", "in_progress").await;
        enqueue(&pool, "k_done", "done").await;
        enqueue(&pool, "k_failed", "failed").await;

        // Only pending + in_progress count.
        assert_eq!(active_queue_depth(&pool).await, 2);
    }

    #[tokio::test]
    async fn active_queue_depth_returns_zero_when_table_missing() {
        // Hook is defensive: missing table must not panic — it falls back
        // to 0 so the post-startup loop can keep running while another
        // migration races to finish.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        assert_eq!(active_queue_depth(&pool).await, 0);
    }

    #[tokio::test]
    async fn wait_for_walk_settle_returns_zero_for_empty_queue() {
        let pool = fresh_pool().await;
        // Empty queue is already settled: two reads in a row both yield 0
        // (after seeing prev = -1, the second loop iteration matches).
        let result =
            wait_for_walk_settle(&pool, Duration::from_millis(1), Duration::from_millis(1)).await;
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn wait_for_walk_settle_returns_stable_depth() {
        let pool = fresh_pool().await;
        enqueue(&pool, "k1", "pending").await;
        enqueue(&pool, "k2", "pending").await;
        enqueue(&pool, "k3", "in_progress").await;

        // Queue is static, so the first two reads (after settle_delay) both
        // see depth=3 and the loop returns it.
        let result =
            wait_for_walk_settle(&pool, Duration::from_millis(1), Duration::from_millis(1)).await;
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn wait_for_walk_settle_observes_growth_then_stabilizes() {
        // Timing budget (generous margins to avoid CI flakiness):
        //   settle_delay = 10ms, poll_interval = 80ms.
        //   Driver inserts later1 at ~t=30ms (within poll window 1) and
        //   later2 at ~t=110ms (within poll window 2). The function should
        //   observe depth 1 → 2 → 3 → 3 and return.
        let pool = fresh_pool().await;
        enqueue(&pool, "seed", "pending").await;

        let pool_for_walk = pool.clone();
        let handle = tokio::spawn(async move {
            wait_for_walk_settle(
                &pool_for_walk,
                Duration::from_millis(10),
                Duration::from_millis(80),
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        enqueue(&pool, "later1", "pending").await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        enqueue(&pool, "later2", "pending").await;

        let depth = handle.await.unwrap();
        assert_eq!(
            depth, 3,
            "function must return only after two consecutive reads match"
        );
    }

    #[tokio::test]
    async fn poll_and_finalize_drains_empty_queue_and_deletes_marker() {
        let pool = fresh_pool().await;
        seed_marker(&pool).await;
        assert_eq!(count_marker_rows(&pool).await, 1);

        poll_and_finalize(&pool, Duration::from_millis(1))
            .await
            .unwrap();

        assert_eq!(count_marker_rows(&pool).await, 0);
        assert!(
            !is_relative_path_migration_in_progress(&pool).await.unwrap(),
            "marker deletion is the 'migration done' signal"
        );
    }

    #[tokio::test]
    async fn poll_and_finalize_waits_for_drain_before_finalizing() {
        let pool = fresh_pool().await;
        seed_marker(&pool).await;
        enqueue(&pool, "pending1", "pending").await;
        enqueue(&pool, "pending2", "in_progress").await;

        let pool_for_drain = pool.clone();
        let handle = tokio::spawn(async move {
            poll_and_finalize(&pool_for_drain, Duration::from_millis(5)).await
        });

        // Wait long enough that the loop has polled at least once with
        // depth > 0 — proves the function actually waits and doesn't
        // short-circuit on a still-populated queue.
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(count_marker_rows(&pool).await, 1, "marker survives drain");

        // Drain the queue. Setting items to 'done' makes
        // active_queue_depth return 0 on the next poll.
        sqlx::query("UPDATE unified_queue SET status = 'done'")
            .execute(&pool)
            .await
            .unwrap();

        handle.await.unwrap().unwrap();
        assert_eq!(count_marker_rows(&pool).await, 0);
    }

    #[tokio::test]
    async fn poll_and_finalize_finalize_runs_even_with_no_marker_row() {
        // No marker row inserted — finalize_relative_path_migration deletes
        // zero rows but still commits successfully. The hook's overall
        // contract treats this as a no-op finalize; verify it doesn't error.
        let pool = fresh_pool().await;
        let result = poll_and_finalize(&pool, Duration::from_millis(1)).await;
        assert!(result.is_ok(), "no-op finalize must succeed: {result:?}");
    }

    #[tokio::test]
    async fn run_short_circuits_when_migration_not_in_progress() {
        // No marker row → the hook must not touch the queue.
        let pool = fresh_pool().await;
        // Pretend a stale item sits in the queue. If `run` proceeded
        // through phase 4 it would block forever waiting for this to drain;
        // since it must short-circuit, the test completes immediately.
        enqueue(&pool, "stale", "pending").await;

        // We exercise the public guard directly rather than `run` because
        // `run` requires a live StorageClient. The guard is the same
        // condition `run` checks first.
        let in_progress = is_relative_path_migration_in_progress(&pool).await.unwrap();
        assert!(!in_progress);

        // Marker still doesn't exist (table is empty).
        assert_eq!(count_marker_rows(&pool).await, 0);
        assert!(get_relative_path_migration_status(&pool)
            .await
            .unwrap()
            .is_none());
    }
}
