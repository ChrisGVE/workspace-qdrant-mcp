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
//!   - Phase 4: poll until migration-scoped queue drains, then finalize
//!     (delete marker row).

use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use workspace_qdrant_core::schema_version::v37::{
    active_queue_depth, finalize_relative_path_migration, is_relative_path_migration_in_progress,
    mark_initial_walk_complete, migration_queue_depth,
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
    let (initial_pending, cutoff_ts) = wait_for_walk_settle(&pool).await?;

    mark_initial_walk_complete(&pool, initial_pending).await?;

    // Phase 4: poll until migration-scoped queue drains, then finalize.
    poll_and_finalize(&pool, &cutoff_ts).await?;

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
/// Strategy: let the walk settle for `WALK_SETTLE_DELAY`, then poll until
/// the queue depth stabilises (two consecutive reads return the same value).
///
/// Returns `(pending_count, cutoff_timestamp)` -- the cutoff is captured at
/// settle time so that `poll_and_finalize` only waits for migration-era
/// items, ignoring unrelated live traffic that arrives later.
async fn wait_for_walk_settle(
    pool: &SqlitePool,
) -> Result<(i64, String), Box<dyn std::error::Error + Send + Sync>> {
    tokio::time::sleep(WALK_SETTLE_DELAY).await;

    let mut prev: i64 = -1;
    loop {
        let depth = active_queue_depth(pool).await?;
        if depth == prev {
            // Capture the cutoff timestamp at settle time. Any items created
            // after this point are live traffic, not part of the migration.
            let cutoff_ts = wqm_common::timestamps::now_utc();
            info!(
                "relative-path migration: initial walk settled, \
                 {depth} items queued (cutoff_ts={cutoff_ts})"
            );
            return Ok((depth, cutoff_ts));
        }
        prev = depth;
        tokio::time::sleep(WALK_POLL_INTERVAL).await;
    }
}

/// Poll until migration-scoped queue items are fully drained, then finalize.
///
/// Only items created at or before `cutoff_ts` are considered. This prevents
/// live traffic from keeping the migration marker alive indefinitely.
async fn poll_and_finalize(
    pool: &SqlitePool,
    cutoff_ts: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("relative-path migration: waiting for queue to drain (cutoff={cutoff_ts})");

    let mut last_logged: i64 = -1;
    loop {
        let depth = migration_queue_depth(pool, cutoff_ts).await?;
        if depth == 0 {
            break;
        }
        if depth != last_logged {
            info!("relative-path migration: {depth} migration items remaining");
            last_logged = depth;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
    }

    finalize_relative_path_migration(pool).await?;
    info!("relative-path migration: complete — marker deleted");
    Ok(())
}
