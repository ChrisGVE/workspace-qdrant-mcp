//! Daemon Startup Recovery (Task 507)
//!
//! On daemon start, reconciles the `tracked_files` table with the filesystem
//! for all enabled watch_folders. Detects files added, deleted, or modified
//! while the daemon was not running and queues appropriate operations.
//!
//! This replaces Qdrant scrolling with fast SQLite queries.

mod queue;
mod reconcile;
pub mod types;

pub use types::{FullRecoveryStats, RecoveryStats};

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::config::StartupConfig;
use crate::patterns::exclusion::should_exclude_file;
use crate::queue_operations::QueueManager;
use crate::tracked_files_schema;
use crate::unified_queue_schema::{ItemType, QueueOperation};
use crate::watching_queue::WatchManager;

use queue::{enqueue_file_op, enqueue_progressive_scan};
use reconcile::reconcile_flagged_files;

/// Check and trigger the one-time base_point migration.
///
/// On first startup after upgrading to the base_point model, enqueues
/// a full (Tenant, Scan) for every active watch folder so that all files
/// get re-ingested with the new point ID scheme.
pub async fn check_base_point_migration(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<bool, String> {
    let done = crate::daemon_state::get_operational_state(
        pool,
        "base_point_migration_done",
        "daemon",
        None,
    )
    .await
    .unwrap_or(None);

    if done.as_deref() == Some("true") {
        debug!("base_point migration already done, skipping");
        return Ok(false);
    }

    info!("base_point migration: triggering full re-scan for all active watch folders");

    let watch_folders = sqlx::query_as::<_, (String, String, String)>(
        "SELECT watch_id, path, tenant_id FROM watch_folders
         WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folders for migration: {}", e))?;

    let mut enqueued = 0u64;
    for (watch_id, path, tenant_id) in &watch_folders {
        let payload = serde_json::json!({
            "project_root": path,
            "scan_reason": "base_point_migration",
        });

        match queue_manager
            .enqueue_unified(
                ItemType::Tenant,
                QueueOperation::Scan,
                tenant_id,
                "projects",
                &serde_json::to_string(&payload).unwrap_or_default(),
                Some("main"),
                None,
            )
            .await
        {
            Ok(_) => enqueued += 1,
            Err(e) => warn!("Failed to enqueue migration scan for {}: {}", watch_id, e),
        }
    }

    crate::daemon_state::set_operational_state(
        pool,
        "base_point_migration_done",
        "daemon",
        "true",
        None,
    )
    .await
    .map_err(|e| format!("Failed to set migration flag: {}", e))?;

    info!("base_point migration: enqueued {} re-scan(s)", enqueued);
    Ok(true)
}

/// Run startup recovery for all enabled watch_folders.
///
/// For each enabled watch_folder:
/// 1. Query tracked_files for all known files
/// 2. Walk filesystem for current eligible files
/// 3. Compare and queue appropriate operations
pub async fn run_startup_recovery(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    allowed_extensions: &AllowedExtensions,
    startup_config: &StartupConfig,
) -> Result<FullRecoveryStats, String> {
    info!(
        "Starting daemon startup recovery (batch_size={}, batch_delay={}ms)...",
        startup_config.startup_enqueue_batch_size, startup_config.startup_enqueue_batch_delay_ms,
    );
    let start = std::time::Instant::now();

    let watch_folders = sqlx::query_as::<_, (String, String, String, String)>(
        "SELECT watch_id, path, collection, tenant_id FROM watch_folders WHERE enabled = 1 AND is_archived = 0"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folders: {}", e))?;

    if watch_folders.is_empty() {
        info!("No enabled watch_folders found, skipping recovery");
        return Ok(FullRecoveryStats::default());
    }

    info!(
        "Running recovery for {} enabled watch_folders",
        watch_folders.len()
    );

    let mut full_stats = FullRecoveryStats::default();

    for (watch_id, path, collection, tenant_id) in &watch_folders {
        let result = recover_watch_folder(
            pool,
            queue_manager,
            watch_id,
            path,
            collection,
            tenant_id,
            allowed_extensions,
            startup_config,
        )
        .await;
        log_folder_recovery(&mut full_stats, watch_id, path, result);
    }

    reconcile_flagged_files(pool, queue_manager, &mut full_stats).await;

    let elapsed = start.elapsed();
    let total = full_stats.total_queued();
    info!(
        "Startup recovery complete: {} folders, {} items queued, {} reconciled ({} reconcile errors) in {:?}",
        full_stats.folders_processed, total, full_stats.reconciled, full_stats.reconcile_errors, elapsed
    );

    Ok(full_stats)
}

/// Log and accumulate per-folder recovery results.
fn log_folder_recovery(
    full_stats: &mut FullRecoveryStats,
    watch_id: &str,
    path: &str,
    result: Result<RecoveryStats, String>,
) {
    match result {
        Ok(s) => {
            if s.files_to_delete > 0
                || s.files_newly_excluded > 0
                || s.progressive_scans_enqueued > 0
                || s.files_to_update > 0
            {
                info!(
                    "Recovery for {} ({}): {} progressive scan(s), ~{} modified, -{} delete, x{} excluded, !{} errors",
                    watch_id, path, s.progressive_scans_enqueued, s.files_to_update,
                    s.files_to_delete, s.files_newly_excluded, s.errors
                );
            } else {
                debug!("Recovery for {} ({}): no changes detected", watch_id, path);
            }
            full_stats.per_folder.push((watch_id.to_string(), s));
        }
        Err(e) => {
            warn!("Recovery failed for {} ({}): {}", watch_id, path, e);
            let mut error_stats = RecoveryStats::default();
            error_stats.errors = 1;
            full_stats
                .per_folder
                .push((watch_id.to_string(), error_stats));
        }
    }
    full_stats.folders_processed += 1;
}

/// Recover a single watch_folder using progressive enqueue-first scanning.
///
/// Instead of walking the entire directory tree upfront (WalkDir), this:
/// 1. Enqueues a `(Tenant, Scan)` item for progressive breadth-first file discovery
/// 2. Checks tracked_files for deletions (files no longer on disk or now excluded)
#[allow(clippy::too_many_arguments)]
async fn recover_watch_folder(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    watch_folder_id: &str,
    base_path: &str,
    collection: &str,
    tenant_id: &str,
    _allowed_extensions: &AllowedExtensions,
    startup_config: &StartupConfig,
) -> Result<RecoveryStats, String> {
    let resolved_root = WatchManager::resolve_local_watch_path(base_path);
    let root = Path::new(&resolved_root);
    if !root.exists() || !root.is_dir() {
        return Err(format!(
            "Watch folder path does not exist or is not a directory: {}",
            base_path
        ));
    }

    let mut stats = RecoveryStats::default();

    enqueue_progressive_scan(queue_manager, root, tenant_id, collection, &mut stats).await?;
    detect_deleted_files(
        pool,
        queue_manager,
        root,
        tenant_id,
        collection,
        watch_folder_id,
        startup_config,
        &mut stats,
    )
    .await;

    Ok(stats)
}

/// Detect deleted/excluded files from tracked_files and queue deletions.
#[allow(clippy::too_many_arguments)]
async fn detect_deleted_files(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    root: &Path,
    tenant_id: &str,
    collection: &str,
    watch_folder_id: &str,
    startup_config: &StartupConfig,
    stats: &mut RecoveryStats,
) {
    let tracked =
        match tracked_files_schema::get_tracked_files_with_hashes(pool, watch_folder_id).await {
            Ok(t) => t,
            Err(e) => {
                warn!("Failed to query tracked_files: {}", e);
                stats.errors += 1;
                return;
            }
        };

    let reconcile_modified = startup_config.reconcile_modified_on_startup;
    let batch_size = startup_config.startup_enqueue_batch_size;
    let batch_delay =
        std::time::Duration::from_millis(startup_config.startup_enqueue_batch_delay_ms);
    let mut enqueued_in_batch: usize = 0;

    for (file_path, stored_hash) in &tracked {
        let abs_path = root.join(file_path);
        enqueued_in_batch += process_tracked_file(
            queue_manager,
            tenant_id,
            collection,
            root,
            &abs_path,
            file_path,
            stored_hash,
            reconcile_modified,
            stats,
        )
        .await;

        if batch_size > 0 && enqueued_in_batch >= batch_size {
            debug!(
                "Recovery deletion batch of {} enqueued, yielding for {:?}",
                enqueued_in_batch, batch_delay
            );
            tokio::task::yield_now().await;
            if !batch_delay.is_zero() {
                tokio::time::sleep(batch_delay).await;
            }
            enqueued_in_batch = 0;
        }
    }
}

/// Check one tracked file and enqueue a deletion if it is missing or now excluded.
/// Returns 1 if an item was enqueued, 0 otherwise.
#[allow(clippy::too_many_arguments)]
async fn process_tracked_file(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    repo_root: &Path,
    abs_path: &Path,
    file_path: &str,
    stored_hash: &str,
    reconcile_modified: bool,
    stats: &mut RecoveryStats,
) -> usize {
    let relative = match wqm_common::paths::RelativePath::from_user_input(file_path) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "tracked_files.relative_path {:?} failed validation: {}",
                file_path, e
            );
            stats.errors += 1;
            return 0;
        }
    };

    if !abs_path.exists() {
        match enqueue_file_op(
            queue_manager,
            tenant_id,
            collection,
            &relative,
            repo_root,
            QueueOperation::Delete,
            None,
        )
        .await
        {
            Ok(_is_new) => {
                stats.files_to_delete += 1;
                return 1;
            }
            Err(e) => {
                warn!(
                    "Failed to queue deletion for missing file: {}: {}",
                    file_path, e
                );
                stats.errors += 1;
            }
        }
    } else if should_exclude_file(file_path) {
        match enqueue_file_op(
            queue_manager,
            tenant_id,
            collection,
            &relative,
            repo_root,
            QueueOperation::Delete,
            None,
        )
        .await
        {
            Ok(_is_new) => {
                stats.files_newly_excluded += 1;
                return 1;
            }
            Err(e) => {
                warn!(
                    "Failed to queue deletion for excluded file: {}: {}",
                    file_path, e
                );
                stats.errors += 1;
            }
        }
    } else if reconcile_modified {
        // File still on disk and not excluded: re-hash and re-index when the
        // content changed while the daemon was down (or in an inactive project,
        // which live watch events skip). enqueue_file_op is idempotent, so this
        // is a no-op when the watcher already queued the same edit.
        match wqm_common::hashing::compute_file_hash(abs_path) {
            Ok(on_disk_hash) if on_disk_hash != stored_hash => {
                match enqueue_file_op(
                    queue_manager,
                    tenant_id,
                    collection,
                    &relative,
                    repo_root,
                    QueueOperation::Update,
                    None,
                )
                .await
                {
                    Ok(_is_new) => {
                        stats.files_to_update += 1;
                        return 1;
                    }
                    Err(e) => {
                        warn!("Failed to queue update for modified file: {}: {}", file_path, e);
                        stats.errors += 1;
                    }
                }
            }
            Ok(_) => stats.files_unchanged += 1,
            Err(e) => debug!("Reconcile hash failed for {}: {}", file_path, e),
        }
    }
    0
}

/// Test-only re-export of `reconcile_flagged_files` so that integration tests
/// in the reconciliation module can drive it directly without going through the
/// full `run_startup_recovery` path.
#[cfg(test)]
pub async fn reconcile_flagged_files_for_test(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    stats: &mut FullRecoveryStats,
) {
    reconcile_flagged_files(pool, queue_manager, stats).await;
}

#[cfg(test)]
mod tests;
