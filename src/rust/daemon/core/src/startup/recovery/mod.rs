//! Daemon Startup Recovery (Task 507)
//!
//! On daemon start, reconciles the `tracked_files` table with the filesystem
//! for all enabled watch_folders. Detects files added, deleted, or modified
//! while the daemon was not running and queues appropriate operations.
//!
//! This replaces Qdrant scrolling with fast SQLite queries.

use std::path::Path;

use wqm_common::timestamps;
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::config::StartupConfig;
use crate::patterns::exclusion::should_exclude_file;
use crate::queue_operations::QueueManager;
use crate::tracked_files_schema;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation};
use crate::file_classification::classify_file_type;

/// Result of a recovery operation for a single watch_folder
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of files queued for ingestion (new on disk)
    pub files_to_ingest: u64,
    /// Number of files queued for deletion (removed from disk)
    pub files_to_delete: u64,
    /// Number of files queued for update (content changed)
    pub files_to_update: u64,
    /// Number of files skipped (unchanged)
    pub files_unchanged: u64,
    /// Number of files routed to libraries collection (from project folders)
    pub files_routed_to_library: u64,
    /// Number of files now excluded (queued for deletion)
    pub files_newly_excluded: u64,
    /// Number of progressive scans enqueued (Tenant, Scan) for async file discovery
    pub progressive_scans_enqueued: u64,
    /// Errors encountered during recovery
    pub errors: u64,
}

/// Result of the full recovery run across all watch_folders
#[derive(Debug, Clone, Default)]
pub struct FullRecoveryStats {
    /// Per-watch_folder stats
    pub per_folder: Vec<(String, RecoveryStats)>,
    /// Total folders processed
    pub folders_processed: u64,
    /// Files re-queued from needs_reconcile markers
    pub reconciled: u64,
    /// Reconciliation errors
    pub reconcile_errors: u64,
}

impl FullRecoveryStats {
    pub fn total_queued(&self) -> u64 {
        self.per_folder.iter().map(|(_, s)| s.progressive_scans_enqueued + s.files_to_delete + s.files_newly_excluded).sum()
    }
}

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
        pool, "base_point_migration_done", "daemon", None,
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
         WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'"
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

        match queue_manager.enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Scan,
            tenant_id,
            "projects",
            &serde_json::to_string(&payload).unwrap_or_default(),
            Some("main"),
            None,
        ).await {
            Ok(_) => enqueued += 1,
            Err(e) => warn!("Failed to enqueue migration scan for {}: {}", watch_id, e),
        }
    }

    crate::daemon_state::set_operational_state(
        pool, "base_point_migration_done", "daemon",
        "true", None,
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
        startup_config.startup_enqueue_batch_size,
        startup_config.startup_enqueue_batch_delay_ms,
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

    info!("Running recovery for {} enabled watch_folders", watch_folders.len());

    let mut full_stats = FullRecoveryStats::default();

    for (watch_id, path, collection, tenant_id) in &watch_folders {
        let result = recover_watch_folder(
            pool, queue_manager, watch_id, path, collection, tenant_id,
            allowed_extensions, startup_config,
        ).await;
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
            if s.files_to_delete > 0 || s.files_newly_excluded > 0 || s.progressive_scans_enqueued > 0 {
                info!(
                    "Recovery for {} ({}): {} progressive scan(s), -{} delete, x{} excluded, !{} errors",
                    watch_id, path, s.progressive_scans_enqueued, s.files_to_delete,
                    s.files_newly_excluded, s.errors
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
            full_stats.per_folder.push((watch_id.to_string(), error_stats));
        }
    }
    full_stats.folders_processed += 1;
}

/// Process tracked_files flagged with needs_reconcile=1.
///
/// For each flagged file, look up its watch_folder to get routing info,
/// then re-queue it for ingestion.
async fn reconcile_flagged_files(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    stats: &mut FullRecoveryStats,
) {
    let flagged = match tracked_files_schema::get_files_needing_reconcile(pool).await {
        Ok(files) => files,
        Err(e) => {
            warn!("Failed to query needs_reconcile files: {}", e);
            stats.reconcile_errors += 1;
            return;
        }
    };

    if flagged.is_empty() {
        debug!("No files need reconciliation");
        return;
    }

    info!("Reconciling {} flagged files", flagged.len());

    for file in &flagged {
        reconcile_single_file(pool, queue_manager, stats, file).await;
    }
}

/// Reconcile a single flagged file: look up watch folder, re-queue, clear flag.
async fn reconcile_single_file(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    stats: &mut FullRecoveryStats,
    file: &tracked_files_schema::TrackedFile,
) {
    let wf = sqlx::query_as::<_, (String, String, String)>(
        "SELECT path, collection, tenant_id FROM watch_folders WHERE watch_id = ?1"
    )
    .bind(&file.watch_folder_id)
    .fetch_optional(pool)
    .await;

    let (base_path, collection, tenant_id) = match wf {
        Ok(Some(row)) => row,
        Ok(None) => {
            warn!(
                "Watch folder {} not found for reconcile file_id={}, clearing flag",
                file.watch_folder_id, file.file_id
            );
            let _ = clear_reconcile_flag(pool, file.file_id).await;
            stats.reconcile_errors += 1;
            return;
        }
        Err(e) => {
            warn!("Failed to query watch_folder {}: {}", file.watch_folder_id, e);
            stats.reconcile_errors += 1;
            return;
        }
    };

    let abs_path = Path::new(&base_path).join(&file.file_path);
    let op = if abs_path.exists() { QueueOperation::Update } else { QueueOperation::Delete };

    if let Err(e) = enqueue_file_op(
        queue_manager, &tenant_id, &collection,
        &abs_path.to_string_lossy(), op.clone(), None,
    ).await {
        warn!("Failed to re-queue reconcile file {}: {}", file.file_path, e);
        stats.reconcile_errors += 1;
        return;
    }

    if let Err(e) = clear_reconcile_flag(pool, file.file_id).await {
        warn!("Failed to clear reconcile flag for file_id={}: {}", file.file_id, e);
        stats.reconcile_errors += 1;
    } else {
        info!(
            "Reconciled file_id={} ({}): re-queued for {}",
            file.file_id, file.file_path, op.as_str()
        );
        stats.reconciled += 1;
    }
}

/// Clear the needs_reconcile flag for a tracked file (non-transactional)
async fn clear_reconcile_flag(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 0, reconcile_reason = NULL, updated_at = ?1
         WHERE file_id = ?2"
    )
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;
    Ok(())
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
    let root = Path::new(base_path);
    if !root.exists() || !root.is_dir() {
        return Err(format!("Watch folder path does not exist or is not a directory: {}", base_path));
    }

    let mut stats = RecoveryStats::default();

    enqueue_progressive_scan(queue_manager, root, tenant_id, collection, &mut stats).await?;
    detect_deleted_files(pool, queue_manager, root, tenant_id, collection, watch_folder_id, startup_config, &mut stats).await;

    Ok(stats)
}

/// Enqueue a progressive scan for async file discovery.
async fn enqueue_progressive_scan(
    queue_manager: &QueueManager,
    root: &Path,
    tenant_id: &str,
    collection: &str,
    stats: &mut RecoveryStats,
) -> Result<(), String> {
    let scan_payload = serde_json::json!({
        "project_root": root.to_string_lossy(),
        "recovery": true,
    }).to_string();

    let branch = crate::watching_queue::get_current_branch(root);

    queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Scan,
        tenant_id,
        collection,
        &scan_payload,
        Some(&branch),
        None,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue progressive scan: {}", e))?;

    stats.progressive_scans_enqueued += 1;
    Ok(())
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
    let tracked = match tracked_files_schema::get_tracked_file_paths(pool, watch_folder_id).await {
        Ok(t) => t,
        Err(e) => {
            warn!("Failed to query tracked_files: {}", e);
            stats.errors += 1;
            return;
        }
    };

    let batch_size = startup_config.startup_enqueue_batch_size;
    let batch_delay = std::time::Duration::from_millis(startup_config.startup_enqueue_batch_delay_ms);
    let mut enqueued_in_batch: usize = 0;

    for (_file_id, file_path, _branch) in &tracked {
        let abs_path = root.join(file_path);

        if !abs_path.exists() {
            match enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                Ok(()) => { stats.files_to_delete += 1; enqueued_in_batch += 1; }
                Err(e) => { warn!("Failed to queue deletion for missing file: {}: {}", file_path, e); stats.errors += 1; }
            }
        } else if should_exclude_file(file_path) {
            match enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                Ok(()) => { stats.files_newly_excluded += 1; enqueued_in_batch += 1; }
                Err(e) => { warn!("Failed to queue deletion for excluded file: {}: {}", file_path, e); stats.errors += 1; }
            }
        }

        if batch_size > 0 && enqueued_in_batch >= batch_size {
            debug!("Recovery deletion batch of {} enqueued, yielding for {:?}", enqueued_in_batch, batch_delay);
            tokio::task::yield_now().await;
            if !batch_delay.is_zero() {
                tokio::time::sleep(batch_delay).await;
            }
            enqueued_in_batch = 0;
        }
    }
}

/// Enqueue a file operation (ingest, update, or delete)
async fn enqueue_file_op(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    abs_file_path: &str,
    op: QueueOperation,
    metadata: Option<&str>,
) -> Result<(), String> {
    let file_type = if op != QueueOperation::Delete {
        Some(classify_file_type(Path::new(abs_file_path)).as_str().to_string())
    } else {
        None
    };

    let file_payload = FilePayload {
        file_path: abs_file_path.to_string(),
        file_type,
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload)
        .map_err(|e| format!("Failed to serialize FilePayload: {}", e))?;

    let branch = crate::watching_queue::get_current_branch(Path::new(abs_file_path));

    queue_manager.enqueue_unified(
        ItemType::File,
        op,
        tenant_id,
        collection,
        &payload_json,
        Some(&branch),
        metadata,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue: {}", e))
}

#[cfg(test)]
mod tests;
