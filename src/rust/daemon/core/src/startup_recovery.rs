//! Daemon Startup Recovery (Task 507)
//!
//! On daemon start, reconciles the `tracked_files` table with the filesystem
//! for all enabled watch_folders. Detects files added, deleted, or modified
//! while the daemon was not running and queues appropriate operations.
//!
//! This replaces Qdrant scrolling with fast SQLite queries.

use std::path::Path;
use std::sync::Arc;

use wqm_common::timestamps;
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::config::StartupConfig;
use crate::patterns::exclusion::should_exclude_file;
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
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
///
/// The migration flag is stored in `operational_state` and checked each
/// startup. Once set, no further re-scans are triggered.
pub async fn check_base_point_migration(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<bool, String> {
    // Check if migration already done
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

    // Get all enabled, non-archived project watch folders
    let watch_folders = sqlx::query_as::<_, (String, String, String)>(
        "SELECT watch_id, path, tenant_id FROM watch_folders
         WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folders for migration: {}", e))?;

    let mut enqueued = 0u64;
    for (watch_id, path, tenant_id) in &watch_folders {
        // Enqueue a (Tenant, Scan) item to trigger full re-processing
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
            5, // Normal priority
            Some("main"),
            None,
        ).await {
            Ok(_) => enqueued += 1,
            Err(e) => warn!("Failed to enqueue migration scan for {}: {}", watch_id, e),
        }
    }

    // Mark migration as done
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

    // Get all enabled, non-archived watch_folders
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
        let stats = recover_watch_folder(
            pool, queue_manager, watch_id, path, collection, tenant_id,
            allowed_extensions, startup_config,
        ).await;

        match stats {
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
                full_stats.per_folder.push((watch_id.clone(), s));
            }
            Err(e) => {
                warn!("Recovery failed for {} ({}): {}", watch_id, path, e);
                let mut error_stats = RecoveryStats::default();
                error_stats.errors = 1;
                full_stats.per_folder.push((watch_id.clone(), error_stats));
            }
        }
        full_stats.folders_processed += 1;
    }

    // Phase 2: Reconcile files flagged with needs_reconcile=1.
    // These were marked during previous runs when SQLite transactions failed
    // after Qdrant writes, leaving potential state drift.
    reconcile_flagged_files(pool, queue_manager, &mut full_stats).await;

    let elapsed = start.elapsed();
    let total = full_stats.total_queued();
    info!(
        "Startup recovery complete: {} folders, {} items queued, {} reconciled ({} reconcile errors) in {:?}",
        full_stats.folders_processed, total, full_stats.reconciled, full_stats.reconcile_errors, elapsed
    );

    Ok(full_stats)
}

/// Process tracked_files flagged with needs_reconcile=1.
///
/// For each flagged file, look up its watch_folder to get routing info,
/// then re-queue it for ingestion. The idempotent processing pipeline
/// will re-verify content hashes and skip unchanged files.
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
        // Look up the watch_folder to get tenant_id and base path
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
                // Orphaned tracked_file: clear the flag since we can't re-queue
                let _ = clear_reconcile_flag(pool, file.file_id).await;
                stats.reconcile_errors += 1;
                continue;
            }
            Err(e) => {
                warn!("Failed to query watch_folder {}: {}", file.watch_folder_id, e);
                stats.reconcile_errors += 1;
                continue;
            }
        };

        // Build absolute path from base_path + relative file_path
        let abs_path = Path::new(&base_path).join(&file.file_path);

        if abs_path.exists() {
            // File still exists: re-queue for ingestion (idempotent — hash check in pipeline)
            if let Err(e) = enqueue_file_op(
                queue_manager, &tenant_id, &collection,
                &abs_path.to_string_lossy(), QueueOperation::Update, None,
            ).await {
                warn!("Failed to re-queue reconcile file {}: {}", file.file_path, e);
                stats.reconcile_errors += 1;
                continue;
            }
        } else {
            // File deleted: re-queue for deletion
            if let Err(e) = enqueue_file_op(
                queue_manager, &tenant_id, &collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                warn!("Failed to queue reconcile deletion for {}: {}", file.file_path, e);
                stats.reconcile_errors += 1;
                continue;
            }
        }

        // Clear the reconcile flag — the re-queued item will handle the rest
        if let Err(e) = clear_reconcile_flag(pool, file.file_id).await {
            warn!("Failed to clear reconcile flag for file_id={}: {}", file.file_id, e);
            stats.reconcile_errors += 1;
        } else {
            info!(
                "Reconciled file_id={} ({}): re-queued for {}",
                file.file_id,
                file.file_path,
                if abs_path.exists() { "update" } else { "deletion" }
            );
            stats.reconciled += 1;
        }
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
///
/// The progressive scan discovers files one directory level at a time through
/// the unified queue, allowing other operations to interleave and preventing
/// queue depth spikes on large projects.
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

    // Phase 1: Enqueue progressive scan for async file discovery.
    // This replaces the full-tree WalkDir with a (Tenant, Scan) queue item
    // that triggers breadth-first directory enumeration via scan_directory_single_level.
    let scan_payload = serde_json::json!({
        "project_root": base_path,
        "recovery": true,
    }).to_string();

    let branch = crate::watching_queue::get_current_branch(root);

    queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Scan,
        tenant_id,
        collection,
        &scan_payload,
        0, // Priority computed at dequeue time
        Some(&branch),
        None,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue progressive scan: {}", e))?;

    stats.progressive_scans_enqueued += 1;

    // Phase 2: Detect deleted/excluded files from tracked_files.
    // Files in tracked_files but no longer on disk or now excluded → queue delete.
    // This is bounded by tracked_files count (SQLite query + stat per file),
    // much cheaper than recursive directory enumeration.
    let tracked = tracked_files_schema::get_tracked_file_paths(pool, watch_folder_id)
        .await
        .map_err(|e| format!("Failed to query tracked_files: {}", e))?;

    let batch_size = startup_config.startup_enqueue_batch_size;
    let batch_delay = std::time::Duration::from_millis(startup_config.startup_enqueue_batch_delay_ms);
    let mut enqueued_in_batch: usize = 0;

    for (_file_id, file_path, _branch) in &tracked {
        let abs_path = root.join(file_path);

        if !abs_path.exists() {
            // File deleted from disk → queue delete
            if let Err(e) = enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                warn!("Failed to queue deletion for missing file: {}: {}", file_path, e);
                stats.errors += 1;
            } else {
                stats.files_to_delete += 1;
                enqueued_in_batch += 1;
            }
        } else if should_exclude_file(file_path) {
            // File now excluded → queue delete
            if let Err(e) = enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                warn!("Failed to queue deletion for excluded file: {}: {}", file_path, e);
                stats.errors += 1;
            } else {
                stats.files_newly_excluded += 1;
                enqueued_in_batch += 1;
            }
        }
        // else: file exists and not excluded — the progressive scan will
        // re-discover it, and the pipeline's hash check will skip if unchanged.

        if batch_size > 0 && enqueued_in_batch >= batch_size {
            debug!("Recovery deletion batch of {} enqueued, yielding for {:?}", enqueued_in_batch, batch_delay);
            tokio::task::yield_now().await;
            if !batch_delay.is_zero() {
                tokio::time::sleep(batch_delay).await;
            }
            enqueued_in_batch = 0;
        }
    }

    Ok(stats)
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

    // Detect branch from git if available
    let branch = crate::watching_queue::get_current_branch(Path::new(abs_file_path));

    queue_manager.enqueue_unified(
        ItemType::File,
        op,
        tenant_id,
        collection,
        &payload_json,
        0, // Priority computed at dequeue time
        Some(&branch),
        metadata,
    )
    .await
    .map(|_| ())
    .map_err(|e| format!("Failed to enqueue: {}", e))
}

/// Backfill `memory_mirror` table from Qdrant memory collection.
///
/// Scrolls all points in the `memory` collection and inserts any missing
/// rows into `memory_mirror` using INSERT OR IGNORE (idempotent).
/// This ensures the SQLite mirror stays in sync after a fresh install,
/// database reset, or if the mirror was added after points already existed.
pub async fn backfill_memory_mirror(
    pool: &SqlitePool,
    storage_client: &Arc<StorageClient>,
) -> Result<MemoryBackfillStats, String> {
    use qdrant_client::qdrant::{Filter, PointId};

    info!("Starting memory_mirror backfill from Qdrant...");
    let start = std::time::Instant::now();

    // Check if memory collection exists before scrolling
    let exists = storage_client
        .collection_exists(wqm_common::constants::COLLECTION_MEMORY)
        .await
        .map_err(|e| format!("Failed to check memory collection: {}", e))?;

    if !exists {
        info!("Memory collection does not exist, skipping backfill");
        return Ok(MemoryBackfillStats::default());
    }

    let mut stats = MemoryBackfillStats::default();
    let mut offset: Option<PointId> = None;
    let batch_size: u32 = 100;

    loop {
        let points = storage_client
            .scroll_with_filter(
                wqm_common::constants::COLLECTION_MEMORY,
                Filter::default(),
                batch_size,
                offset.clone(),
            )
            .await
            .map_err(|e| format!("Failed to scroll memory collection: {}", e))?;

        if points.is_empty() {
            break;
        }

        stats.points_scanned += points.len() as u64;

        // Track last point ID for pagination
        if let Some(last) = points.last() {
            offset = last.id.clone();
        }

        for point in &points {
            // Extract label as memory_id (required — skip points without label)
            let label = match point.payload.get("label").and_then(|v| v.as_str()) {
                Some(l) => l,
                None => {
                    debug!("Skipping memory point without label");
                    stats.skipped_no_label += 1;
                    continue;
                }
            };

            let empty = String::new();
            let content = point.payload.get("content").and_then(|v| v.as_str()).unwrap_or(&empty);
            let scope = point.payload.get("scope").and_then(|v| v.as_str());
            let tenant_id = point.payload.get("tenant_id").and_then(|v| v.as_str());
            let created_at = point.payload.get("created_at").and_then(|v| v.as_str());
            let now = timestamps::now_utc();
            let ts = created_at.unwrap_or(&now);

            let result = sqlx::query(
                "INSERT OR IGNORE INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )
            .bind(label)
            .bind(content)
            .bind(scope)
            .bind(tenant_id)
            .bind(ts)
            .bind(ts)
            .execute(pool)
            .await;

            match result {
                Ok(r) => {
                    if r.rows_affected() > 0 {
                        stats.inserted += 1;
                    } else {
                        stats.already_exists += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to insert memory_mirror row for label={}: {}", label, e);
                    stats.errors += 1;
                }
            }
        }

        // If we got fewer than batch_size, we've reached the end
        if (points.len() as u32) < batch_size {
            break;
        }

        // Advance offset past the last returned point
        // scroll_with_filter already returned the offset-based page;
        // the next page starts from the last point's ID
    }

    let elapsed = start.elapsed();
    info!(
        "Memory mirror backfill complete: {} scanned, {} inserted, {} existed, {} skipped (no label), {} errors in {:?}",
        stats.points_scanned, stats.inserted, stats.already_exists, stats.skipped_no_label, stats.errors, elapsed
    );

    Ok(stats)
}

/// Statistics from memory mirror backfill
#[derive(Debug, Clone, Default)]
pub struct MemoryBackfillStats {
    /// Total Qdrant points scanned
    pub points_scanned: u64,
    /// New rows inserted into memory_mirror
    pub inserted: u64,
    /// Rows already present (INSERT OR IGNORE)
    pub already_exists: u64,
    /// Points skipped because they lack a label
    pub skipped_no_label: u64,
    /// Insert errors
    pub errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_stats_default() {
        let stats = RecoveryStats::default();
        assert_eq!(stats.files_to_ingest, 0);
        assert_eq!(stats.files_to_delete, 0);
        assert_eq!(stats.files_to_update, 0);
        assert_eq!(stats.files_unchanged, 0);
        assert_eq!(stats.files_routed_to_library, 0);
        assert_eq!(stats.files_newly_excluded, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_full_recovery_stats_total() {
        let mut stats = FullRecoveryStats::default();
        stats.per_folder.push(("w1".to_string(), RecoveryStats {
            progressive_scans_enqueued: 1,
            files_to_delete: 2,
            files_newly_excluded: 1,
            ..RecoveryStats::default()
        }));
        stats.per_folder.push(("w2".to_string(), RecoveryStats {
            progressive_scans_enqueued: 1,
            files_to_delete: 3,
            files_newly_excluded: 0,
            ..RecoveryStats::default()
        }));

        // total_queued = progressive_scans + deletes + newly_excluded
        assert_eq!(stats.total_queued(), 1 + 2 + 1 + 1 + 3 + 0);
    }

    #[test]
    fn test_compute_relative_path_for_recovery() {
        let root = Path::new("/home/user/project");
        let abs = Path::new("/home/user/project/src/main.rs");
        let rel = abs.strip_prefix(root).unwrap().to_string_lossy().to_string();
        assert_eq!(rel, "src/main.rs");
    }

    use std::time::Duration;
    use sqlx::sqlite::SqlitePoolOptions;
    use crate::tracked_files_schema::{self as tfs, CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL};
    use crate::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL};
    use crate::watch_folders_schema;

    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    async fn setup_reconcile_tables(pool: &SqlitePool) {
        sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await.unwrap();
        sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(pool).await.unwrap();
        sqlx::query(CREATE_TRACKED_FILES_SQL).execute(pool).await.unwrap();
        for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await.unwrap();
        for idx in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }
    }

    /// Insert a watch_folder and a tracked_file with needs_reconcile=1
    async fn insert_reconcile_fixture(pool: &SqlitePool, base_path: &str, rel_path: &str) -> i64 {
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
             VALUES ('wf-rc', ?1, 'projects', 'tenant-rc', 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(base_path)
        .execute(pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, chunk_count, collection, needs_reconcile, reconcile_reason, created_at, updated_at)
             VALUES ('wf-rc', ?1, '2025-01-01T00:00:00Z', 'abc123', 3, 'projects', 1, 'ingest_tx_failed: test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(rel_path)
        .execute(pool).await.unwrap();

        sqlx::query_scalar::<_, i64>("SELECT last_insert_rowid()").fetch_one(pool).await.unwrap()
    }

    #[tokio::test]
    async fn test_reconcile_flagged_files_requeues_existing() {
        let pool = create_test_pool().await;
        setup_reconcile_tables(&pool).await;

        // Use a real temp dir so the file "exists"
        let tmp = tempfile::tempdir().unwrap();
        let base_path = tmp.path().to_string_lossy().to_string();
        let rel_path = "src/main.rs";
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join(rel_path), "fn main() {}").unwrap();

        let file_id = insert_reconcile_fixture(&pool, &base_path, rel_path).await;

        let queue_manager = QueueManager::new(pool.clone());
        let mut stats = FullRecoveryStats::default();

        reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

        assert_eq!(stats.reconciled, 1);
        assert_eq!(stats.reconcile_errors, 0);

        // Verify flag was cleared
        let flagged = tfs::get_files_needing_reconcile(&pool).await.unwrap();
        assert!(flagged.is_empty(), "needs_reconcile should be cleared after reconciliation");

        // Verify an item was enqueued
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(count, 1, "Should have enqueued one update item");

        // Verify the enqueued item is an update operation
        let op: String = sqlx::query_scalar("SELECT op FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(op, "update");

        drop(tmp);
        let _ = file_id; // used for insert verification
    }

    #[tokio::test]
    async fn test_reconcile_flagged_files_deleted_file_queues_delete() {
        let pool = create_test_pool().await;
        setup_reconcile_tables(&pool).await;

        // Use a path that does NOT exist on disk
        let base_path = "/tmp/nonexistent_recovery_test_dir";
        let rel_path = "gone.rs";

        insert_reconcile_fixture(&pool, base_path, rel_path).await;

        let queue_manager = QueueManager::new(pool.clone());
        let mut stats = FullRecoveryStats::default();

        reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

        assert_eq!(stats.reconciled, 1);
        assert_eq!(stats.reconcile_errors, 0);

        // Verify the enqueued item is a delete operation
        let op: String = sqlx::query_scalar("SELECT op FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(op, "delete");
    }

    #[tokio::test]
    async fn test_reconcile_no_flagged_files_is_noop() {
        let pool = create_test_pool().await;
        setup_reconcile_tables(&pool).await;

        let queue_manager = QueueManager::new(pool.clone());
        let mut stats = FullRecoveryStats::default();

        reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

        assert_eq!(stats.reconciled, 0);
        assert_eq!(stats.reconcile_errors, 0);
    }

    #[test]
    fn test_memory_backfill_stats_default() {
        let stats = MemoryBackfillStats::default();
        assert_eq!(stats.points_scanned, 0);
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.already_exists, 0);
        assert_eq!(stats.skipped_no_label, 0);
        assert_eq!(stats.errors, 0);
    }

    /// Test that memory_mirror INSERT OR IGNORE is idempotent
    #[tokio::test]
    async fn test_memory_mirror_insert_idempotent() {
        let pool = create_test_pool().await;
        // Create memory_mirror table
        sqlx::query(crate::watch_folders_schema::CREATE_MEMORY_MIRROR_SQL)
            .execute(&pool).await.unwrap();

        let now = timestamps::now_utc();

        // First insert succeeds
        let r1 = sqlx::query(
            "INSERT OR IGNORE INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-1")
        .bind("Always use snake_case")
        .bind("global")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r1.rows_affected(), 1);

        // Duplicate insert is silently ignored
        let r2 = sqlx::query(
            "INSERT OR IGNORE INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-1")
        .bind("Different text")
        .bind("project")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r2.rows_affected(), 0);

        // Original text preserved
        let text: String = sqlx::query_scalar(
            "SELECT rule_text FROM memory_mirror WHERE memory_id = 'rule-1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(text, "Always use snake_case");

        // Different key inserts fine
        let r3 = sqlx::query(
            "INSERT OR IGNORE INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-2")
        .bind("Use Rust for performance")
        .bind("global")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r3.rows_affected(), 1);

        // Total count is 2
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_mirror")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(count, 2);
    }
}
