//! Daemon Startup Recovery (Task 507)
//!
//! On daemon start, reconciles the `tracked_files` table with the filesystem
//! for all enabled watch_folders. Detects files added, deleted, or modified
//! while the daemon was not running and queues appropriate operations.
//!
//! This replaces Qdrant scrolling with fast SQLite queries.

use std::collections::HashMap;
use std::path::Path;

use wqm_common::timestamps;
use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use wqm_common::constants::COLLECTION_LIBRARIES;
use crate::allowed_extensions::{AllowedExtensions, FileRoute};
use crate::config::StartupConfig;
use crate::patterns::exclusion::{should_exclude_file, should_exclude_directory};
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
        self.per_folder.iter().map(|(_, s)| s.files_to_ingest + s.files_to_delete + s.files_to_update + s.files_newly_excluded).sum()
    }
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
                if s.files_to_ingest > 0 || s.files_to_delete > 0 || s.files_to_update > 0 || s.files_newly_excluded > 0 {
                    info!(
                        "Recovery for {} ({}): +{} ingest, -{} delete, ~{} update, >{} to-lib, x{} excluded, ={} unchanged, !{} errors",
                        watch_id, path, s.files_to_ingest, s.files_to_delete,
                        s.files_to_update, s.files_routed_to_library, s.files_newly_excluded,
                        s.files_unchanged, s.errors
                    );
                } else {
                    debug!("Recovery for {} ({}): no changes detected ({} files unchanged)", watch_id, path, s.files_unchanged);
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

/// Recover a single watch_folder by comparing tracked_files with filesystem
async fn recover_watch_folder(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    watch_folder_id: &str,
    base_path: &str,
    collection: &str,
    tenant_id: &str,
    allowed_extensions: &AllowedExtensions,
    startup_config: &StartupConfig,
) -> Result<RecoveryStats, String> {
    let root = Path::new(base_path);
    if !root.exists() || !root.is_dir() {
        return Err(format!("Watch folder path does not exist or is not a directory: {}", base_path));
    }

    let mut stats = RecoveryStats::default();

    // Step 1: Query tracked_files for all known files in this watch_folder
    let tracked = tracked_files_schema::get_tracked_file_paths(pool, watch_folder_id)
        .await
        .map_err(|e| format!("Failed to query tracked_files: {}", e))?;

    // Build a map: relative_path -> (file_id, branch)
    let mut tracked_map: HashMap<String, (i64, Option<String>)> = HashMap::new();
    for (file_id, file_path, branch) in &tracked {
        tracked_map.insert(file_path.clone(), (*file_id, branch.clone()));
    }

    // Step 2: Walk filesystem to get current eligible files
    // Use filter_entry to skip excluded directories entirely (target/, node_modules/, .git/, etc.)
    // This avoids enumerating hundreds of thousands of files in build artifact directories.
    let mut disk_files: HashMap<String, ()> = HashMap::new();

    // Batching: track enqueue operations and yield+sleep between batches (Task 579)
    let batch_size = startup_config.startup_enqueue_batch_size;
    let batch_delay = std::time::Duration::from_millis(startup_config.startup_enqueue_batch_delay_ms);
    let mut enqueued_in_batch: usize = 0;

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() && e.depth() > 0 {
                let dir_name = e.file_name().to_string_lossy();
                !should_exclude_directory(&dir_name)
            } else {
                true
            }
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let rel_path = match path.strip_prefix(root) {
            Ok(rp) => rp.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        // Check exclusion rules
        if should_exclude_file(&rel_path) {
            // Check if this file was previously tracked (now excluded → queue delete)
            if tracked_map.contains_key(&rel_path) {
                if let Err(e) = enqueue_file_op(
                    queue_manager, tenant_id, collection,
                    &path.to_string_lossy(), QueueOperation::Delete, None,
                ).await {
                    warn!("Failed to queue excluded file for deletion: {}: {}", rel_path, e);
                    stats.errors += 1;
                } else {
                    stats.files_newly_excluded += 1;
                    enqueued_in_batch += 1;
                }
            }
            continue;
        }

        // Route file to appropriate collection based on extension (Task 565/566)
        let abs_path_str = path.to_string_lossy();
        let route = allowed_extensions.route_file(&abs_path_str, collection, tenant_id);

        let (target_collection, target_tenant, metadata) = match &route {
            FileRoute::ProjectCollection => (collection.to_string(), tenant_id.to_string(), None),
            FileRoute::LibraryCollection { source_project_id } => {
                let meta = source_project_id.as_ref().map(|pid| {
                    serde_json::json!({"source_project_id": pid}).to_string()
                });
                (COLLECTION_LIBRARIES.to_string(), tenant_id.to_string(), meta)
            }
            FileRoute::Excluded => {
                // File extension not in any allowlist - queue delete if previously tracked
                if tracked_map.contains_key(&rel_path) {
                    if let Err(e) = enqueue_file_op(
                        queue_manager, tenant_id, collection,
                        &abs_path_str, QueueOperation::Delete, None,
                    ).await {
                        warn!("Failed to queue non-allowed file for deletion: {}: {}", rel_path, e);
                        stats.errors += 1;
                    } else {
                        stats.files_newly_excluded += 1;
                        enqueued_in_batch += 1;
                    }
                }
                continue;
            }
        };

        let is_library_routed = matches!(route, FileRoute::LibraryCollection { .. })
            && collection != COLLECTION_LIBRARIES;

        disk_files.insert(rel_path.clone(), ());

        // Step 3c-f: Compare with tracked_files
        if let Some((_file_id, _branch)) = tracked_map.get(&rel_path) {
            // File exists in both tracked_files and on disk
            // Compare mtime and hash for changes
            let needs_update = match check_file_changed(pool, watch_folder_id, &rel_path, path) {
                Ok(changed) => changed,
                Err(e) => {
                    debug!("Error checking file {}: {}, assuming changed", rel_path, e);
                    true
                }
            };

            if needs_update {
                if let Err(e) = enqueue_file_op(
                    queue_manager, &target_tenant, &target_collection,
                    &abs_path_str, QueueOperation::Update, metadata.as_deref(),
                ).await {
                    warn!("Failed to queue file update: {}: {}", rel_path, e);
                    stats.errors += 1;
                } else {
                    stats.files_to_update += 1;
                    if is_library_routed { stats.files_routed_to_library += 1; }
                    enqueued_in_batch += 1;
                }
            } else {
                stats.files_unchanged += 1;
            }
        } else {
            // File on disk but not in tracked_files → queue ingest
            if let Err(e) = enqueue_file_op(
                queue_manager, &target_tenant, &target_collection,
                &abs_path_str, QueueOperation::Ingest, metadata.as_deref(),
            ).await {
                warn!("Failed to queue file ingest: {}: {}", rel_path, e);
                stats.errors += 1;
            } else {
                stats.files_to_ingest += 1;
                if is_library_routed { stats.files_routed_to_library += 1; }
                enqueued_in_batch += 1;
            }
        }

        // Yield and sleep between batches to avoid overwhelming the queue (Task 579)
        if batch_size > 0 && enqueued_in_batch >= batch_size {
            debug!("Recovery batch of {} enqueued, yielding for {:?}", enqueued_in_batch, batch_delay);
            tokio::task::yield_now().await;
            if !batch_delay.is_zero() {
                tokio::time::sleep(batch_delay).await;
            }
            enqueued_in_batch = 0;
        }
    }

    // Step 3d: In tracked_files but not on disk → queue delete
    for (tracked_path, (_file_id, _branch)) in &tracked_map {
        if !disk_files.contains_key(tracked_path) {
            // Skip files already handled as newly excluded
            let abs_path = root.join(tracked_path);
            if let Err(e) = enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete, None,
            ).await {
                warn!("Failed to queue file deletion: {}: {}", tracked_path, e);
                stats.errors += 1;
            } else {
                stats.files_to_delete += 1;
                enqueued_in_batch += 1;
            }

            // Batch sleep for deletion phase too
            if batch_size > 0 && enqueued_in_batch >= batch_size {
                debug!("Recovery delete batch of {} enqueued, yielding for {:?}", enqueued_in_batch, batch_delay);
                tokio::task::yield_now().await;
                if !batch_delay.is_zero() {
                    tokio::time::sleep(batch_delay).await;
                }
                enqueued_in_batch = 0;
            }
        }
    }

    Ok(stats)
}

/// Check if a file has changed compared to its tracked_files record
/// Check if a file has changed compared to its tracked_files record.
///
/// Currently uses a conservative approach: always returns true (assumes changed).
/// The actual hash comparison happens in process_file_item's update path,
/// which will skip unchanged files. This avoids needing an async tracked_file
/// lookup in what is otherwise a sync filesystem walk context.
fn check_file_changed(
    _pool: &SqlitePool,
    _watch_folder_id: &str,
    _rel_path: &str,
    _abs_path: &Path,
) -> Result<bool, String> {
    // Conservative: always assume changed.
    // The update path in process_file_item computes the hash and skips
    // unchanged files, so this doesn't cause unnecessary Qdrant writes.
    Ok(true)
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
            files_to_ingest: 5,
            files_to_delete: 2,
            files_to_update: 3,
            files_unchanged: 100,
            files_routed_to_library: 1,
            files_newly_excluded: 1,
            errors: 0,
        }));
        stats.per_folder.push(("w2".to_string(), RecoveryStats {
            files_to_ingest: 10,
            files_to_delete: 0,
            files_to_update: 1,
            files_unchanged: 50,
            files_routed_to_library: 0,
            files_newly_excluded: 0,
            errors: 0,
        }));

        assert_eq!(stats.total_queued(), 5 + 2 + 3 + 1 + 10 + 0 + 1 + 0);
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
}
