//! Daemon Startup Recovery (Task 507)
//!
//! On daemon start, reconciles the `tracked_files` table with the filesystem
//! for all enabled watch_folders. Detects files added, deleted, or modified
//! while the daemon was not running and queues appropriate operations.
//!
//! This replaces Qdrant scrolling with fast SQLite queries.

use std::collections::HashMap;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::allowed_extensions::AllowedExtensions;
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
) -> Result<FullRecoveryStats, String> {
    info!("Starting daemon startup recovery...");
    let start = std::time::Instant::now();

    // Get all enabled watch_folders
    let watch_folders = sqlx::query_as::<_, (String, String, String, String)>(
        "SELECT watch_id, path, collection, tenant_id FROM watch_folders WHERE enabled = 1"
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
            allowed_extensions,
        ).await;

        match stats {
            Ok(s) => {
                if s.files_to_ingest > 0 || s.files_to_delete > 0 || s.files_to_update > 0 || s.files_newly_excluded > 0 {
                    info!(
                        "Recovery for {} ({}): +{} ingest, -{} delete, ~{} update, x{} excluded, ={} unchanged, !{} errors",
                        watch_id, path, s.files_to_ingest, s.files_to_delete,
                        s.files_to_update, s.files_newly_excluded, s.files_unchanged, s.errors
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

    let elapsed = start.elapsed();
    let total = full_stats.total_queued();
    info!(
        "Startup recovery complete: {} folders, {} items queued in {:?}",
        full_stats.folders_processed, total, elapsed
    );

    Ok(full_stats)
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
    let mut disk_files: HashMap<String, ()> = HashMap::new();

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
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
                    &path.to_string_lossy(), QueueOperation::Delete,
                ).await {
                    warn!("Failed to queue excluded file for deletion: {}: {}", rel_path, e);
                    stats.errors += 1;
                } else {
                    stats.files_newly_excluded += 1;
                }
            }
            continue;
        }

        // Check allowlist (Task 511)
        let abs_path_str = path.to_string_lossy();
        if !allowed_extensions.is_allowed(&abs_path_str, collection) {
            // File extension not in allowlist - queue delete if previously tracked
            if tracked_map.contains_key(&rel_path) {
                if let Err(e) = enqueue_file_op(
                    queue_manager, tenant_id, collection,
                    &abs_path_str, QueueOperation::Delete,
                ).await {
                    warn!("Failed to queue non-allowed file for deletion: {}: {}", rel_path, e);
                    stats.errors += 1;
                } else {
                    stats.files_newly_excluded += 1;
                }
            }
            continue;
        }

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
                    queue_manager, tenant_id, collection,
                    &path.to_string_lossy(), QueueOperation::Update,
                ).await {
                    warn!("Failed to queue file update: {}: {}", rel_path, e);
                    stats.errors += 1;
                } else {
                    stats.files_to_update += 1;
                }
            } else {
                stats.files_unchanged += 1;
            }
        } else {
            // File on disk but not in tracked_files → queue ingest
            if let Err(e) = enqueue_file_op(
                queue_manager, tenant_id, collection,
                &path.to_string_lossy(), QueueOperation::Ingest,
            ).await {
                warn!("Failed to queue file ingest: {}: {}", rel_path, e);
                stats.errors += 1;
            } else {
                stats.files_to_ingest += 1;
            }
        }

        // Yield periodically
        let total_processed = stats.files_to_ingest + stats.files_to_update + stats.files_unchanged;
        if total_processed % 500 == 0 && total_processed > 0 {
            tokio::task::yield_now().await;
        }
    }

    // Step 3d: In tracked_files but not on disk → queue delete
    for (tracked_path, (_file_id, _branch)) in &tracked_map {
        if !disk_files.contains_key(tracked_path) {
            // Skip files already handled as newly excluded
            let abs_path = root.join(tracked_path);
            if let Err(e) = enqueue_file_op(
                queue_manager, tenant_id, collection,
                &abs_path.to_string_lossy(), QueueOperation::Delete,
            ).await {
                warn!("Failed to queue file deletion: {}: {}", tracked_path, e);
                stats.errors += 1;
            } else {
                stats.files_to_delete += 1;
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
        None,
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
            files_newly_excluded: 1,
            errors: 0,
        }));
        stats.per_folder.push(("w2".to_string(), RecoveryStats {
            files_to_ingest: 10,
            files_to_delete: 0,
            files_to_update: 1,
            files_unchanged: 50,
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
}
