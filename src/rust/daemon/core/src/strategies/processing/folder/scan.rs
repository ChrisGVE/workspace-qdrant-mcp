//! Progressive single-level directory scan logic.

use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use chrono::DateTime;
use tracing::{debug, warn};
use wqm_common::paths::{CanonicalPath, RelativePath};

use crate::allowed_extensions::AllowedExtensions;
use crate::file_classification::classify_file_type;
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file_in_root};
use crate::patterns::gitignore::ProjectIgnoreMatcher;
use crate::queue_operations::QueueManager;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, FolderPayload, ItemType, ProjectPayload, QueueOperation, UnifiedQueueItem,
};

/// Progressive single-level directory scan with mtime-based pruning.
///
/// Enumerates only the immediate children of `dir_path`:
/// - Files: check exclusion + allowlist + mtime, enqueue `(File, Add)`
/// - Directories: check exclusion + mtime, enqueue `(Folder, Scan)`
/// - Directories with `.git`: submodule detection, enqueue `(Tenant, Add)`
///
/// `last_scan` is an ISO 8601 timestamp string. Entries with mtime <= this
/// value are skipped -- they are unchanged since the previous scan. Pass
/// `None` for a full scan (first-time or forced rescan).
///
/// Returns `(files_queued, dirs_queued, files_excluded, errors)`.
pub(crate) async fn scan_directory_single_level(
    dir_path: &Path,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
    last_scan: Option<&str>,
) -> UnifiedProcessorResult<(u64, u64, u64, u64)> {
    let mut files_queued = 0u64;
    let mut dirs_queued = 0u64;
    let mut files_excluded = 0u64;
    let mut errors = 0u64;

    let baseline: Option<SystemTime> = last_scan.and_then(parse_iso8601_to_system_time);
    // Anchor at the watch root so root-level .gitignore/.wqmignore patterns
    // cascade into every subdirectory scan (#105). Passing `None` here applied
    // them only when scanning the root itself — subdirectory scans then
    // enqueued thousands of ignored dirs (session-env scan storm, #103).
    let ignore_matcher =
        ProjectIgnoreMatcher::for_dir(dir_path, Some(Path::new(watch_folder_root.as_str())));

    let entries = std::fs::read_dir(dir_path).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to read directory {}: {}",
            dir_path.display(),
            e
        ))
    })?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to read dir entry in {}: {}", dir_path.display(), e);
                errors += 1;
                continue;
            }
        };

        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                warn!("Failed to get file type for {}: {}", path.display(), e);
                errors += 1;
                continue;
            }
        };

        if file_type.is_dir() {
            if is_ignored_by_matcher(&ignore_matcher, &path, true) {
                files_excluded += 1;
                continue;
            }
            dirs_queued += process_directory_entry(
                &path,
                &entry.file_name().to_string_lossy().to_string(),
                watch_folder_root,
                item,
                queue_manager,
                last_scan,
                &mut errors,
            )
            .await;
        } else if file_type.is_file() {
            if is_ignored_by_matcher(&ignore_matcher, &path, false) {
                files_excluded += 1;
                continue;
            }
            files_queued += process_file_entry(
                &path,
                watch_folder_root,
                item,
                queue_manager,
                allowed_extensions,
                baseline.as_ref(),
                &mut files_excluded,
                &mut errors,
            )
            .await;
        }
        // Symlinks are skipped (no follow)
    }

    Ok((files_queued, dirs_queued, files_excluded, errors))
}

/// Check if a path is ignored by the project-level ignore matcher.
fn is_ignored_by_matcher(
    ignore_matcher: &Option<ProjectIgnoreMatcher>,
    path: &Path,
    is_dir: bool,
) -> bool {
    if let Some(ref m) = ignore_matcher {
        if m.is_ignored(path, is_dir) {
            let label = if is_dir { "directory" } else { "file" };
            debug!(
                "Gate 0: {} excluded by ignore file: {}",
                label,
                path.display()
            );
            return true;
        }
    }
    false
}

/// Parse an ISO 8601 / RFC 3339 timestamp string into a `SystemTime`.
/// Returns `None` on parse failure (safe fallback: scan everything).
fn parse_iso8601_to_system_time(s: &str) -> Option<SystemTime> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| SystemTime::from(dt))
}

/// Process a single directory entry encountered during scan.
///
/// `last_scan` is propagated into the child `FolderPayload` so that
/// mtime pruning continues through the entire directory tree.
///
/// Returns 1 if an item was enqueued, 0 otherwise.
async fn process_directory_entry(
    path: &Path,
    dir_name: &str,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    last_scan: Option<&str>,
    errors: &mut u64,
) -> u64 {
    // Check directory exclusion
    if should_exclude_directory(dir_name) {
        return 0;
    }

    // Submodule detection: directory with .git -> (Tenant, Add)
    if path.join(".git").exists() {
        return enqueue_submodule(path, item, queue_manager, errors).await;
    }

    // Regular subdirectory -> (Folder, Scan)
    enqueue_subdirectory(path, watch_folder_root, item, queue_manager, last_scan).await
}

/// Enqueue a submodule directory as a Tenant/Add item.
///
/// Returns 1 if enqueued successfully, 0 otherwise.
async fn enqueue_submodule(
    path: &Path,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    errors: &mut u64,
) -> u64 {
    let submodule_payload = ProjectPayload {
        project_root: path.to_string_lossy().to_string(),
        git_remote: None,
        project_type: None,
        old_tenant_id: None,
        is_active: None,
        rebuild: false,
    };
    let payload_json = serde_json::to_string(&submodule_payload)
        .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, path.display()));

    let submodule_tenant = wqm_common::project_id::calculate_tenant_id(path);

    match queue_manager
        .enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Add,
            &submodule_tenant,
            &item.collection,
            &payload_json,
            None,
            None,
        )
        .await
    {
        Ok((_, true)) => {
            debug!("Enqueued submodule as Tenant/Add: {}", path.display());
            1
        }
        Ok((_, false)) => 0,
        Err(e) => {
            warn!("Failed to enqueue submodule {}: {}", path.display(), e);
            *errors += 1;
            0
        }
    }
}

/// Enqueue a regular subdirectory as a Folder/Scan item.
///
/// `last_scan` is embedded in the payload so the child scan can prune
/// unchanged entries without an extra DB query.
///
/// Returns 1 if enqueued successfully, 0 otherwise.
async fn enqueue_subdirectory(
    path: &Path,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    last_scan: Option<&str>,
) -> u64 {
    // Build a CanonicalPath for the absolute subdir so we can derive the
    // relative form. If the path is not UTF-8 or contains `..` we skip
    // enqueueing rather than store an invalid payload.
    let abs_str = match path.to_str() {
        Some(s) => s,
        None => {
            warn!("Non-UTF-8 subdir path skipped: {}", path.display());
            return 0;
        }
    };
    let abs = match CanonicalPath::from_user_input(abs_str) {
        Ok(p) => p,
        Err(e) => {
            warn!("Subdir path failed canonicalization ({}): {}", abs_str, e);
            return 0;
        }
    };
    let relative = match RelativePath::from_absolute_and_root(&abs, watch_folder_root) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "Subdir {} is not under watch_folder root {} ({}); skipping",
                abs.as_str(),
                watch_folder_root.as_str(),
                e
            );
            return 0;
        }
    };

    let folder_payload = FolderPayload {
        folder_path: Some(relative),
        recursive: false,
        recursive_depth: 0,
        patterns: vec![],
        ignore_patterns: vec![],
        old_path: None,
        last_scan: last_scan.map(|s| s.to_string()),
    };
    let payload_json = match serde_json::to_string(&folder_payload) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                "Failed to serialize FolderPayload for {}: {}",
                path.display(),
                e
            );
            return 0;
        }
    };

    match queue_manager
        .enqueue_unified(
            ItemType::Folder,
            QueueOperation::Scan,
            &item.tenant_id,
            &item.collection,
            &payload_json,
            None,
            None,
        )
        .await
    {
        Ok((_, true)) => 1,
        _ => 0,
    }
}

/// Maximum file size (100 MB) for scan ingestion.
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Process a single file entry encountered during scan.
///
/// `baseline` is the parsed mtime pruning threshold: files with
/// `mtime <= baseline` are skipped as unchanged.
///
/// Returns 1 if the file was enqueued, 0 otherwise.
#[allow(clippy::too_many_arguments)]
async fn process_file_entry(
    path: &Path,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
    baseline: Option<&SystemTime>,
    files_excluded: &mut u64,
    errors: &mut u64,
) -> u64 {
    let abs_path = path.to_string_lossy();

    // Root-anchored check (#97): hidden components ABOVE the watch root
    // (e.g. `.config` in `~/.config/main-docker/...`) must not exclude.
    if should_exclude_file_in_root(&abs_path, watch_folder_root.as_str()) {
        *files_excluded += 1;
        return 0;
    }

    if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
        *files_excluded += 1;
        return 0;
    }

    let metadata = match path.metadata() {
        Ok(m) => m,
        Err(e) => {
            warn!("Failed to get metadata for {}: {}", abs_path, e);
            *errors += 1;
            return 0;
        }
    };

    if should_prune_by_mtime(baseline, &metadata) {
        debug!("mtime prune: skipping unchanged file {}", abs_path);
        *files_excluded += 1;
        return 0;
    }

    if metadata.len() > MAX_FILE_SIZE {
        debug!(
            "Skipping large file: {} ({} bytes)",
            abs_path,
            metadata.len()
        );
        *files_excluded += 1;
        return 0;
    }

    enqueue_scanned_file(
        path,
        &abs_path,
        watch_folder_root,
        &metadata,
        item,
        queue_manager,
        errors,
    )
    .await
}

/// Check mtime pruning: returns `true` if the file is unchanged since baseline.
fn should_prune_by_mtime(baseline: Option<&SystemTime>, metadata: &std::fs::Metadata) -> bool {
    if let Some(bl) = baseline {
        metadata.modified().map(|m| m <= *bl).unwrap_or(false)
    } else {
        false
    }
}

/// Build the file payload (anchored to `watch_folder_root`) and enqueue
/// the file. Returns 1 on success, 0 on failure.
async fn enqueue_scanned_file(
    path: &Path,
    abs_path: &std::borrow::Cow<'_, str>,
    watch_folder_root: &CanonicalPath,
    metadata: &std::fs::Metadata,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    errors: &mut u64,
) -> u64 {
    let file_type_class = classify_file_type(path);

    let abs = match CanonicalPath::from_user_input(abs_path) {
        Ok(a) => a,
        Err(e) => {
            warn!("File path failed canonicalization ({}): {}", abs_path, e);
            *errors += 1;
            return 0;
        }
    };
    let relative = match RelativePath::from_absolute_and_root(&abs, watch_folder_root) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "File {} not under watch_folder root {} ({}); skipping",
                abs.as_str(),
                watch_folder_root.as_str(),
                e
            );
            *errors += 1;
            return 0;
        }
    };

    let file_payload = FilePayload {
        file_path: relative,
        file_type: Some(file_type_class.as_str().to_string()),
        file_hash: None,
        size_bytes: Some(metadata.len()),
        old_path: None,
    };

    let payload_json = match serde_json::to_string(&file_payload) {
        Ok(j) => j,
        Err(e) => {
            warn!("Failed to serialize FilePayload for {}: {}", abs_path, e);
            *errors += 1;
            return 0;
        }
    };

    match queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            &item.tenant_id,
            &item.collection,
            &payload_json,
            Some(&item.branch),
            None,
        )
        .await
    {
        Ok((_, true)) => 1,
        Ok((_, false)) => 0,
        Err(e) => {
            warn!("Failed to queue file {}: {}", abs_path, e);
            *errors += 1;
            0
        }
    }
}
