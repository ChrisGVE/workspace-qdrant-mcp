//! Content-hash deduplication for cross-branch file ingestion.
//!
//! When a file is ingested and identical content (same `file_hash`) already
//! exists at the same path for a different branch, this module skips the
//! entire embedding pipeline and instead adds the current branch to the
//! existing Qdrant points' `branches` array and the SQLite `tracked_files`
//! row's `branches` JSON column.
//!
//! This saves embedding costs proportional to the number of branches sharing
//! the same file content.

use std::collections::HashMap;
use std::path::Path;

use qdrant_client::qdrant::{Condition, Filter};
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::tracked_files_schema::{self, TrackedFile};
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{DestinationStatus, UnifiedQueueItem};

/// Attempt content-hash deduplication before running the full ingest pipeline.
///
/// Returns `Some(())` if dedup succeeded (caller should return early),
/// or `None` if no dedup candidate was found (caller should proceed with
/// normal embedding).
#[allow(clippy::too_many_arguments)]
pub(super) async fn try_dedup(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    detected_branch: &str,
) -> UnifiedProcessorResult<Option<()>> {
    // Compute the file hash to check for cross-branch duplicates.
    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .unwrap_or_else(|_| "unknown".to_string());

    if file_hash == "unknown" {
        return Ok(None);
    }

    // Check if this exact content already exists under a different branch.
    let existing = tracked_files_schema::lookup_tracked_file_by_hash(
        pool,
        watch_folder_id,
        relative_path,
        &file_hash,
        detected_branch,
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("dedup hash lookup failed: {}", e))
    })?;

    let existing = match existing {
        Some(f) => f,
        None => return Ok(None),
    };

    // Found a cross-branch duplicate. Run the fast path.
    info!(
        "Content-hash dedup: {} on branch '{}' matches existing file_id={} (branch '{}'). \
         Skipping embedding.",
        relative_path,
        detected_branch,
        existing.file_id,
        existing.primary_branch.as_deref().unwrap_or("?"),
    );

    handle_dedup_branch_add(
        ctx,
        &existing,
        detected_branch,
        item,
        pool,
        watch_folder_id,
        relative_path,
        abs_file_path,
    )
    .await?;

    Ok(Some(()))
}

/// Handle the dedup fast path: add the branch to the existing tracked file
/// and update the Qdrant points' `branches` payload.
#[allow(clippy::too_many_arguments)]
async fn handle_dedup_branch_add(
    ctx: &ProcessingContext,
    existing: &TrackedFile,
    branch: &str,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
) -> UnifiedProcessorResult<()> {
    // 1. Check if branch already present in the existing row's branches array.
    let current_branches: Vec<String> =
        serde_json::from_str(&existing.branches).unwrap_or_default();
    if current_branches.iter().any(|b| b == branch) {
        debug!(
            "Dedup: branch '{}' already in branches array for file_id={}, nothing to do",
            branch, existing.file_id
        );
        mark_dedup_done(ctx, item).await;
        return Ok(());
    }

    // 2. Acquire per-tenant lock to serialize branch-array mutations.
    let lock = ctx.branch_locks.get(&item.tenant_id);
    let _guard = lock.lock().await;

    // 3. Update SQLite: append branch to the branches JSON array.
    let updated_branches =
        tracked_files_schema::add_branch_to_tracked_file(pool, existing.file_id, branch)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "dedup: failed to add branch to tracked_files: {}",
                    e
                ))
            })?;

    // 4. Also create a tracked_files row for this branch so that
    //    `lookup_tracked_file(pool, wfid, path, Some(branch))` finds it
    //    in future update/delete operations. We clone the existing row's
    //    metadata.
    create_branch_tracked_file(
        pool,
        existing,
        branch,
        watch_folder_id,
        relative_path,
        &item.collection,
    )
    .await?;

    // 5. Update Qdrant: set the branches payload on all points for this file.
    if let Some(ref base_point) = existing.base_point {
        let new_branches: Vec<String> = serde_json::from_str(&updated_branches).unwrap_or_default();
        update_qdrant_branches(ctx, base_point, &item.collection, &new_branches).await;
    }

    // 6. Update FTS5 search index for this branch (if search_db is available).
    if let Some(sdb) = &ctx.search_db {
        let fts_result = super::fts5_index::update_fts5_for_file(
            sdb,
            pool,
            existing.file_id,
            abs_file_path,
            &item.tenant_id,
            Some(branch),
            existing.base_point.as_deref(),
            Some(relative_path),
            Some(existing.file_hash.as_str()),
        )
        .await;
        if let Err(e) = fts_result {
            warn!(
                "Dedup: FTS5 update failed for branch '{}' of {}: {}",
                branch, relative_path, e
            );
        }
    }

    mark_dedup_done(ctx, item).await;

    info!(
        "Content-hash dedup complete: added branch '{}' to file_id={} ({})",
        branch, existing.file_id, relative_path
    );

    Ok(())
}

/// Create a new `tracked_files` row for the dedup branch.
///
/// This ensures that future `lookup_tracked_file(pool, wfid, path, Some(branch))`
/// calls find the file and can handle updates/deletes correctly.
#[allow(clippy::too_many_arguments)]
async fn create_branch_tracked_file(
    pool: &SqlitePool,
    existing: &TrackedFile,
    branch: &str,
    watch_folder_id: &str,
    relative_path: &str,
    collection: &str,
) -> UnifiedProcessorResult<()> {
    // Check if a row already exists for this branch (idempotency).
    let already_exists = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(branch),
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("dedup: branch row lookup failed: {}", e))
    })?;

    if already_exists.is_some() {
        return Ok(());
    }

    tracked_files_schema::insert_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(branch),
        existing.file_type.as_deref(),
        existing.language.as_deref(),
        &existing.file_mtime,
        &existing.file_hash,
        existing.chunk_count,
        existing.chunking_method.as_deref(),
        existing.lsp_status,
        existing.treesitter_status,
        Some(collection),
        existing.extension.as_deref(),
        existing.is_test,
        existing.base_point.as_deref(),
        existing.component.as_deref(),
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!(
            "dedup: failed to insert branch tracked_file row: {}",
            e
        ))
    })?;

    Ok(())
}

/// Update the `branches` payload field on all Qdrant points for a file.
///
/// Uses `set_payload_by_filter` to atomically update all chunks that share
/// the same `base_point`. The caller must hold the per-tenant branch lock.
async fn update_qdrant_branches(
    ctx: &ProcessingContext,
    base_point: &str,
    collection: &str,
    branches: &[String],
) {
    let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);

    let mut payload = HashMap::new();
    payload.insert("branches".to_string(), serde_json::json!(branches));

    if let Err(e) = ctx
        .storage_client
        .set_payload_by_filter(collection, filter, payload)
        .await
    {
        warn!(
            "Dedup: failed to update Qdrant branches for base_point={}: {}",
            base_point, e
        );
    }
}

/// Mark both qdrant and search destinations as done for a dedup-handled item.
async fn mark_dedup_done(ctx: &ProcessingContext, item: &UnifiedQueueItem) {
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_already_present_detection() {
        let branches_json = r#"["main","feature/auth"]"#;
        let branches: Vec<String> = serde_json::from_str(branches_json).unwrap_or_default();
        assert!(branches.iter().any(|b| b == "main"));
        assert!(branches.iter().any(|b| b == "feature/auth"));
        assert!(!branches.iter().any(|b| b == "develop"));
    }

    #[test]
    fn test_empty_branches_array() {
        let branches_json = "[]";
        let branches: Vec<String> = serde_json::from_str(branches_json).unwrap_or_default();
        assert!(branches.is_empty());
        assert!(!branches.iter().any(|b| b == "main"));
    }

    #[test]
    fn test_malformed_branches_falls_back_to_empty() {
        let branches_json = "not-valid-json";
        let branches: Vec<String> = serde_json::from_str(branches_json).unwrap_or_default();
        assert!(branches.is_empty());
    }
}
