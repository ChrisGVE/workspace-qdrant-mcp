//! Database operations for branch switch: branch-array updates and commit hash tracking.

use std::collections::{HashMap, HashSet};

use qdrant_client::qdrant::{Condition, Filter};
use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use super::types::BranchUpdateContext;

/// Tracked file info needed for branch-add operations.
struct BranchAddCandidate {
    file_id: i64,
    relative_path: String,
    base_point: Option<String>,
    branches: String,
}

/// Add new branch to unchanged files' `branches[]` arrays in SQLite and Qdrant.
///
/// For each unchanged file (not in `changed_paths`), adds `new_branch` to:
/// 1. `tracked_files.branches` JSON array in state.db
/// 2. Qdrant points' `branches` payload field
/// 3. `file_metadata` row in search.db (for FTS5 branch-scoped search)
pub async fn batch_add_branch_to_unchanged_files(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    watch_folder_id: &str,
    tenant_id: &str,
    old_branch: &str,
    new_branch: &str,
    changed_paths: &HashSet<String>,
) -> Result<u64, String> {
    // Hold per-tenant lock across the full read-modify-write to prevent
    // concurrent branch updates from clobbering each other.
    let lock = branch_ctx.branch_locks.get(tenant_id);
    let _guard = lock.lock().await;

    let candidates =
        fetch_unchanged_candidates(pool, watch_folder_id, old_branch, new_branch, changed_paths)
            .await?;

    if candidates.is_empty() {
        return Ok(0);
    }

    let count = candidates.len() as u64;

    // 1. Batch update SQLite tracked_files.branches[]
    add_branch_to_tracked_files_batch(pool, &candidates, new_branch).await?;

    // 2. Batch update Qdrant points' branches[] payload
    update_qdrant_branches_batch(branch_ctx, &candidates, new_branch).await;

    // 3. Insert file_metadata rows for the new branch in search.db
    if let Some(ref sdb) = branch_ctx.search_db {
        let watch_root = match fetch_watch_folder(pool, watch_folder_id).await {
            Ok((path, _, _)) => path,
            Err(e) => {
                warn!("Failed to fetch watch folder path for file_metadata: {}", e);
                String::new()
            }
        };
        insert_file_metadata_for_branch(sdb, pool, &candidates, tenant_id, new_branch, &watch_root)
            .await;
    }

    info!(
        "Added branch '{}' to {} unchanged files (was '{}')",
        new_branch, count, old_branch
    );

    Ok(count)
}

/// Fetch candidates for branch-add: files on old_branch NOT in changed_paths,
/// excluding files that already have new_branch in their branches array.
async fn fetch_unchanged_candidates(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_branch: &str,
    new_branch: &str,
    changed_paths: &HashSet<String>,
) -> Result<Vec<BranchAddCandidate>, String> {
    let rows: Vec<(i64, String, Option<String>, String)> = sqlx::query_as(
        "SELECT file_id, relative_path, base_point, COALESCE(branches, '[]')
         FROM tracked_files
         WHERE watch_folder_id = ?1
           AND EXISTS (
               SELECT 1 FROM json_each(branches) WHERE json_each.value = ?2
           )",
    )
    .bind(watch_folder_id)
    .bind(old_branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query tracked files for branch add: {}", e))?;

    Ok(rows
        .into_iter()
        .filter(|(_, rel_path, _, _)| !changed_paths.contains(rel_path.as_str()))
        .filter(|(_, _, _, branches_json)| {
            let branches: Vec<String> = serde_json::from_str(branches_json).unwrap_or_default();
            !branches.iter().any(|b| b == new_branch)
        })
        .map(
            |(file_id, relative_path, base_point, branches)| BranchAddCandidate {
                file_id,
                relative_path,
                base_point,
                branches,
            },
        )
        .collect())
}

/// Batch-add new_branch to tracked_files.branches[] for all candidates.
async fn add_branch_to_tracked_files_batch(
    pool: &SqlitePool,
    candidates: &[BranchAddCandidate],
    new_branch: &str,
) -> Result<(), String> {
    let now = timestamps::now_utc();

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    for candidate in candidates {
        sqlx::query(
            "UPDATE tracked_files
             SET branches = json_insert(branches, '$[#]', ?1),
                 updated_at = ?2
             WHERE file_id = ?3",
        )
        .bind(new_branch)
        .bind(&now)
        .bind(candidate.file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            format!(
                "Failed to add branch to file_id={}: {}",
                candidate.file_id, e
            )
        })?;
    }

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit batch branch add: {}", e))?;

    Ok(())
}

/// Batch-update Qdrant points' `branches` payload for all candidates.
///
/// Groups candidates by base_point (one Qdrant set_payload per unique base_point)
/// since all chunks of a file share the same base_point.
async fn update_qdrant_branches_batch(
    branch_ctx: &BranchUpdateContext,
    candidates: &[BranchAddCandidate],
    new_branch: &str,
) {
    let mut by_base_point: HashMap<&str, Vec<String>> = HashMap::new();

    for c in candidates {
        if let Some(ref bp) = c.base_point {
            let current: Vec<String> = serde_json::from_str(&c.branches).unwrap_or_default();
            by_base_point
                .entry(bp.as_str())
                .or_insert_with(|| current)
                .push(new_branch.to_string());
        }
    }

    // Deduplicate branches in each set
    for branches in by_base_point.values_mut() {
        branches.sort();
        branches.dedup();
    }

    let collection = "projects";
    let mut success = 0u64;
    let mut errors = 0u64;

    for (base_point, branches) in &by_base_point {
        let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);

        let mut payload = HashMap::new();
        payload.insert("branches".to_string(), serde_json::json!(branches));

        match branch_ctx
            .storage_client
            .set_payload_by_filter(collection, filter, payload)
            .await
        {
            Ok(_) => success += 1,
            Err(e) => {
                warn!(
                    "Failed to update Qdrant branches for base_point={}: {}",
                    base_point, e
                );
                errors += 1;
            }
        }
    }

    if success > 0 || errors > 0 {
        debug!(
            "Qdrant branch update: {} base_points updated, {} errors",
            success, errors
        );
    }
}

/// Insert file_metadata rows for the new branch in search.db.
async fn insert_file_metadata_for_branch(
    sdb: &crate::search_db::SearchDbManager,
    state_pool: &SqlitePool,
    candidates: &[BranchAddCandidate],
    tenant_id: &str,
    new_branch: &str,
    watch_root: &str,
) {
    let search_pool = sdb.pool();

    for candidate in candidates {
        let file_hash: Option<String> =
            sqlx::query_scalar("SELECT file_hash FROM tracked_files WHERE file_id = ?1")
                .bind(candidate.file_id)
                .fetch_optional(state_pool)
                .await
                .unwrap_or(None);

        let abs_path = format!(
            "{}/{}",
            watch_root.trim_end_matches('/'),
            &candidate.relative_path
        );
        let result = sqlx::query(
            "INSERT INTO file_metadata (file_id, tenant_id, branch, file_path, base_point, relative_path, file_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(file_id, branch) DO NOTHING",
        )
        .bind(candidate.file_id)
        .bind(tenant_id)
        .bind(new_branch)
        .bind(&abs_path)
        .bind(candidate.base_point.as_deref())
        .bind(&candidate.relative_path)
        .bind(file_hash.as_deref())
        .execute(search_pool)
        .await;

        if let Err(e) = result {
            warn!(
                "Failed to insert file_metadata for file_id={}, branch='{}': {}",
                candidate.file_id, new_branch, e
            );
        }
    }
}

/// Update last_commit_hash in watch_folders.
pub async fn update_last_commit_hash(
    pool: &SqlitePool,
    watch_folder_id: &str,
    commit_hash: &str,
) -> Result<(), String> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE watch_folders SET last_commit_hash = ?1, updated_at = ?2 WHERE watch_id = ?3",
    )
    .bind(commit_hash)
    .bind(&now)
    .bind(watch_folder_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to update last_commit_hash: {}", e))?;
    Ok(())
}

/// Test-only: SQLite-only batch branch add (no Qdrant, no search.db).
///
/// Returns the number of files where the branch was added.
#[cfg(test)]
pub async fn add_branch_to_tracked_files_batch_pub(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_branch: &str,
    new_branch: &str,
    changed_paths: &std::collections::HashSet<String>,
) -> Result<u64, String> {
    let candidates =
        fetch_unchanged_candidates(pool, watch_folder_id, old_branch, new_branch, changed_paths)
            .await?;
    if candidates.is_empty() {
        return Ok(0);
    }
    let count = candidates.len() as u64;
    add_branch_to_tracked_files_batch(pool, &candidates, new_branch).await?;
    Ok(count)
}

/// Fetch watch folder info: (path, collection, tenant_id).
pub async fn fetch_watch_folder(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<(String, String, String), String> {
    sqlx::query_as::<_, (String, String, String)>(
        "SELECT path, collection, tenant_id FROM watch_folders WHERE watch_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folder: {}", e))?
    .ok_or_else(|| format!("Watch folder {} not found", watch_folder_id))
}
