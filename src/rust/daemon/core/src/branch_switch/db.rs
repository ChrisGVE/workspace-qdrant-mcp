//! Database operations for branch switch: batch updates and commit hash tracking.

use std::collections::HashSet;

use sqlx::SqlitePool;
use wqm_common::timestamps;

/// Batch update branch column in tracked_files for unchanged files.
///
/// Within a single transaction, updates all tracked files on the old branch
/// that are NOT in the changed_paths set. Also recomputes base_point since
/// it includes the branch in its hash input.
pub async fn batch_update_branch(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_branch: &str,
    new_branch: &str,
    changed_paths: &HashSet<String>,
) -> Result<u64, String> {
    let now = timestamps::now_utc();

    // Get all tracked files on the old branch for this watch folder
    let files: Vec<(i64, String, String, Option<String>)> = sqlx::query_as(
        "SELECT file_id, file_path, COALESCE(file_hash, ''), relative_path
         FROM tracked_files
         WHERE watch_folder_id = ?1 AND branch = ?2"
    )
    .bind(watch_folder_id)
    .bind(old_branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query tracked files: {}", e))?;

    if files.is_empty() {
        return Ok(0);
    }

    // Look up tenant_id for base_point computation
    let tenant_id: String = sqlx::query_scalar(
        "SELECT tenant_id FROM watch_folders WHERE watch_id = ?1"
    )
    .bind(watch_folder_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Failed to query tenant_id: {}", e))?;

    let mut tx = pool.begin().await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut updated = 0u64;

    for (file_id, file_path, file_hash, relative_path) in &files {
        // Skip files that changed (those will be re-ingested)
        let rel = relative_path.as_deref().unwrap_or(file_path);
        if changed_paths.contains(rel) || changed_paths.contains(file_path.as_str()) {
            continue;
        }

        // Recompute base_point with new branch
        let new_bp = wqm_common::hashing::compute_base_point(
            &tenant_id, new_branch, rel, file_hash,
        );

        sqlx::query(
            "UPDATE tracked_files
             SET branch = ?1, base_point = ?2, updated_at = ?3
             WHERE file_id = ?4"
        )
        .bind(new_branch)
        .bind(&new_bp)
        .bind(&now)
        .bind(file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to update file_id={}: {}", file_id, e))?;

        updated += 1;
    }

    tx.commit().await
        .map_err(|e| format!("Failed to commit batch branch update: {}", e))?;

    Ok(updated)
}

/// Update last_commit_hash in watch_folders.
pub async fn update_last_commit_hash(
    pool: &SqlitePool,
    watch_folder_id: &str,
    commit_hash: &str,
) -> Result<(), String> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE watch_folders SET last_commit_hash = ?1, updated_at = ?2 WHERE watch_id = ?3"
    )
    .bind(commit_hash)
    .bind(&now)
    .bind(watch_folder_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to update last_commit_hash: {}", e))?;
    Ok(())
}

/// Fetch watch folder info: (path, collection, tenant_id).
pub async fn fetch_watch_folder(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<(String, String, String), String> {
    sqlx::query_as::<_, (String, String, String)>(
        "SELECT path, collection, tenant_id FROM watch_folders WHERE watch_id = ?1"
    )
    .bind(watch_folder_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folder: {}", e))?
    .ok_or_else(|| format!("Watch folder {} not found", watch_folder_id))
}
