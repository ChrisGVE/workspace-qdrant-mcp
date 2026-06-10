//! Database operations for branch cleanup.

use std::collections::HashMap;

use qdrant_client::qdrant::{Condition, Filter};
use sqlx::SqlitePool;
use tracing::warn;
use wqm_common::timestamps;

use crate::branch_switch::BranchUpdateContext;

/// A tracked file affected by branch deletion.
pub struct AffectedFile {
    pub file_id: i64,
    pub base_point: Option<String>,
    pub branches: Vec<String>,
    pub remaining_branches: usize,
}

/// Fetch all tracked_files rows that reference the deleted branch.
pub async fn fetch_affected_files(
    pool: &SqlitePool,
    watch_folder_id: &str,
    branch: &str,
) -> Result<Vec<AffectedFile>, String> {
    let rows: Vec<(i64, Option<String>, String)> = sqlx::query_as(
        "SELECT file_id, base_point, COALESCE(branches, '[]')
         FROM tracked_files
         WHERE watch_folder_id = ?1
           AND EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = ?2)",
    )
    .bind(watch_folder_id)
    .bind(branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to fetch affected files: {}", e))?;

    Ok(rows
        .into_iter()
        .map(|(file_id, base_point, branches_json)| {
            let branches: Vec<String> = serde_json::from_str(&branches_json).unwrap_or_default();
            let remaining = branches.iter().filter(|b| b.as_str() != branch).count();
            AffectedFile {
                file_id,
                base_point,
                branches,
                remaining_branches: remaining,
            }
        })
        .collect())
}

/// Remove a branch from tracked_files.branches[] for the given files.
pub async fn remove_branch_from_tracked_files(
    pool: &SqlitePool,
    files: &[&AffectedFile],
    branch: &str,
) -> Result<u64, String> {
    let now = timestamps::now_utc();
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut updated = 0u64;
    for f in files {
        let result = sqlx::query(
            "UPDATE tracked_files
             SET branches = (
                 SELECT json_group_array(j.value)
                 FROM json_each(branches) AS j
                 WHERE j.value != ?1
             ),
             updated_at = ?2
             WHERE file_id = ?3",
        )
        .bind(branch)
        .bind(&now)
        .bind(f.file_id)
        .execute(&mut *tx)
        .await;

        match result {
            Ok(r) => updated += r.rows_affected(),
            Err(e) => {
                warn!("Failed to remove branch from file_id={}: {}", f.file_id, e);
            }
        }
    }

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit branch removal: {}", e))?;

    Ok(updated)
}

/// Delete orphaned tracked_files and qdrant_chunks rows.
pub async fn delete_orphaned_files(pool: &SqlitePool, file_ids: &[i64]) -> Result<u64, String> {
    if file_ids.is_empty() {
        return Ok(0);
    }

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut deleted = 0u64;
    for &file_id in file_ids {
        // Delete qdrant_chunks first (foreign key)
        let _ = sqlx::query("DELETE FROM qdrant_chunks WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await;

        let result = sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await;

        match result {
            Ok(r) => deleted += r.rows_affected(),
            Err(e) => {
                warn!("Failed to delete tracked_file {}: {}", file_id, e);
            }
        }
    }

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit orphan deletion: {}", e))?;

    Ok(deleted)
}

/// Update Qdrant points' branches payload to the new set (without deleted branch).
pub async fn update_qdrant_branches(
    branch_ctx: &BranchUpdateContext,
    base_point: &str,
    branches: &[&str],
) {
    let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);

    let mut payload = HashMap::new();
    payload.insert("branches".to_string(), serde_json::json!(branches));

    if let Err(e) = branch_ctx
        .storage_client
        .set_payload_by_filter("projects", filter, payload)
        .await
    {
        warn!(
            "Cleanup: failed to update Qdrant branches for base_point={}: {}",
            base_point, e
        );
    }
}

/// Delete all Qdrant points for a base_point (orphaned content).
///
/// Returns `Err` when Qdrant rejects the deletion. Callers must then keep
/// the corresponding local rows so the next cleanup/reconcile pass can
/// retry — deleting them anyway would orphan the vectors with no repair
/// path (#127).
pub async fn delete_qdrant_points(
    branch_ctx: &BranchUpdateContext,
    base_point: &str,
) -> Result<(), String> {
    let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);

    branch_ctx
        .storage_client
        .delete_points_with_filter("projects", filter)
        .await
        .map(|_deleted_count| ())
        .map_err(|e| {
            format!(
                "failed to delete Qdrant points for base_point={}: {}",
                base_point, e
            )
        })
}

/// Rename a branch in tracked_files.branches[] for the given watch folder.
pub async fn rename_branch_in_tracked_files(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_name: &str,
    new_name: &str,
) -> Result<u64, String> {
    let now = timestamps::now_utc();
    let affected = fetch_affected_files(pool, watch_folder_id, old_name).await?;
    if affected.is_empty() {
        return Ok(0);
    }

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut updated = 0u64;
    for f in &affected {
        let mut branches: Vec<String> = f.branches.clone();
        for b in &mut branches {
            if b == old_name {
                *b = new_name.to_string();
            }
        }
        branches.sort();
        branches.dedup();
        let new_json = serde_json::to_string(&branches).unwrap_or_else(|_| "[]".to_string());

        let result = sqlx::query(
            "UPDATE tracked_files SET branches = ?1, updated_at = ?2 WHERE file_id = ?3",
        )
        .bind(&new_json)
        .bind(&now)
        .bind(f.file_id)
        .execute(&mut *tx)
        .await;

        match result {
            Ok(r) => updated += r.rows_affected(),
            Err(e) => {
                warn!("Failed to rename branch in file_id={}: {}", f.file_id, e);
            }
        }
    }

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit branch rename: {}", e))?;

    Ok(updated)
}

/// Rename a branch in Qdrant points' branches payload for affected files.
pub async fn rename_qdrant_branches(
    branch_ctx: &BranchUpdateContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_name: &str,
    new_name: &str,
) {
    let affected = match fetch_affected_files(pool, watch_folder_id, old_name).await {
        Ok(files) => files,
        Err(e) => {
            warn!("Failed to fetch files for Qdrant branch rename: {}", e);
            return;
        }
    };

    for f in &affected {
        if let Some(ref bp) = f.base_point {
            let mut branches: Vec<String> = f.branches.clone();
            for b in &mut branches {
                if b == old_name {
                    *b = new_name.to_string();
                }
            }
            branches.sort();
            branches.dedup();
            let branch_refs: Vec<&str> = branches.iter().map(|b| b.as_str()).collect();
            update_qdrant_branches(branch_ctx, bp, &branch_refs).await;
        }
    }
}

/// Rename a branch in file_metadata rows (search.db).
pub async fn rename_file_metadata_branch(
    search_pool: &SqlitePool,
    tenant_id: &str,
    old_name: &str,
    new_name: &str,
) {
    let result =
        sqlx::query("UPDATE file_metadata SET branch = ?1 WHERE tenant_id = ?2 AND branch = ?3")
            .bind(new_name)
            .bind(tenant_id)
            .bind(old_name)
            .execute(search_pool)
            .await;

    match result {
        Ok(r) => {
            if r.rows_affected() > 0 {
                tracing::info!(
                    "Renamed {} file_metadata rows: '{}' -> '{}'",
                    r.rows_affected(),
                    old_name,
                    new_name
                );
            }
        }
        Err(e) => {
            warn!(
                "Failed to rename file_metadata branch '{}' -> '{}': {}",
                old_name, new_name, e
            );
        }
    }
}

/// Check if any other watch folder references the same base_point.
pub async fn has_other_base_point_references(
    pool: &SqlitePool,
    base_point: &str,
    watch_folder_id: &str,
) -> bool {
    let count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files WHERE base_point = ?1 AND watch_folder_id != ?2",
    )
    .bind(base_point)
    .bind(watch_folder_id)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    count > 0
}

/// Delete file_metadata rows for the deleted branch and prune orphaned
/// code_lines (#102).
///
/// A file whose LAST file_metadata row is removed becomes unreachable by
/// every scoped FTS query, but its code_lines (and FTS5 entries) would keep
/// matching unscoped queries with stale content.
///
/// Transactions are deliberately SHORT — one for the branch rows, then one
/// per orphaned file — because FTS5 'delete' entries are expensive on a large
/// trigram index and search.db has a single writer: one branch-wide
/// transaction was measured holding the write lock for minutes, stalling all
/// ingestion. If interrupted between transactions, leftover orphaned
/// code_lines are reclaimed by the reconcile sweep's orphan pass.
pub async fn delete_file_metadata_for_branch(
    search_pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
) {
    let file_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT DISTINCT file_id FROM file_metadata WHERE tenant_id = ?1 AND branch = ?2",
    )
    .bind(tenant_id)
    .bind(branch)
    .fetch_all(search_pool)
    .await
    .unwrap_or_default();

    let deleted_rows =
        match sqlx::query("DELETE FROM file_metadata WHERE tenant_id = ?1 AND branch = ?2")
            .bind(tenant_id)
            .bind(branch)
            .execute(search_pool)
            .await
        {
            Ok(r) => r.rows_affected(),
            Err(e) => {
                warn!(
                    "Failed to delete file_metadata for branch '{}': {}",
                    branch, e
                );
                return;
            }
        };

    let mut orphaned_lines = 0u64;
    for file_id in &file_ids {
        match delete_code_lines_if_orphaned(search_pool, *file_id).await {
            Ok(n) => orphaned_lines += n,
            Err(e) => warn!(
                "Orphan prune failed for file_id={} (branch '{}'): {}",
                file_id, branch, e
            ),
        }
    }

    if deleted_rows > 0 || orphaned_lines > 0 {
        tracing::info!(
            "Deleted {} file_metadata rows for branch '{}' ({} orphaned code_lines pruned)",
            deleted_rows,
            branch,
            orphaned_lines
        );
    }
}

/// Delete one file's code_lines (and FTS5 index entries) iff the file has no
/// file_metadata rows left. One short transaction per file (#102).
///
/// The FTS5 'delete' entries are issued as a single set-based INSERT..SELECT
/// instead of one statement per row.
pub async fn delete_code_lines_if_orphaned(
    search_pool: &SqlitePool,
    file_id: i64,
) -> Result<u64, String> {
    let mut tx = search_pool
        .begin()
        .await
        .map_err(|e| format!("begin: {}", e))?;

    let remaining: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| format!("metadata check: {}", e))?;
    if remaining > 0 {
        return Ok(0); // tx dropped — nothing written
    }

    // External-content FTS5: remove index entries before the content rows.
    sqlx::query(
        "INSERT INTO code_lines_fts(code_lines_fts, rowid, content) \
         SELECT 'delete', line_id, content FROM code_lines WHERE file_id = ?1",
    )
    .bind(file_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("fts5 delete: {}", e))?;

    let result = sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
        .bind(file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("code_lines delete: {}", e))?;

    tx.commit().await.map_err(|e| format!("commit: {}", e))?;
    Ok(result.rows_affected())
}
