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
pub async fn delete_qdrant_points(branch_ctx: &BranchUpdateContext, base_point: &str) {
    let filter = Filter::must([Condition::matches("base_point", base_point.to_string())]);

    if let Err(e) = branch_ctx
        .storage_client
        .delete_points_with_filter("projects", filter)
        .await
    {
        warn!(
            "Cleanup: failed to delete Qdrant points for base_point={}: {}",
            base_point, e
        );
    }
}

/// Delete file_metadata rows for the deleted branch.
pub async fn delete_file_metadata_for_branch(
    search_pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
) {
    let result = sqlx::query("DELETE FROM file_metadata WHERE tenant_id = ?1 AND branch = ?2")
        .bind(tenant_id)
        .bind(branch)
        .execute(search_pool)
        .await;

    match result {
        Ok(r) => {
            if r.rows_affected() > 0 {
                tracing::info!(
                    "Deleted {} file_metadata rows for branch '{}'",
                    r.rows_affected(),
                    branch
                );
            }
        }
        Err(e) => {
            warn!(
                "Failed to delete file_metadata for branch '{}': {}",
                branch, e
            );
        }
    }
}
