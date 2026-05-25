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

    let (tenant_id, files) = fetch_branch_data(pool, watch_folder_id, old_branch).await?;

    if files.is_empty() {
        return Ok(0);
    }

    // Pre-compute (file_id, new_base_point) for unchanged files
    let updates = compute_unchanged_updates(&files, changed_paths, &tenant_id);

    if updates.is_empty() {
        return Ok(0);
    }

    let updated = updates.len() as u64;

    execute_batch_update(pool, &updates, new_branch, &now).await?;

    Ok(updated)
}

/// Fetch tenant_id and tracked files for the given watch folder and branch.
async fn fetch_branch_data(
    pool: &SqlitePool,
    watch_folder_id: &str,
    old_branch: &str,
) -> Result<(String, Vec<(i64, String, String)>), String> {
    // Post-v37: the row carries only `relative_path`; the legacy absolute
    // `file_path` column is gone. Tuple shape: `(file_id, relative_path, file_hash)`.
    let files: Vec<(i64, String, String)> = sqlx::query_as(
        "SELECT file_id, relative_path, COALESCE(file_hash, '')
         FROM tracked_files
         WHERE watch_folder_id = ?1 AND branch = ?2",
    )
    .bind(watch_folder_id)
    .bind(old_branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query tracked files: {}", e))?;

    if files.is_empty() {
        return Ok((String::new(), files));
    }

    let tenant_id: String =
        sqlx::query_scalar("SELECT tenant_id FROM watch_folders WHERE watch_id = ?1")
            .bind(watch_folder_id)
            .fetch_one(pool)
            .await
            .map_err(|e| format!("Failed to query tenant_id: {}", e))?;

    Ok((tenant_id, files))
}

/// Compute new base_points for unchanged files.
fn compute_unchanged_updates(
    files: &[(i64, String, String)],
    changed_paths: &HashSet<String>,
    tenant_id: &str,
) -> Vec<(i64, String)> {
    let mut updates: Vec<(i64, String)> = Vec::new();
    for (file_id, relative_path, file_hash) in files {
        if changed_paths.contains(relative_path.as_str()) {
            continue;
        }
        let new_bp =
            wqm_common::hashing::compute_base_point(tenant_id, relative_path.as_str(), file_hash);
        updates.push((*file_id, new_bp));
    }
    updates
}

/// Execute the batch update within a transaction using a temp table join.
async fn execute_batch_update(
    pool: &SqlitePool,
    updates: &[(i64, String)],
    new_branch: &str,
    now: &str,
) -> Result<(), String> {
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    // Create temp table for batch join-update
    sqlx::query(
        "CREATE TEMP TABLE IF NOT EXISTS _bp_update(file_id INTEGER PRIMARY KEY, base_point TEXT)",
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("Failed to create temp table: {}", e))?;

    sqlx::query("DELETE FROM _bp_update")
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to clear temp table: {}", e))?;

    // Batch-insert into temp table (chunks of 200 to stay under SQLite's 999 param limit)
    insert_temp_batches(updates, &mut tx).await?;

    // Single join-update replaces N individual UPDATEs
    sqlx::query(
        "UPDATE tracked_files
         SET branch = ?1, base_point = _bp_update.base_point, updated_at = ?2
         FROM _bp_update
         WHERE tracked_files.file_id = _bp_update.file_id",
    )
    .bind(new_branch)
    .bind(now)
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("Failed to batch-update branch: {}", e))?;

    sqlx::query("DROP TABLE IF EXISTS _bp_update")
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("Failed to drop temp table: {}", e))?;

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit batch branch update: {}", e))?;

    Ok(())
}

/// Batch-insert update pairs into the temp table in chunks.
async fn insert_temp_batches(
    updates: &[(i64, String)],
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
) -> Result<(), String> {
    const CHUNK_SIZE: usize = 200;
    for chunk in updates.chunks(CHUNK_SIZE) {
        let placeholders: Vec<String> = (0..chunk.len())
            .map(|i| format!("(?{}, ?{})", i * 2 + 1, i * 2 + 2))
            .collect();
        let insert_sql = format!(
            "INSERT INTO _bp_update(file_id, base_point) VALUES {}",
            placeholders.join(", ")
        );
        let mut q = sqlx::query(&insert_sql);
        for (file_id, base_point) in chunk {
            q = q.bind(file_id).bind(base_point);
        }
        q.execute(&mut **tx)
            .await
            .map_err(|e| format!("Failed to batch-insert temp data: {}", e))?;
    }
    Ok(())
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
