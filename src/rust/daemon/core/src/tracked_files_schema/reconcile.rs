//! Reconciliation operations for tracked files

use sqlx::{Sqlite, SqlitePool};
use wqm_common::timestamps;

use super::operations::tracked_file_from_row;
use super::types::TrackedFile;

/// Mark a tracked file as needing reconciliation (using pool, not transaction)
///
/// Used when Qdrant succeeds but the SQLite transaction for tracked_files fails.
/// This allows startup recovery to prioritize fixing inconsistencies.
pub async fn mark_needs_reconcile(
    pool: &SqlitePool,
    file_id: i64,
    reason: &str,
) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 1, reconcile_reason = ?1, updated_at = ?2
         WHERE file_id = ?3",
    )
    .bind(reason)
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get all tracked files that need reconciliation
pub async fn get_files_needing_reconcile(
    pool: &SqlitePool,
) -> Result<Vec<TrackedFile>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT file_id, watch_folder_id, file_path, branch, file_type, language,
                file_mtime, file_hash, chunk_count, chunking_method,
                lsp_status, treesitter_status, last_error,
                needs_reconcile, reconcile_reason, extension, is_test,
                collection, base_point, relative_path, incremental,
                component, created_at, updated_at
         FROM tracked_files
         WHERE needs_reconcile = 1",
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(tracked_file_from_row).collect())
}

/// Clear the reconcile flag for a tracked file within a transaction
pub async fn clear_reconcile_flag_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    file_id: i64,
) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 0, reconcile_reason = NULL, updated_at = ?1
         WHERE file_id = ?2",
    )
    .bind(&now)
    .bind(file_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}
