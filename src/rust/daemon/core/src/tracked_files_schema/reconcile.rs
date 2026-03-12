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

/// Capability upgrade reason codes for Uplift operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpgradeReason {
    /// Tree-sitter grammar became available (file was text-chunked)
    GrammarAvailable,
    /// LSP server became available (file has no LSP enrichment)
    LspAvailable,
    /// Previous enrichment failed and should be retried
    EnrichmentRetry,
}

impl UpgradeReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GrammarAvailable => "grammar_available",
            Self::LspAvailable => "lsp_available",
            Self::EnrichmentRetry => "enrichment_retry",
        }
    }
}

/// Find files needing capability upgrade for a given tenant and language.
///
/// Returns `(file_id, file_path, branch, collection)` tuples for files where:
/// - `treesitter_status` is 'none', 'failed', or 'skipped' (grammar upgrade), or
/// - `lsp_status` is 'none' or 'failed' (LSP upgrade)
pub async fn get_files_needing_upgrade(
    pool: &SqlitePool,
    tenant_id: &str,
    reason: UpgradeReason,
    language: Option<&str>,
) -> Result<Vec<(i64, String, String, String)>, sqlx::Error> {
    let status_filter = match reason {
        UpgradeReason::GrammarAvailable => "treesitter_status IN ('none', 'failed', 'skipped')",
        UpgradeReason::LspAvailable => "lsp_status IN ('none', 'failed')",
        UpgradeReason::EnrichmentRetry => "lsp_status = 'failed' OR treesitter_status = 'failed'",
    };

    let query = if language.is_some() {
        format!(
            "SELECT tf.file_id, tf.file_path, tf.branch, tf.collection
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.id
             WHERE wf.tenant_id = ?1
               AND ({})
               AND tf.language = ?2",
            status_filter
        )
    } else {
        format!(
            "SELECT tf.file_id, tf.file_path, tf.branch, tf.collection
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.id
             WHERE wf.tenant_id = ?1
               AND ({})",
            status_filter
        )
    };

    let rows = if let Some(lang) = language {
        sqlx::query(&query)
            .bind(tenant_id)
            .bind(lang)
            .fetch_all(pool)
            .await?
    } else {
        sqlx::query(&query).bind(tenant_id).fetch_all(pool).await?
    };

    use sqlx::Row;
    Ok(rows
        .iter()
        .map(|r| {
            (
                r.get::<i64, _>("file_id"),
                r.get::<String, _>("file_path"),
                r.get::<String, _>("branch"),
                r.get::<String, _>("collection"),
            )
        })
        .collect())
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
