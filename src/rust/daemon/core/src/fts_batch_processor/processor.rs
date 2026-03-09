//! FtsBatchProcessor: adaptive batch/single-file processing for code_lines.
//!
//! FTS5 index is maintained incrementally — individual rows are inserted/deleted
//! in the FTS index inline with code_lines changes, avoiding expensive full rebuilds.

use tracing::{debug, info};

use crate::code_lines_schema::{initial_seq, FTS5_DELETE_ROW_SQL, FTS5_INSERT_ROW_SQL};
use crate::line_diff::compute_line_diff;
use crate::search_db::{SearchDbError, SearchDbManager};

use super::diff_apply::apply_diff_to_code_lines;
use super::{BatchStats, FileChange, FtsBatchConfig};

/// Adaptive batch processor for FTS5 code_lines updates.
///
/// Accumulates file changes and flushes them either individually (single-file mode)
/// or as a batch (batch mode) depending on queue depth.
///
/// FTS5 updates are performed incrementally within each transaction —
/// no full rebuild is needed after processing.
pub struct FtsBatchProcessor<'a> {
    search_db: &'a SearchDbManager,
    config: FtsBatchConfig,
    pending: Vec<FileChange>,
}

impl<'a> FtsBatchProcessor<'a> {
    /// Create a new batch processor with the given search database and config.
    pub fn new(search_db: &'a SearchDbManager, config: FtsBatchConfig) -> Self {
        Self {
            search_db,
            config,
            pending: Vec::new(),
        }
    }

    /// Add a file change to the pending queue.
    pub fn add_change(&mut self, change: FileChange) {
        self.pending.push(change);
    }

    /// Number of pending file changes.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Determine processing mode based on queue depth.
    pub fn should_use_batch_mode(&self, queue_depth: usize) -> bool {
        queue_depth > self.config.burst_threshold
    }

    /// Flush all pending changes, choosing single-file or batch mode.
    ///
    /// - If `queue_depth > burst_threshold`, uses batch mode (single transaction).
    /// - Otherwise, processes each file individually.
    ///
    /// Returns statistics about the processing run.
    pub async fn flush(&mut self, queue_depth: usize) -> Result<BatchStats, SearchDbError> {
        if self.pending.is_empty() {
            return Ok(BatchStats::default());
        }

        let start = std::time::Instant::now();
        let batch_mode = self.should_use_batch_mode(queue_depth);
        let changes = std::mem::take(&mut self.pending);

        let mut stats = if batch_mode {
            info!(
                "FTS5 batch mode: processing {} file changes in single transaction (queue_depth={})",
                changes.len(),
                queue_depth
            );
            self.process_batch(changes).await?
        } else {
            debug!(
                "FTS5 single-file mode: processing {} file changes individually (queue_depth={})",
                changes.len(),
                queue_depth
            );
            self.process_individually(changes).await?
        };

        stats.batch_mode = batch_mode;
        stats.processing_time_ms = start.elapsed().as_millis() as u64;

        info!(
            "FTS5 flush complete: {} files, {} inserted, {} updated, {} deleted, {} unchanged, {} rebalances, {}ms ({})",
            stats.files_processed,
            stats.lines_inserted,
            stats.lines_updated,
            stats.lines_deleted,
            stats.lines_unchanged,
            stats.rebalances_triggered,
            stats.processing_time_ms,
            if stats.batch_mode { "batch" } else { "single-file" }
        );

        Ok(stats)
    }

    /// Process all file changes in a single batch transaction.
    ///
    /// 1. Compute diffs for all files
    /// 2. BEGIN TRANSACTION
    /// 3. Apply all INSERT/UPDATE/DELETE to code_lines (with inline FTS5 updates)
    /// 4. Upsert file_metadata for all files
    /// 5. COMMIT
    async fn process_batch(&self, changes: Vec<FileChange>) -> Result<BatchStats, SearchDbError> {
        let pool = self.search_db.pool();
        let mut stats = BatchStats::default();

        // Phase 1: Compute all diffs upfront (CPU-bound, no DB)
        let mut file_diffs: Vec<(FileChange, crate::line_diff::DiffResult)> =
            Vec::with_capacity(changes.len());
        for change in changes {
            let diff = compute_line_diff(&change.old_content, &change.new_content);
            file_diffs.push((change, diff));
        }

        // Phase 2: Apply all changes in a single transaction
        // (diff_apply handles FTS5 incrementally within the transaction)
        let mut tx = pool.begin().await?;

        for (change, diff) in &file_diffs {
            let file_stats =
                apply_diff_to_code_lines(&mut tx, change.file_id, diff, &change.new_content)
                    .await?;

            stats.lines_inserted += file_stats.lines_inserted;
            stats.lines_updated += file_stats.lines_updated;
            stats.lines_deleted += file_stats.lines_deleted;
            stats.lines_unchanged += file_stats.lines_unchanged;
            stats.files_processed += 1;

            // Upsert file_metadata for search scoping
            sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
                .bind(change.file_id)
                .bind(&change.tenant_id)
                .bind(&change.branch)
                .bind(&change.file_path)
                .bind(&change.base_point)
                .bind(&change.relative_path)
                .bind(&change.file_hash)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;

        // Phase 3: Check if any files need rebalancing (outside transaction)
        for (change, _) in &file_diffs {
            if let Some(min_gap) = self.search_db.min_seq_gap(change.file_id).await? {
                if SearchDbManager::needs_rebalance(min_gap) {
                    self.search_db.rebalance_file_seqs(change.file_id).await?;
                    stats.rebalances_triggered += 1;
                }
            }
        }

        // No FTS5 rebuild needed — updates were applied incrementally

        Ok(stats)
    }

    /// Process each file change individually with incremental FTS5 updates.
    async fn process_individually(
        &self,
        changes: Vec<FileChange>,
    ) -> Result<BatchStats, SearchDbError> {
        let pool = self.search_db.pool();
        let mut stats = BatchStats::default();

        for change in changes {
            let diff = compute_line_diff(&change.old_content, &change.new_content);

            // Apply diff in a per-file transaction
            // (diff_apply handles FTS5 incrementally within the transaction)
            let mut tx = pool.begin().await?;

            let file_stats =
                apply_diff_to_code_lines(&mut tx, change.file_id, &diff, &change.new_content)
                    .await?;

            // Upsert file_metadata
            sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
                .bind(change.file_id)
                .bind(&change.tenant_id)
                .bind(&change.branch)
                .bind(&change.file_path)
                .bind(&change.base_point)
                .bind(&change.relative_path)
                .bind(&change.file_hash)
                .execute(&mut *tx)
                .await?;

            tx.commit().await?;

            stats.lines_inserted += file_stats.lines_inserted;
            stats.lines_updated += file_stats.lines_updated;
            stats.lines_deleted += file_stats.lines_deleted;
            stats.lines_unchanged += file_stats.lines_unchanged;
            stats.files_processed += 1;

            // Check rebalance per file
            if let Some(min_gap) = self.search_db.min_seq_gap(change.file_id).await? {
                if SearchDbManager::needs_rebalance(min_gap) {
                    self.search_db.rebalance_file_seqs(change.file_id).await?;
                    stats.rebalances_triggered += 1;
                }
            }

            // No FTS5 rebuild needed — updates were applied incrementally
        }

        Ok(stats)
    }

    /// Replace all code_lines for a file with new content (full rewrite).
    ///
    /// Use when there is no old content to diff against (new file ingestion).
    /// FTS5 index is updated incrementally (old entries deleted, new entries inserted).
    #[allow(clippy::too_many_arguments)]
    pub async fn full_rewrite(
        &self,
        file_id: i64,
        content: &str,
        tenant_id: &str,
        branch: Option<&str>,
        file_path: &str,
        base_point: Option<&str>,
        relative_path: Option<&str>,
        file_hash: Option<&str>,
    ) -> Result<BatchStats, SearchDbError> {
        let start = std::time::Instant::now();
        let pool = self.search_db.pool();
        let lines: Vec<&str> = content.split('\n').collect();

        let mut tx = pool.begin().await?;

        // Fetch existing rows for incremental FTS5 delete
        let old_rows: Vec<(i64, String)> = {
            let rows = sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
                .bind(file_id)
                .fetch_all(&mut *tx)
                .await?;
            rows.iter()
                .map(|r| {
                    use sqlx::Row;
                    (r.get::<i64, _>("line_id"), r.get::<String, _>("content"))
                })
                .collect()
        };

        // Delete old FTS5 entries incrementally
        for (line_id, old_content) in &old_rows {
            sqlx::query(FTS5_DELETE_ROW_SQL)
                .bind(*line_id)
                .bind(old_content.as_str())
                .execute(&mut *tx)
                .await?;
        }

        // Delete existing lines for this file
        sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await?;

        // Insert all new lines with standard gaps and 1-based line numbers
        for (i, line) in lines.iter().enumerate() {
            let seq = initial_seq(i);
            let line_number = (i + 1) as i64;
            let result = sqlx::query(
                "INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)",
            )
            .bind(file_id)
            .bind(seq)
            .bind(*line)
            .bind(line_number)
            .execute(&mut *tx)
            .await?;

            // Insert new FTS5 entry
            let new_line_id = result.last_insert_rowid();
            sqlx::query(FTS5_INSERT_ROW_SQL)
                .bind(new_line_id)
                .bind(*line)
                .execute(&mut *tx)
                .await?;
        }

        // Upsert file_metadata
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(file_id)
            .bind(tenant_id)
            .bind(branch)
            .bind(file_path)
            .bind(base_point)
            .bind(relative_path)
            .bind(file_hash)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;

        // No FTS5 rebuild needed — updates were applied incrementally

        Ok(BatchStats {
            files_processed: 1,
            lines_inserted: lines.len(),
            batch_mode: false,
            processing_time_ms: start.elapsed().as_millis() as u64,
            ..Default::default()
        })
    }

    /// Delete all code_lines and file_metadata for a file.
    /// FTS5 entries are removed incrementally.
    pub async fn delete_file(&self, file_id: i64) -> Result<usize, SearchDbError> {
        let pool = self.search_db.pool();

        // Fetch existing rows for incremental FTS5 delete
        let old_rows: Vec<(i64, String)> = {
            let rows = sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
                .bind(file_id)
                .fetch_all(pool)
                .await?;
            rows.iter()
                .map(|r| {
                    use sqlx::Row;
                    (r.get::<i64, _>("line_id"), r.get::<String, _>("content"))
                })
                .collect()
        };

        if old_rows.is_empty() {
            // Still clean up file_metadata if it exists
            sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_SQL)
                .bind(file_id)
                .execute(pool)
                .await?;
            return Ok(0);
        }

        let mut tx = pool.begin().await?;

        // Delete FTS5 entries incrementally
        for (line_id, old_content) in &old_rows {
            sqlx::query(FTS5_DELETE_ROW_SQL)
                .bind(*line_id)
                .bind(old_content.as_str())
                .execute(&mut *tx)
                .await?;
        }

        // Delete code_lines
        let result = sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await?;

        // Delete file_metadata
        sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_SQL)
            .bind(file_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;

        Ok(result.rows_affected() as usize)
    }

    /// Delete all code_lines and file_metadata for a tenant (project deletion).
    /// FTS5 entries are removed incrementally.
    pub async fn delete_tenant(&self, tenant_id: &str) -> Result<usize, SearchDbError> {
        let pool = self.search_db.pool();

        // Get all file_ids for the tenant from file_metadata
        let file_ids: Vec<i64> =
            sqlx::query_scalar("SELECT file_id FROM file_metadata WHERE tenant_id = ?1")
                .bind(tenant_id)
                .fetch_all(pool)
                .await?;

        if file_ids.is_empty() {
            return Ok(0);
        }

        let mut tx = pool.begin().await?;
        let mut total_deleted = 0usize;

        for file_id in &file_ids {
            // Fetch rows for incremental FTS5 delete
            let old_rows: Vec<(i64, String)> = {
                let rows =
                    sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
                        .bind(*file_id)
                        .fetch_all(&mut *tx)
                        .await?;
                rows.iter()
                    .map(|r| {
                        use sqlx::Row;
                        (r.get::<i64, _>("line_id"), r.get::<String, _>("content"))
                    })
                    .collect()
            };

            // Delete FTS5 entries incrementally
            for (line_id, old_content) in &old_rows {
                sqlx::query(FTS5_DELETE_ROW_SQL)
                    .bind(*line_id)
                    .bind(old_content.as_str())
                    .execute(&mut *tx)
                    .await?;
            }

            let result = sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
                .bind(*file_id)
                .execute(&mut *tx)
                .await?;
            total_deleted += result.rows_affected() as usize;
        }

        // Delete all file_metadata for tenant
        sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_BY_TENANT_SQL)
            .bind(tenant_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;

        Ok(total_deleted)
    }
}
