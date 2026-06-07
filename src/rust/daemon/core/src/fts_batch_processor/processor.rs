//! FtsBatchProcessor: adaptive batch/single-file processing for code_lines.
//!
//! FTS5 index is maintained incrementally -- individual rows are inserted/deleted
//! in the FTS index inline with code_lines changes, avoiding expensive full rebuilds.

use tracing::{debug, info, warn};

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
/// FTS5 updates are performed incrementally within each transaction --
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
    /// On processing failure the batch is requeued into `pending` so that a
    /// subsequent `flush` call can retry without data loss.
    ///
    /// Returns statistics about the processing run.
    pub async fn flush(&mut self, queue_depth: usize) -> Result<BatchStats, SearchDbError> {
        if self.pending.is_empty() {
            return Ok(BatchStats::default());
        }

        let start = std::time::Instant::now();
        let mut changes = std::mem::take(&mut self.pending);

        // Hard cap (#103): drop files whose content alone would blow the
        // memory budget. Line search of huge minified/generated files is
        // not worth multi-GB diff memory; Qdrant ingestion is unaffected.
        let before = changes.len();
        changes.retain(|c| {
            let bytes = c.content_bytes();
            if bytes > self.config.hard_cap_bytes {
                warn!(
                    "FTS5: skipping {} — {} content bytes exceed hard cap {} (WQM_FTS5_HARD_CAP)",
                    c.file_path, bytes, self.config.hard_cap_bytes
                );
                return false;
            }
            true
        });
        let files_skipped_too_large = before - changes.len();
        if changes.is_empty() {
            return Ok(BatchStats {
                files_skipped_too_large,
                processing_time_ms: start.elapsed().as_millis() as u64,
                ..Default::default()
            });
        }

        // Bytes budget (#103): batch mode holds every pending file's old+new
        // content in one transaction. Above the budget, fall back to
        // single-file mode so memory stays bounded per file.
        let total_bytes: usize = changes.iter().map(FileChange::content_bytes).sum();
        let over_budget = total_bytes > self.config.single_mode_threshold_bytes;
        let batch_mode = self.should_use_batch_mode(queue_depth) && !over_budget;
        if over_budget {
            info!(
                "FTS5: accumulated {} content bytes exceed single-mode threshold {} \
                 (WQM_FTS5_SINGLE_MODE_THRESHOLD); forcing single-file mode",
                total_bytes, self.config.single_mode_threshold_bytes
            );
        }

        let result = if batch_mode {
            info!(
                "FTS5 batch mode: processing {} file changes in single transaction (queue_depth={})",
                changes.len(),
                queue_depth
            );
            self.process_batch(changes).await
        } else {
            debug!(
                "FTS5 single-file mode: processing {} file changes individually (queue_depth={})",
                changes.len(),
                queue_depth
            );
            self.process_individually(changes).await
        };

        match result {
            Ok(mut stats) => {
                stats.batch_mode = batch_mode;
                stats.files_skipped_too_large = files_skipped_too_large;
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
            Err((requeue, err)) => {
                let count = requeue.len();
                warn!(
                    "FTS5 flush failed, requeuing {} file changes: {}",
                    count, err
                );
                self.pending = requeue;
                Err(err)
            }
        }
    }

    /// Process all file changes in a single batch transaction.
    ///
    /// On error, returns the full change list for requeue alongside the error.
    async fn process_batch(
        &self,
        changes: Vec<FileChange>,
    ) -> Result<BatchStats, (Vec<FileChange>, SearchDbError)> {
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
        let mut tx = pool.begin().await.map_err(|e| {
            let requeue: Vec<FileChange> = file_diffs.iter().map(|(c, _)| c.clone()).collect();
            (requeue, SearchDbError::from(e))
        })?;

        for (change, diff) in &file_diffs {
            let file_stats =
                apply_diff_to_code_lines(&mut tx, change.file_id, diff, &change.new_content)
                    .await
                    .map_err(|e| {
                        let requeue: Vec<FileChange> =
                            file_diffs.iter().map(|(c, _)| c.clone()).collect();
                        (requeue, e)
                    })?;

            stats.lines_inserted += file_stats.lines_inserted;
            stats.lines_updated += file_stats.lines_updated;
            stats.lines_deleted += file_stats.lines_deleted;
            stats.lines_unchanged += file_stats.lines_unchanged;
            stats.files_processed += 1;

            upsert_file_metadata(&mut tx, change).await.map_err(|e| {
                let requeue: Vec<FileChange> = file_diffs.iter().map(|(c, _)| c.clone()).collect();
                (requeue, e)
            })?;
        }

        tx.commit().await.map_err(|e| {
            let requeue: Vec<FileChange> = file_diffs.iter().map(|(c, _)| c.clone()).collect();
            (requeue, SearchDbError::from(e))
        })?;

        // Phase 3: Check if any files need rebalancing (outside transaction)
        for (change, _) in &file_diffs {
            // Rebalance failures are non-fatal -- log and continue
            if let Err(e) = self.maybe_rebalance(change.file_id, &mut stats).await {
                warn!(
                    "FTS5 rebalance failed for file_id={}: {}",
                    change.file_id, e
                );
            }
        }

        Ok(stats)
    }

    /// Process each file change individually with incremental FTS5 updates.
    ///
    /// On error, returns unprocessed changes for requeue alongside the error.
    async fn process_individually(
        &self,
        changes: Vec<FileChange>,
    ) -> Result<BatchStats, (Vec<FileChange>, SearchDbError)> {
        let pool = self.search_db.pool();
        let mut stats = BatchStats::default();

        for (i, change) in changes.iter().enumerate() {
            let diff = compute_line_diff(&change.old_content, &change.new_content);

            let mut tx = pool.begin().await.map_err(|e| {
                let remaining = changes[i..].to_vec();
                (remaining, SearchDbError::from(e))
            })?;

            let file_stats =
                apply_diff_to_code_lines(&mut tx, change.file_id, &diff, &change.new_content)
                    .await
                    .map_err(|e| {
                        let remaining = changes[i..].to_vec();
                        (remaining, e)
                    })?;

            upsert_file_metadata(&mut tx, change).await.map_err(|e| {
                let remaining = changes[i..].to_vec();
                (remaining, e)
            })?;

            tx.commit().await.map_err(|e| {
                let remaining = changes[i..].to_vec();
                (remaining, SearchDbError::from(e))
            })?;

            stats.lines_inserted += file_stats.lines_inserted;
            stats.lines_updated += file_stats.lines_updated;
            stats.lines_deleted += file_stats.lines_deleted;
            stats.lines_unchanged += file_stats.lines_unchanged;
            stats.files_processed += 1;

            self.maybe_rebalance(change.file_id, &mut stats)
                .await
                .map_err(|e| {
                    let remaining = changes[i + 1..].to_vec();
                    (remaining, e)
                })?;
        }

        Ok(stats)
    }

    /// Check rebalance condition for a single file and trigger if needed.
    async fn maybe_rebalance(
        &self,
        file_id: i64,
        stats: &mut BatchStats,
    ) -> Result<(), SearchDbError> {
        if let Some(min_gap) = self.search_db.min_seq_gap(file_id).await? {
            if SearchDbManager::needs_rebalance(min_gap) {
                self.search_db.rebalance_file_seqs(file_id).await?;
                stats.rebalances_triggered += 1;
            }
        }
        Ok(())
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

        // Hard cap (#103): mirror the flush() guard for the no-diff path.
        if content.len() > self.config.hard_cap_bytes {
            warn!(
                "FTS5: skipping full rewrite of file_id={} — {} content bytes exceed \
                 hard cap {} (WQM_FTS5_HARD_CAP)",
                file_id,
                content.len(),
                self.config.hard_cap_bytes
            );
            return Ok(BatchStats {
                files_skipped_too_large: 1,
                processing_time_ms: start.elapsed().as_millis() as u64,
                ..Default::default()
            });
        }

        let pool = self.search_db.pool();
        let lines: Vec<&str> = content.split('\n').collect();

        let mut tx = pool.begin().await?;

        // Delete old FTS5 entries incrementally
        delete_old_fts5_entries(&mut tx, file_id).await?;

        // Delete existing lines for this file
        sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await?;

        // Insert all new lines with standard gaps and 1-based line numbers
        insert_new_lines(&mut tx, file_id, &lines).await?;

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

        Ok(BatchStats {
            files_processed: 1,
            lines_inserted: lines.len(),
            batch_mode: false,
            processing_time_ms: start.elapsed().as_millis() as u64,
            ..Default::default()
        })
    }

    /// Delete all code_lines and file_metadata for a file.
    ///
    /// FTS5 row reads and deletes happen within the same transaction to
    /// prevent TOCTOU races with concurrent writers.
    pub async fn delete_file(&self, file_id: i64) -> Result<usize, SearchDbError> {
        let pool = self.search_db.pool();

        let mut tx = pool.begin().await?;

        // Fetch FTS5 rows inside the transaction to avoid TOCTOU race:
        // a concurrent writer inserting rows between a pre-transaction read
        // and the delete loop would leave orphaned FTS5 entries.
        let old_rows = fetch_fts5_rows_in_tx(&mut tx, file_id).await?;

        if old_rows.is_empty() {
            // Still clean up file_metadata if it exists
            sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_SQL)
                .bind(file_id)
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            return Ok(0);
        }

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

/// Upsert file_metadata for search scoping within a transaction.
async fn upsert_file_metadata(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    change: &FileChange,
) -> Result<(), SearchDbError> {
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(change.file_id)
        .bind(&change.tenant_id)
        .bind(&change.branch)
        .bind(&change.file_path)
        .bind(&change.base_point)
        .bind(&change.relative_path)
        .bind(&change.file_hash)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

/// Fetch existing FTS5 rows (line_id, content) for incremental deletion
/// within a transaction. This ensures reads and subsequent deletes are
/// atomic, preventing TOCTOU races with concurrent writers.
async fn fetch_fts5_rows_in_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
) -> Result<Vec<(i64, String)>, SearchDbError> {
    let rows = sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
        .bind(file_id)
        .fetch_all(&mut **tx)
        .await?;
    Ok(rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            (r.get::<i64, _>("line_id"), r.get::<String, _>("content"))
        })
        .collect())
}

/// Delete old FTS5 entries incrementally for a file within a transaction.
async fn delete_old_fts5_entries(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
) -> Result<(), SearchDbError> {
    let old_rows: Vec<(i64, String)> = {
        let rows = sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .fetch_all(&mut **tx)
            .await?;
        rows.iter()
            .map(|r| {
                use sqlx::Row;
                (r.get::<i64, _>("line_id"), r.get::<String, _>("content"))
            })
            .collect()
    };

    for (line_id, old_content) in &old_rows {
        sqlx::query(FTS5_DELETE_ROW_SQL)
            .bind(*line_id)
            .bind(old_content.as_str())
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Insert all new lines with standard gaps and 1-based line numbers,
/// plus corresponding FTS5 entries, within a transaction.
async fn insert_new_lines(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_id: i64,
    lines: &[&str],
) -> Result<(), SearchDbError> {
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
        .execute(&mut **tx)
        .await?;

        let new_line_id = result.last_insert_rowid();
        sqlx::query(FTS5_INSERT_ROW_SQL)
            .bind(new_line_id)
            .bind(*line)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}
