//! FtsBatchProcessor: adaptive batch/single-file processing for code_lines.
//!
//! FTS5 index is maintained incrementally -- individual rows are inserted/deleted
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

    /// Determine processing mode based on queue depth and pending change sizes.
    ///
    /// Returns `true` (batch mode) only when both:
    /// 1. `queue_depth > burst_threshold` — there's enough work to amortize
    ///    the batched transaction.
    /// 2. No pending change exceeds [`single_mode_line_threshold`] — batch
    ///    mode's Phase 1 loads every diff into RAM before opening the
    ///    transaction, so one giant file in the same window as small ones
    ///    can spike RSS into the multi-GB range and trip the queue
    ///    processor's memory-pressure pause-loop.
    pub fn should_use_batch_mode(&self, queue_depth: usize) -> bool {
        self.should_use_batch_mode_with(queue_depth, single_mode_line_threshold())
    }

    /// Same decision as [`should_use_batch_mode`], with the line threshold
    /// passed in explicitly. Lets tests cover the size-guard branch without
    /// fighting the env-cached OnceLock in [`single_mode_line_threshold`].
    pub(super) fn should_use_batch_mode_with(
        &self,
        queue_depth: usize,
        line_threshold: usize,
    ) -> bool {
        if queue_depth <= self.config.burst_threshold {
            return false;
        }
        if line_threshold > 0
            && self
                .pending
                .iter()
                .any(|c| line_count_estimate(&c.new_content) > line_threshold)
        {
            return false;
        }
        true
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

        // Collapse duplicate file_ids, keeping the LAST change per file.
        //
        // Two queue items for the same file (e.g. a re-enqueue plus a
        // reconciliation pass, or two quick edits) can be flushed together.
        // Each change's diff is computed against the content baseline captured
        // when its item was prepared, but both target the same code_lines
        // rows. Applying both — whether in one batch transaction or as two
        // sequential single-file transactions — makes the second insert the
        // same `(file_id, seq)` rows the first already wrote, hitting
        // `UNIQUE constraint failed: code_lines.file_id, code_lines.seq`
        // (SQLite error 2067) and failing the item.
        //
        // The last change carries the newest content and supersedes the
        // earlier ones, so keeping only it reduces the flush to the
        // already-correct one-change-per-file case. The dropped items' queue
        // rows are still finalized as success by the batch writer (the file
        // ends up correctly indexed by the surviving change).
        let changes = dedup_changes_keep_last(changes);

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
    async fn process_batch(&self, changes: Vec<FileChange>) -> Result<BatchStats, SearchDbError> {
        let pool = self.search_db.pool();
        let mut stats = BatchStats::default();

        // NOTE: callers reach this via `flush`, which has already collapsed
        // duplicate file_ids (see `dedup_changes_keep_last`). Applying two
        // changes for the same file_id in this single transaction would
        // violate `UNIQUE(code_lines.file_id, code_lines.seq)`.

        // Phase 1: Compute all diffs upfront (CPU-bound, no DB)
        let mut file_diffs: Vec<(FileChange, crate::line_diff::DiffResult)> =
            Vec::with_capacity(changes.len());
        for change in changes {
            let diff = compute_line_diff(&change.old_content, &change.new_content);
            file_diffs.push((change, diff));
        }

        // Phase 2: Apply all changes in a single transaction
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

            upsert_file_metadata(&mut tx, change).await?;
        }

        tx.commit().await?;

        // Phase 3: Check if any files need rebalancing (outside transaction)
        for (change, _) in &file_diffs {
            self.maybe_rebalance(change.file_id, &mut stats).await?;
        }

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

            let mut tx = pool.begin().await?;

            let file_stats =
                apply_diff_to_code_lines(&mut tx, change.file_id, &diff, &change.new_content)
                    .await?;

            upsert_file_metadata(&mut tx, &change).await?;

            tx.commit().await?;

            stats.lines_inserted += file_stats.lines_inserted;
            stats.lines_updated += file_stats.lines_updated;
            stats.lines_deleted += file_stats.lines_deleted;
            stats.lines_unchanged += file_stats.lines_unchanged;
            stats.files_processed += 1;

            self.maybe_rebalance(change.file_id, &mut stats).await?;
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

        // Upsert file_metadata. v7 added size_bytes (8th param); v8 added
        // fts5_skipped (9th). full_rewrite always writes fts5_skipped=0 —
        // see upsert_file_metadata.
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(file_id)
            .bind(tenant_id)
            .bind(branch)
            .bind(file_path)
            .bind(base_point)
            .bind(relative_path)
            .bind(file_hash)
            .bind(content.len() as i64) // size_bytes
            .bind(0_i64) // fts5_skipped
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
    /// FTS5 entries are removed incrementally.
    pub async fn delete_file(&self, file_id: i64) -> Result<usize, SearchDbError> {
        let pool = self.search_db.pool();

        let old_rows = fetch_fts5_rows(pool, file_id).await?;

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

/// Collapse a batch's file changes so each `file_id` appears at most once,
/// keeping the LAST change (newest content) and preserving its relative
/// order. Applying two changes for the same file in one transaction causes a
/// `UNIQUE constraint failed: code_lines.file_id, code_lines.seq` violation —
/// see the call site in [`FtsBatchProcessor::process_batch`].
fn dedup_changes_keep_last(changes: Vec<FileChange>) -> Vec<FileChange> {
    use std::collections::HashMap;
    // Index of the last occurrence of each file_id.
    let mut last_idx: HashMap<i64, usize> = HashMap::with_capacity(changes.len());
    for (i, change) in changes.iter().enumerate() {
        last_idx.insert(change.file_id, i);
    }
    changes
        .into_iter()
        .enumerate()
        .filter(|(i, change)| last_idx.get(&change.file_id) == Some(i))
        .map(|(_, change)| change)
        .collect()
}

/// Upsert file_metadata for search scoping within a transaction.
///
/// Always writes `fts5_skipped = 0` because reaching this function means
/// the file went through the normal batch/diff path — the hard cap fork in
/// [`enforce_fts5_hard_cap_skip`] is the only place that ever writes 1.
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
        .bind(change.size_bytes)
        .bind(0_i64) // fts5_skipped: not skipped — we did the FTS5 work
        .execute(&mut **tx)
        .await?;
    Ok(())
}

/// Fetch existing FTS5 rows (line_id, content) for incremental deletion.
async fn fetch_fts5_rows(
    pool: &sqlx::SqlitePool,
    file_id: i64,
) -> Result<Vec<(i64, String)>, SearchDbError> {
    let rows = sqlx::query("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
        .bind(file_id)
        .fetch_all(pool)
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

/// Threshold (in lines) above which a single oversize change forces the
/// processor into single-file mode regardless of queue depth.
///
/// Read from `WQM_FTS5_SINGLE_MODE_THRESHOLD` once at first call and
/// cached. Returns 0 when unset, malformed, or non-positive — that
/// disables the guard, restoring the queue-depth-only decision.
///
/// Rationale: batch mode loads ALL pending diffs into RAM upfront before
/// opening the SQLite transaction. When one file in the window is 600k+
/// lines, that working set alone can drive RSS over 10GB and trip the
/// `UnifiedQueueProcessor` memory-pressure pause-loop. Forcing single-mode
/// isolates the giant file into its own transaction so the rest of the
/// window commits quickly while we still pay the per-file cost for the
/// giant.
pub(super) fn single_mode_line_threshold() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("WQM_FTS5_SINGLE_MODE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Cheap line-count estimate by counting `\n` bytes. Off by one when the
/// final line lacks a trailing newline — harmless for threshold checks.
pub fn line_count_estimate(s: &str) -> usize {
    s.bytes().filter(|b| *b == b'\n').count()
}

/// Hard line cap above which a file is excluded from the FTS5 / `code_lines`
/// pipeline entirely.
///
/// Read from `WQM_FTS5_HARD_CAP` once at first call and cached. Returns 0
/// when unset, malformed, or non-positive — that disables the cap.
///
/// When the cap fires, [`enforce_fts5_hard_cap_skip`] writes a
/// `file_metadata` row with `fts5_skipped = 1` (so the admin UI / Grafana
/// can surface the skip) but inserts NO `code_lines` and NO FTS5 trigrams.
/// The file remains searchable via the semantic-embedding pipeline, which
/// is separately bounded and not affected by this cap.
///
/// Rationale: even single-mode processing of a 600k-line file holds
/// `old_content + new_content + DiffResult` in RAM and runs hundreds of
/// thousands of FTS5 inserts in a single transaction. Both push RSS into
/// the multi-GB range. The cap stops that work from starting at all.
pub fn hard_cap_line_threshold() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("WQM_FTS5_HARD_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Mark a file as FTS5-skipped: clear any existing `code_lines` / FTS5
/// rows for it, then upsert `file_metadata` with `fts5_skipped = 1`.
///
/// Called by the FTS5 ingestion entry points
/// ([`crate::strategies::processing::file::fts5_index`]) when
/// [`hard_cap_line_threshold`] fires for a file. Two SQL steps (delete +
/// upsert) because they run against different statement scopes; a crash
/// between them leaves the file with no rows but no metadata, which the
/// next ingestion attempt will heal idempotently.
///
/// `line_count` is logged as the reason. The actual decision must already
/// have been made by the caller — this function does NOT re-check the cap.
#[allow(clippy::too_many_arguments)]
pub async fn enforce_fts5_hard_cap_skip(
    search_db: &SearchDbManager,
    file_id: i64,
    tenant_id: &str,
    branch: Option<&str>,
    file_path: &str,
    base_point: Option<&str>,
    relative_path: Option<&str>,
    file_hash: Option<&str>,
    size_bytes: Option<i64>,
    line_count: usize,
) -> Result<(), SearchDbError> {
    // Step 1: clear any prior code_lines + FTS5 rows for this file.
    // `delete_file` is idempotent — returns Ok(0) if nothing was indexed
    // (the common case: hard cap fires on first ingestion of a file).
    let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
    let _ = processor.delete_file(file_id).await?;

    // Step 2: upsert file_metadata with fts5_skipped = 1.
    // delete_file also wipes file_metadata, so this is a fresh INSERT in
    // most cases; the ON CONFLICT clause handles the unlikely race where
    // a concurrent path re-created the row between the two steps.
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(file_id)
        .bind(tenant_id)
        .bind(branch)
        .bind(file_path)
        .bind(base_point)
        .bind(relative_path)
        .bind(file_hash)
        .bind(size_bytes)
        .bind(1_i64) // fts5_skipped = TRUE
        .execute(search_db.pool())
        .await?;

    // Bump the per-(tenant, branch) hard-cap counter so Grafana can chart
    // how often the guard fires. NULL branch → "(none)" matches the gauge
    // convention used by `set_file_metadata_stats`.
    crate::monitoring::METRICS.inc_fts5_skipped(tenant_id, branch.unwrap_or("(none)"));

    info!(
        "FTS5 hard-cap: skipping {} ({} lines > cap {}) — marked fts5_skipped=1, no code_lines written",
        file_path,
        line_count,
        hard_cap_line_threshold()
    );

    Ok(())
}

#[cfg(test)]
mod dedup_tests {
    use super::{dedup_changes_keep_last, FileChange};

    fn change(file_id: i64, new_content: &str) -> FileChange {
        FileChange {
            file_id,
            old_content: String::new(),
            new_content: new_content.to_string(),
            tenant_id: "t".to_string(),
            branch: None,
            file_path: format!("f{file_id}.rs"),
            base_point: None,
            relative_path: None,
            file_hash: None,
            size_bytes: Some(new_content.len() as i64),
        }
    }

    #[test]
    fn keeps_last_change_per_file_id_and_preserves_order() {
        // file 1 appears twice (v1 then v3), file 2 once. Applying both
        // copies of file 1 in one batch tx is what triggers the UNIQUE
        // constraint violation, so only the newest (v3) must survive.
        let input = vec![
            change(1, "v1"),
            change(2, "v2"),
            change(1, "v3"),
        ];
        let out = dedup_changes_keep_last(input);

        assert_eq!(out.len(), 2, "one entry per file_id");
        // file 2 keeps its original position (before file 1's last slot).
        assert_eq!(out[0].file_id, 2);
        assert_eq!(out[1].file_id, 1);
        // The surviving file-1 change is the LATEST content.
        assert_eq!(out[1].new_content, "v3");
    }

    #[test]
    fn no_duplicates_is_identity() {
        let input = vec![change(1, "a"), change(2, "b"), change(3, "c")];
        let out = dedup_changes_keep_last(input);
        let ids: Vec<i64> = out.iter().map(|c| c.file_id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }
}
