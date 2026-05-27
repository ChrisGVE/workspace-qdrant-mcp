//! FTS5 code search index updates.
//!
//! Updates the FTS5 full-text search index for a single file after Qdrant
//! ingestion. Reads file from disk, diffs against cached content, and applies
//! line-level changes to code_lines + FTS5.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, warn};

use crate::fts_batch_processor::{
    enforce_fts5_hard_cap_skip, hard_cap_line_threshold, line_count_estimate, BatchStats,
    FileChange, FtsBatchConfig, FtsBatchProcessor,
};
use crate::indexed_content_schema;
use crate::search_db::{Fts5WorkItem, SearchDbError, SearchDbManager};
use wqm_common::hashing::compute_content_hash;

/// Outcome of `update_fts5_for_file_or_enqueue` — tells the caller how to
/// handle `search_status` for the queue item.
pub(super) enum Fts5Outcome {
    /// File was processed inline; caller should flip `search_status=done`.
    /// The inner bool mirrors the legacy `update_fts5_for_file` return: true
    /// if anything was actually written, false if the file was skipped.
    Inline(bool),
    /// File was handed off to the global FTS5 batch writer. The batch
    /// actor takes responsibility for flipping `search_status` and
    /// finalizing the queue item — the caller MUST NOT touch
    /// `search_status` again, or it will race the actor.
    Enqueued,
    /// File was skipped before any write attempt (binary content, hash
    /// match against `indexed_content` cache). Caller should still flip
    /// `search_status=done` because no search work is owed for this row.
    Skipped,
}

/// Hand the FTS5 work off to the global batch writer when one is installed;
/// otherwise fall back to the legacy inline path.
///
/// The batched path eliminates `SQLITE_BUSY` contention on `search.db` by
/// funneling all writes through one transaction-batched task. Workers
/// still do the disk read + hash + old-content lookup, so file IO stays
/// parallel — only the database write is serialized.
#[allow(clippy::too_many_arguments)]
pub(super) async fn update_fts5_for_file_or_enqueue(
    search_db: &Arc<SearchDbManager>,
    state_pool: &SqlitePool,
    file_id: i64,
    file_path: &str,
    tenant_id: &str,
    branch: Option<&str>,
    base_point: Option<&str>,
    relative_path: Option<&str>,
    file_hash: Option<&str>,
    queue_id: &str,
) -> Result<Fts5Outcome, String> {
    let Some(sender) = crate::search_db::batch_writer::global_sender() else {
        return update_fts5_for_file(
            search_db,
            state_pool,
            file_id,
            file_path,
            tenant_id,
            branch,
            base_point,
            relative_path,
            file_hash,
        )
        .await
        .map(Fts5Outcome::Inline);
    };

    // Batched path: do disk + hash + cache-lookup here so workers stay
    // parallel for that work, then `send` and return — the actor owns
    // every write after this point.
    let new_content = match tokio::fs::read_to_string(file_path).await {
        Ok(c) => c,
        Err(e) => {
            debug!(
                "FTS5: cannot read file for indexing (may be binary): {}: {}",
                file_path, e
            );
            return Ok(Fts5Outcome::Skipped);
        }
    };

    // FTS5 hard cap: bypass code_lines / FTS5 entirely for giant files
    // (CSV dumps, generated proto, lockfiles, etc.) — the batch processor's
    // Phase 1 diff materialization is the RSS killer. file_metadata is
    // still upserted with fts5_skipped=1 so admin UI / Grafana can surface
    // the skip. Semantic embedding for the file proceeds independently.
    let hard_cap = hard_cap_line_threshold();
    if hard_cap > 0 {
        let line_count = line_count_estimate(&new_content);
        if line_count > hard_cap {
            if let Err(e) = enforce_fts5_hard_cap_skip(
                search_db.as_ref(),
                file_id,
                tenant_id,
                branch,
                file_path,
                base_point,
                relative_path,
                file_hash,
                Some(new_content.len() as i64),
                line_count,
            )
            .await
            {
                warn!(
                    "FTS5 hard-cap metadata upsert failed for {} ({} lines): {}",
                    file_path, line_count, e
                );
            }
            return Ok(Fts5Outcome::Skipped);
        }
    }

    let new_hash = compute_content_hash(&new_content);
    let old_content = match fetch_old_content(state_pool, file_id, file_path, &new_hash).await {
        Some(c) => c,
        None => return Ok(Fts5Outcome::Skipped),
    };

    let change = FileChange {
        file_id,
        size_bytes: Some(new_content.len() as i64),
        old_content,
        new_content: new_content.clone(),
        tenant_id: tenant_id.to_string(),
        branch: branch.map(|s| s.to_string()),
        file_path: file_path.to_string(),
        base_point: base_point.map(|s| s.to_string()),
        relative_path: relative_path.map(|s| s.to_string()),
        file_hash: file_hash.map(|s| s.to_string()),
    };

    let work = Fts5WorkItem {
        change,
        new_content_bytes: new_content.into_bytes(),
        new_hash,
        queue_id: queue_id.to_string(),
    };

    sender
        .send(work)
        .await
        .map(|()| Fts5Outcome::Enqueued)
        .map_err(|e| format!("FTS5 batch writer channel closed: {}", e))
}

/// Update the FTS5 code search index for a single file (Task 52).
///
/// Reads the file from disk, compares content hash against indexed_content cache.
/// If changed (or new), computes line diff and applies to code_lines + FTS5.
/// Returns `Ok(true)` if updated, `Ok(false)` if skipped (unchanged/binary),
/// `Err` if the FTS5 write failed.
#[allow(clippy::too_many_arguments)]
pub(super) async fn update_fts5_for_file(
    search_db: &Arc<SearchDbManager>,
    state_pool: &SqlitePool,
    file_id: i64,
    file_path: &str,
    tenant_id: &str,
    branch: Option<&str>,
    base_point: Option<&str>,
    relative_path: Option<&str>,
    file_hash: Option<&str>,
) -> Result<bool, String> {
    let fts_start = std::time::Instant::now();

    // Read file content from disk
    let new_content = match tokio::fs::read_to_string(file_path).await {
        Ok(content) => content,
        Err(e) => {
            debug!(
                "FTS5: cannot read file for indexing (may be binary): {}: {}",
                file_path, e
            );
            return Ok(false);
        }
    };

    // FTS5 hard cap: same guard as the batched path
    // ([`update_fts5_for_file_or_enqueue`]) — bypass code_lines / FTS5 for
    // giant files, mark them fts5_skipped=1, return as if work was done.
    let hard_cap = hard_cap_line_threshold();
    if hard_cap > 0 {
        let line_count = line_count_estimate(&new_content);
        if line_count > hard_cap {
            if let Err(e) = enforce_fts5_hard_cap_skip(
                search_db.as_ref(),
                file_id,
                tenant_id,
                branch,
                file_path,
                base_point,
                relative_path,
                file_hash,
                Some(new_content.len() as i64),
                line_count,
            )
            .await
            {
                warn!(
                    "FTS5 hard-cap metadata upsert failed for {} ({} lines): {}",
                    file_path, line_count, e
                );
            }
            return Ok(false);
        }
    }

    let new_hash = compute_content_hash(&new_content);

    // Check indexed_content cache for skip detection
    let old_content = fetch_old_content(state_pool, file_id, file_path, &new_hash).await;
    if old_content.is_none() {
        return Ok(false);
    }
    let old_content = old_content.unwrap();

    // Apply diff to code_lines via FtsBatchProcessor (single-file mode)
    let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
    let change = FileChange {
        file_id,
        // Record the post-upsert size. Spec 20 §3.2 — grep's `bytes_in`
        // is computed against this column when present.
        size_bytes: Some(new_content.len() as i64),
        old_content,
        new_content: new_content.clone(),
        tenant_id: tenant_id.to_string(),
        branch: branch.map(|s| s.to_string()),
        file_path: file_path.to_string(),
        base_point: base_point.map(|s| s.to_string()),
        relative_path: relative_path.map(|s| s.to_string()),
        file_hash: file_hash.map(|s| s.to_string()),
    };

    let fts_result = execute_fts_update(
        processor,
        change,
        file_id,
        tenant_id,
        branch,
        file_path,
        base_point,
        relative_path,
        file_hash,
    )
    .await;

    handle_fts_result(
        fts_result,
        state_pool,
        file_id,
        file_path,
        &new_content,
        &new_hash,
        fts_start,
    )
    .await
}

/// Fetch old content from the indexed_content cache.
///
/// Returns `Some(old_content)` to proceed, `None` if content is unchanged (skip).
async fn fetch_old_content(
    state_pool: &SqlitePool,
    file_id: i64,
    file_path: &str,
    new_hash: &str,
) -> Option<String> {
    match indexed_content_schema::get_indexed_content(state_pool, file_id).await {
        Ok(Some((cached_bytes, cached_hash))) => {
            if cached_hash == new_hash {
                debug!(
                    "FTS5: content unchanged (hash match), skipping: {}",
                    file_path
                );
                return None;
            }
            // Content changed -- use cached content as diff base
            Some(String::from_utf8(cached_bytes).unwrap_or_default())
        }
        Ok(None) => {
            // New file -- no old content to diff against
            Some(String::new())
        }
        Err(e) => {
            warn!(
                "FTS5: failed to read indexed_content cache for file_id={}: {}",
                file_id, e
            );
            Some(String::new())
        }
    }
}

/// Execute the FTS5 update using either full_rewrite or diff mode.
#[allow(clippy::too_many_arguments)]
async fn execute_fts_update<'a>(
    processor: FtsBatchProcessor<'a>,
    change: FileChange,
    file_id: i64,
    tenant_id: &str,
    branch: Option<&str>,
    file_path: &str,
    base_point: Option<&str>,
    relative_path: Option<&str>,
    file_hash: Option<&str>,
) -> Result<BatchStats, SearchDbError> {
    if change.old_content.is_empty() {
        processor
            .full_rewrite(
                file_id,
                &change.new_content,
                tenant_id,
                branch,
                file_path,
                base_point,
                relative_path,
                file_hash,
            )
            .await
    } else {
        // Use flush() with queue_depth=0 (single-file mode)
        let mut processor = processor;
        processor.add_change(change);
        processor.flush(0).await
    }
}

/// Handle the result of an FTS5 update, updating the cache on success.
async fn handle_fts_result(
    fts_result: Result<BatchStats, SearchDbError>,
    state_pool: &SqlitePool,
    file_id: i64,
    file_path: &str,
    new_content: &str,
    new_hash: &str,
    fts_start: std::time::Instant,
) -> Result<bool, String> {
    match fts_result {
        Ok(stats) => {
            debug!(
                "FTS5: updated {} (+{} ~{} -{}) for {} in {}ms",
                file_path,
                stats.lines_inserted,
                stats.lines_updated,
                stats.lines_deleted,
                file_path,
                fts_start.elapsed().as_millis()
            );

            // Update indexed_content cache with new content + hash
            if let Err(e) = indexed_content_schema::upsert_indexed_content(
                state_pool,
                file_id,
                new_content.as_bytes(),
                new_hash,
            )
            .await
            {
                warn!(
                    "FTS5: failed to update indexed_content cache for file_id={}: {}",
                    file_id, e
                );
            }
            Ok(true)
        }
        Err(e) => {
            warn!("FTS5: failed to update code_lines for {}: {}", file_path, e);
            Err(format!("FTS5 indexing failed for {}: {}", file_path, e))
        }
    }
}
