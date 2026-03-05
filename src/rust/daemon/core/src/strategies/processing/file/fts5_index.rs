//! FTS5 code search index updates.
//!
//! Updates the FTS5 full-text search index for a single file after Qdrant
//! ingestion. Reads file from disk, diffs against cached content, and applies
//! line-level changes to code_lines + FTS5.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, warn};

use crate::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use crate::indexed_content_schema;
use crate::search_db::SearchDbManager;
use wqm_common::hashing::compute_content_hash;

/// Update the FTS5 code search index for a single file (Task 52).
///
/// Reads the file from disk, compares content hash against indexed_content cache.
/// If changed (or new), computes line diff and applies to code_lines + FTS5.
/// Failures are logged but non-fatal -- they don't block Qdrant ingestion.
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
) {
    let fts_start = std::time::Instant::now();

    // Read file content from disk
    let new_content = match tokio::fs::read_to_string(file_path).await {
        Ok(content) => content,
        Err(e) => {
            debug!(
                "FTS5: cannot read file for indexing (may be binary): {}: {}",
                file_path, e
            );
            return;
        }
    };

    let new_hash = compute_content_hash(&new_content);

    // Check indexed_content cache for skip detection
    let old_content = match indexed_content_schema::get_indexed_content(state_pool, file_id).await {
        Ok(Some((cached_bytes, cached_hash))) => {
            if cached_hash == new_hash {
                debug!(
                    "FTS5: content unchanged (hash match), skipping: {}",
                    file_path
                );
                return;
            }
            // Content changed -- use cached content as diff base
            String::from_utf8(cached_bytes).unwrap_or_default()
        }
        Ok(None) => {
            // New file -- no old content to diff against
            String::new()
        }
        Err(e) => {
            warn!(
                "FTS5: failed to read indexed_content cache for file_id={}: {}",
                file_id, e
            );
            String::new()
        }
    };

    // Apply diff to code_lines via FtsBatchProcessor (single-file mode)
    let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
    let change = FileChange {
        file_id,
        old_content,
        new_content: new_content.clone(),
        tenant_id: tenant_id.to_string(),
        branch: branch.map(|s| s.to_string()),
        file_path: file_path.to_string(),
        base_point: base_point.map(|s| s.to_string()),
        relative_path: relative_path.map(|s| s.to_string()),
        file_hash: file_hash.map(|s| s.to_string()),
    };

    // Use full_rewrite for new files (empty old content), diff for updates
    let fts_result = if change.old_content.is_empty() {
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
    };

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
                &new_hash,
            )
            .await
            {
                warn!(
                    "FTS5: failed to update indexed_content cache for file_id={}: {}",
                    file_id, e
                );
            }
        }
        Err(e) => {
            warn!(
                "FTS5: failed to update code_lines for {}: {} (non-fatal)",
                file_path, e
            );
        }
    }
}
