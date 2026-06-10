//! Cross-branch ingestion fast-path.
//!
//! When a file/add or file/update item is processed and an identical row
//! (same `watch_folder_id`, `relative_path`, `file_hash`) already exists
//! in `tracked_files` for a **different branch**, the expensive
//! parse + embed pipeline is replaced with a Qdrant scroll + re-upsert:
//!
//! 1. Look up the existing entry's `base_point` (one SQL hit on the
//!    `idx_tracked_files_dedup` index added in 2026-05-27).
//! 2. Scroll all Qdrant points whose payload `base_point` matches, **with
//!    vectors**.
//! 3. Re-upsert each point under the new branch's base_point + payload
//!    (re-keys the point_id but reuses the dense/sparse vectors verbatim).
//! 4. Insert the `tracked_files` row for the new branch with the new
//!    base_point + same chunk_count.
//! 5. Enqueue FTS5 work via the global batch sender so search.db gets
//!    `code_lines` + `file_metadata` rows for the current branch
//!    (search filters by `fm.branch = ?`, so we can't share FTS5 rows).
//! 6. Flip qdrant_status=done, search_status=in_progress (batch writer
//!    finishes search_status).
//!
//! This skips the parse + embed phases — the dominant cost per file
//! (FastEmbed ONNX inference per chunk). Designed to make `git checkout`
//! between branches near-free on the indexed-data side.
//!
//! See [docs/specs/21-cross-branch-dedup.md](../../../../../../../docs/specs/21-cross-branch-dedup.md)
//! for the design rationale and the Layer 2 follow-up (drop branch from
//! base_point entirely).

use std::collections::HashMap;
use std::path::Path;

use qdrant_client::qdrant::{Condition, Filter};
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::fts_batch_processor::FileChange;
use crate::search_db::Fts5WorkItem;
use crate::storage::DocumentPoint;
use crate::tracked_files_schema::{self, ProcessingStatus};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::{DestinationStatus, FilePayload, UnifiedQueueItem};
use wqm_common::hashing::{compute_base_point, compute_content_hash, compute_point_id};

/// Outcome of [`try_branch_dedup`] — `Some` means the dedup fast-path
/// completed and the caller must return early; `None` means the file is
/// novel and the normal ingest pipeline should run.
pub(super) struct DedupHit;

/// Scroll limit per dedup probe. Default chunk_count per file is bounded
/// by the chunking config (typically <100); 256 keeps comfortable margin
/// and a single round-trip is enough.
const SCROLL_LIMIT: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub(super) async fn try_branch_dedup(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
    file_path: &Path,
    abs_file_path: &str,
    relative_path: &str,
    watch_folder_id: &str,
) -> Result<Option<DedupHit>, UnifiedProcessorError> {
    // Compute file_hash up-front — the dedup key depends on it.
    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;

    // ── 1. SQL lookup: does another branch have an entry for the same content? ──
    let existing: Option<(String, String, i32, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT base_point, branch, chunk_count, file_type, language
             FROM tracked_files
             WHERE watch_folder_id = ?1
               AND relative_path = ?2
               AND file_hash = ?3
               AND branch != ?4
               AND base_point IS NOT NULL
             ORDER BY updated_at DESC
             LIMIT 1",
    )
    .bind(watch_folder_id)
    .bind(relative_path)
    .bind(&file_hash)
    .bind(&item.branch)
    .fetch_optional(&ctx.pool)
    .await
    .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("dedup lookup: {e}")))?;

    let Some((old_base_point, old_branch, chunk_count, file_type, language)) = existing else {
        return Ok(None);
    };

    // ── 2. Scroll the old points (with vectors) ──
    let filter = Filter::must([Condition::matches("base_point", old_base_point.clone())]);
    let old_points = ctx
        .storage_client
        .scroll_with_filter_and_vectors(&item.collection, filter, SCROLL_LIMIT)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    if old_points.is_empty() {
        // tracked_files said there were chunks, but Qdrant has nothing —
        // probably a stale row or a partial cleanup. Fall back to normal
        // ingest so the file gets re-embedded fresh.
        warn!(
            "branch_dedup: tracked entry for {} (branch={}) has base_point {} but Qdrant returned 0 points — falling back to normal ingest",
            relative_path, old_branch, old_base_point
        );
        return Ok(None);
    }

    // ── 3. Re-upsert under new base_point + branch ──
    let new_base_point =
        compute_base_point(&item.tenant_id, &item.branch, relative_path, &file_hash);

    let new_points = old_points
        .into_iter()
        .filter_map(|p| rekey_point(p, &new_base_point, &item.branch, abs_file_path))
        .collect::<Vec<DocumentPoint>>();

    if new_points.is_empty() {
        warn!(
            "branch_dedup: failed to decode any retrieved points for {} — falling back",
            relative_path
        );
        return Ok(None);
    }

    // Capture the qdrant_chunks mirror rows before the upsert consumes the
    // points. Without them the cloned generation is invisible to every
    // per-file Qdrant deletion path (update replacement, file delete,
    // missing-file cleanup, delete triage) — exactly the drift the mirror
    // reconcile task repairs.
    let chunk_tuples = build_mirror_tuples(&new_points);

    let scrolled = new_points.len();
    ctx.storage_client
        .insert_points_batch(&item.collection, new_points, None)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    // ── 4. tracked_files row + qdrant_chunks mirror for the new branch ──
    let file_mtime = std::fs::metadata(file_path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default();
    let extension = file_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());
    let is_test = crate::file_classification::is_test_file(file_path);

    let mut tx = ctx
        .pool
        .begin()
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("dedup tx begin: {e}")))?;
    let file_id = tracked_files_schema::insert_tracked_file_tx(
        &mut tx,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
        file_type.as_deref(),
        language.as_deref(),
        &file_mtime,
        &file_hash,
        chunk_count,
        Some("dedup_clone"),
        ProcessingStatus::None,
        ProcessingStatus::None,
        Some(item.collection.as_str()),
        extension.as_deref(),
        is_test,
        Some(&new_base_point),
        None,
    )
    .await
    .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("insert_tracked_file: {e}")))?;
    tracked_files_schema::insert_qdrant_chunks_tx(&mut tx, file_id, &chunk_tuples)
        .await
        .map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!("insert_qdrant_chunks: {e}"))
        })?;
    tx.commit()
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("dedup tx commit: {e}")))?;

    // ── 5. Enqueue FTS5 work (batch writer owns search.db writes) ──
    if let Some(sender) = crate::search_db::batch_writer::global_sender() {
        // Read file content once for FTS5. The batch writer needs the new
        // content + old_content (empty since this file_id is brand new in
        // search.db) to compute the diff.
        match tokio::fs::read_to_string(file_path).await {
            Ok(new_content) => {
                let new_hash = compute_content_hash(&new_content);
                let change = FileChange {
                    file_id,
                    size_bytes: Some(new_content.len() as i64),
                    old_content: String::new(),
                    new_content: new_content.clone(),
                    tenant_id: item.tenant_id.clone(),
                    branch: Some(item.branch.clone()),
                    file_path: abs_file_path.to_string(),
                    base_point: Some(new_base_point.clone()),
                    relative_path: Some(relative_path.to_string()),
                    file_hash: Some(file_hash.clone()),
                };
                let work = Fts5WorkItem {
                    change,
                    new_content_bytes: new_content.into_bytes(),
                    new_hash,
                    queue_id: item.queue_id.clone(),
                };
                let _ = ctx
                    .queue_manager
                    .update_destination_status(
                        &item.queue_id,
                        "search",
                        DestinationStatus::InProgress,
                    )
                    .await;
                if let Err(e) = sender.send(work).await {
                    warn!(
                        "branch_dedup: failed to enqueue FTS5 work for {}: {} — marking search=failed",
                        relative_path, e
                    );
                    let _ = ctx
                        .queue_manager
                        .update_destination_status(
                            &item.queue_id,
                            "search",
                            DestinationStatus::Failed,
                        )
                        .await;
                }
            }
            Err(e) => {
                // Binary or unreadable — skip search but qdrant work still
                // counts as done.
                debug!(
                    "branch_dedup: skipping FTS5 for {} ({}): {}",
                    relative_path, abs_file_path, e
                );
                let _ = ctx
                    .queue_manager
                    .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
                    .await;
            }
        }
    } else {
        // Library/test mode with no batch writer — mark search=done so
        // the orchestration-only path completes.
        let _ = ctx
            .queue_manager
            .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
            .await;
    }

    // ── 6. Destination markers + return ──
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;

    info!(
        "branch_dedup hit: {} ({}→{}) skipped embed, copied {} chunks from old base_point {} → {}",
        relative_path, old_branch, item.branch, scrolled, old_base_point, new_base_point
    );

    // Suppress unused warnings on payload — kept in the signature to
    // mirror the normal ingest entry-point and ease future field reuse
    // (e.g. honoring payload.file_type override).
    let _ = payload;
    Ok(Some(DedupHit))
}

/// Chunk tuple accepted by `insert_qdrant_chunks_tx`.
type ChunkTuple = (
    String,
    i32,
    String,
    Option<tracked_files_schema::ChunkType>,
    Option<String>,
    Option<i32>,
    Option<i32>,
);

/// Build the `qdrant_chunks` mirror tuples for the re-keyed points.
///
/// Point IDs are already in `compute_point_id`'s bare-hex form (assigned by
/// [`rekey_point`]); chunk metadata comes from the cloned payload
/// (`chunk_index` as a number, the forwarded `chunk_*` fields as strings).
/// Duplicate chunk indexes are dropped to respect UNIQUE(file_id, chunk_index).
fn build_mirror_tuples(points: &[DocumentPoint]) -> Vec<ChunkTuple> {
    let int_of = |v: &serde_json::Value| {
        v.as_i64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))
    };

    let mut seen_indexes = std::collections::HashSet::new();
    points
        .iter()
        .filter_map(|p| {
            let chunk_index = i32::try_from(p.payload.get("chunk_index").and_then(int_of)?).ok()?;
            let content_hash = p
                .payload
                .get("content")
                .and_then(|v| v.as_str())
                .map(compute_content_hash)
                .unwrap_or_default();
            let chunk_type = p
                .payload
                .get("chunk_chunk_type")
                .and_then(|v| v.as_str())
                .and_then(tracked_files_schema::ChunkType::from_str);
            let symbol_name = p
                .payload
                .get("chunk_symbol_name")
                .and_then(|v| v.as_str())
                .map(String::from);
            let start_line = p
                .payload
                .get("chunk_start_line")
                .and_then(int_of)
                .and_then(|n| i32::try_from(n).ok());
            let end_line = p
                .payload
                .get("chunk_end_line")
                .and_then(int_of)
                .and_then(|n| i32::try_from(n).ok());
            Some((
                p.id.clone(),
                chunk_index,
                content_hash,
                chunk_type,
                symbol_name,
                start_line,
                end_line,
            ))
        })
        .filter(|t| seen_indexes.insert(t.1))
        .collect()
}

/// Translate a `RetrievedPoint` (with vectors) into a `DocumentPoint`
/// keyed by the **new** base_point + branch.
///
/// Returns `None` when the point cannot be decoded (missing chunk_index,
/// missing dense vector). Sparse vector is optional.
fn rekey_point(
    p: qdrant_client::qdrant::RetrievedPoint,
    new_base_point: &str,
    new_branch: &str,
    new_abs_path: &str,
) -> Option<DocumentPoint> {
    use qdrant_client::qdrant::{value, vector_output, vectors_output};

    // chunk_index lives in the payload — needed to recompute the point_id.
    let chunk_index = p.payload.get("chunk_index").and_then(|v| match &v.kind {
        Some(value::Kind::IntegerValue(n)) => Some(*n as u32),
        Some(value::Kind::DoubleValue(f)) => Some(*f as u32),
        _ => None,
    })?;

    // Decode vectors.
    let (dense, sparse) = match p.vectors.as_ref()?.vectors_options.as_ref()? {
        vectors_output::VectorsOptions::Vectors(named) => {
            let dense = named.vectors.get("dense").and_then(|v| match &v.vector {
                Some(vector_output::Vector::Dense(dv)) => Some(dv.data.clone()),
                _ => None,
            })?;
            let sparse = named.vectors.get("sparse").and_then(|v| match &v.vector {
                Some(vector_output::Vector::Sparse(sv)) => {
                    let map: HashMap<u32, f32> = sv
                        .indices
                        .iter()
                        .zip(sv.values.iter())
                        .map(|(&i, &v)| (i, v))
                        .collect();
                    Some(map)
                }
                _ => None,
            });
            (dense, sparse)
        }
        _ => return None,
    };

    // Re-key the payload to the new branch.
    let mut payload: HashMap<String, serde_json::Value> = p
        .payload
        .into_iter()
        .map(|(k, v)| (k, crate::storage::convert::convert_qdrant_value_to_json(v)))
        .collect();
    payload.insert(
        "base_point".to_string(),
        serde_json::Value::String(new_base_point.to_string()),
    );
    payload.insert(
        "branch".to_string(),
        serde_json::Value::String(new_branch.to_string()),
    );
    payload.insert(
        "absolute_path".to_string(),
        serde_json::Value::String(new_abs_path.to_string()),
    );

    Some(DocumentPoint {
        id: compute_point_id(new_base_point, chunk_index),
        dense_vector: dense,
        sparse_vector: sparse,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn point(id: &str, payload: HashMap<String, serde_json::Value>) -> DocumentPoint {
        DocumentPoint {
            id: id.to_string(),
            dense_vector: vec![0.0],
            sparse_vector: None,
            payload,
        }
    }

    /// The dedup fast-path used to insert the tracked_files row with
    /// `chunk_count > 0` but never wrote the qdrant_chunks mirror — the
    /// cloned generation was then invisible to every per-file Qdrant
    /// deletion path. The mirror tuples must come out of the re-keyed
    /// points themselves.
    #[test]
    fn build_mirror_tuples_extracts_chunk_metadata() {
        let mut payload = HashMap::new();
        payload.insert("chunk_index".to_string(), serde_json::json!(2));
        payload.insert("content".to_string(), serde_json::json!("fn main() {}"));
        payload.insert(
            "chunk_chunk_type".to_string(),
            serde_json::json!("function"),
        );
        payload.insert("chunk_symbol_name".to_string(), serde_json::json!("main"));
        payload.insert("chunk_start_line".to_string(), serde_json::json!("10"));
        payload.insert("chunk_end_line".to_string(), serde_json::json!("20"));

        let tuples = build_mirror_tuples(&[point("aabbccdd", payload)]);
        assert_eq!(tuples.len(), 1);
        let (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line) =
            &tuples[0];
        assert_eq!(point_id, "aabbccdd");
        assert_eq!(*chunk_index, 2);
        assert_eq!(content_hash, &compute_content_hash("fn main() {}"));
        assert_eq!(*chunk_type, Some(tracked_files_schema::ChunkType::Function));
        assert_eq!(symbol_name.as_deref(), Some("main"));
        assert_eq!(*start_line, Some(10));
        assert_eq!(*end_line, Some(20));
    }

    #[test]
    fn build_mirror_tuples_skips_undecodable_and_duplicate_indexes() {
        let mut ok = HashMap::new();
        ok.insert("chunk_index".to_string(), serde_json::json!(0));
        let mut dup = HashMap::new();
        dup.insert("chunk_index".to_string(), serde_json::json!(0));
        let no_index = HashMap::new();

        let tuples =
            build_mirror_tuples(&[point("p0", ok), point("p0-dup", dup), point("p1", no_index)]);
        assert_eq!(
            tuples.len(),
            1,
            "duplicate index and indexless point dropped"
        );
        assert_eq!(tuples[0].0, "p0");
        // Missing content still produces a row (hash defaults to empty).
        assert_eq!(tuples[0].2, "");
    }
}
