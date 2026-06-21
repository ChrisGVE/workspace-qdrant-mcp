//! The branch-tagging three-case dedup ladder (branch-lineage F6, arch §5.1).
//!
//! File: `daemon/core/src/branch_index/tagger.rs`
//! Context: the body of the chokepoint declared in `branch_index/mod.rs`. Every
//! file-ingest path calls [`tag_and_store`]; it computes the path-independent
//! `content_key`, classifies the write into one of three cases, and performs the
//! state.db + Qdrant + search.db writes in the recovery-safe order (FP
//! order-by-recoverability / arch §4.2): tracked_files INSERT first (with
//! `needs_reconcile=1` for the real-point cases) → qdrant_chunks → Qdrant upsert
//! → search.db file_metadata → clear `needs_reconcile` last.
//!
//! ## The ladder (arch §5.1)
//! * **Case 1 — content_key HIT** (this file-identity already has these bytes):
//!   write a virtual (vectorless) view point, no embed. Same-branch + same-path
//!   is an idempotent refresh; same-branch + different-path is a MOVE.
//! * **Case 2 — content_key MISS, byte-identical locator HIT**: copy the existing
//!   real point's vector into this identity's OWN real point, no re-embed.
//! * **Case 3 — both MISS** (genuinely new content): embed via the production
//!   `embed_chunks` and RE-KEY the points to the content-key scheme (Option C,
//!   locked 2026-06-21 — preserves LSP/sparse/payload richness, project FP-2).
//!
//! Re-add of a previously-deleted path resurrects the tombstone (§3.2). The
//! `write_tombstone` helper exists for the delete path (task 23/24) but is not
//! wired from this ADD chokepoint.

use std::collections::HashMap;

use sqlx::{Row, SqlitePool};
use wqm_common::hashing::content_key;
use wqm_common::timestamps;

use super::{EmbedInputs, IngestItem, TagOutcome, TagStored, TaggerError};
use crate::context::ProcessingContext;
use crate::processing_timings::PhaseTiming;
use crate::storage::DocumentPoint;
use crate::strategies::processing::file::chunk_embed::{
    build_chunk_payload, embed_chunks, ChunkRecord,
};
use crate::strategies::processing::file::component::inject_component;
use crate::strategies::processing::file::ingest::run_tier2_tagging;
use crate::strategies::processing::file::keyword_extract::run_keyword_extraction;
use crate::strategies::processing::file::keyword_persist::persist_extraction;
use crate::tracked_files_schema::{
    self, compute_content_hash, insert_qdrant_chunks, insert_tracked_file_v48,
    locate_byte_identical, real_point_id_for, ChunkType, ProcessingStatus,
};

/// Tag and store one file's chunks, routing through the dedup ladder.
///
/// See [`TagStored`] for the return contract: `points`/`records` are non-empty
/// only for the real-point cases (2/3 + resurrection) that feed the downstream
/// concept / narrative / graph phases; virtual / move / idempotent outcomes
/// carry empty vecs (no re-embed → no downstream graph work for this branch).
pub(crate) async fn tag_and_store(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
) -> Result<TagStored, TaggerError> {
    let content_key = content_key(
        item.tenant_id,
        &item.file_identity_id.to_string(),
        item.file_hash,
    );

    // Resurrection: a re-add of a path that currently carries a tombstone flips
    // it back to present rather than inserting a colliding row (§3.2/§7.2).
    if let Some(file_id) = tombstone_file_id(&ctx.pool, item).await? {
        let (points, records) = write_resurrection(ctx, item, embed, &content_key, file_id).await?;
        return Ok(TagStored {
            outcome: TagOutcome::EmbeddedNew,
            file_id,
            points,
            records,
        });
    }

    // Case 1 — content_key already present somewhere for this identity.
    if content_key_present(&ctx.pool, item.tenant_id, &content_key).await? {
        return route_content_key_hit(ctx, item, embed, &content_key).await;
    }

    // Case 2 — byte-identical content exists tenant-wide under a different
    // identity/collection: copy its vector instead of re-embedding.
    if let Some(hit) = locate_byte_identical(&ctx.pool, item.tenant_id, item.file_hash).await? {
        let (file_id, points, records) =
            write_real_copy(ctx, item, embed, &content_key, &hit).await?;
        return Ok(TagStored {
            outcome: TagOutcome::CopiedVector,
            file_id,
            points,
            records,
        });
    }

    // Case 3 — genuinely new content: embed + re-key.
    let (file_id, points, records) = write_real_embed(ctx, item, embed, &content_key).await?;
    Ok(TagStored {
        outcome: TagOutcome::EmbeddedNew,
        file_id,
        points,
        records,
    })
}

// ── Classification probes ──────────────────────────────────────────────────

/// True if any live (`state='present'`) row exists for this `content_key`.
async fn content_key_present(
    pool: &SqlitePool,
    tenant_id: &str,
    content_key: &str,
) -> Result<bool, sqlx::Error> {
    let row = sqlx::query(
        "SELECT 1 FROM tracked_files \
         WHERE tenant_id = ?1 AND content_key = ?2 AND state = 'present' LIMIT 1",
    )
    .bind(tenant_id)
    .bind(content_key)
    .fetch_optional(pool)
    .await?;
    Ok(row.is_some())
}

/// The `file_id` of a tombstone (`state='deleted'`) for this branch+path, if any.
async fn tombstone_file_id(
    pool: &SqlitePool,
    item: &IngestItem<'_>,
) -> Result<Option<i64>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT file_id FROM tracked_files \
         WHERE tenant_id = ?1 AND branch = ?2 AND relative_path = ?3 AND state = 'deleted' \
         LIMIT 1",
    )
    .bind(item.tenant_id)
    .bind(item.branch)
    .bind(item.relative_path)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| r.get::<i64, _>("file_id")))
}

/// Route a content_key HIT: same-branch+same-path → idempotent refresh;
/// same-branch+different-path → MOVE; different branch → virtual write.
async fn route_content_key_hit(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    content_key: &str,
) -> Result<TagStored, TaggerError> {
    // Same-branch live row for this content_key (at most one — live-view index).
    let same_branch = sqlx::query(
        "SELECT file_id, relative_path FROM tracked_files \
         WHERE tenant_id = ?1 AND content_key = ?2 AND branch = ?3 AND state = 'present' LIMIT 1",
    )
    .bind(item.tenant_id)
    .bind(content_key)
    .bind(item.branch)
    .fetch_optional(&ctx.pool)
    .await?;

    if let Some(r) = same_branch {
        let file_id: i64 = r.get("file_id");
        let existing_path: String = r.get("relative_path");
        if existing_path == item.relative_path {
            // Idempotent re-ingest: refresh the timestamp, no store writes.
            touch_row(&ctx.pool, file_id).await?;
            return Ok(empty_stored(TagOutcome::SharedExisting, file_id));
        }
        // Same content, same branch, new path → MOVE (a Case-1 INSERT would
        // collide with idx_tracked_files_live_view).
        write_move_metadata(ctx, item, content_key, file_id, &existing_path).await?;
        return Ok(empty_stored(TagOutcome::MovedMetadataOnly, file_id));
    }

    // Different branch already holds these bytes → virtual shadow on this branch.
    let file_id = write_virtual(ctx, item, embed, content_key).await?;
    Ok(empty_stored(TagOutcome::SharedExisting, file_id))
}

/// A [`TagStored`] for the vectorless outcomes (Case 1 / move / idempotent):
/// no points, no records, so the pipeline skips the graph phases for this branch.
fn empty_stored(outcome: TagOutcome, file_id: i64) -> TagStored {
    TagStored {
        outcome,
        file_id,
        points: Vec::new(),
        records: Vec::new(),
    }
}

// ── Case 1: virtual write ──────────────────────────────────────────────────

/// Insert a virtual (vectorless) view row + Qdrant point for this branch,
/// sharing the real point's `content_key` (axis A — no embed, no copy).
async fn write_virtual(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    content_key: &str,
) -> Result<i64, TaggerError> {
    let file_id = insert_row(ctx, item, content_key, true, ProcessingStatus::None, false).await?;

    let mut points = Vec::with_capacity(item.chunks.len());
    for (i, chunk) in item.chunks.iter().enumerate() {
        let real_point_id = real_point_id_for(content_key, i as u32).to_string();
        let mut payload = base_payload(item, embed, chunk);
        apply_lineage_payload(&mut payload, item, content_key, true, Some(&real_point_id));
        points.push(DocumentPoint {
            id: real_point_id,        // virtual point id == real point id it shadows
            dense_vector: Vec::new(), // no vector (convert.rs omits the dense slot)
            sparse_vector: None,
            payload,
        });
    }

    ctx.storage_client
        .insert_points_batch(item.collection, points, None)
        .await?;
    upsert_file_metadata(ctx, item, file_id, "present").await?;
    Ok(file_id)
}

// ── Case 2: copy-vector write ──────────────────────────────────────────────

/// Build this identity's OWN real point by COPYING a byte-identical existing
/// point's vector (axes B/C — own point, copied compute, no re-embed).
async fn write_real_copy(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    content_key: &str,
    hit: &tracked_files_schema::ByteIdenticalHit,
) -> Result<(i64, Vec<DocumentPoint>, Vec<ChunkRecord>), TaggerError> {
    let file_id = insert_row(ctx, item, content_key, false, ProcessingStatus::None, true).await?;

    let mut points = Vec::with_capacity(item.chunks.len());
    let mut records = Vec::with_capacity(item.chunks.len());
    let mut tuples = Vec::with_capacity(item.chunks.len());
    for (i, chunk) in item.chunks.iter().enumerate() {
        let src_id = real_point_id_for(&hit.content_key, i as u32).to_string();
        let vector = ctx
            .storage_client
            .retrieve_point_with_vector(&hit.collection, &src_id)
            .await?
            .unwrap_or_default();
        let new_id = real_point_id_for(content_key, i as u32).to_string();
        let mut payload = base_payload(item, embed, chunk);
        apply_lineage_payload(&mut payload, item, content_key, false, None);
        let record = chunk_record(&new_id, i, chunk);
        tuples.push(record_tuple(&record));
        records.push(record);
        points.push(DocumentPoint {
            id: new_id,
            dense_vector: vector,
            sparse_vector: None,
            payload,
        });
    }

    inject_component(
        ctx,
        &ctx.pool,
        item.watch_folder_id,
        embed.base_path,
        item.relative_path,
        &mut points,
    )
    .await;

    insert_qdrant_chunks(&ctx.pool, file_id, &tuples).await?;
    ctx.storage_client
        .insert_points_batch(item.collection, points.clone(), None)
        .await?;
    upsert_file_metadata(ctx, item, file_id, "present").await?;
    clear_reconcile(&ctx.pool, file_id).await?;
    Ok((file_id, points, records))
}

// ── Case 3: embed + re-key (Option C) ──────────────────────────────────────

/// Embed genuinely-new content via the production `embed_chunks`, then RE-KEY
/// each point/record to the content-key scheme and rewrite the branch-lineage
/// payload fields. Reuses the one embed path (LSP, lexicon sparse, oversize
/// splitting, rich payload) — project FP-2, no regression.
async fn write_real_embed(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    content_key: &str,
) -> Result<(i64, Vec<DocumentPoint>, Vec<ChunkRecord>), TaggerError> {
    let file_id = insert_row(ctx, item, content_key, false, ProcessingStatus::None, true).await?;

    let embed_result = embed_chunks(
        ctx,
        embed.queue_item,
        embed.document_content,
        embed.file_path,
        embed.file_document_id,
        item.relative_path,
        item.base_point.unwrap_or(""),
        item.file_hash,
        item.file_type,
        item.branch,
    )
    .await
    .map_err(|e| TaggerError::Embed(e.to_string()))?;

    let lsp_status = embed_result.lsp_status;
    let treesitter_status = embed_result.treesitter_status;
    let mut points = embed_result.points;
    let mut records = embed_result.chunk_records;
    let tuples = rekey_to_content_key(content_key, item, &mut points, &mut records);

    let mut timings: Vec<PhaseTiming> = Vec::new();
    run_tier2_tagging(ctx, &mut points, &mut timings).await;
    // Keyword/ranking-aid payloads must be injected BEFORE the upsert below —
    // the tagger owns the single point write, so this runs here (not in a
    // post-tagger phase, where it would never reach Qdrant). F6e Issue-1.
    enrich_and_persist_keywords(ctx, item, embed, &mut points).await;
    inject_component(
        ctx,
        &ctx.pool,
        item.watch_folder_id,
        embed.base_path,
        item.relative_path,
        &mut points,
    )
    .await;

    insert_qdrant_chunks(&ctx.pool, file_id, &tuples).await?;
    ctx.storage_client
        .insert_points_batch(item.collection, points.clone(), None)
        .await?;
    upsert_file_metadata(ctx, item, file_id, "present").await?;
    finalize_embed_row(&ctx.pool, file_id, lsp_status, treesitter_status).await?;
    Ok((file_id, points, records))
}

// ── Move / resurrection / tombstone ────────────────────────────────────────

/// A pure rename (same content_key, same branch, new path): rewrite the view
/// row's `relative_path` and the matching Qdrant point payloads + search.db,
/// never a re-embed (§3.3).
async fn write_move_metadata(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    content_key: &str,
    file_id: i64,
    _old_path: &str,
) -> Result<(), TaggerError> {
    let now = timestamps::now_utc();
    sqlx::query("UPDATE tracked_files SET relative_path = ?1, updated_at = ?2 WHERE file_id = ?3")
        .bind(item.relative_path)
        .bind(&now)
        .bind(file_id)
        .execute(&ctx.pool)
        .await?;

    // Rewrite the relative_path on every chunk point of this content_key on this
    // branch. The point ids are derivable; patch each payload in place.
    for i in 0..item.chunks.len() {
        let pid = real_point_id_for(content_key, i as u32).to_string();
        let mut patch = HashMap::new();
        patch.insert(
            "relative_path".to_string(),
            serde_json::json!(item.relative_path),
        );
        ctx.storage_client
            .set_payload_on_point(item.collection, &pid, patch)
            .await?;
    }

    upsert_file_metadata(ctx, item, file_id, "present").await?;
    Ok(())
}

/// Flip a tombstoned path back to present and re-drive its stores (§7.2 — the
/// inverse of `write_tombstone`). Re-embeds the current content so a changed
/// file resurrected with new bytes is correct.
async fn write_resurrection(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    content_key: &str,
    file_id: i64,
) -> Result<(Vec<DocumentPoint>, Vec<ChunkRecord>), TaggerError> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET state = 'present', content_key = ?1, file_hash = ?2, \
         needs_reconcile = 1, reconcile_reason = 'resurrection', updated_at = ?3 \
         WHERE file_id = ?4",
    )
    .bind(content_key)
    .bind(item.file_hash)
    .bind(&now)
    .bind(file_id)
    .execute(&ctx.pool)
    .await?;

    let embed_result = embed_chunks(
        ctx,
        embed.queue_item,
        embed.document_content,
        embed.file_path,
        embed.file_document_id,
        item.relative_path,
        item.base_point.unwrap_or(""),
        item.file_hash,
        item.file_type,
        item.branch,
    )
    .await
    .map_err(|e| TaggerError::Embed(e.to_string()))?;

    let lsp_status = embed_result.lsp_status;
    let treesitter_status = embed_result.treesitter_status;
    let mut points = embed_result.points;
    let mut records = embed_result.chunk_records;
    let tuples = rekey_to_content_key(content_key, item, &mut points, &mut records);

    let mut timings: Vec<PhaseTiming> = Vec::new();
    run_tier2_tagging(ctx, &mut points, &mut timings).await;
    // Ranking-aids before the upsert (F6e Issue-1; see write_real_embed).
    enrich_and_persist_keywords(ctx, item, embed, &mut points).await;
    insert_qdrant_chunks(&ctx.pool, file_id, &tuples).await?;
    ctx.storage_client
        .insert_points_batch(item.collection, points.clone(), None)
        .await?;
    upsert_file_metadata(ctx, item, file_id, "present").await?;
    finalize_embed_row(&ctx.pool, file_id, lsp_status, treesitter_status).await?;
    Ok((points, records))
}

/// Write a delete tombstone for this branch+path: flip state.db to `deleted` and
/// search.db file_metadata to `deleted` (the inverse of resurrection). Helper
/// for the per-file delete path (task 23/24); NOT wired from the ADD chokepoint.
#[allow(dead_code)] // wired by the delete path (task 23/24), not the ADD chokepoint
async fn write_tombstone(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    file_id: i64,
) -> Result<(), TaggerError> {
    let now = timestamps::now_utc();
    sqlx::query("UPDATE tracked_files SET state = 'deleted', updated_at = ?1 WHERE file_id = ?2")
        .bind(&now)
        .bind(file_id)
        .execute(&ctx.pool)
        .await?;
    upsert_file_metadata(ctx, item, file_id, "deleted").await?;
    Ok(())
}

// ── Shared helpers ─────────────────────────────────────────────────────────

/// Insert a v48 tracked_files row for this item, returning the file_id.
async fn insert_row(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    content_key: &str,
    is_virtual: bool,
    status: ProcessingStatus,
    needs_reconcile: bool,
) -> Result<i64, sqlx::Error> {
    let reason = if needs_reconcile {
        Some("additive_crash")
    } else {
        None
    };
    insert_tracked_file_v48(
        &ctx.pool,
        item.watch_folder_id,
        item.tenant_id,
        item.branch,
        &item.file_identity_id.to_string(),
        content_key,
        is_virtual,
        "present",
        item.file_type,
        item.language,
        item.file_mtime,
        item.file_hash,
        item.chunks.len() as i32,
        None,
        status,
        status,
        item.collection,
        item.extension,
        item.is_test,
        item.base_point,
        item.component,
        item.relative_path,
        needs_reconcile,
        reason,
    )
    .await
}

/// Build the rich Qdrant payload for one chunk (no embedding), reusing the
/// production `build_chunk_payload` so virtual/copy points carry the same
/// metadata as embedded points.
fn base_payload(
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    chunk: &crate::core_types::TextChunk,
) -> HashMap<String, serde_json::Value> {
    build_chunk_payload(
        &chunk.content,
        chunk.chunk_index,
        embed.queue_item,
        embed.document_content,
        embed.file_path,
        embed.file_document_id,
        item.relative_path,
        item.base_point.unwrap_or(""),
        item.file_hash,
        item.file_type,
        &chunk.metadata,
        None,
        item.branch,
    )
}

/// Overwrite the legacy `branches` array with the branch-lineage fields the
/// tagger owns: singular `branch`, `virtual`, `content_key`, `file_identity_id`,
/// `state`, and (for virtual points) the shadowed `real_point_id`.
fn apply_lineage_payload(
    payload: &mut HashMap<String, serde_json::Value>,
    item: &IngestItem<'_>,
    content_key: &str,
    is_virtual: bool,
    real_point_id: Option<&str>,
) {
    payload.remove("branches");
    payload.insert(
        wqm_common::constants::field::BRANCH.to_string(),
        serde_json::json!(item.branch),
    );
    payload.insert("virtual".to_string(), serde_json::json!(is_virtual));
    payload.insert("content_key".to_string(), serde_json::json!(content_key));
    payload.insert(
        "file_identity_id".to_string(),
        serde_json::json!(item.file_identity_id.to_string()),
    );
    payload.insert(
        wqm_common::constants::field::STATE.to_string(),
        serde_json::json!("present"),
    );
    if let Some(rid) = real_point_id {
        payload.insert("real_point_id".to_string(), serde_json::json!(rid));
    }
    for (k, v) in &item.extra_payload {
        payload.insert(k.clone(), v.clone());
    }
}

/// The tuple shape `insert_qdrant_chunks` expects per chunk.
type ChunkTuple = (
    String,
    i32,
    String,
    Option<ChunkType>,
    Option<String>,
    Option<i32>,
    Option<i32>,
);

/// Build a `ChunkRecord` for one chunk + its content-key point id. Used by the
/// no-embed cases (1/2), where there is no `embed_chunks` call to produce
/// records; semantic metadata is read from the chunk's `metadata` map.
fn chunk_record(
    point_id: &str,
    chunk_index: usize,
    chunk: &crate::core_types::TextChunk,
) -> ChunkRecord {
    ChunkRecord {
        point_id: point_id.to_string(),
        chunk_index: chunk_index as i32,
        content_hash: compute_content_hash(&chunk.content),
        chunk_type: chunk
            .metadata
            .get("chunk_type")
            .and_then(|s| ChunkType::from_str(s)),
        symbol_name: chunk.metadata.get("symbol_name").cloned(),
        start_line: chunk
            .metadata
            .get("start_line")
            .and_then(|s| s.parse().ok()),
        end_line: chunk.metadata.get("end_line").and_then(|s| s.parse().ok()),
    }
}

/// The `insert_qdrant_chunks` tuple for a `ChunkRecord`.
fn record_tuple(r: &ChunkRecord) -> ChunkTuple {
    (
        r.point_id.clone(),
        r.chunk_index,
        r.content_hash.clone(),
        r.chunk_type,
        r.symbol_name.clone(),
        r.start_line,
        r.end_line,
    )
}

/// Re-key embedded points + records to the content-key scheme (Option C). The
/// production `embed_chunks` keys points by `base_point`; the branch-lineage
/// index keys them by `content_key`. Rewrites each point id + lineage payload
/// and each record's `point_id` (by position), returning the
/// `insert_qdrant_chunks` tuples. Shared by Case 3 and resurrection.
fn rekey_to_content_key(
    content_key: &str,
    item: &IngestItem<'_>,
    points: &mut [DocumentPoint],
    records: &mut [ChunkRecord],
) -> Vec<ChunkTuple> {
    let mut tuples = Vec::with_capacity(points.len());
    for (i, (p, r)) in points.iter_mut().zip(records.iter_mut()).enumerate() {
        let new_id = real_point_id_for(content_key, i as u32).to_string();
        p.id = new_id.clone();
        apply_lineage_payload(&mut p.payload, item, content_key, false, None);
        r.point_id = new_id;
        tuples.push(record_tuple(r));
    }
    tuples
}

/// Run keyword extraction over freshly-embedded points — injecting the
/// ranking-aid payload keys (`keywords`/`concept_tags`/`structural_tags`/
/// `keyword_baskets`) into the payloads BEFORE the tagger's single upsert so
/// they reach Qdrant — and persist the extraction to state.db. F6e Issue-1: the
/// embed path ran this before its upsert; the tagger now owns the upsert, so it
/// runs here. Real-point embed cases only (Case 3 + resurrection).
async fn enrich_and_persist_keywords(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    embed: &EmbedInputs<'_>,
    points: &mut [DocumentPoint],
) {
    if let Some(extraction) = run_keyword_extraction(
        ctx,
        embed.queue_item,
        embed.file_path,
        embed.document_content,
        points,
    )
    .await
    {
        persist_extraction(
            &ctx.pool,
            embed.file_document_id,
            item.tenant_id,
            item.collection,
            &extraction,
        )
        .await;
    }
}

/// Upsert the search.db `file_metadata` row (v8, with `state`).
async fn upsert_file_metadata(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    file_id: i64,
    state: &str,
) -> Result<(), TaggerError> {
    let search_db = match &ctx.search_db {
        Some(db) => db,
        None => return Ok(()), // FTS5 disabled — nothing to mirror.
    };
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_V8_SQL)
        .bind(file_id)
        .bind(item.tenant_id)
        .bind(item.branch)
        .bind(item.abs_file_path)
        .bind(item.base_point)
        .bind(item.relative_path)
        .bind(item.file_hash)
        .bind(state)
        .execute(search_db.pool())
        .await?;
    Ok(())
}

/// Touch `updated_at` for an idempotent re-ingest (no store writes).
async fn touch_row(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query("UPDATE tracked_files SET updated_at = ?1 WHERE file_id = ?2")
        .bind(&now)
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Clear the crash-safety flag after the real-point cases complete (FP
/// order-by-recoverability: cleared LAST, arch §4.2). Used by Case 2 (copy),
/// which performs no embed and so has no enrichment status to record.
async fn clear_reconcile(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 0, reconcile_reason = NULL WHERE file_id = ?1",
    )
    .bind(file_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Clear the crash-safety flag AND record the embed's real LSP / tree-sitter
/// enrichment status (Case 3 + resurrection). The enrichment-upgrade reconciler
/// selects files by `lsp_status`/`treesitter_status IN ('none','failed',…)`
/// (`reconcile.rs`); leaving these at the row's `'none'` default would re-flag
/// every freshly-embedded file for re-enrichment. Cleared/set LAST (FP
/// order-by-recoverability, arch §4.2).
async fn finalize_embed_row(
    pool: &SqlitePool,
    file_id: i64,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 0, reconcile_reason = NULL, \
         lsp_status = ?1, treesitter_status = ?2 WHERE file_id = ?3",
    )
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(file_id)
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    const TENANT: &str = "tenant1";

    fn text_chunk(
        content: &str,
        idx: usize,
        meta: &[(&str, &str)],
    ) -> crate::core_types::TextChunk {
        let mut m = HashMap::new();
        for (k, v) in meta {
            m.insert(k.to_string(), v.to_string());
        }
        crate::core_types::TextChunk {
            content: content.to_string(),
            chunk_index: idx,
            start_char: 0,
            end_char: content.chars().count(),
            metadata: m,
        }
    }

    fn item_for<'a>(
        branch: &'a str,
        relative_path: &'a str,
        fid: uuid::Uuid,
        chunks: &'a [crate::core_types::TextChunk],
        extra: HashMap<String, serde_json::Value>,
    ) -> IngestItem<'a> {
        IngestItem {
            watch_folder_id: "wf1",
            tenant_id: TENANT,
            branch,
            collection: "projects",
            relative_path,
            abs_file_path: "/abs/src/a.rs",
            file_identity_id: fid,
            file_hash: "hash1",
            chunks,
            file_mtime: "2025-01-01T00:00:00.000Z",
            file_type: Some("code"),
            language: Some("rust"),
            is_test: false,
            extension: Some("rs"),
            component: Some("core"),
            base_point: Some("bp1"),
            extra_payload: extra,
        }
    }

    /// apply_lineage_payload removes the legacy `branches` array and installs the
    /// singular branch-lineage fields (+ real_point_id for virtual points) and
    /// merges extra_payload.
    #[test]
    fn apply_lineage_payload_rewrites_fields() {
        let fid = uuid::Uuid::new_v4();
        let chunks = vec![text_chunk("x", 0, &[])];
        let mut extra = HashMap::new();
        extra.insert("custom".to_string(), serde_json::json!("v"));
        let item = item_for("feature/x", "src/a.rs", fid, &chunks, extra);

        let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
        payload.insert("branches".to_string(), serde_json::json!(["feature/x"]));
        payload.insert("content".to_string(), serde_json::json!("x"));

        apply_lineage_payload(&mut payload, &item, "ck1", true, Some("real-pid"));

        assert!(!payload.contains_key("branches"), "legacy array removed");
        assert_eq!(payload["branch"], serde_json::json!("feature/x"));
        assert_eq!(payload["virtual"], serde_json::json!(true));
        assert_eq!(payload["content_key"], serde_json::json!("ck1"));
        assert_eq!(
            payload["file_identity_id"],
            serde_json::json!(fid.to_string())
        );
        assert_eq!(payload["state"], serde_json::json!("present"));
        assert_eq!(payload["real_point_id"], serde_json::json!("real-pid"));
        assert_eq!(payload["custom"], serde_json::json!("v"), "extra merged");
        assert_eq!(payload["content"], serde_json::json!("x"), "rich kept");
    }

    /// Real points carry no `real_point_id` and `virtual=false`.
    #[test]
    fn apply_lineage_payload_real_has_no_real_point_id() {
        let fid = uuid::Uuid::new_v4();
        let chunks = vec![text_chunk("x", 0, &[])];
        let item = item_for("main", "src/a.rs", fid, &chunks, HashMap::new());
        let mut payload = HashMap::new();
        apply_lineage_payload(&mut payload, &item, "ck1", false, None);
        assert_eq!(payload["virtual"], serde_json::json!(false));
        assert!(!payload.contains_key("real_point_id"));
    }

    /// chunk_record parses semantic metadata into the ChunkRecord; record_tuple
    /// projects it into the qdrant_chunks tuple shape.
    #[test]
    fn chunk_record_extracts_metadata() {
        let chunk = text_chunk(
            "fn f() {}",
            2,
            &[
                ("symbol_name", "f"),
                ("start_line", "10"),
                ("end_line", "12"),
            ],
        );
        let record = chunk_record("pid-2", 2, &chunk);
        assert_eq!(record.point_id, "pid-2");
        assert_eq!(record.chunk_index, 2);
        assert_eq!(record.content_hash, compute_content_hash("fn f() {}"));
        assert_eq!(record.symbol_name.as_deref(), Some("f"));
        assert_eq!(record.start_line, Some(10));
        assert_eq!(record.end_line, Some(12));

        let (pid, idx, hash, _ct, sym, sl, el) = record_tuple(&record);
        assert_eq!(pid, "pid-2");
        assert_eq!(idx, 2);
        assert_eq!(hash, compute_content_hash("fn f() {}"));
        assert_eq!(sym.as_deref(), Some("f"));
        assert_eq!(sl, Some(10));
        assert_eq!(el, Some(12));
    }

    async fn v48_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        sqlx::query(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES ('wf1', '/tmp/wf1', 'projects', ?1, '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')",
        )
        .bind(TENANT)
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    /// content_key_present sees live rows and ignores tombstones; tombstone_file_id
    /// is the inverse. touch_row and clear_reconcile mutate the expected columns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn db_probes_and_mutators() {
        let pool = v48_pool().await;
        let fid = uuid::Uuid::new_v4().to_string();

        // Insert a live present row + a tombstone on a different path.
        let live_id = insert_tracked_file_v48(
            &pool,
            "wf1",
            TENANT,
            "main",
            &fid,
            "ck_live",
            false,
            "present",
            None,
            None,
            "2025-01-01T00:00:00.000Z",
            "h1",
            1,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            "projects",
            None,
            false,
            None,
            None,
            "src/a.rs",
            true,
            Some("additive_crash"),
        )
        .await
        .unwrap();
        insert_tracked_file_v48(
            &pool,
            "wf1",
            TENANT,
            "main",
            &fid,
            "ck_dead",
            false,
            "deleted",
            None,
            None,
            "2025-01-01T00:00:00.000Z",
            "h2",
            1,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            "projects",
            None,
            false,
            None,
            None,
            "src/gone.rs",
            false,
            None,
        )
        .await
        .unwrap();

        assert!(content_key_present(&pool, TENANT, "ck_live").await.unwrap());
        assert!(!content_key_present(&pool, TENANT, "ck_dead").await.unwrap());

        // tombstone_file_id finds the deleted path's row, not the live one.
        let item_chunks: Vec<crate::core_types::TextChunk> = vec![];
        let item_gone = item_for(
            "main",
            "src/gone.rs",
            uuid::Uuid::new_v4(),
            &item_chunks,
            HashMap::new(),
        );
        assert!(tombstone_file_id(&pool, &item_gone)
            .await
            .unwrap()
            .is_some());
        let item_live = item_for(
            "main",
            "src/a.rs",
            uuid::Uuid::new_v4(),
            &item_chunks,
            HashMap::new(),
        );
        assert!(tombstone_file_id(&pool, &item_live)
            .await
            .unwrap()
            .is_none());

        // clear_reconcile flips the flag + reason on the live row.
        clear_reconcile(&pool, live_id).await.unwrap();
        let row = sqlx::query(
            "SELECT needs_reconcile, reconcile_reason FROM tracked_files WHERE file_id = ?1",
        )
        .bind(live_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.get::<i64, _>("needs_reconcile"), 0);
        assert!(row.get::<Option<String>, _>("reconcile_reason").is_none());

        // touch_row updates updated_at without error.
        touch_row(&pool, live_id).await.unwrap();
    }
}
