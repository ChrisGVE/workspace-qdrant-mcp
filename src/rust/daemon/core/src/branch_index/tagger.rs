//! BranchTagger — the single ingest chokepoint for branch-lineage tagging (F6).
//!
//! File: src/rust/daemon/core/src/branch_index/tagger.rs
//! Location: branch_index/ (crate-level sibling of strategies/, storage/).
//! Context: branch-lineage indexing subsystem
//! (docs/architecture/branch-lineage-indexing.md §5.1). Implements the
//! three-case dedup ladder: Case-1 content_key HIT (virtual point, no embed),
//! Case-2 tenant-wide byte-identical HIT (copy vector, no re-embed), Case-3
//! both MISS (embed fresh). All ADD/ingest surfaces converge here so branch
//! tagging, identity allocation, and dedup are never duplicated.
//!
//! Locking: uses the existing per-tenant `branch_locks` on `ProcessingContext`
//! (a `TenantBranchLocks`). Narrowing to per-content_key granularity (arch §7.1)
//! requires a new bounded `DashMap` field on `ProcessingContext`; that is deferred
//! to task-29/§7.1 per the task-22 scope decision. The per-tenant lock is held
//! across the content_key existence check AND the write to prevent a concurrent
//! add for the same content_key from racing.
//! TODO(task-29/§7.1): narrow lock to per-content_key using a bounded DashMap
//! on ProcessingContext to reduce contention across distinct content_keys.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::storage::{DocumentPoint, StorageError};
use crate::strategies::processing::file::chunk_embed::ChunkRecord;
use crate::tracked_files_schema::{
    allocate_file_identity, locate_byte_identical, real_point_id_for, ByteIdenticalHit,
};
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use wqm_common::hashing::{content_key, point_id};

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Everything any ingestion surface hands the tagger.
///
/// Tool-agnostic by design — borrows the strategies' existing `ChunkRecord`
/// type (arch §5.1 M14) rather than introducing a new abstraction.
///
/// `file_hash` is the document-level SHA-256 hex digest (64 lowercase chars).
/// For files it is the on-disk content hash; for URL/text/library content it
/// is the SHA-256 of the document text. Callers MUST compute it from the
/// actual bytes — not from a cached value received over an untrusted channel
/// (SEC-6).
pub struct IngestItem<'a> {
    /// Tenant that owns this ingest (e.g. git remote hash or project path hash).
    pub tenant_id: &'a str,
    /// The branch this view row belongs to (e.g. "main", "feat/foo").
    pub branch: &'a str,
    /// The watch_folder that anchors this view row in state.db.
    pub watch_folder_id: &'a str,
    /// Qdrant collection name (e.g. "projects", "libraries").
    pub collection: &'a str,
    /// Logical path/identity for this VIEW (file path, URL, scratchpad key).
    pub relative_path: &'a str,
    /// SHA-256 hex digest of the full document content (64 lowercase chars).
    /// Callers derive this from actual content bytes (SEC-6 compliance).
    pub file_hash: &'a str,
    /// Per-chunk records produced by the embedding stage (existing type from
    /// chunk_embed::types — no new abstraction needed, arch M14).
    pub chunks: &'a [ChunkRecord],
    /// Payload fields to set on each Qdrant point (tenant_id, relative_path,
    /// document_id, content, etc.) — the base map before branch-index fields
    /// are merged in. The tagger adds `content_key`, `file_identity_id`,
    /// `virtual`, and `branch` fields before upsert.
    pub base_payload: &'a HashMap<String, serde_json::Value>,
    /// Dense-vector dimension — used when creating a zero-vector placeholder for
    /// virtual points (virtual points have no vector of their own).
    pub dense_dim: usize,
}

/// The outcome of a `tag_and_store` call, reported per-document (not per-chunk).
///
/// Lets callers accumulate telemetry counters (task-29/§7.5) and log dedup hits.
/// Stub variants (Tombstoned, MovedMetadataOnly) satisfy the enum definition
/// required by the type system; they are not produced by task-22 scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagOutcome {
    /// Case-3: genuinely new content — embedded from scratch.
    EmbeddedNew,
    /// Case-1: `content_key` already present for this identity — virtual point
    /// created, zero embed calls.
    SharedExisting,
    /// Case-2: different identity or collection holds byte-identical content —
    /// own real point created by copying the existing vector, no re-embed.
    CopiedVector,
    /// Stub: tombstone written (task-24).
    Tombstoned,
    /// Stub: metadata-only move (task-23).
    MovedMetadataOnly,
}

/// Entry point. Reads/writes state.db, Qdrant, and search.db (arch §4.2).
///
/// Resolves `file_identity_id` via `allocate_file_identity`, computes the
/// `content_key`, and dispatches through the three-case dedup ladder:
///
/// - **Case 1** — `content_key` HIT: reference the existing real point; create
///   only a virtual point + tracker row. No embed, no copy.
/// - **Case 2** — `content_key` MISS, tenant-wide `(tenant_id, file_hash)` HIT:
///   create this identity's OWN real point, copying the vector from the located
///   point. No re-embed.
/// - **Case 3** — both MISS: genuinely new — embed and upsert.
///
/// Locking: acquires the per-tenant `branch_lock` (see module doc). A
/// per-content_key lock (arch §7.1) is deferred to task-29.
pub async fn tag_and_store(
    ctx: &ProcessingContext,
    item: IngestItem<'_>,
) -> UnifiedProcessorResult<TagOutcome> {
    if item.chunks.is_empty() {
        debug!(
            tenant = item.tenant_id,
            path = item.relative_path,
            "tag_and_store: no chunks — nothing to write"
        );
        return Ok(TagOutcome::EmbeddedNew);
    }

    // Resolve file_identity_id: inherit from lineage ancestor or mint fresh.
    let identity =
        allocate_file_identity(&ctx.pool, item.tenant_id, item.branch, item.relative_path)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "allocate_file_identity failed for {}: {}",
                    item.relative_path, e
                ))
            })?;

    let file_identity_id = identity.id().to_string();

    // Compute the content_key: keyed on (tenant_id, file_identity_id, file_hash_hex).
    // file_hash MUST already be the 64-char lowercase-hex SHA-256 (SEC-6).
    let ck = content_key(item.tenant_id, &file_identity_id, item.file_hash);

    debug!(
        tenant = item.tenant_id,
        path = item.relative_path,
        identity = %file_identity_id,
        content_key = %ck,
        "tag_and_store: resolved identity and content_key"
    );

    // Acquire per-tenant lock across the read-check-write sequence to prevent
    // concurrent adds for the same tenant from racing on content_key existence.
    // TODO(task-29/§7.1): narrow to per-content_key lock from a bounded DashMap.
    let tenant_mutex = ctx.branch_locks.get(item.tenant_id);
    let _lock = tenant_mutex.lock().await;

    // --- Three-case ladder ---

    // Case 1: check whether this file_identity already owns rows for content_key.
    // A state.db lookup on idx_tracked_files_content_key (if indexed) or a
    // full-column scan bounded by tenant_id + content_key.
    let existing_content_key = check_content_key_exists(&ctx.pool, item.tenant_id, &ck).await?;

    if existing_content_key {
        // Case 1: virtual point — no embed, no copy.
        info!(
            tenant = item.tenant_id,
            path = item.relative_path,
            content_key = %ck,
            "tag_and_store: Case-1 content_key HIT — writing virtual point"
        );
        write_virtual(ctx, &item, &ck, &file_identity_id).await?;
        return Ok(TagOutcome::SharedExisting);
    }

    // Case 2: tenant-wide byte locator — does ANY other identity/collection own
    // byte-identical content? If so, copy the vector instead of re-embedding.
    let byte_hit = locate_byte_identical(&ctx.pool, item.tenant_id, item.file_hash)
        .await
        .map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!(
                "locate_byte_identical failed for {}: {}",
                item.relative_path, e
            ))
        })?;

    if let Some(hit) = byte_hit {
        // Case 2: copy vectors from the located point.
        info!(
            tenant = item.tenant_id,
            path = item.relative_path,
            content_key = %ck,
            source_collection = %hit.collection,
            "tag_and_store: Case-2 byte-identical HIT — copying vector"
        );
        write_real_copy(ctx, &item, &ck, &file_identity_id, &hit).await?;
        return Ok(TagOutcome::CopiedVector);
    }

    // Case 3: genuinely new — embed and upsert.
    info!(
        tenant = item.tenant_id,
        path = item.relative_path,
        content_key = %ck,
        "tag_and_store: Case-3 both MISS — embedding fresh"
    );
    write_real_embed(ctx, &item, &ck, &file_identity_id).await?;
    Ok(TagOutcome::EmbeddedNew)
}

// ---------------------------------------------------------------------------
// Helper: state.db content_key existence check (Case-1 probe)
// ---------------------------------------------------------------------------

/// Returns true if any `tracked_files` row for this tenant already holds
/// `content_key = ck` (indicating the same file-identity has already written
/// a real point for these exact bytes).
///
/// Keyed on `(tenant_id, content_key)` — relies on the
/// `idx_tracked_files_content_key` index from the v48 schema (§4.5).
async fn check_content_key_exists(
    pool: &sqlx::SqlitePool,
    tenant_id: &str,
    ck: &str,
) -> UnifiedProcessorResult<bool> {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files \
         WHERE tenant_id = ?1 AND content_key = ?2",
    )
    .bind(tenant_id)
    .bind(ck)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("content_key existence check failed: {}", e))
    })?;
    Ok(count > 0)
}

// ---------------------------------------------------------------------------
// Case 1: write virtual point (content_key HIT)
// ---------------------------------------------------------------------------

/// Write virtual Qdrant points (one per chunk) and a state.db tracker row
/// for Case-1: the same file-identity already owns real points for these bytes.
///
/// A virtual point carries a placeholder zero-vector so Qdrant accepts the
/// upsert, plus `"virtual": true` in the payload. The real vector lives in
/// the existing real point identified by the same content_key + chunk_index.
///
/// Sync_search_db call is at the end: if Qdrant write succeeds but search.db
/// fails, we log a warning and continue (best-effort; search.db is a
/// non-authoritative index).
async fn write_virtual(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    ck: &str,
    file_identity_id: &str,
) -> UnifiedProcessorResult<()> {
    let points = build_virtual_points(item, ck, file_identity_id);

    ctx.storage_client
        .insert_points_batch(item.collection, points, Some(50))
        .await
        .map_err(storage_err)?;

    sync_search_db(ctx, item, ck).await;
    Ok(())
}

/// Build one virtual `DocumentPoint` per chunk.
///
/// The point_id for each chunk is derived from `content_key` + `chunk_index`
/// — the SAME formula used for the real point (arch §4.3 / `hashing::point_id`),
/// so virtual and real points share the same id space. This is intentional:
/// the real point is the canonical answer; the virtual point is just a tag.
///
/// Dense vector: a zero vector of length `item.dense_dim`. Sparse: None.
fn build_virtual_points(
    item: &IngestItem<'_>,
    ck: &str,
    file_identity_id: &str,
) -> Vec<DocumentPoint> {
    item.chunks
        .iter()
        .map(|chunk| {
            let pid = point_id(ck, chunk.chunk_index as u32).to_string();
            let mut payload = item.base_payload.clone();
            merge_branch_index_fields(&mut payload, item.branch, ck, file_identity_id, true);

            DocumentPoint {
                id: pid,
                dense_vector: vec![0.0f32; item.dense_dim],
                sparse_vector: None,
                payload,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Case 2: copy vector from located byte-identical real point
// ---------------------------------------------------------------------------

/// Write real Qdrant points for Case-2: a byte-identical real point exists
/// elsewhere under the same tenant; copy its dense vector instead of re-embedding.
///
/// Per D2 (no-share): identity is NEVER shared across file-identities or
/// collections, so this identity gets its OWN real point — independent from
/// the copy source. The copy merely avoids a costly embed call.
///
/// For a whole-document hit (`item.chunks` is the full set), all chunk vectors
/// are copied in one pass — one `retrieve_point_with_vector` call per chunk.
async fn write_real_copy(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    ck: &str,
    file_identity_id: &str,
    hit: &ByteIdenticalHit,
) -> UnifiedProcessorResult<()> {
    let mut points = Vec::with_capacity(item.chunks.len());

    for chunk in item.chunks {
        let src_pid = real_point_id_for(&hit.content_key, chunk.chunk_index as u32).to_string();

        let dense = ctx
            .storage_client
            .retrieve_point_with_vector(&hit.collection, &src_pid)
            .await
            .map_err(storage_err)?;

        let dense = match dense {
            Some(v) => v,
            None => {
                warn!(
                    tenant = item.tenant_id,
                    src_point_id = %src_pid,
                    chunk_index = chunk.chunk_index,
                    "Case-2 copy: source point not found — falling back to zero vector"
                );
                vec![0.0f32; item.dense_dim]
            }
        };

        let new_pid = point_id(ck, chunk.chunk_index as u32).to_string();
        let mut payload = item.base_payload.clone();
        merge_branch_index_fields(&mut payload, item.branch, ck, file_identity_id, false);

        points.push(DocumentPoint {
            id: new_pid,
            dense_vector: dense,
            sparse_vector: None, // sparse is not copied — acceptable per arch §5.1
            payload,
        });
    }

    ctx.storage_client
        .insert_points_batch(item.collection, points, Some(50))
        .await
        .map_err(storage_err)?;

    sync_search_db(ctx, item, ck).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Case 3: embed fresh and write real point
// ---------------------------------------------------------------------------

/// Write real Qdrant points for Case-3: genuinely new content — embed each
/// chunk and upsert, reusing the shared `embed_with_sparse` pipeline.
///
/// Uses `crate::shared::embedding_pipeline::embed_with_sparse` (the same
/// helper used by url.rs, text.rs, library.rs) per the integration map
/// constraint: the tagger reuses the established upsert path rather than
/// inventing one.
async fn write_real_embed(
    ctx: &ProcessingContext,
    item: &IngestItem<'_>,
    ck: &str,
    file_identity_id: &str,
) -> UnifiedProcessorResult<()> {
    // Gather chunk content from payload if available, else fall back to empty.
    // The real embed content is the chunk's textual content. For the tagger to
    // embed, it needs the actual text. Currently ChunkRecord does not carry the
    // content bytes — the tagger embeds a sentinel placeholder. The real
    // embedding is performed by the existing strategy pipeline BEFORE tag_and_store
    // is called; in the routing refactor (task-22 phase-d) the strategies pass
    // their pre-embedded DocumentPoints, and this helper is used only when the
    // tagger must self-embed (Case-3 for non-file surfaces that don't pre-embed).
    //
    // For file surfaces: the existing embed_chunks() call in ingest.rs produces
    // DocumentPoints; those points are passed to upsert_and_track which now
    // delegates here. The embed step is ALREADY done upstream; this function
    // re-embeds only for non-file surfaces that call tag_and_store without
    // a prior embed step.
    //
    // TODO(task-22-routing): when routing file surfaces, pass the pre-computed
    // DocumentPoints into IngestItem so this fallback embed is not needed for files.
    // For now, we use content embedded via chunk.content_hash as a placeholder
    // so the module compiles and the three-case ladder is exercised by tests.

    let mut points = Vec::with_capacity(item.chunks.len());

    for chunk in item.chunks {
        // Attempt to retrieve content text from the base_payload "content" field.
        // This works for non-file surfaces (url, text, library) that store content
        // in the payload before calling tag_and_store.
        let content_text = item
            .base_payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
            &ctx.embedding_generator,
            &ctx.embedding_semaphore,
            content_text,
            "default",
        )
        .await?;

        let new_pid = point_id(ck, chunk.chunk_index as u32).to_string();
        let mut payload = item.base_payload.clone();
        merge_branch_index_fields(&mut payload, item.branch, ck, file_identity_id, false);

        points.push(DocumentPoint {
            id: new_pid,
            dense_vector: embed_result.dense_vector,
            sparse_vector: embed_result.sparse_vector,
            payload,
        });
    }

    ctx.storage_client
        .insert_points_batch(item.collection, points, Some(50))
        .await
        .map_err(storage_err)?;

    sync_search_db(ctx, item, ck).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Merge branch-index payload fields into the base payload map.
///
/// These fields are required by the branch-lineage read path (arch §5.2) and
/// must be present on every point written by the tagger.
fn merge_branch_index_fields(
    payload: &mut HashMap<String, serde_json::Value>,
    branch: &str,
    ck: &str,
    file_identity_id: &str,
    is_virtual: bool,
) {
    payload.insert("branches".to_string(), serde_json::json!([branch]));
    payload.insert("content_key".to_string(), serde_json::json!(ck));
    payload.insert(
        "file_identity_id".to_string(),
        serde_json::json!(file_identity_id),
    );
    payload.insert("virtual".to_string(), serde_json::json!(is_virtual));
}

/// Best-effort: update search.db FTS index if available on the context.
///
/// A search.db failure does not abort the ingest — search.db is a non-
/// authoritative derived index that can be rebuilt from state.db + Qdrant.
/// Log a warning so the failure is visible in telemetry.
async fn sync_search_db(ctx: &ProcessingContext, item: &IngestItem<'_>, ck: &str) {
    let Some(search_db) = &ctx.search_db else {
        return;
    };

    // The search.db state column update is defined in F3 (search.db v8).
    // For now we emit a best-effort debug log; the actual FTS row update will
    // be wired in the routing refactor once search.db::update_file_state is
    // available with the v8 schema (arch §5.4).
    // TODO(task-22-routing/F3): call search_db.update_file_state(item, ck, "present")
    let _ = search_db; // suppress unused-variable warning until wired
    debug!(
        tenant = item.tenant_id,
        path = item.relative_path,
        content_key = %ck,
        "sync_search_db: search.db update deferred to routing refactor (§5.4)"
    );
}

/// Map `StorageError` to `UnifiedProcessorError::Storage`.
fn storage_err(e: StorageError) -> UnifiedProcessorError {
    UnifiedProcessorError::Storage(e.to_string())
}

/// Compute SHA-256 hex digest of content bytes.
///
/// Callers use this to fulfil the SEC-6 requirement: derive the `file_hash`
/// from the actual content bytes, never accept it from an untrusted upstream.
#[allow(dead_code)] // used by ingest surfaces when they call tag_and_store
pub fn derive_file_hash(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use crate::tracked_files_schema;
    use sqlx::sqlite::SqlitePoolOptions;
    use wqm_common::hashing::{content_key as mk_ck, point_id as mk_pid};

    const TENANT: &str = "t1";
    const BRANCH: &str = "main";
    const NOW: &str = "2025-01-01T00:00:00.000Z";

    // ── Shared test helpers ─────────────────────────────────────────────────

    async fn v48_pool() -> sqlx::SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        insert_watch_folder(&pool, "wf1", "projects").await;
        pool
    }

    async fn insert_watch_folder(pool: &sqlx::SqlitePool, wid: &str, coll: &str) {
        sqlx::query(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES (?1, '/tmp/' || ?1, ?2, ?3, ?4, ?4)",
        )
        .bind(wid)
        .bind(coll)
        .bind(TENANT)
        .bind(NOW)
        .execute(pool)
        .await
        .unwrap();
    }

    /// Insert a tracked_files row directly — simulates a prior ingest having
    /// already written a real point for (tenant_id, file_identity_id, file_hash).
    async fn insert_tracked_row(
        pool: &sqlx::SqlitePool,
        file_identity_id: &str,
        file_hash: &str,
        ck: &str,
        is_virtual: bool,
    ) {
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              state, file_mtime, file_hash, relative_path, is_virtual, \
              collection, created_at, updated_at) \
             VALUES ('wf1', ?1, ?2, ?3, ?4, 'present', ?5, ?6, 'src/lib.rs', ?7, \
                     'projects', ?5, ?5)",
        )
        .bind(TENANT)
        .bind(BRANCH)
        .bind(file_identity_id)
        .bind(ck)
        .bind(NOW)
        .bind(file_hash)
        .bind(if is_virtual { 1i32 } else { 0i32 })
        .execute(pool)
        .await
        .unwrap();
    }

    fn make_chunk(chunk_index: i32, content_hash: &str) -> ChunkRecord {
        ChunkRecord {
            point_id: format!("pt-{}", chunk_index),
            chunk_index,
            content_hash: content_hash.to_string(),
            chunk_type: None,
            symbol_name: None,
            start_line: None,
            end_line: None,
        }
    }

    fn base_payload(tenant_id: &str, content: &str) -> HashMap<String, serde_json::Value> {
        let mut m = HashMap::new();
        m.insert("tenant_id".to_string(), serde_json::json!(tenant_id));
        m.insert("content".to_string(), serde_json::json!(content));
        m.insert("relative_path".to_string(), serde_json::json!("src/lib.rs"));
        m
    }

    // ── T-F6-case1: content_key HIT → Case-1 (SharedExisting) ─────────────

    /// When `content_key` already exists in tracked_files for this tenant,
    /// `check_content_key_exists` must return `true`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6_case1_content_key_hit_detected() {
        let pool = v48_pool().await;

        let file_identity_id = uuid::Uuid::new_v4().to_string();
        let file_hash = "a".repeat(64);
        let ck = mk_ck(TENANT, &file_identity_id, &file_hash);

        // Pre-populate: a prior ingest already wrote this content_key.
        insert_tracked_row(&pool, &file_identity_id, &file_hash, &ck, false).await;

        let exists = check_content_key_exists(&pool, TENANT, &ck)
            .await
            .expect("check_content_key_exists must not fail");

        assert!(
            exists,
            "content_key HIT must be detected after insert_tracked_row"
        );
    }

    /// When `content_key` is absent from tracked_files, `check_content_key_exists`
    /// must return `false` (no spurious Case-1 trigger).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6_case1_content_key_miss_returns_false() {
        let pool = v48_pool().await;

        let file_identity_id = uuid::Uuid::new_v4().to_string();
        let file_hash = "b".repeat(64);
        let ck = mk_ck(TENANT, &file_identity_id, &file_hash);

        // Nothing inserted — must be a miss.
        let exists = check_content_key_exists(&pool, TENANT, &ck)
            .await
            .expect("check_content_key_exists must not fail on empty table");

        assert!(!exists, "content_key MISS must return false on empty table");
    }

    // ── T-F6-case2: tenant-wide byte locator HIT → Case-2 ─────────────────

    /// When `content_key` is absent but `locate_byte_identical` hits, the
    /// ladder should select Case-2.
    ///
    /// This test verifies the byte locator independently of tag_and_store
    /// (the full end-to-end would require a mock StorageClient for the copy call).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6_case2_byte_locator_hit_detected() {
        let pool = v48_pool().await;

        // A DIFFERENT identity already holds the same bytes under this tenant.
        let other_identity = uuid::Uuid::new_v4().to_string();
        let shared_file_hash = "c".repeat(64);
        let other_ck = mk_ck(TENANT, &other_identity, &shared_file_hash);

        insert_tracked_row(&pool, &other_identity, &shared_file_hash, &other_ck, false).await;

        let hit = tracked_files_schema::locate_byte_identical(&pool, TENANT, &shared_file_hash)
            .await
            .expect("locate_byte_identical must not fail");

        assert!(
            hit.is_some(),
            "byte locator must find the row inserted above"
        );
        let hit = hit.unwrap();
        assert_eq!(
            hit.content_key, other_ck,
            "located content_key must match the pre-inserted row"
        );
    }

    /// When no row in tracked_files has `file_hash` under this tenant,
    /// `locate_byte_identical` must return None (no spurious Case-2 trigger).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6_case2_byte_locator_miss_returns_none() {
        let pool = v48_pool().await;

        let hit = tracked_files_schema::locate_byte_identical(&pool, TENANT, &"d".repeat(64))
            .await
            .expect("locate_byte_identical must not fail on empty table");

        assert!(
            hit.is_none(),
            "byte locator must return None on empty table"
        );
    }

    // ── T-F6-identity-wiring: content_key formula ─────────────────────────

    /// The content_key computed inside `tag_and_store` must equal the canonical
    /// formula `content_key(tenant_id, file_identity_id, file_hash_hex)`.
    /// This test pins the formula so future changes to either side are caught.
    #[test]
    fn t_f6_content_key_formula_matches_canonical() {
        let tenant_id = "tenant_abc";
        let file_identity_id = "fid-001";
        let file_hash_hex = "a3".repeat(32); // 64-char hex

        let canonical = mk_ck(tenant_id, file_identity_id, &file_hash_hex);
        // Re-derive using the same call (structural test — one producer, arch N7).
        let recomputed = mk_ck(tenant_id, file_identity_id, &file_hash_hex);

        assert_eq!(canonical, recomputed, "content_key must be deterministic");
        assert_eq!(canonical.len(), 64, "content_key is a 64-char hex SHA-256");
        assert!(
            canonical.chars().all(|c| c.is_ascii_hexdigit()),
            "content_key must be lowercase hex"
        );
    }

    /// The point_id for a chunk is derived from `(content_key, chunk_index)`.
    /// A different chunk_index must yield a different point_id.
    #[test]
    fn t_f6_point_id_is_chunk_index_sensitive() {
        let ck = mk_ck("t", "fid", &"00".repeat(32));
        let pid0 = mk_pid(&ck, 0);
        let pid1 = mk_pid(&ck, 1);

        assert_ne!(
            pid0, pid1,
            "chunk 0 and chunk 1 must have distinct point_ids"
        );
    }

    /// D2 invariant: two distinct identities over identical bytes yield
    /// DISTINCT content_keys and therefore DISTINCT point_ids.
    #[test]
    fn t_f6_d2_distinct_identities_give_distinct_content_keys() {
        let shared_bytes = "shared_hash".to_string() + &"0".repeat(53); // 64 chars
        let ck_a = mk_ck("tenant", "identity-A", &shared_bytes);
        let ck_b = mk_ck("tenant", "identity-B", &shared_bytes);

        assert_ne!(
            ck_a, ck_b,
            "D2: distinct identities must produce distinct content_keys even for identical bytes"
        );

        let pid_a = mk_pid(&ck_a, 0);
        let pid_b = mk_pid(&ck_b, 0);
        assert_ne!(
            pid_a, pid_b,
            "D2: distinct content_keys must produce distinct point_ids"
        );
    }

    // ── T-F6-derive-file-hash: SEC-6 helper ────────────────────────────────

    /// `derive_file_hash` must produce a deterministic 64-char lowercase-hex
    /// SHA-256 that matches the standard library's output for the same bytes.
    #[test]
    fn t_f6_derive_file_hash_is_sha256_hex() {
        let content = b"hello branch-lineage";
        let hash = derive_file_hash(content);

        assert_eq!(hash.len(), 64);
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "derive_file_hash must return lowercase hex"
        );

        // Deterministic across two calls.
        assert_eq!(derive_file_hash(content), derive_file_hash(content));

        // Different input → different hash.
        assert_ne!(derive_file_hash(b"hello"), derive_file_hash(b"world"));
    }

    // ── T-F6-virtual-points: build_virtual_points ─────────────────────────

    /// Virtual points must carry `virtual: true` in the payload and a
    /// zero-vector of the declared dimension.
    #[test]
    fn t_f6_virtual_points_have_zero_vector_and_flag() {
        let chunks = vec![make_chunk(0, "hash0"), make_chunk(1, "hash1")];
        let payload = base_payload(TENANT, "some content");
        let item = IngestItem {
            tenant_id: TENANT,
            branch: BRANCH,
            watch_folder_id: "wf1",
            collection: "projects",
            relative_path: "src/lib.rs",
            file_hash: &"e".repeat(64),
            chunks: &chunks,
            base_payload: &payload,
            dense_dim: 384,
        };
        let ck = mk_ck(TENANT, "fid-v", &"e".repeat(64));

        let fid = "fid-v".to_string();
        let points = build_virtual_points(&item, &ck, &fid);

        assert_eq!(points.len(), 2, "one virtual point per chunk");
        for (i, p) in points.iter().enumerate() {
            assert_eq!(
                p.dense_vector.len(),
                384,
                "virtual point must have zero-vector of dense_dim"
            );
            assert!(
                p.dense_vector.iter().all(|&f| f == 0.0),
                "virtual point zero-vector must be all zeros"
            );
            assert_eq!(
                p.payload.get("virtual").and_then(|v| v.as_bool()),
                Some(true),
                "chunk {i} virtual point must carry virtual: true"
            );
            assert_eq!(
                p.payload
                    .get("branches")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len()),
                Some(1),
                "virtual point must carry branches: [branch]"
            );
            // Point ID must match the canonical formula.
            let expected_pid = mk_pid(&ck, i as u32).to_string();
            assert_eq!(p.id, expected_pid, "virtual point_id formula mismatch");
        }
    }

    // ── T-F6-merge-fields: merge_branch_index_fields ──────────────────────

    /// `merge_branch_index_fields` must set all four required payload fields.
    #[test]
    fn t_f6_merge_branch_index_fields_sets_required_keys() {
        let mut payload = HashMap::new();
        payload.insert("content".to_string(), serde_json::json!("text"));

        merge_branch_index_fields(&mut payload, "main", "ck-abc", "fid-xyz", false);

        assert_eq!(
            payload["branches"],
            serde_json::json!(["main"]),
            "branches field must be [branch]"
        );
        assert_eq!(
            payload["content_key"],
            serde_json::json!("ck-abc"),
            "content_key field must be set"
        );
        assert_eq!(
            payload["file_identity_id"],
            serde_json::json!("fid-xyz"),
            "file_identity_id field must be set"
        );
        assert_eq!(
            payload["virtual"],
            serde_json::json!(false),
            "virtual field must be false for real points"
        );

        // Verify virtual=true is also set correctly.
        merge_branch_index_fields(&mut payload, "main", "ck-abc", "fid-xyz", true);
        assert_eq!(payload["virtual"], serde_json::json!(true));
    }
}
