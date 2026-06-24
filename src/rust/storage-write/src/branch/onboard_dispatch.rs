//! Per-file dispatch helpers for `branch_onboard` and `apply_git_diff` (arch §4.2/§4.6, F8).
//!
//! Extracted from `onboard.rs` for codesize compliance (coding.md §X / 500-line limit).
//! Contains: `ChangeOutcome`, `apply_one_change`, `ChangeCtx`, `dispatch_change`,
//! arm-specific helpers (`dispatch_ingest`, `dispatch_delete`, `dispatch_rename`),
//! `ingest_one`, and `delete_one`.
//!
//! Visibility: `pub(super)` functions are callable from `onboard.rs` only.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::debug;
use wqm_common::error::StorageError;
use wqm_common::git::file_change::{FileChange, FileChangeStatus};
use wqm_storage::types::stats::IngestOutcome;

use crate::blob::dedup::{ingest_file, IngestParams};
use crate::blob::embed::Embedder;
use crate::blob::file_delete::delete_file_from_branch;
use crate::blob::ladder::QdrantSink;
use crate::blob::lock::ContentKeyLockManager;
use crate::qdrant::membership_batch::MembershipPutBatch;

use super::FileContentProvider;

// ---------------------------------------------------------------------------
// ChangeOutcome (public within crate for use in sql accumulator)
// ---------------------------------------------------------------------------

/// Per-`FileChange` outcome from one dispatch (ingest stats).
pub(crate) struct ChangeOutcome {
    pub ingest: IngestOutcome,
}

// ---------------------------------------------------------------------------
// apply_one_change (pub(crate) -- called from tests and onboard.rs)
// ---------------------------------------------------------------------------

/// Dispatch one `FileChange`: ingest, delete, or rename (ingest-then-delete).
///
/// `provider.chunk_file` is called at most once per path per call (AC-F8.3).
/// For `Renamed`: ingest(new) BEFORE delete(old) so refcount never transiently
/// hits 0 (DOM-02, AC-F8.6).
pub(crate) async fn apply_one_change(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    collection_name: &str,
    change: &FileChange,
) -> Result<ChangeOutcome, StorageError> {
    let ctx = ChangeCtx {
        pool,
        locks,
        embedder,
        sink,
        batch,
        provider,
        tenant_id,
        branch_id,
        collection_id,
        collection_name,
    };
    dispatch_change(ctx, change).await
}

// ---------------------------------------------------------------------------
// ChangeCtx: parameter bundle (reduces fn arg counts in dispatch arms)
// ---------------------------------------------------------------------------

/// Parameter bundle for `dispatch_change` (avoids repetition in match arms).
struct ChangeCtx<'a> {
    pool: &'a SqlitePool,
    locks: &'a Arc<ContentKeyLockManager>,
    embedder: &'a dyn Embedder,
    sink: &'a mut dyn QdrantSink,
    batch: &'a mut MembershipPutBatch,
    provider: &'a dyn FileContentProvider,
    tenant_id: &'a str,
    branch_id: &'a str,
    collection_id: &'a str,
    collection_name: &'a str,
}

// ---------------------------------------------------------------------------
// dispatch_change: route to arm
// ---------------------------------------------------------------------------

/// Route a `FileChange` to the appropriate arm.
async fn dispatch_change(
    ctx: ChangeCtx<'_>,
    change: &FileChange,
) -> Result<ChangeOutcome, StorageError> {
    let ChangeCtx {
        pool,
        locks,
        embedder,
        sink,
        batch,
        provider,
        tenant_id,
        branch_id,
        collection_id,
        collection_name,
    } = ctx;
    match &change.status {
        FileChangeStatus::Added
        | FileChangeStatus::Modified
        | FileChangeStatus::TypeChanged
        | FileChangeStatus::Copied { .. } => {
            dispatch_ingest(
                pool,
                locks,
                embedder,
                sink,
                provider,
                tenant_id,
                branch_id,
                collection_id,
                &change.path,
            )
            .await
        }
        FileChangeStatus::Deleted => {
            dispatch_delete(
                pool,
                locks,
                sink,
                batch,
                branch_id,
                tenant_id,
                collection_id,
                collection_name,
                &change.path,
            )
            .await
        }
        FileChangeStatus::Renamed { old_path, .. } => {
            dispatch_rename(
                pool,
                locks,
                embedder,
                sink,
                batch,
                provider,
                tenant_id,
                branch_id,
                collection_id,
                collection_name,
                &change.path,
                old_path,
            )
            .await
        }
    }
}

// ---------------------------------------------------------------------------
// Arm-specific dispatch helpers
// ---------------------------------------------------------------------------

/// Ingest arm: Added / Modified / TypeChanged / Copied.
async fn dispatch_ingest(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    path: &str,
) -> Result<ChangeOutcome, StorageError> {
    let outcome = ingest_one(
        pool,
        locks,
        embedder,
        sink,
        provider,
        tenant_id,
        branch_id,
        collection_id,
        path,
    )
    .await?;
    Ok(ChangeOutcome { ingest: outcome })
}

/// Delete arm: Deleted.
async fn dispatch_delete(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    branch_id: &str,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
    path: &str,
) -> Result<ChangeOutcome, StorageError> {
    delete_one(
        pool,
        locks,
        sink,
        batch,
        branch_id,
        tenant_id,
        collection_id,
        collection_name,
        path,
    )
    .await?;
    Ok(ChangeOutcome {
        ingest: IngestOutcome::default(),
    })
}

/// Rename arm: ingest(new) BEFORE delete(old) (DOM-02, AC-F8.6).
#[allow(clippy::too_many_arguments)]
async fn dispatch_rename(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    collection_name: &str,
    new_path: &str,
    old_path: &str,
) -> Result<ChangeOutcome, StorageError> {
    // Ingest new path FIRST: HIT path fires for same content, so no re-embed occurs.
    let outcome = ingest_one(
        pool,
        locks,
        embedder,
        sink,
        provider,
        tenant_id,
        branch_id,
        collection_id,
        new_path,
    )
    .await?;
    // Delete old path AFTER: shared blob refcount stays >= 1 throughout.
    delete_one(
        pool,
        locks,
        sink,
        batch,
        branch_id,
        tenant_id,
        collection_id,
        collection_name,
        old_path,
    )
    .await?;
    Ok(ChangeOutcome { ingest: outcome })
}

// ---------------------------------------------------------------------------
// Leaf helpers: ingest_one / delete_one
// ---------------------------------------------------------------------------

/// Ingest one file via `provider.chunk_file` then `ingest_file` (F6 ladder, AC-F8.3).
pub(super) async fn ingest_one(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    path: &str,
) -> Result<IngestOutcome, StorageError> {
    let req = provider.chunk_file(tenant_id, branch_id, path).await?;
    // file_hash: content-address the whole file via first chunk hash, or empty for empty files.
    let file_hash = req
        .chunks
        .first()
        .map(|c| c.content_hash.as_str())
        .unwrap_or("");
    let params = IngestParams {
        tenant_id,
        branch_id,
        collection_id,
        content_key_version: 4, // four-slot collection-discriminated (arch §5.1)
        file_hash,
    };
    ingest_file(pool, locks, embedder, sink, &params, &req).await
}

/// Look up file_id for `(branch_id, path)` and call `delete_file_from_branch`.
pub(super) async fn delete_one(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    branch_id: &str,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
    path: &str,
) -> Result<(), StorageError> {
    let file_id: Option<i64> =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch_id)
            .bind(path)
            .fetch_optional(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("lookup file_id for delete: {e}")))?;

    if let Some(fid) = file_id {
        delete_file_from_branch(
            pool,
            locks,
            sink,
            batch,
            branch_id,
            fid,
            tenant_id,
            collection_id,
            collection_name,
        )
        .await?;
    } else {
        debug!("delete_one: no files row for branch={branch_id} path={path}; skipping");
    }
    Ok(())
}
