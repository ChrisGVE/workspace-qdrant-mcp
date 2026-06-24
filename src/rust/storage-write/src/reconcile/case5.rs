//! Case 5 -- cross-DB tenant-mismatch heal (arch §4.7 case 5, AC-F15.6).
//!
//! File: `wqm-storage-write/src/reconcile/case5.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: A Qdrant point whose payload `tenant_id` disagrees with the owning
//!   store's `store_meta.tenant_id`. Produced by crash mid-re-tenant (F16 AC-F16.2
//!   / AC-F16.5): the copy-then-delete migration writes the destination row, PUTs the
//!   Qdrant payload, then deletes the source row LAST -- a crash between the PUT and
//!   the source-row delete leaves BOTH stores with a `blob_refs` row but the point's
//!   payload already naming the destination tenant.
//!
//!   **MUST run BEFORE case 2.** Without this fix, case 2 sees a point whose payload
//!   tenant finds zero referrers in the WRONG store and culls it = silent data loss
//!   (DATA-R7-04).
//!
//!   **M4 detection bound (AC-F15.6):** case-5 is bounded by the migration journal.
//!   It runs ONLY when `watermark.tenant_move_since_last_pass()` is true -- i.e., the
//!   `maintenance_meta.last_tenant_move_at` timestamp postdates the last successful
//!   reconcile. On an EMPTY journal (no tenant-moves ever), case-5 performs NO
//!   full-collection scan (asserted by AC-F15.6 test iii).
//!
//!   **M1 disambiguation (AC-F15.6):** for each candidate point (sourced from the
//!   migration journal via the injectable `QdrantPointReader` seam):
//!
//!   - Read the current Qdrant payload `tenant_id` via `reader.payload_tenant_id`.
//!   - Query ALL candidate stores for `store_meta.tenant_id`.
//!   - If exactly ONE store's `store_meta.tenant_id` EQUALS the payload tenant:
//!     NO-OP -- the payload already names the intended owner; the pending source-row
//!     delete (which resolves the duplicate) is still in flight. Do NOT re-PUT back
//!     to the source tenant (would revert the migration and oscillate).
//!   - If NO store matches the payload tenant: genuine stale mismatch. Re-derive the
//!     authoritative tenant from the store that holds the `blob_refs`/`concrete` row
//!     and enqueue `QdrantOp::OverwritePayload` (additive-first, no re-embed, no cull).
//!
//! The `QdrantPointReader` seam is injected for offline testing. The LIVE wiring
//! (real Qdrant scroll driven by the migration journal) RIDES #175.
//!
//! Neighbors: [`super::seams::QdrantPointReader`] (injectable reader),
//!   [`crate::blob::ladder::{QdrantOp, QdrantSink}`] (op sink).

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::blob::ladder::{BlobPayload, QdrantOp, QdrantSink};

use super::seams::QdrantPointReader;
use super::watermark::ReconcileWatermark;

/// A candidate for case-5 inspection: a point whose payload tenant may disagree
/// with the owning store's `store_meta.tenant_id`.
///
/// In the real daemon (#175) this set is sourced from the migration journal (the
/// per-tenant `maintenance_meta` tenant-move records). In tests, callers inject
/// the candidate set directly so no Qdrant scroll is needed.
#[derive(Debug, Clone)]
pub struct TenantMismatchCandidate {
    /// The Qdrant point id to inspect.
    pub point_id: String,
    /// The `tenant_id` recorded in `store_meta` for this store (the owning store's
    /// authoritative tenant).
    pub store_tenant_id: String,
    /// The `blob_id` in `blobs` for this point (for re-deriving the owner if needed).
    pub blob_id: i64,
    /// The `collection_id` for the payload.
    pub collection_id: String,
    /// All candidate store tenant IDs involved in the migration that produced this
    /// candidate (source AND destination). Used for M1 disambiguation: if the current
    /// Qdrant payload tenant is present in this set, the payload already names a
    /// legitimate owner (transient copy-then-delete window) -- NO-OP.
    ///
    /// In the live daemon (#175) this is sourced from the migration journal entries
    /// for this point. In tests, inject the set directly.
    pub candidate_store_tenants: Vec<String>,
}

/// Read the `store_meta.tenant_id` from the store.
pub async fn read_store_tenant(pool: &SqlitePool) -> Result<String, StorageError> {
    sqlx::query_scalar("SELECT tenant_id FROM store_meta")
        .fetch_one(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("case5 read_store_tenant: {e}")))
}

/// Reconcile case 5: heal cross-DB tenant-mismatch points.
///
/// **Skipped entirely when `watermark.tenant_move_since_last_pass()` is false**
/// (M4 detection bound, AC-F15.6 test iii). This means no Qdrant scroll or
/// SQLite scan is performed when the migration journal is empty or predates the
/// last reconcile.
///
/// For each candidate point in `candidates`:
///   1. Read the current payload `tenant_id` via `reader.payload_tenant_id`.
///   2. **M1 disambiguation**: if the payload tenant is present in
///      `candidate.candidate_store_tenants` -> NO-OP. The payload already names
///      a legitimate owner (either already correct, or the transient
///      copy-then-delete window where the destination PUT succeeded but the source
///      row delete is still pending). Do NOT re-PUT back to the source tenant --
///      that would revert the migration and oscillate.
///   3. If NO candidate store tenant matches the payload AND this store holds a
///      `blob_refs` row -> genuine stale mismatch; enqueue `QdrantOp::OverwritePayload`
///      to fix (additive-first: no Delete).
///
/// Returns the number of points where an `OverwritePayload` was enqueued.
pub async fn run_case5<S, R>(
    pool: &SqlitePool,
    sink: &mut S,
    reader: &R,
    watermark: &ReconcileWatermark,
    candidates: &[TenantMismatchCandidate],
    collection_id: &str,
) -> Result<u64, StorageError>
where
    S: QdrantSink,
    R: QdrantPointReader,
{
    // M4 detection bound: skip entirely if no tenant-move since last pass.
    if !watermark.tenant_move_since_last_pass() {
        return Ok(0);
    }

    let store_tenant = read_store_tenant(pool).await?;
    let mut healed = 0u64;

    for candidate in candidates {
        let payload_tenant = match reader.payload_tenant_id(&candidate.point_id).await {
            None => {
                // Point absent from Qdrant: case-1 handles the upsert; skip here.
                continue;
            }
            Some(pt) => pt,
        };

        // M1 disambiguation: if the payload tenant is present in the candidate
        // store set, at least one store legitimately owns this payload -- NO-OP.
        // This covers BOTH directions of the transient copy-then-delete window:
        //   - Destination-store run: payload == destination tenant (in set) -> NO-OP.
        //   - Source-store run: payload == destination tenant (in set because source
        //     and destination are both candidates) -> NO-OP. This prevents the
        //     dangerous oscillation where case-5 would re-PUT back to the source.
        if candidate
            .candidate_store_tenants
            .iter()
            .any(|t| t == &payload_tenant)
        {
            continue;
        }

        // No candidate store matches the payload: genuine stale mismatch.
        // Verify this store holds a blob_refs row (is the rightful owner).
        let ref_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
            .bind(candidate.blob_id)
            .fetch_one(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("case5 ref_count: {e}")))?;

        if ref_count == 0 {
            // This store does not own the blob; let the owning store's pass fix it.
            continue;
        }

        // This store owns the blob. Re-derive branch membership and re-PUT payload.
        let branch_ids =
            crate::blob::membership::compute_membership(pool, candidate.blob_id).await?;

        let payload = BlobPayload {
            tenant_id: store_tenant.clone(),
            branch_id: branch_ids,
            collection_id: collection_id.to_owned(),
        };

        sink.enqueue(QdrantOp::OverwritePayload {
            point_id: candidate.point_id.clone(),
            payload,
        });

        healed += 1;
    }

    Ok(healed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "case5_tests.rs"]
mod tests;
