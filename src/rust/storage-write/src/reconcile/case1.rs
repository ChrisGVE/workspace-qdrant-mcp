//! Case 1 -- missing Qdrant membership heal (arch §4.7 case 1, AC-F15.1 / AC-F15.3).
//!
//! File: `wqm-storage-write/src/reconcile/case1.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: Reconcile case 1 handles the drift where SQLite says branch B owns blob X
//!   but the Qdrant point's `branch_id[]` payload lacks B. Root cause: crash after the
//!   §4.1 SQLite write, before the Qdrant payload flush.
//!
//!   Fix (additive-first, FP-1):
//!     1. Scan `blobs` in the watermark window (`blob_id > max_seen_blob_id`).
//!     2. For each blob, call `compute_membership` (the canonical SELECT DISTINCT
//!        producer) and enqueue a `QdrantOp::OverwritePayload` (PUT) via the sink.
//!     3. Sub-case -- point absent from Qdrant: also enqueue a `QdrantOp::Upsert`
//!        that carries the durable dense+sparse vectors (no re-embed, AC-F15.3).
//!     4. In FULL mode (`watermark = 0`) the scan covers all blobs.
//!
//!   The Qdrant existence check is provided by the injectable `QdrantPointReader` seam
//!   so offline tests can mock it without a live Qdrant connection. The LIVE
//!   implementation RIDES #175 (daemon assembly).
//!
//! Neighbors: [`crate::blob::membership::compute_membership`] (canonical producer),
//!   [`super::seams::QdrantPointReader`] (injectable scroll seam),
//!   [`crate::blob::ladder::{QdrantOp, QdrantSink}`] (op sink).

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

use crate::blob::ladder::{BlobPayload, QdrantOp, QdrantSink};
use crate::blob::membership::compute_membership;
use crate::blob::vector_codec::{decode_dense, decode_sparse};

use super::seams::QdrantPointReader;

/// One blob row read from `blobs` for the case-1 scan.
struct BlobRow {
    blob_id: i64,
    point_id: String,
    tenant_id: String,
    dense_vec: Vec<u8>,
    sparse_vec: Vec<u8>,
}

/// Fetch blob rows above the watermark for the incremental scan.
///
/// Pass `watermark = 0` for a FULL pass (scan every blob).
async fn fetch_blobs_above_watermark(
    pool: &SqlitePool,
    watermark: i64,
) -> Result<Vec<BlobRow>, StorageError> {
    let rows = sqlx::query(
        "SELECT blob_id, point_id, tenant_id, dense_vec, sparse_vec \
         FROM blobs WHERE blob_id > ? ORDER BY blob_id",
    )
    .bind(watermark)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("case1 fetch_blobs: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| BlobRow {
            blob_id: r.get("blob_id"),
            point_id: r.get("point_id"),
            tenant_id: r.get("tenant_id"),
            dense_vec: r.get("dense_vec"),
            sparse_vec: r.get("sparse_vec"),
        })
        .collect())
}

/// Reconcile case 1: heal missing Qdrant membership for blobs in the scan window.
///
/// For each blob in the watermark window:
///   - Enqueues `QdrantOp::OverwritePayload` (PUT) with the full membership set
///     from `compute_membership` -- idempotent if Qdrant is correct, heals drift
///     if it is not.
///   - If `reader.point_exists(point_id)` returns false, also enqueues
///     `QdrantOp::Upsert` with the durable vectors (no re-embed, AC-F15.3).
///
/// Returns the maximum `blob_id` scanned (for watermark advancement). Returns
/// `watermark` unchanged if no blobs exist in the window.
pub async fn run_case1<S, R>(
    pool: &SqlitePool,
    sink: &mut S,
    reader: &R,
    watermark: i64,
    collection_id: &str,
    collection_name: &str,
) -> Result<i64, StorageError>
where
    S: QdrantSink,
    R: QdrantPointReader,
{
    let blobs = fetch_blobs_above_watermark(pool, watermark).await?;
    let mut max_blob_id = watermark;

    for blob in &blobs {
        if blob.blob_id > max_blob_id {
            max_blob_id = blob.blob_id;
        }

        let branch_ids = compute_membership(pool, blob.blob_id).await?;
        let payload = BlobPayload {
            tenant_id: blob.tenant_id.clone(),
            branch_id: branch_ids,
            collection_id: collection_id.to_owned(),
        };

        // PUT the full membership -- heals missing branch_id[] entries (additive-first).
        sink.enqueue(QdrantOp::OverwritePayload {
            point_id: blob.point_id.clone(),
            payload: payload.clone(),
        });

        // Sub-case: point absent from Qdrant -- re-upsert from durable vectors.
        if !reader.point_exists(&blob.point_id).await {
            let dense = decode_dense(&blob.dense_vec);
            let sparse = decode_sparse(&blob.sparse_vec);
            sink.enqueue(QdrantOp::Upsert {
                point_id: blob.point_id.clone(),
                dense,
                sparse,
                payload,
            });
        }

        let _ = collection_name; // used by caller when flushing the sink ops
    }

    Ok(max_blob_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "case1_tests.rs"]
mod tests;
