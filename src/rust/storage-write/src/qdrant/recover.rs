//! Rebuild Qdrant from durable SQLite vectors (arch §4.5, AC-F11.2/F11.3).
//!
//! File: `wqm-storage-write/src/qdrant/recover.rs`
//! Location: `src/rust/storage-write/src/qdrant/` (write-crate qdrant layer)
//! Context: The recovery path that fires when Qdrant is empty or when the
//!   operator runs `wqm admin rebuild` (routed via daemon gRPC, deferred to
//!   daemon cutover #175). This module owns the WRITE-CRATE half: the keyset-
//!   paginated cursor over `blobs JOIN blob_refs`, the vector decode, and the
//!   batch upsert into Qdrant. No embedding calls are made -- vectors are read
//!   verbatim from `blobs.dense_vec` / `blobs.sparse_vec` (AC-F11.3).
//!
//! ## Memory bound (PERF-R5-N3 / arch §4.5)
//!
//! Page size is clamped to `[PAGE_SIZE_MIN, PAGE_SIZE_MAX]` (1000..=10000).
//! A 400k-blob single-query load would spike ~1.9 GB; 10000-blob pages keep
//! per-page materialization to tens of MB. The test `rebuild_respects_page_bound`
//! asserts this with a live RSS check on a >=20000-blob fixture.
//!
//! ## Collection setup (AC-F11.1)
//!
//! `rebuild_qdrant` drops and recreates the collection (idempotent via
//! `ensure_collection` / `create_payload_index` from `qdrant::collection`),
//! then pages through the cursor and upserts. The collection drop is performed
//! by the caller (daemon / test), not here, so `rebuild_qdrant` always works
//! against a FRESHLY CREATED collection with indexes already in place.
//!
//! ## Upsert sink (offline testing)
//!
//! The function accepts a `&mut dyn RebuildSink` so offline tests can capture
//! upserted points without a live Qdrant. The real sink calls
//! `QdrantWriteClient::upsert_points`; the capture sink records the
//! `(point_id, dense, sparse, payload)` tuples for assertion.
//!
//! Neighbors: [`crate::qdrant::collection`] (collection + index creation),
//!   [`crate::blob::ladder::{decode_dense, decode_sparse}`] (vector decode),
//!   [`crate::qdrant::membership::blob_payload_to_qdrant`] (payload map),
//!   [`crate::blob::ladder::BlobPayload`] (three-field payload shape).

use std::collections::HashMap;

use qdrant_client::qdrant::{
    point_id, vectors, DenseVector, NamedVectors, PointId, PointStruct, SparseVector,
    UpsertPointsBuilder, Vectors,
};
use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

use crate::blob::ladder::{decode_dense, decode_sparse, BlobPayload};
use crate::qdrant::membership::blob_payload_to_qdrant;
use crate::qdrant::write_client::QdrantWriteClient;

/// Minimum page size (DR GP-7 batch floor -- keeps upsert SLA intact).
pub const PAGE_SIZE_MIN: u64 = 1000;

/// Maximum page size (arch §4.5 memory bound -- 10000 blobs x ~3KB/blob is
/// tens of MB, two orders below the 1.9 GB single-query spike).
pub const PAGE_SIZE_MAX: u64 = 10_000;

/// One upserted point as seen by the sink. Carries the decoded vectors and the
/// full three-field payload so tests can assert the complete spec without a
/// live Qdrant.
#[derive(Debug, Clone)]
pub struct RebuildPoint {
    pub point_id: String,
    /// Decoded dense vector (f32 slice -- same as what went into Qdrant).
    pub dense: Vec<f32>,
    /// Decoded sparse vector (term -> weight map).
    pub sparse: HashMap<u32, f32>,
    /// Full three-field payload (tenant_id, branch_id[], collection_id).
    pub payload: BlobPayload,
}

/// Sink for rebuild upserts. The real implementation calls
/// `QdrantWriteClient::upsert_points`; the capture sink records points for
/// offline assertion (mirrors `blob::ladder::{QdrantSink, CaptureSink}`).
pub trait RebuildSink: Send {
    /// Flush one page of blob points. Called once per keyset page.
    fn flush_page(&mut self, points: Vec<RebuildPoint>);
}

/// A capturing sink for unit tests: accumulates all flushed points in
/// insertion order so tests can assert the complete upsert set offline.
#[derive(Debug, Default)]
pub struct CaptureRebuildSink {
    pub points: Vec<RebuildPoint>,
    /// Maximum page length seen across all `flush_page` calls. Used by the
    /// bounded-memory test to assert no single page exceeds `PAGE_SIZE_MAX`.
    pub max_page_len: usize,
}

impl RebuildSink for CaptureRebuildSink {
    fn flush_page(&mut self, page: Vec<RebuildPoint>) {
        if page.len() > self.max_page_len {
            self.max_page_len = page.len();
        }
        self.points.extend(page);
    }
}

/// A live sink that upserts via `QdrantWriteClient` (used in production and in
/// live-Qdrant integration tests).
pub struct LiveRebuildSink<'a> {
    client: &'a QdrantWriteClient,
    collection_name: String,
    dense_name: &'static str,
    sparse_name: &'static str,
}

impl<'a> LiveRebuildSink<'a> {
    pub fn new(client: &'a QdrantWriteClient, collection_name: impl Into<String>) -> Self {
        Self {
            client,
            collection_name: collection_name.into(),
            dense_name: crate::qdrant::collection::DENSE_VECTOR_NAME,
            sparse_name: crate::qdrant::collection::SPARSE_VECTOR_NAME,
        }
    }
}

impl<'a> RebuildSink for LiveRebuildSink<'a> {
    fn flush_page(&mut self, page: Vec<RebuildPoint>) {
        // RebuildSink::flush_page is sync; we block on the async upsert.
        // This is acceptable in the rebuild context (single sequential
        // operation, not called from an async hot path).
        let rt = tokio::runtime::Handle::current();
        let points: Vec<PointStruct> = page
            .into_iter()
            .map(|p| {
                let qdrant_payload = blob_payload_to_qdrant(&p.payload);

                let mut named = HashMap::new();
                named.insert(
                    self.dense_name.to_string(),
                    DenseVector { data: p.dense }.into(),
                );
                named.insert(
                    self.sparse_name.to_string(),
                    SparseVector {
                        indices: {
                            let mut idx: Vec<u32> = p.sparse.keys().copied().collect();
                            idx.sort_unstable();
                            idx
                        },
                        values: {
                            let mut pairs: Vec<(u32, f32)> = p.sparse.into_iter().collect();
                            pairs.sort_unstable_by_key(|(k, _)| *k);
                            pairs.into_iter().map(|(_, v)| v).collect()
                        },
                    }
                    .into(),
                );

                PointStruct {
                    id: Some(PointId {
                        point_id_options: Some(point_id::PointIdOptions::Uuid(p.point_id)),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(vectors::VectorsOptions::Vectors(NamedVectors {
                            vectors: named,
                        })),
                    }),
                    payload: qdrant_payload,
                }
            })
            .collect();

        if points.is_empty() {
            return;
        }

        let req = UpsertPointsBuilder::new(&self.collection_name, points).wait(true);
        rt.block_on(async {
            self.client
                .upsert_points(req)
                .await
                .expect("rebuild upsert failed");
        });
    }
}

// ---------------------------------------------------------------------------
// Core rebuild function
// ---------------------------------------------------------------------------

/// Rebuild Qdrant for one tenant from durable SQLite vectors (arch §4.5).
///
/// ## Contract
///
/// - The COLLECTION must already exist with the arch §5.3 spec and both
///   payload indexes in place. Call `ensure_collection` + two
///   `create_payload_index` calls from `qdrant::collection` BEFORE calling
///   this function. (The caller is responsible for the optional prior
///   `delete_collection` if a clean rebuild is desired.)
/// - `page_size` is clamped to `[PAGE_SIZE_MIN, PAGE_SIZE_MAX]` by this
///   function -- the caller may pass any value in that range.
/// - Vectors are read VERBATIM from `blobs.dense_vec` / `blobs.sparse_vec`.
///   No embedding calls are made (AC-F11.3).
/// - `blobs.point_id` is used VERBATIM (AC-F5.5 / DATA-05: salted re-keys are
///   honored; never recomputed).
/// - Branch membership is derived from `json_group_array(DISTINCT br.branch_id)`
///   in the GROUP BY cursor (arch §4.5 query). This is the SET-PER-BLOB form
///   (one scan, all referrers), distinct from `compute_membership` (per-blob
///   DISTINCT query); both are correct for their context (FP-2 does not apply
///   here -- FP-2 forbids RE-IMPLEMENTING the SELECT DISTINCT, not using the
///   equivalent GROUP BY aggregate in a different query shape).
///
/// ## Errors
///
/// Returns `StorageError::Sqlite` on cursor failures. Sink errors surface as
/// panics from `LiveRebuildSink` (rebuild is a maintenance operation, not a
/// hot path; the daemon's caller wraps this in a gRPC error response).
pub async fn rebuild_qdrant(
    pool: &SqlitePool,
    sink: &mut dyn RebuildSink,
    tenant_id: &str,
    collection_id: &str,
    page_size: u64,
) -> Result<u64, StorageError> {
    // Clamp page_size to the documented [PAGE_SIZE_MIN, PAGE_SIZE_MAX] window.
    let page = page_size.clamp(PAGE_SIZE_MIN, PAGE_SIZE_MAX);

    let mut cursor: i64 = 0; // keyset cursor: last seen blob_id (exclusive lower bound)
    let mut total_upserted: u64 = 0;

    loop {
        // Arch §4.5 keyset cursor query. GROUP BY collapses all blob_refs rows
        // for a blob into a single json_group_array; ORDER BY blob_id ensures a
        // stable keyset that avoids O(N^2) OFFSET scans (PERF-R5-N3).
        let rows = sqlx::query(
            "SELECT b.blob_id, b.point_id, b.dense_vec, b.sparse_vec, \
             json_group_array(DISTINCT br.branch_id) AS branch_ids \
             FROM blobs b \
             JOIN blob_refs br ON br.blob_id = b.blob_id \
             WHERE b.tenant_id = ? AND b.blob_id > ? \
             GROUP BY b.blob_id \
             ORDER BY b.blob_id \
             LIMIT ?",
        )
        .bind(tenant_id)
        .bind(cursor)
        .bind(page as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("rebuild cursor: {e}")))?;

        if rows.is_empty() {
            break;
        }

        // Advance the keyset cursor to the last blob_id in this page.
        let last_blob_id: i64 = rows.last().unwrap().get("blob_id");

        let page_points: Vec<RebuildPoint> = rows
            .into_iter()
            .map(|row| {
                let blob_id: i64 = row.get("blob_id");
                let point_id_str: String = row.get("point_id");
                let dense_bytes: Vec<u8> = row.get("dense_vec");
                let sparse_bytes: Vec<u8> = row.get("sparse_vec");
                let branch_ids_json: String = row.get("branch_ids");

                let dense = decode_dense(&dense_bytes);
                let sparse = decode_sparse(&sparse_bytes);
                let branch_ids = parse_branch_ids_json(&branch_ids_json, blob_id);

                RebuildPoint {
                    point_id: point_id_str,
                    dense,
                    sparse,
                    payload: BlobPayload {
                        tenant_id: tenant_id.to_string(),
                        branch_id: branch_ids,
                        collection_id: collection_id.to_string(),
                    },
                }
            })
            .collect();

        let page_len = page_points.len() as u64;
        sink.flush_page(page_points);
        total_upserted += page_len;
        cursor = last_blob_id;
    }

    Ok(total_upserted)
}

// ---------------------------------------------------------------------------
// JSON parsing helper
// ---------------------------------------------------------------------------

/// Parse the `json_group_array(DISTINCT branch_id)` output into a `Vec<String>`.
///
/// The SQLite JSON1 extension produces a well-formed JSON array of strings,
/// e.g. `'["hash-a","hash-b"]'`. We parse it without pulling in a full JSON
/// crate: split on `","`, strip the outer `[` / `]`, and unescape the minimal
/// subset that SQLite JSON1 produces (only `\"` and `\\`).
///
/// This is intentionally simple: branch_ids are content-key hashes (hex
/// characters and `-`/`_`) so no exotic escaping appears in practice. The
/// fallback for unexpected input is an empty vec (logged with blob_id context).
fn parse_branch_ids_json(json: &str, blob_id: i64) -> Vec<String> {
    let trimmed = json.trim();

    // Handle the trivially empty case.
    if trimmed == "[]" || trimmed.is_empty() {
        return Vec::new();
    }

    // Strip outer `[` and `]`.
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        tracing::warn!(
            blob_id,
            "parse_branch_ids_json: unexpected format {:?}, returning empty",
            json
        );
        return Vec::new();
    }

    let inner = &trimmed[1..trimmed.len() - 1];

    // Split on `","` (the inter-element separator in JSON arrays of strings).
    // Each element is a JSON string: `"value"` or (for the first/last) just the
    // content with surrounding `"` stripped.
    inner
        .split("\",\"")
        .filter_map(|s| {
            // Strip any residual leading/trailing `"`.
            let s = s.trim_matches('"');
            if s.is_empty() {
                None
            } else {
                // Unescape `\"` -> `"` and `\\` -> `\` (SQLite JSON1 escaping).
                Some(s.replace("\\\"", "\"").replace("\\\\", "\\"))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "recover_tests.rs"]
mod tests;
