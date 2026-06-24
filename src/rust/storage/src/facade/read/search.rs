//! Hybrid Qdrant search + SQLite enrichment (AC-F10.1, AC-F10.2, §4.4).
//!
//! File: `wqm-storage/src/facade/read/search.rs`
//! Location: `src/rust/storage/src/facade/read/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch §4.4, §5.3, §6.2).
//!
//!   The search path (arch §4.4):
//!     1. Deserialize dense + sparse vectors from `blobs` in `store.db`.
//!     2. Issue two Qdrant queries (dense + sparse) via `QdrantReadClient`,
//!        each pre-filtered with `branch_id ANY [branch_id] AND tenant_id = ?`
//!        (both payload indexes mandatory — arch §5.3, AC-F10.1).
//!     3. RRF-fuse the two ranked lists (from `wqm_common::search::rrf`).
//!     4. Enrich the top-K point_ids via the `idx_blob_refs_covering` JOIN
//!        (arch §4.4, AC-F10.1).
//!
//!   AC-F10.2 (SEC-3): `tenant_id = ?` appears on EVERY Qdrant query. When the
//!   caller supplies no resolved tenant the function returns an error immediately
//!   — never an all-tenant fall-through.
//!
//!   DR GP-9 composition note: this module composes `QdrantReadClient` (the
//!   storage-crate newtype) for Qdrant I/O. The `wqm-client` search pipeline
//!   (`run_search_pipeline`) targets the daemon gRPC path and is a different
//!   composition point (used by the MCP server proper); this module implements
//!   the direct facade-level branch-filtered search that does NOT require a
//!   running daemon.
//!
//! Neighbors: `crate::qdrant::QdrantReadClient` (read-only Qdrant newtype),
//!   `wqm_common::search::rrf` (RRF fusion), `crate::types::results`.

use std::collections::HashMap;

use qdrant_client::qdrant::{Condition, Filter, QueryPointsBuilder, VectorInput};
use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::qdrant::QdrantReadClient;

// The collection that holds project blob points (arch §5.3).
const PROJECTS_COLLECTION: &str = "projects";
// Named vector keys must match the collection spec (arch §5.3).
const DENSE_VECTOR_NAME: &str = "dense";
const SPARSE_VECTOR_NAME: &str = "sparse";
// RRF constant k (standard 60, same as wqm-client pipeline).
const RRF_K: f64 = 60.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An enriched search hit returned by `branch_search`.
#[derive(Debug, Clone)]
pub struct EnrichedHit {
    /// The Qdrant point_id (UUID string).
    pub point_id: String,
    /// RRF-fused relevance score.
    pub score: f32,
    /// `blob_id` from `store.db`.
    pub blob_id: i64,
    /// Branch-relative path of the referencing file.
    pub path: String,
    /// Symbol name when the blob is a symbol chunk.
    pub symbol_name: Option<String>,
    /// Start source line (1-based), when known.
    pub start_line: Option<i64>,
    /// End source line (1-based), when known.
    pub end_line: Option<i64>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Hybrid branch-scoped search: dense + sparse Qdrant query, RRF fusion,
/// SQLite enrichment.
///
/// `tenant_id` and `branch_id` are MANDATORY (AC-F10.2). Returns
/// `StorageError::Validation` when either is empty -- never falls through to
/// an all-tenant query.
///
/// `top_k` caps the number of Qdrant results per leg (returned after fusion).
pub async fn branch_search(
    qdrant: &QdrantReadClient,
    store_pool: &SqlitePool,
    tenant_id: &str,
    branch_id: &str,
    dense_vec: Vec<f32>,
    sparse_indices: Vec<u32>,
    sparse_values: Vec<f32>,
    top_k: u64,
) -> Result<Vec<EnrichedHit>, StorageError> {
    // AC-F10.2 / SEC-3: both identifiers must be present.
    if tenant_id.is_empty() {
        return Err(StorageError::Validation(
            "tenant_id is required for search -- no all-tenant fall-through (SEC-3)".into(),
        ));
    }
    if branch_id.is_empty() {
        return Err(StorageError::Validation(
            "branch_id is required for search".into(),
        ));
    }

    // Build the Qdrant pre-filter: branch_id ANY [branch_id] AND tenant_id = ?
    // Both payload indexes are mandatory (arch §5.3, AC-F10.1).
    let filter = build_branch_tenant_filter(tenant_id, branch_id);

    // Issue dense and sparse queries in parallel via the read-only newtype.
    let (dense_hits, sparse_hits) = tokio::try_join!(
        query_dense(qdrant, filter.clone(), dense_vec, top_k),
        query_sparse(qdrant, filter, sparse_indices, sparse_values, top_k),
    )?;

    // RRF fusion over the two ranked lists.
    let fused = rrf_fuse(dense_hits, sparse_hits);

    // Truncate to top_k before enrichment.
    let top: Vec<(String, f32)> = fused.into_iter().take(top_k as usize).collect();
    if top.is_empty() {
        return Ok(vec![]);
    }

    // Enrich from SQLite using the idx_blob_refs_covering JOIN (arch §4.4).
    enrich_from_sqlite(store_pool, branch_id, top).await
}

// ---------------------------------------------------------------------------
// Qdrant query helpers
// ---------------------------------------------------------------------------

/// Build a Qdrant filter for `branch_id ANY [branch_id] AND tenant_id = ?`.
fn build_branch_tenant_filter(tenant_id: &str, branch_id: &str) -> Filter {
    Filter {
        must: vec![
            // branch_id is a keyword array payload field; ANY matches one element.
            Condition::matches("branch_id", branch_id.to_string()),
            // tenant_id is a scalar keyword payload field.
            Condition::matches("tenant_id", tenant_id.to_string()),
        ],
        ..Default::default()
    }
}

/// Issue a dense named-vector query.
async fn query_dense(
    qdrant: &QdrantReadClient,
    filter: Filter,
    vector: Vec<f32>,
    limit: u64,
) -> Result<Vec<(String, f32)>, StorageError> {
    if vector.is_empty() {
        return Ok(vec![]);
    }

    let req = QueryPointsBuilder::new(PROJECTS_COLLECTION)
        .query(VectorInput::new_dense(vector))
        .using(DENSE_VECTOR_NAME)
        .filter(filter)
        .limit(limit)
        .with_payload(false)
        .build();

    let resp = qdrant
        .query(req)
        .await
        .map_err(|e| StorageError::Search(format!("dense Qdrant query failed: {e}")))?;

    let hits = resp
        .result
        .into_iter()
        .filter_map(|p| {
            let id = point_id_to_string(&p.id?)?;
            Some((id, p.score))
        })
        .collect();

    Ok(hits)
}

/// Issue a sparse named-vector query.
async fn query_sparse(
    qdrant: &QdrantReadClient,
    filter: Filter,
    indices: Vec<u32>,
    values: Vec<f32>,
    limit: u64,
) -> Result<Vec<(String, f32)>, StorageError> {
    if indices.is_empty() {
        return Ok(vec![]);
    }

    let sparse_input = VectorInput::new_sparse(indices, values);
    let req = QueryPointsBuilder::new(PROJECTS_COLLECTION)
        .query(sparse_input)
        .using(SPARSE_VECTOR_NAME)
        .filter(filter)
        .limit(limit)
        .with_payload(false)
        .build();

    let resp = qdrant
        .query(req)
        .await
        .map_err(|e| StorageError::Search(format!("sparse Qdrant query failed: {e}")))?;

    let hits = resp
        .result
        .into_iter()
        .filter_map(|p| {
            let id = point_id_to_string(&p.id?)?;
            Some((id, p.score))
        })
        .collect();

    Ok(hits)
}

/// Convert a Qdrant `PointId` to a string UUID.
fn point_id_to_string(id: &qdrant_client::qdrant::PointId) -> Option<String> {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    match &id.point_id_options {
        Some(PointIdOptions::Uuid(u)) => Some(u.clone()),
        Some(PointIdOptions::Num(n)) => Some(n.to_string()),
        None => None,
    }
}

// ---------------------------------------------------------------------------
// RRF fusion
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion of two ranked hit lists.
///
/// RRF score(d) = sum_over_lists( 1 / (k + rank(d)) )
/// where rank is 1-based. Standard k = 60 (same as wqm-client pipeline).
fn rrf_fuse(dense: Vec<(String, f32)>, sparse: Vec<(String, f32)>) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (id, _score)) in dense.into_iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
    }
    for (rank, (id, _score)) in sparse.into_iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
    }

    let mut fused: Vec<(String, f32)> = scores.into_iter().map(|(id, s)| (id, s as f32)).collect();

    // Stable descending sort (higher score first).
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

// ---------------------------------------------------------------------------
// SQLite enrichment (arch §4.4)
// ---------------------------------------------------------------------------

/// Enrich Qdrant point_ids with metadata from `store.db`.
///
/// Uses the `idx_blob_refs_covering` JOIN (arch §4.4):
///   `blobs JOIN blob_refs ON blob_refs.blob_id = blobs.blob_id
///    JOIN files ON files.file_id = blob_refs.file_id
///    WHERE blob_refs.branch_id = ? AND blobs.point_id IN (...)`
///
/// All externally-originated values (branch_id, point_ids) are bound
/// parameters (arch §6.5 GP-8).
async fn enrich_from_sqlite(
    pool: &SqlitePool,
    branch_id: &str,
    hits: Vec<(String, f32)>,
) -> Result<Vec<EnrichedHit>, StorageError> {
    // Build an ordered score map so we can restore ranking after the JOIN.
    let score_map: HashMap<String, f32> = hits.iter().cloned().collect();
    let point_ids: Vec<String> = hits.into_iter().map(|(id, _)| id).collect();

    if point_ids.is_empty() {
        return Ok(vec![]);
    }

    // SQLite does not support IN (?) with a Vec natively via sqlx; we build
    // a parameterised placeholder string. This is safe because we control the
    // placeholder count (not user input) — the actual values are bound.
    let placeholders: String = (0..point_ids.len())
        .map(|i| format!("?{}", i + 2)) // ?1 reserved for branch_id
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        r#"
        SELECT
            b.point_id,
            b.blob_id,
            f.relative_path  AS path,
            b.symbol_name,
            b.start_line,
            b.end_line
        FROM blobs b
        JOIN blob_refs br ON br.blob_id = b.blob_id AND br.branch_id = ?1
        JOIN files f      ON f.file_id  = br.file_id AND f.branch_id = ?1
        WHERE b.point_id IN ({placeholders})
        GROUP BY b.blob_id
        "#
    );

    let mut query = sqlx::query_as::<_, EnrichRow>(&sql).bind(branch_id);
    for pid in &point_ids {
        query = query.bind(pid);
    }

    let rows = query
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("SQLite enrichment failed: {e}")))?;

    // Reconstruct ordered results using the RRF scores.
    let mut enriched: Vec<EnrichedHit> = rows
        .into_iter()
        .filter_map(|row| {
            let score = *score_map.get(&row.point_id)?;
            Some(EnrichedHit {
                point_id: row.point_id,
                score,
                blob_id: row.blob_id,
                path: row.path,
                symbol_name: row.symbol_name,
                start_line: row.start_line,
                end_line: row.end_line,
            })
        })
        .collect();

    enriched.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(enriched)
}

#[derive(sqlx::FromRow)]
struct EnrichRow {
    point_id: String,
    blob_id: i64,
    path: String,
    symbol_name: Option<String>,
    start_line: Option<i64>,
    end_line: Option<i64>,
}

// ---------------------------------------------------------------------------
// Tests (AC-F10.1, AC-F10.2)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
