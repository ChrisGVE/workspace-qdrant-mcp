//! Idempotent Qdrant collection + payload-index creation (AC-F11.1).
//!
//! File: `wqm-storage-write/src/qdrant/collection.rs`
//! Location: `src/rust/storage-write/src/qdrant/` (write-crate qdrant layer)
//! Context: The collection specification from arch §5.3:
//!   - Dense named vector: "dense", 768-dim, Cosine distance.
//!   - Sparse named vector: "sparse", dot-product (declared under the top-level
//!     `sparse_vectors` map, NOT inside `vectors` -- the latter requires a
//!     mandatory `distance` field and is an invalid shape for sparse vectors).
//!   - Keyword payload indexes on `branch_id` and `tenant_id`, created BEFORE
//!     the first upsert (required for exact pre-filter; without them Qdrant
//!     degrades to O(N) post-filter, negating the "exact top-K" guarantee).
//!
//! Both `ensure_collection` and the two `create_payload_index` calls are
//! IDEMPOTENT: Qdrant silently ignores a `create_collection` when the
//! collection already exists, and `create_field_index` is a no-op on a field
//! that already carries the same index type. `rebuild_qdrant` (arch §4.5)
//! recreates them after every collection DROP+CREATE so the indexes are always
//! present after a rebuild.
//!
//! Neighbors: [`crate::qdrant::write_client::QdrantWriteClient`] (the PUT
//! transport), [`crate::qdrant::recover`] (the rebuild caller).

use std::collections::HashMap;

use qdrant_client::qdrant::{
    vectors_config, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance,
    FieldType, SparseVectorConfig, SparseVectorParams, VectorParamsBuilder, VectorParamsMap,
    VectorsConfig,
};
use wqm_common::error::StorageError;

use crate::qdrant::write_client::QdrantWriteClient;

/// Dense vector dimension for the branch-storage collection (arch §5.3).
/// Must match the embedding model output (768-dim for the current model).
pub const DENSE_DIM: u64 = 768;

/// The named dense vector key in Qdrant (arch §5.3).
pub const DENSE_VECTOR_NAME: &str = "dense";

/// The named sparse vector key in Qdrant (arch §5.3). Declared under
/// `sparse_vectors`, not inside `vectors`.
pub const SPARSE_VECTOR_NAME: &str = "sparse";

/// Build the `VectorsConfig` for the branch-storage collection (arch §5.3):
/// one named dense vector ("dense", 768-dim, Cosine). The sparse vector is
/// declared separately under `sparse_vectors_config` (see `build_sparse_config`).
///
/// Exported so tests can assert the exact spec without a live Qdrant.
pub fn build_vectors_config() -> VectorsConfig {
    let mut dense_map = HashMap::new();
    dense_map.insert(
        DENSE_VECTOR_NAME.to_string(),
        VectorParamsBuilder::new(DENSE_DIM, Distance::Cosine).build(),
    );
    VectorsConfig {
        config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
            map: dense_map,
        })),
    }
}

/// Build the `SparseVectorConfig` for the branch-storage collection (arch §5.3):
/// one sparse vector slot ("sparse", dot-product implicit). The empty
/// `SparseVectorParams {}` is correct: sparse similarity is always dot-product
/// for Qdrant sparse vectors; no `distance` field is used or accepted here.
///
/// Exported so tests can assert the exact spec without a live Qdrant.
pub fn build_sparse_config() -> SparseVectorConfig {
    let mut sparse_map = HashMap::new();
    sparse_map.insert(
        SPARSE_VECTOR_NAME.to_string(),
        SparseVectorParams {
            index: None,
            modifier: None,
        },
    );
    SparseVectorConfig { map: sparse_map }
}

/// Ensure a branch-storage collection exists with the exact arch §5.3 spec.
///
/// Idempotent: if the collection already exists Qdrant returns success and
/// this function succeeds without re-creating it. This is the CORRECT behavior
/// for both initial setup and for a rebuild that follows a `delete_collection`
/// (the caller deletes first, then calls this). On rebuild, the caller MUST
/// delete the old collection first so the indexes are re-created fresh; calling
/// this function alone after a delete satisfies that contract.
///
/// ## Collection spec (arch §5.3)
///
/// - Dense: named "dense", 768-dim, Cosine.
/// - Sparse: named "sparse", dot-product (under `sparse_vectors`, NOT inside
///   `vectors`).
/// - No explicit HNSW tuning (m=16, ef_construct=100 are Qdrant defaults;
///   performance tuning is deferred per arch §5.3 note).
pub async fn ensure_collection(
    client: &QdrantWriteClient,
    collection_name: &str,
) -> Result<(), StorageError> {
    let request = CreateCollectionBuilder::new(collection_name)
        .vectors_config(build_vectors_config())
        .sparse_vectors_config(build_sparse_config());

    client
        .create_collection(request)
        .await
        .map_err(StorageError::from)?;

    Ok(())
}

/// Create a keyword payload index on `field_name` in the given collection.
///
/// Idempotent: Qdrant accepts a repeated `create_field_index` call on an
/// already-indexed field with the same type as a no-op. On rebuild, the
/// collection is freshly created so both `branch_id` and `tenant_id` indexes
/// must be created before the first upsert (arch §5.3 MANDATORY note).
///
/// `branch_id` and `tenant_id` are the two MANDATORY fields (arch §5.3).
/// `collection_id` is OPTIONAL (future collection filter); this function is
/// called for both mandatory fields from `rebuild_qdrant`.
pub async fn create_payload_index(
    client: &QdrantWriteClient,
    collection_name: &str,
    field_name: &str,
) -> Result<(), StorageError> {
    let request =
        CreateFieldIndexCollectionBuilder::new(collection_name, field_name, FieldType::Keyword);

    client
        .create_field_index(request)
        .await
        .map_err(StorageError::from)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "collection_tests.rs"]
mod tests;
