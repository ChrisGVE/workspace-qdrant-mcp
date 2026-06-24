//! `QdrantWriteClient` — the mutating newtype over `qdrant_client::Qdrant`.
//!
//! Location: `wqm-storage-write/src/qdrant/write_client.rs`. Logical context:
//! the single home of Qdrant mutation. The six mutating methods below
//! (`upsert_points`, `delete_points`, `overwrite_payload`, `set_payload`,
//! `create_collection`, `create_field_index`) appear in this crate ONLY; the
//! read crate's `QdrantReadClient` exposes none of them. Guard 3 scans the
//! `mcp-server` / `wqm-cli` release binaries to confirm these symbols are
//! never reachable there. Signatures mirror the live qdrant-client 1.17
//! surface.
//!
//! Neighbors: `wqm_storage::qdrant::QdrantReadClient` (read sibling),
//! [`crate::registry::WriteHandleRegistry`] (serializes writes per branch).

use qdrant_client::qdrant::{
    CollectionOperationResponse, CreateCollection, CreateFieldIndexCollection, DeletePoints,
    PointsOperationResponse, SetPayloadPoints, UpsertPoints,
};
use qdrant_client::{Qdrant, QdrantError};

/// Mutating wrapper around a Qdrant client. Construct with
/// [`QdrantWriteClient::new`]; the inner client is private.
pub struct QdrantWriteClient {
    inner: Qdrant,
}

impl QdrantWriteClient {
    /// Wrap an existing Qdrant client for mutation.
    pub fn new(inner: Qdrant) -> Self {
        Self { inner }
    }

    /// Insert or overwrite points (`UpsertPoints`).
    pub async fn upsert_points(
        &self,
        request: impl Into<UpsertPoints>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        self.inner.upsert_points(request).await
    }

    /// Delete points by id or filter (`DeletePoints`).
    pub async fn delete_points(
        &self,
        request: impl Into<DeletePoints>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        self.inner.delete_points(request).await
    }

    /// Replace the entire payload of matched points (`SetPayloadPoints`).
    pub async fn overwrite_payload(
        &self,
        request: impl Into<SetPayloadPoints>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        self.inner.overwrite_payload(request).await
    }

    /// Merge keys into the payload of matched points (`SetPayloadPoints`).
    pub async fn set_payload(
        &self,
        request: impl Into<SetPayloadPoints>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        self.inner.set_payload(request).await
    }

    /// Create a collection (`CreateCollection`).
    pub async fn create_collection(
        &self,
        request: impl Into<CreateCollection>,
    ) -> Result<CollectionOperationResponse, QdrantError> {
        self.inner.create_collection(request).await
    }

    /// Create a keyword payload index on a collection field
    /// (`CreateFieldIndexCollection`). Used by rebuild to create `branch_id`
    /// and `tenant_id` indexes before the first upsert (arch §5.3 / AC-F11.1).
    pub async fn create_field_index(
        &self,
        request: impl Into<CreateFieldIndexCollection>,
    ) -> Result<PointsOperationResponse, QdrantError> {
        self.inner.create_field_index(request).await
    }
}
