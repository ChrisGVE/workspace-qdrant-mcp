//! `QdrantReadClient` — a read-only newtype over `qdrant_client::Qdrant`.
//!
//! Location: `wqm-storage/src/qdrant/read_client.rs`. Logical context: SEC-01 /
//! D-PRD-2. The read crate never holds a raw `Qdrant`; it holds this newtype,
//! which re-exports ONLY the four read methods (`search_points`, `query`,
//! `scroll`, `retrieve`). The inner client is private — no `pub` field, no
//! `Deref` to the raw client — so a read-crate caller has no path to
//! `upsert_points`, `delete_points`, `overwrite_payload`, `set_payload`, or
//! `create_collection`. Those mutating calls live exclusively in
//! `wqm-storage-write/qdrant/`. Method signatures mirror the live qdrant-client
//! 1.17 surface used by daemon-core (`storage/scroll.rs`, `storage/search.rs`).
//!
//! Neighbors: `wqm-storage-write::qdrant` (write sibling), `wqm_common::StorageError`.

use qdrant_client::qdrant::{
    GetPoints, GetResponse, QueryPoints, QueryResponse, ScrollPoints, ScrollResponse, SearchPoints,
    SearchResponse,
};
use qdrant_client::{Qdrant, QdrantError};

/// Read-only wrapper around a Qdrant client.
///
/// Construct with [`QdrantReadClient::new`] from an already-built `Qdrant`
/// handle. Only the read methods below are reachable; the wrapped client is
/// private and never exposed (no field access, no `Deref`).
///
/// `Clone` is derived because `Qdrant` is internally `Arc`-based (cheap to
/// clone); needed by the F17 fan-out tasks which capture a client per task.
#[derive(Clone)]
pub struct QdrantReadClient {
    inner: Qdrant,
}

impl QdrantReadClient {
    /// Wrap an existing Qdrant client, surfacing only its read methods.
    pub fn new(inner: Qdrant) -> Self {
        Self { inner }
    }

    /// Vector similarity search (`SearchPoints`).
    pub async fn search_points(
        &self,
        request: impl Into<SearchPoints>,
    ) -> Result<SearchResponse, QdrantError> {
        self.inner.search_points(request).await
    }

    /// Universal query API (`QueryPoints`) — dense, sparse, or fusion queries.
    pub async fn query(
        &self,
        request: impl Into<QueryPoints>,
    ) -> Result<QueryResponse, QdrantError> {
        self.inner.query(request).await
    }

    /// Paginated point retrieval with filtering (`ScrollPoints`).
    pub async fn scroll(
        &self,
        request: impl Into<ScrollPoints>,
    ) -> Result<ScrollResponse, QdrantError> {
        self.inner.scroll(request).await
    }

    /// Fetch specific points by id (`GetPoints`) — the `get_points` read method.
    pub async fn retrieve(
        &self,
        request: impl Into<GetPoints>,
    ) -> Result<GetResponse, QdrantError> {
        self.inner.get_points(request).await
    }
}
