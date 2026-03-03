//! CollectionService gRPC implementation
//!
//! Handles Qdrant collection lifecycle and alias management operations.
//! Provides 7 RPCs: CreateCollection, DeleteCollection, ListCollections, GetCollection,
//! CreateCollectionAlias, DeleteCollectionAlias, RenameCollectionAlias.

use std::sync::Arc;
use workspace_qdrant_core::storage::StorageClient;

mod handlers;
pub(super) mod validation;

#[cfg(test)]
mod tests;

/// CollectionService implementation with Qdrant integration
pub struct CollectionServiceImpl {
    pub(super) storage_client: Arc<StorageClient>,
}

impl CollectionServiceImpl {
    /// Create a new CollectionService with the provided storage client
    pub fn new(storage_client: Arc<StorageClient>) -> Self {
        Self { storage_client }
    }

    /// Create with default storage client
    pub fn default() -> Self {
        Self {
            storage_client: Arc::new(StorageClient::new()),
        }
    }
}
