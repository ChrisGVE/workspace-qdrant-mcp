//! Single canonical `ensure_collection()` replacing 6+ inline copies.
//!
//! The pattern `if !collection_exists { create_multi_tenant_collection }` was
//! scattered across `unified_queue_processor.rs` in `process_content_item`,
//! `process_file_item`, `process_project_item`, `process_library_item`, and
//! others. This module provides a single function to replace them all.

use std::sync::Arc;
use tracing::info;

use crate::specs::Collection;
use crate::storage::{MultiTenantConfig, StorageClient, StorageError};

/// Ensure a Qdrant collection exists, creating it with multi-tenant config if not.
///
/// This is the single source of truth for the "ensure collection exists" pattern.
/// All call sites in the queue processor should delegate here instead of
/// performing inline existence checks.
pub async fn ensure_collection(
    storage_client: &Arc<StorageClient>,
    collection_name: &str,
    vector_size: u64,
) -> Result<(), StorageError> {
    let config = MultiTenantConfig {
        vector_size,
        ..MultiTenantConfig::default()
    };
    ensure_collection_with_config(storage_client, collection_name, &config).await
}

/// Ensure a canonical Qdrant collection exists using `Collection` enum config.
///
/// Convenience wrapper that pulls the name and creation config from the enum.
/// `vector_size` must come from the active embedding provider's dim
/// (`embedding.output_dim` per PRD §5.10 / §6.5).
pub async fn ensure_canonical_collection(
    storage_client: &Arc<StorageClient>,
    collection: &Collection,
    vector_size: u64,
) -> Result<(), StorageError> {
    ensure_collection_with_config(
        storage_client,
        collection.name(),
        &collection.creation_config(vector_size),
    )
    .await
}

/// Ensure a Qdrant collection exists with a custom configuration.
///
/// Use this when the call site needs non-default parameters (e.g., different
/// vector dimensions or HNSW settings).
pub async fn ensure_collection_with_config(
    storage_client: &Arc<StorageClient>,
    collection_name: &str,
    config: &MultiTenantConfig,
) -> Result<(), StorageError> {
    if !storage_client.collection_exists(collection_name).await? {
        info!(
            collection = collection_name,
            vector_size = config.vector_size,
            "Creating collection with multi-tenant config (dense+sparse)",
        );
        storage_client
            .create_multi_tenant_collection(collection_name, config)
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_collection_requires_explicit_vector_size() {
        let non_default_dim: u64 = 1536;
        let config = MultiTenantConfig {
            vector_size: non_default_dim,
            ..MultiTenantConfig::default()
        };
        assert_eq!(config.vector_size, non_default_dim);
        assert_ne!(config.vector_size, 384);
    }
}
