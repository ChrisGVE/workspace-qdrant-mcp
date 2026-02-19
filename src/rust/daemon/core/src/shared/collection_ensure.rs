//! Single canonical `ensure_collection()` replacing 6+ inline copies.
//!
//! The pattern `if !collection_exists { create_multi_tenant_collection }` was
//! scattered across `unified_queue_processor.rs` in `process_content_item`,
//! `process_file_item`, `process_project_item`, `process_library_item`, and
//! others. This module provides a single function to replace them all.

use std::sync::Arc;
use tracing::info;

use crate::storage::{MultiTenantConfig, StorageClient, StorageError};

/// Ensure a Qdrant collection exists, creating it with multi-tenant config if not.
///
/// This is the single source of truth for the "ensure collection exists" pattern.
/// All call sites in the queue processor should delegate here instead of
/// performing inline existence checks.
pub async fn ensure_collection(
    storage_client: &Arc<StorageClient>,
    collection_name: &str,
) -> Result<(), StorageError> {
    ensure_collection_with_config(
        storage_client,
        collection_name,
        &MultiTenantConfig::default(),
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
            "Creating collection '{}' with multi-tenant config (dense+sparse)",
            collection_name
        );
        storage_client
            .create_multi_tenant_collection(collection_name, config)
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests for ensure_collection require a running Qdrant instance.
    // Unit-level verification is done by checking that the function signature
    // matches what all 6+ call sites expect (StorageClient + name + optional config).
    //
    // The refactoring correctness is verified by the existing integration tests
    // in `file_ingestion_pipeline_tests.rs` and `fts5_integration_tests.rs`.
}
