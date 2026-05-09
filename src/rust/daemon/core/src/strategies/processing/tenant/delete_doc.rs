//! Document deletion and tenant rename operations.
//!
//! These operations act on individual documents or rename a tenant's
//! `tenant_id` payload field across Qdrant points, in contrast to the
//! full-cascade deletion handled by `delete.rs`.

use std::sync::Arc;

use tracing::info;

use crate::specs::parse_payload;
use crate::storage::StorageClient;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ProjectPayload, UnifiedQueueItem};

/// Process delete document item -- delete specific document by document_id.
pub(crate) async fn process_delete_document_item(
    item: &UnifiedQueueItem,
    storage_client: &Arc<StorageClient>,
) -> UnifiedProcessorResult<()> {
    info!("Processing delete document item: {}", item.queue_id);

    let payload = item.parse_delete_document_payload().map_err(|e| {
        UnifiedProcessorError::InvalidPayload(format!(
            "Failed to parse DeleteDocumentPayload: {}",
            e
        ))
    })?;

    if payload.document_id.trim().is_empty() {
        return Err(UnifiedProcessorError::InvalidPayload(
            "document_id must not be empty".to_string(),
        ));
    }

    if storage_client
        .collection_exists(&item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        storage_client
            .delete_points_by_document_id(&item.collection, &payload.document_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }

    info!(
        "Successfully deleted document {} from {} (tenant={})",
        payload.document_id, item.collection, item.tenant_id
    );
    Ok(())
}

/// Process tenant rename item -- update tenant_id on all matching Qdrant points.
///
/// Uses `ProjectPayload` with `old_tenant_id` field.
pub(crate) async fn process_tenant_rename_item(
    item: &UnifiedQueueItem,
    storage_client: &Arc<StorageClient>,
) -> UnifiedProcessorResult<()> {
    let payload: ProjectPayload = parse_payload(item)?;

    let old_tenant = payload.old_tenant_id.as_deref().ok_or_else(|| {
        UnifiedProcessorError::InvalidPayload(
            "Missing old_tenant_id in tenant rename payload".to_string(),
        )
    })?;
    let new_tenant = &item.tenant_id;

    let reason = item
        .metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .and_then(|v| v.get("reason").and_then(|r| r.as_str().map(String::from)))
        .unwrap_or_else(|| "unknown".to_string());

    info!(
        "Processing tenant rename: {} -> {} in collection '{}' (reason: {})",
        old_tenant, new_tenant, item.collection, reason
    );

    use qdrant_client::qdrant::{Condition, Filter};
    let filter = Filter::must([Condition::matches("tenant_id", old_tenant.to_string())]);

    let mut new_payload = std::collections::HashMap::new();
    new_payload.insert(
        "tenant_id".to_string(),
        serde_json::Value::String(new_tenant.to_string()),
    );

    storage_client
        .set_payload_by_filter(&item.collection, filter, new_payload)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!(
        "Successfully processed tenant rename {} -> {} in '{}'",
        old_tenant, new_tenant, item.collection
    );
    Ok(())
}
