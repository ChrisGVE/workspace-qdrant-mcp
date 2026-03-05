//! Text/content processing strategy.
//!
//! Handles `ItemType::Text` queue items, routing to collection-specific
//! processing for rules, scratchpad items, or generic content.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::storage::DocumentPoint;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    ContentPayload, ItemType, MemoryPayload, QueueOperation, ScratchpadPayload, UnifiedQueueItem,
};

/// Strategy for processing text/content queue items.
///
/// Routes to rules, scratchpad, or generic content processing based on
/// the target collection.
pub struct TextStrategy;

#[async_trait]
impl ProcessingStrategy for TextStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Text
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_content_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "text"
    }
}

impl TextStrategy {
    /// Route content item to collection-specific or generic processing.
    ///
    /// Ensures the target collection exists, then dispatches to the appropriate
    /// handler based on the collection name.
    pub(crate) async fn process_content_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing content item: {} -> collection: {}",
            item.queue_id, item.collection
        );

        crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        // Route to collection-specific or generic content processing
        if item.collection == wqm_common::constants::COLLECTION_RULES {
            Self::process_rules_item(ctx, item).await
        } else if item.collection == wqm_common::constants::COLLECTION_SCRATCHPAD {
            Self::process_scratchpad_item(ctx, item).await
        } else {
            Self::process_generic_content_item(ctx, item).await
        }
    }

    /// Process a rule item -- preserves all rule-specific metadata.
    async fn process_rules_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let payload: MemoryPayload = parse_payload(item)?;
        let action = payload.action.as_deref().unwrap_or("add");

        if action == "remove" {
            return Self::handle_rules_remove(ctx, &payload).await;
        }

        let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
            &ctx.embedding_generator,
            &ctx.embedding_semaphore,
            &payload.content,
            "bge-small-en-v1.5",
        )
        .await?;

        if action == "update" {
            Self::delete_rule_by_label(ctx, &payload.label).await?;
        }

        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);
        let point_payload = build_rules_payload(item, &payload, &content_doc_id, action);

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embed_result.dense_vector,
            sparse_vector: embed_result.sparse_vector,
            payload: point_payload,
        };

        ctx.storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed rules item {} (action={}, label={:?}) -> {}",
            item.queue_id, action, payload.label, item.collection
        );

        Ok(())
    }

    /// Handle the `remove` action: delete the point from Qdrant and clean `rules_mirror`.
    async fn handle_rules_remove(
        ctx: &ProcessingContext,
        payload: &MemoryPayload,
    ) -> UnifiedProcessorResult<()> {
        if let Some(label) = &payload.label {
            info!("Removing rule with label: {}", label);
            ctx.storage_client
                .delete_points_by_payload_field(
                    wqm_common::constants::COLLECTION_RULES,
                    "label",
                    label,
                )
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

            // Best-effort: remove from rules_mirror (Qdrant delete already succeeded)
            let pool = ctx.queue_manager.pool();
            if let Err(e) = sqlx::query("DELETE FROM rules_mirror WHERE rule_id = ?1")
                .bind(label)
                .execute(pool)
                .await
            {
                warn!(
                    "Failed to delete rules_mirror row for label={}: {}",
                    label, e
                );
            }
        }
        Ok(())
    }

    /// Delete an existing rule point by label (used for `update` action).
    async fn delete_rule_by_label(
        ctx: &ProcessingContext,
        label: &Option<String>,
    ) -> UnifiedProcessorResult<()> {
        if let Some(label) = label {
            info!("Updating rule with label: {} (delete + re-insert)", label);
            let _ = ctx
                .storage_client
                .delete_points_by_payload_field(
                    wqm_common::constants::COLLECTION_RULES,
                    "label",
                    label,
                )
                .await;
        }
        Ok(())
    }

    /// Process a scratchpad item -- persistent LLM scratch space.
    async fn process_scratchpad_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let payload: ScratchpadPayload = parse_payload(item)?;

        let now = wqm_common::timestamps::now_utc();

        // Generate document ID from content hash (for idempotent updates)
        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Generate embedding (semaphore-gated)
        let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
            &ctx.embedding_generator,
            &ctx.embedding_semaphore,
            &payload.content,
            "all-MiniLM-L6-v2",
        )
        .await?;

        // Build Qdrant payload
        let mut point_payload = HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("source_type".to_string(), serde_json::json!("scratchpad"));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("created_at".to_string(), serde_json::json!(&now));
        point_payload.insert("updated_at".to_string(), serde_json::json!(&now));

        if let Some(ref title) = payload.title {
            point_payload.insert("title".to_string(), serde_json::json!(title));
        }
        if !payload.tags.is_empty() {
            point_payload.insert("tags".to_string(), serde_json::json!(payload.tags));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embed_result.dense_vector,
            sparse_vector: embed_result.sparse_vector,
            payload: point_payload,
        };

        ctx.storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed scratchpad item {} (tenant={}, title={:?}) -> {}",
            item.queue_id, item.tenant_id, payload.title, item.collection
        );

        Ok(())
    }

    /// Process a generic content item (non-rules, non-scratchpad).
    async fn process_generic_content_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let payload: ContentPayload = parse_payload(item)?;

        // Generate embedding (semaphore-gated, Task 504)
        let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
            &ctx.embedding_generator,
            &ctx.embedding_semaphore,
            &payload.content,
            "bge-small-en-v1.5",
        )
        .await?;

        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Build payload with metadata
        let mut point_payload = HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert(
            "source_type".to_string(),
            serde_json::json!(payload.source_type.to_lowercase()),
        );

        if let Some(main_tag) = &payload.main_tag {
            point_payload.insert("main_tag".to_string(), serde_json::json!(main_tag));
        }
        if let Some(full_tag) = &payload.full_tag {
            point_payload.insert("full_tag".to_string(), serde_json::json!(full_tag));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embed_result.dense_vector,
            sparse_vector: embed_result.sparse_vector,
            payload: point_payload,
        };

        ctx.storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed content item {} -> {}",
            item.queue_id, item.collection
        );

        Ok(())
    }
}

/// Build the Qdrant payload map for a rules item.
fn build_rules_payload(
    item: &UnifiedQueueItem,
    payload: &MemoryPayload,
    content_doc_id: &str,
    action: &str,
) -> HashMap<String, serde_json::Value> {
    let now = wqm_common::timestamps::now_utc();
    let mut point_payload = HashMap::new();
    point_payload.insert("content".to_string(), serde_json::json!(payload.content));
    point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
    point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
    point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
    point_payload.insert("item_type".to_string(), serde_json::json!("content"));
    point_payload.insert(
        "source_type".to_string(),
        serde_json::json!(payload.source_type.to_lowercase()),
    );
    if let Some(label) = &payload.label {
        point_payload.insert("label".to_string(), serde_json::json!(label));
    }
    if let Some(scope) = &payload.scope {
        point_payload.insert("scope".to_string(), serde_json::json!(scope));
    }
    if let Some(project_id) = &payload.project_id {
        point_payload.insert("project_id".to_string(), serde_json::json!(project_id));
    }
    if let Some(title) = &payload.title {
        point_payload.insert("title".to_string(), serde_json::json!(title));
    }
    if let Some(tags) = &payload.tags {
        // Store as comma-separated string for Qdrant keyword matching
        point_payload.insert("tags".to_string(), serde_json::json!(tags.join(",")));
    }
    if let Some(priority) = payload.priority {
        point_payload.insert("priority".to_string(), serde_json::json!(priority));
    }
    if action == "add" {
        point_payload.insert("created_at".to_string(), serde_json::json!(&now));
    }
    point_payload.insert("updated_at".to_string(), serde_json::json!(&now));
    point_payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_strategy_handles_text_items() {
        let strategy = TextStrategy;
        assert!(strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Text, &QueueOperation::Update));
        assert!(strategy.handles(&ItemType::Text, &QueueOperation::Delete));
    }

    #[test]
    fn test_text_strategy_rejects_non_text_items() {
        let strategy = TextStrategy;
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Folder, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Url, &QueueOperation::Add));
    }

    #[test]
    fn test_text_strategy_name() {
        let strategy = TextStrategy;
        assert_eq!(strategy.name(), "text");
    }
}
