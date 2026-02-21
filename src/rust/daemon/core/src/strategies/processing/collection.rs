//! Collection processing strategy.
//!
//! Handles `ItemType::Collection` queue items: collection-level uplift
//! (cascade to all tenants) and collection reset (delete all data).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ItemType, QueueOperation, UnifiedQueueItem};

/// Strategy for processing collection-level queue items.
///
/// Handles uplift (cascade to all tenants) and reset (delete all data,
/// preserve schema/watch_folders) operations.
pub struct CollectionStrategy;

#[async_trait]
impl ProcessingStrategy for CollectionStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Collection
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_collection_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "collection"
    }
}

impl CollectionStrategy {
    /// Process collection item -- Uplift or Reset operations.
    ///
    /// - Uplift: cascade to all tenants in the collection via (Tenant, Uplift)
    /// - Reset: delete all Qdrant points + SQLite data for all tenants, preserve schema
    pub(crate) async fn process_collection_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing collection item: {} (op={:?}, collection={})",
            item.queue_id, item.op, item.collection
        );

        match item.op {
            QueueOperation::Uplift => {
                Self::handle_uplift(ctx, item).await?;
            }
            QueueOperation::Reset => {
                Self::handle_reset(ctx, item).await?;
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for collection item {}",
                    item.op, item.queue_id
                );
            }
        }

        Ok(())
    }

    /// Cascade uplift to all tenants in the collection.
    async fn handle_uplift(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let tenants: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1",
        )
        .bind(&item.collection)
        .fetch_all(&ctx.pool)
        .await
        .map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to query tenants for collection uplift: {}",
                e
            ))
        })?;

        let mut enqueued = 0u32;
        for (tenant_id,) in &tenants {
            let payload = serde_json::json!({
                "project_root": "",
            })
            .to_string();

            if let Ok((_, true)) = ctx
                .queue_manager
                .enqueue_unified(
                    ItemType::Tenant,
                    QueueOperation::Uplift,
                    tenant_id,
                    &item.collection,
                    &payload,
                    None,
                    None,
                )
                .await
            {
                enqueued += 1;
            }
        }
        info!(
            "Collection uplift: enqueued {}/{} tenant uplift items for collection={}",
            enqueued,
            tenants.len(),
            item.collection
        );

        Ok(())
    }

    /// Delete all Qdrant points + SQLite data for all tenants, preserve watch_folders.
    async fn handle_reset(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        // 1. Delete all Qdrant points in the collection
        if ctx
            .storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            let tenants: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1",
            )
            .bind(&item.collection)
            .fetch_all(&ctx.pool)
            .await
            .map_err(|e| {
                UnifiedProcessorError::ProcessingFailed(format!(
                    "Failed to query tenants for reset: {}",
                    e
                ))
            })?;

            for (tenant_id,) in &tenants {
                ctx.storage_client
                    .delete_points_by_tenant(&item.collection, tenant_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
            info!(
                "Reset: deleted Qdrant points for {} tenants in collection={}",
                tenants.len(),
                item.collection
            );
        }

        // 2. Delete SQLite data in a transaction (preserve watch_folders)
        Self::reset_sqlite_data(&ctx.pool, &item.collection).await?;

        Ok(())
    }

    /// Delete tracked_files and qdrant_chunks for all tenants in the collection.
    async fn reset_sqlite_data(
        pool: &SqlitePool,
        collection: &str,
    ) -> UnifiedProcessorResult<()> {
        let mut tx = pool.begin().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to begin reset transaction: {}",
                e
            ))
        })?;

        let tenant_ids: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1",
        )
        .bind(collection)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to query tenants: {}",
                e
            ))
        })?;

        for (tenant_id,) in &tenant_ids {
            let _ = sqlx::query(
                r#"DELETE FROM qdrant_chunks WHERE file_id IN (
                    SELECT file_id FROM tracked_files WHERE tenant_id = ?1
                )"#,
            )
            .bind(tenant_id)
            .execute(&mut *tx)
            .await;

            let _ = sqlx::query("DELETE FROM tracked_files WHERE tenant_id = ?1")
                .bind(tenant_id)
                .execute(&mut *tx)
                .await;
        }

        tx.commit().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to commit reset transaction: {}",
                e
            ))
        })?;

        info!(
            "Reset: cleared SQLite data for {} tenants in collection={} (watch_folders preserved)",
            tenant_ids.len(),
            collection
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_strategy_handles_collection_items() {
        let strategy = CollectionStrategy;
        assert!(strategy.handles(&ItemType::Collection, &QueueOperation::Uplift));
        assert!(strategy.handles(&ItemType::Collection, &QueueOperation::Reset));
    }

    #[test]
    fn test_collection_strategy_rejects_non_collection_items() {
        let strategy = CollectionStrategy;
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
    }

    #[test]
    fn test_collection_strategy_name() {
        let strategy = CollectionStrategy;
        assert_eq!(strategy.name(), "collection");
    }
}
