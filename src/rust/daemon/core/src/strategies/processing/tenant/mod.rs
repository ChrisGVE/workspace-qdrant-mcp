//! Tenant processing strategy.
//!
//! Handles `ItemType::Tenant` and `ItemType::Doc` queue items:
//! project registration, scanning, library management, tenant deletion,
//! document deletion, and tenant renaming.
//!
//! Submodules handle specific operation families:
//! - `project`: project add/scan/update/uplift/delete
//! - `library`: library add/scan/delete
//! - `cleanup`: post-scan exclusion file cleanup
//! - `delete`: surgical tenant deletion cascade, document deletion, rename

pub(crate) mod cleanup;
mod delete;
mod library;
mod project;
#[cfg(test)]
mod tests;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ItemType, QueueOperation, UnifiedQueueItem};

// Re-export submodule items used by sibling modules within the crate
pub(crate) use library::scan_library_directory;

/// Strategy for processing tenant and document queue items.
///
/// Routes `ItemType::Tenant` items by operation and collection, and handles
/// `ItemType::Doc` delete operations.
pub struct TenantStrategy;

#[async_trait]
impl ProcessingStrategy for TenantStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Tenant || *item_type == ItemType::Doc
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        match item.item_type {
            ItemType::Doc => Self::process_doc_item(ctx, item).await,
            _ => Self::process_tenant_item(ctx, item).await,
        }
    }

    fn name(&self) -> &'static str {
        "tenant"
    }
}

impl TenantStrategy {
    // =========================================================================
    // Tenant dispatch
    // =========================================================================

    /// Main tenant processing entry point.
    ///
    /// Routes by operation: Delete, Rename, or Add/Scan/Update (further
    /// sub-routed by collection).
    pub(crate) async fn process_tenant_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        match item.op {
            QueueOperation::Delete => delete::process_delete_tenant_item(ctx, item).await,
            QueueOperation::Rename => {
                delete::process_tenant_rename_item(item, &ctx.storage_client).await
            }
            _ => {
                // Add, Scan, Update -- route by collection
                match item.collection.as_str() {
                    "libraries" => library::process_library_item(ctx, item).await,
                    _ => project::process_project_item(ctx, item).await,
                }
            }
        }
    }

    /// Main Doc dispatch -- currently only Delete and Uplift (placeholder).
    pub(crate) async fn process_doc_item(
        _ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        match item.op {
            QueueOperation::Delete => {
                delete::process_delete_document_item(item, &_ctx.storage_client).await
            }
            QueueOperation::Uplift => {
                // Placeholder: no enrichment logic yet
                info!(
                    "Doc uplift placeholder for queue_id={} tenant={}",
                    item.queue_id, item.tenant_id
                );
                Ok(())
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for Doc item {}",
                    item.op, item.queue_id
                );
                Ok(())
            }
        }
    }
}
