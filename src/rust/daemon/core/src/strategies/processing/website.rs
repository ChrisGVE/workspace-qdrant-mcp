//! Website processing strategy.
//!
//! Handles `ItemType::Website` queue items: website crawling with link
//! extraction, page-level URL enqueuing, and tenant-scoped deletion.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    ItemType, QueueOperation, UnifiedQueueItem, WebsitePayload,
};

/// Strategy for processing website queue items.
///
/// Handles website crawl orchestration: validates URLs, enqueues page-level
/// (Url, Add) items for embedding, and extracts same-domain links.
pub struct WebsiteStrategy;

#[async_trait]
impl ProcessingStrategy for WebsiteStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Website
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_website_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "website"
    }
}

impl WebsiteStrategy {
    /// Process website crawl and ingestion item.
    pub(crate) async fn process_website_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let payload: WebsitePayload = serde_json::from_str(&item.payload_json).map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!(
                "Failed to parse WebsitePayload: {}",
                e
            ))
        })?;

        info!(
            "Processing website item: {} (op={:?}, url={})",
            item.queue_id, item.op, payload.url
        );

        match item.op {
            QueueOperation::Add => {
                Self::handle_add(ctx, item, &payload).await?;
            }
            QueueOperation::Scan => {
                Self::handle_scan(ctx, item, &payload).await?;
            }
            QueueOperation::Update => {
                Self::handle_update(ctx, item, &payload).await?;
            }
            QueueOperation::Delete => {
                Self::handle_delete(ctx, item).await?;
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for website item {}",
                    item.op, item.queue_id
                );
            }
        }

        Ok(())
    }

    /// Validate URL and enqueue (Website, Scan) for crawling.
    async fn handle_add(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &WebsitePayload,
    ) -> UnifiedProcessorResult<()> {
        let parsed = url::Url::parse(&payload.url).map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!(
                "Invalid URL {}: {}",
                payload.url, e
            ))
        })?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "Unsupported URL scheme: {}",
                parsed.scheme()
            )));
        }

        let scan_payload = serde_json::json!({
            "url": payload.url,
            "max_depth": payload.max_depth,
            "max_pages": payload.max_pages,
        })
        .to_string();

        match ctx
            .queue_manager
            .enqueue_unified(
                ItemType::Website,
                QueueOperation::Scan,
                &item.tenant_id,
                &item.collection,
                &scan_payload,
                0,
                None,
                None,
            )
            .await
        {
            Ok((queue_id, _)) => {
                info!(
                    "Enqueued website scan for url={} queue_id={}",
                    payload.url, queue_id
                );
            }
            Err(e) => {
                warn!("Failed to enqueue website scan: {}", e);
            }
        }

        Ok(())
    }

    /// Fetch page HTML, enqueue root page as (Url, Add), extract same-domain links.
    async fn handle_scan(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &WebsitePayload,
    ) -> UnifiedProcessorResult<()> {
        let response = reqwest::get(&payload.url).await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to fetch {}: {}",
                payload.url, e
            ))
        })?;

        if !response.status().is_success() {
            return Err(UnifiedProcessorError::ProcessingFailed(format!(
                "HTTP {} for {}",
                response.status(),
                payload.url
            )));
        }

        let body = response.text().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to read response from {}: {}",
                payload.url, e
            ))
        })?;

        // Enqueue the root page itself as (Url, Add) for embedding
        let root_url_payload = serde_json::json!({
            "url": payload.url,
            "crawl": false,
            "max_depth": 0,
            "max_pages": 1,
        })
        .to_string();

        let _ = ctx
            .queue_manager
            .enqueue_unified(
                ItemType::Url,
                QueueOperation::Add,
                &item.tenant_id,
                &item.collection,
                &root_url_payload,
                0,
                None,
                None,
            )
            .await;

        // Extract same-domain links if depth allows
        if payload.max_depth > 0 {
            Self::extract_and_enqueue_links(ctx, item, payload, &body).await;
        }

        Ok(())
    }

    /// Extract same-domain links from HTML and enqueue as (Url, Add).
    async fn extract_and_enqueue_links(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &WebsitePayload,
        body: &str,
    ) {
        let base_url = url::Url::parse(&payload.url).ok();
        let base_host = base_url
            .as_ref()
            .and_then(|u| u.host_str().map(|s| s.to_string()));

        let Some(host) = base_host else {
            return;
        };

        let link_re = regex::Regex::new(r#"href=["']([^"']+)["']"#).unwrap();
        let mut enqueued = 0u32;
        let mut seen = HashSet::new();
        seen.insert(payload.url.clone());

        for cap in link_re.captures_iter(body) {
            if enqueued >= payload.max_pages {
                break;
            }

            let href = &cap[1];
            let resolved = if let Some(ref base) = base_url {
                base.join(href).ok()
            } else {
                url::Url::parse(href).ok()
            };

            if let Some(resolved_url) = resolved {
                if resolved_url.host_str() != Some(&host) {
                    continue;
                }
                if resolved_url.scheme() != "http" && resolved_url.scheme() != "https" {
                    continue;
                }

                let url_str = resolved_url.as_str().to_string();
                if !seen.insert(url_str.clone()) {
                    continue;
                }

                let url_payload = serde_json::json!({
                    "url": url_str,
                    "crawl": false,
                    "max_depth": 0,
                    "max_pages": 1,
                })
                .to_string();

                if let Ok((_, true)) = ctx
                    .queue_manager
                    .enqueue_unified(
                        ItemType::Url,
                        QueueOperation::Add,
                        &item.tenant_id,
                        &item.collection,
                        &url_payload,
                        0,
                        None,
                        None,
                    )
                    .await
                {
                    enqueued += 1;
                }
            }
        }
        info!(
            "Website scan: extracted {} same-domain URLs from {}",
            enqueued, payload.url
        );
    }

    /// Re-enqueue as (Website, Scan) for re-crawl.
    async fn handle_update(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &WebsitePayload,
    ) -> UnifiedProcessorResult<()> {
        let scan_payload = serde_json::json!({
            "url": payload.url,
            "max_depth": payload.max_depth,
            "max_pages": payload.max_pages,
        })
        .to_string();

        let _ = ctx
            .queue_manager
            .enqueue_unified(
                ItemType::Website,
                QueueOperation::Scan,
                &item.tenant_id,
                &item.collection,
                &scan_payload,
                0,
                None,
                None,
            )
            .await;
        info!("Re-enqueued website scan for url={}", payload.url);
        Ok(())
    }

    /// Delete all Qdrant points for this tenant (website scoped).
    async fn handle_delete(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        if ctx
            .storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            ctx.storage_client
                .delete_points_by_tenant(&item.collection, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            info!("Deleted website points for tenant={}", item.tenant_id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_website_strategy_handles_website_items() {
        let strategy = WebsiteStrategy;
        assert!(strategy.handles(&ItemType::Website, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Website, &QueueOperation::Scan));
        assert!(strategy.handles(&ItemType::Website, &QueueOperation::Delete));
    }

    #[test]
    fn test_website_strategy_rejects_non_website_items() {
        let strategy = WebsiteStrategy;
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Url, &QueueOperation::Add));
    }

    #[test]
    fn test_website_strategy_name() {
        let strategy = WebsiteStrategy;
        assert_eq!(strategy.name(), "website");
    }
}
