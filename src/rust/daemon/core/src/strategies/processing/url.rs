//! URL processing strategy.
//!
//! Handles `ItemType::Url` queue items: fetching content from a URL,
//! extracting text (using html2text for HTML), generating embeddings,
//! and storing in Qdrant.

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::storage::DocumentPoint;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ItemType, QueueOperation, UnifiedQueueItem, UrlPayload};

/// Strategy for processing URL queue items.
///
/// Fetches content from a URL, extracts text, generates embeddings,
/// and stores the result in Qdrant.
pub struct UrlStrategy;

#[async_trait]
impl ProcessingStrategy for UrlStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Url
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_url_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "url"
    }
}

impl UrlStrategy {
    /// Process URL fetch and ingestion item.
    ///
    /// Fetches content from a URL, extracts text (using html2text for HTML),
    /// generates embeddings, and stores in Qdrant. Supports both single-page
    /// fetch and crawl mode.
    pub(crate) async fn process_url_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        let payload: UrlPayload = parse_payload(item)?;

        info!(
            "Processing URL item: {} (url={})",
            item.queue_id, payload.url
        );

        // Fetch URL content
        let response = reqwest::get(&payload.url).await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to fetch URL {}: {}",
                payload.url, e
            ))
        })?;

        let status = response.status();
        if !status.is_success() {
            return Err(UnifiedProcessorError::ProcessingFailed(format!(
                "HTTP {} for URL {}",
                status, payload.url
            )));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();

        let body = response.text().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!("Failed to read response body: {}", e))
        })?;

        let is_html = content_type.contains("text/html");

        // Extract title from HTML before text extraction
        let title = payload.title.unwrap_or_else(|| {
            if is_html {
                // Simple title extraction: find <title>...</title>
                let lower = body.to_lowercase();
                if let Some(start) = lower.find("<title>") {
                    let title_start = start + 7;
                    if let Some(end) = lower[title_start..].find("</title>") {
                        return body[title_start..title_start + end].trim().to_string();
                    }
                }
            }
            payload.url.clone()
        });

        // Extract text based on content type
        let extracted_text = if is_html {
            html2text::from_read(body.as_bytes(), 80)
        } else {
            body
        };

        if extracted_text.trim().is_empty() {
            warn!("URL {} yielded empty content after extraction", payload.url);
            return Ok(());
        }

        // Generate document ID from URL (stable across re-fetches)
        let document_id = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(payload.url.as_bytes());
            format!("{:x}", hasher.finalize())[..32].to_string()
        };

        // Generate embedding (semaphore-gated)
        let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
            &ctx.embedding_generator,
            &ctx.embedding_semaphore,
            &extracted_text,
            "all-MiniLM-L6-v2",
        )
        .await?;

        // Build Qdrant payload
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(extracted_text));
        point_payload.insert("document_id".to_string(), serde_json::json!(document_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("source_url".to_string(), serde_json::json!(payload.url));
        point_payload.insert("title".to_string(), serde_json::json!(title));
        point_payload.insert("source_type".to_string(), serde_json::json!("web"));
        point_payload.insert("item_type".to_string(), serde_json::json!("url"));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));

        if let Some(ref lib_name) = payload.library_name {
            point_payload.insert("library_name".to_string(), serde_json::json!(lib_name));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &payload.url, 0),
            dense_vector: embed_result.dense_vector,
            sparse_vector: embed_result.sparse_vector,
            payload: point_payload,
        };

        ctx.storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed URL item {} (url={}, content_length={})",
            item.queue_id,
            payload.url,
            extracted_text.len()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_strategy_handles_url_items() {
        let strategy = UrlStrategy;
        assert!(strategy.handles(&ItemType::Url, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Url, &QueueOperation::Update));
        assert!(strategy.handles(&ItemType::Url, &QueueOperation::Delete));
    }

    #[test]
    fn test_url_strategy_rejects_non_url_items() {
        let strategy = UrlStrategy;
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Website, &QueueOperation::Add));
    }

    #[test]
    fn test_url_strategy_name() {
        let strategy = UrlStrategy;
        assert_eq!(strategy.name(), "url");
    }
}
