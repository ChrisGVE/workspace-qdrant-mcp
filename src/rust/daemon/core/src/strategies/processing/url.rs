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

        if item.op == QueueOperation::Delete {
            return Self::handle_url_delete(ctx, item, &payload).await;
        }

        let fetched = super::url_fetch::fetch_url_secured(&payload.url, &ctx.url_ingestion)
            .await
            .map_err(|e| {
                UnifiedProcessorError::ProcessingFailed(format!(
                    "URL fetch failed ({}): {}",
                    payload.url, e
                ))
            })?;

        let is_html = fetched.content_type.contains("text/html")
            || fetched.content_type.contains("xhtml")
            || fetched.content_type.contains("application/xml");
        let title = extract_title(&payload, &fetched.body, is_html);
        let extracted_text = extract_text(&fetched.body, is_html);

        if extracted_text.trim().is_empty() {
            warn!("URL {} yielded empty content after extraction", payload.url);
            return Ok(());
        }

        if fetched.truncated {
            warn!(
                "URL {} body truncated at {} bytes (config cap reached)",
                payload.url, ctx.url_ingestion.max_body_bytes
            );
        }

        let document_id = compute_url_document_id(&payload.url);

        // Chunk extracted text using shared chunker so large pages produce
        // multiple embeddings rather than one massive vector.
        let chunks = chunk_url_text(&extracted_text, ctx.document_processor.chunking_config());
        let chunk_count = chunks.len();

        let mut points: Vec<DocumentPoint> = Vec::with_capacity(chunk_count);
        for chunk in &chunks {
            let embed_result = crate::shared::embedding_pipeline::embed_with_sparse(
                &ctx.embedding_generator,
                &ctx.embedding_semaphore,
                &chunk.content,
                "all-MiniLM-L6-v2",
            )
            .await?;

            let point_payload = build_url_payload(
                item,
                &payload,
                &chunk.content,
                &document_id,
                &title,
                &fetched.final_url,
                &fetched.content_type,
                fetched.truncated,
                chunk.chunk_index,
                chunk_count,
                chunk.start_char,
                chunk.end_char,
            );

            points.push(DocumentPoint {
                id: wqm_common::hashing::content_point_id(
                    &item.tenant_id,
                    &payload.url,
                    chunk.chunk_index as u32,
                )
                .to_string(),
                dense_vector: embed_result.dense_vector,
                sparse_vector: embed_result.sparse_vector,
                payload: point_payload,
            });
        }

        let batch_size = points.len();
        ctx.storage_client
            .insert_points_batch(&item.collection, points, Some(batch_size.max(1)))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed URL item {} (url={}, content_length={}, chunks={})",
            item.queue_id,
            payload.url,
            extracted_text.len(),
            chunk_count
        );

        Ok(())
    }
}

impl UrlStrategy {
    /// Delete URL points scoped by `(tenant_id, document_id)`.
    ///
    /// `document_id` is derived from the URL (same derivation used on insert),
    /// so the delete matches the previously stored point. Tenant scope is
    /// enforced to prevent cross-tenant eviction.
    async fn handle_url_delete(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &UrlPayload,
    ) -> UnifiedProcessorResult<()> {
        let document_id = compute_url_document_id(&payload.url);

        info!(
            "Deleting URL point: tenant={} url={} document_id={} -> collection={}",
            item.tenant_id, payload.url, document_id, item.collection
        );

        ctx.storage_client
            .delete_points_by_payload_fields(
                &item.collection,
                &[
                    ("tenant_id", item.tenant_id.as_str()),
                    ("document_id", document_id.as_str()),
                ],
            )
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        Ok(())
    }
}

/// Extract a title from the payload or from the HTML body.
fn extract_title(payload: &UrlPayload, body: &str, is_html: bool) -> String {
    payload.title.clone().unwrap_or_else(|| {
        if is_html {
            let lower = body.to_lowercase();
            if let Some(start) = lower.find("<title>") {
                let title_start = start + 7;
                if let Some(end) = lower[title_start..].find("</title>") {
                    return body[title_start..title_start + end].trim().to_string();
                }
            }
        }
        payload.url.clone()
    })
}

/// Extract text from content based on content type.
fn extract_text(body: &str, is_html: bool) -> String {
    if is_html {
        html2text::from_read(body.as_bytes(), 80)
    } else {
        body.to_string()
    }
}

/// Compute a stable document ID from a URL.
fn compute_url_document_id(url: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    format!("{:x}", hasher.finalize())[..32].to_string()
}

/// Chunk extracted URL text. Uses the shared `chunk_text` chunker so URLs
/// share the same paragraph-aware boundary logic as file ingestion. Empty
/// input collapses to a single empty chunk (callers short-circuit before this).
fn chunk_url_text(text: &str, chunking_config: &crate::ChunkingConfig) -> Vec<crate::TextChunk> {
    let mut chunks = crate::document_processor::chunking::chunk_text(
        text,
        &std::collections::HashMap::new(),
        chunking_config,
    );
    if chunks.is_empty() && !text.is_empty() {
        // Defensive: if the chunker returns no output for non-empty text
        // (paragraph-mode edge case with no `\n\n` separators), wrap it.
        chunks.push(crate::TextChunk {
            content: text.to_string(),
            chunk_index: 0,
            start_char: 0,
            end_char: text.len(),
            metadata: std::collections::HashMap::new(),
        });
    }
    chunks
}

/// Build the Qdrant payload map for a URL chunk.
#[allow(clippy::too_many_arguments)]
fn build_url_payload(
    item: &UnifiedQueueItem,
    payload: &UrlPayload,
    chunk_content: &str,
    document_id: &str,
    title: &str,
    final_url: &str,
    content_type: &str,
    truncated: bool,
    chunk_index: usize,
    chunk_count: usize,
    start_char: usize,
    end_char: usize,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut point_payload = std::collections::HashMap::new();
    point_payload.insert("content".to_string(), serde_json::json!(chunk_content));
    point_payload.insert("document_id".to_string(), serde_json::json!(document_id));
    point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
    point_payload.insert("source_url".to_string(), serde_json::json!(payload.url));
    point_payload.insert("final_url".to_string(), serde_json::json!(final_url));
    point_payload.insert("content_type".to_string(), serde_json::json!(content_type));
    point_payload.insert("truncated".to_string(), serde_json::json!(truncated));
    point_payload.insert("title".to_string(), serde_json::json!(title));
    point_payload.insert("source_type".to_string(), serde_json::json!("web"));
    point_payload.insert("item_type".to_string(), serde_json::json!("url"));
    point_payload.insert("branches".to_string(), serde_json::json!([&item.branch]));
    point_payload.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
    point_payload.insert("chunk_count".to_string(), serde_json::json!(chunk_count));
    point_payload.insert("chunk_start".to_string(), serde_json::json!(start_char));
    point_payload.insert("chunk_end".to_string(), serde_json::json!(end_char));

    if let Some(ref lib_name) = payload.library_name {
        point_payload.insert("library_name".to_string(), serde_json::json!(lib_name));
    }

    point_payload
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

    /// F-030: long extracted text produces multiple chunks.
    #[test]
    fn test_chunk_url_text_splits_long_input() {
        // Build text larger than default chunk_size (384 chars) with paragraph
        // boundaries so the paragraph-aware chunker yields multiple chunks.
        let paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(8);
        let text = format!("{p}\n\n{p}\n\n{p}\n\n{p}\n\n{p}", p = paragraph);
        let cfg = crate::ChunkingConfig::default();
        let chunks = chunk_url_text(&text, &cfg);
        assert!(
            chunks.len() > 1,
            "expected >1 chunk for {}-char text, got {}",
            text.len(),
            chunks.len()
        );
        // Chunk indices are contiguous from 0.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_index, i);
        }
    }

    /// F-030: short extracted text produces exactly one chunk.
    #[test]
    fn test_chunk_url_text_short_input_single_chunk() {
        let text = "Short page content.";
        let cfg = crate::ChunkingConfig::default();
        let chunks = chunk_url_text(text, &cfg);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].content, text);
    }

    /// F-006: URL Delete derives the same `document_id` as the Add path.
    ///
    /// The Add path (build_url_payload) writes `document_id` derived from
    /// `compute_url_document_id(&payload.url)`. The Delete path
    /// (`handle_url_delete`) MUST derive the same id, or Delete becomes a
    /// no-op against the point inserted by Add.
    #[test]
    fn test_url_delete_uses_same_document_id_as_add() {
        let url = "https://example.com/docs/page";
        let id_a = compute_url_document_id(url);
        let id_b = compute_url_document_id(url);
        assert_eq!(id_a, id_b);
        // Distinct URL → distinct id
        assert_ne!(id_a, compute_url_document_id("https://example.com/other"));
        // 32-char hex prefix (per implementation)
        assert_eq!(id_a.len(), 32);
        assert!(id_a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
