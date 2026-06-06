//! Text chunking, storage interaction, and batch ingestion
//!
//! Handles text splitting into overlapping chunks, collection lifecycle,
//! storage error mapping, and the core ingestion pipeline that combines
//! chunking, embedding, and Qdrant storage.

use std::collections::HashMap;
use std::sync::Arc;

use tonic::Status;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use workspace_qdrant_core::embedding::provider::DenseProvider;
use workspace_qdrant_core::storage::{
    DocumentPoint, MultiTenantConfig, StorageClient, StorageError,
};
use wqm_common::timestamps;

use super::embedding;
use crate::proto::IngestTextResponse;

/// Default chunk size in characters for text chunking
pub(crate) const DEFAULT_CHUNK_SIZE: usize = 1000;
/// Default overlap size in characters for sliding window chunking
pub(crate) const DEFAULT_CHUNK_OVERLAP: usize = 200;

/// Chunk text into overlapping segments.
/// Returns: Vec<(chunk_content, chunk_index)>
pub(crate) fn chunk_text(
    text: &str,
    enable_chunking: bool,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(String, usize)> {
    if !enable_chunking || text.len() <= chunk_size {
        return vec![(text.to_string(), 0)];
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < chars.len() {
        let end = std::cmp::min(start + chunk_size, chars.len());

        let chunk: String = chars[start..end].iter().collect();

        if !chunk.trim().is_empty() {
            chunks.push((chunk, chunk_index));
            chunk_index += 1;
        }

        if end == chars.len() {
            break;
        }

        start = if chunk_overlap < chunk_size {
            end - chunk_overlap
        } else {
            end
        };

        if start >= end {
            start = end;
        }
    }

    if chunks.is_empty() {
        chunks.push((text.to_string(), 0));
    }

    chunks
}

/// Ensure collection exists, create if not.
pub(crate) async fn ensure_collection_exists(
    storage_client: &StorageClient,
    collection_name: &str,
    vector_size: u64,
) -> Result<(), Status> {
    match storage_client.collection_exists(collection_name).await {
        Ok(true) => {
            debug!("Collection '{}' already exists", collection_name);
            Ok(())
        }
        Ok(false) => {
            info!(
                "Creating collection '{}' with multi-tenant config (dense+sparse)",
                collection_name
            );
            let config = MultiTenantConfig {
                vector_size,
                ..MultiTenantConfig::default()
            };
            storage_client
                .create_multi_tenant_collection(collection_name, &config)
                .await
                .map_err(map_storage_error)?;
            info!("Successfully created collection '{}'", collection_name);
            Ok(())
        }
        Err(e) => {
            error!("Failed to check collection existence: {:?}", e);
            Err(map_storage_error(e))
        }
    }
}

/// Map storage errors to gRPC Status.
pub(crate) fn map_storage_error(err: StorageError) -> Status {
    match err {
        StorageError::Collection(msg) if msg.contains("already exists") => {
            Status::already_exists(format!("Collection already exists: {}", msg))
        }
        StorageError::Collection(msg) if msg.contains("not found") => {
            Status::not_found(format!("Collection not found: {}", msg))
        }
        StorageError::Collection(msg) => {
            Status::failed_precondition(format!("Collection error: {}", msg))
        }
        StorageError::Connection(msg) => Status::unavailable(format!("Connection error: {}", msg)),
        StorageError::Timeout(msg) => Status::deadline_exceeded(format!("Timeout: {}", msg)),
        StorageError::Qdrant(err) => {
            let err_msg = format!("{:?}", err);
            if err_msg.contains("rate limit") || err_msg.contains("too many requests") {
                Status::resource_exhausted("Rate limit exceeded")
            } else if err_msg.contains("not found") {
                Status::not_found(err_msg)
            } else {
                Status::internal(format!("Qdrant error: {}", err_msg))
            }
        }
        _ => Status::internal(format!("Storage error: {}", err)),
    }
}

/// Build a single DocumentPoint from chunk content, its precomputed dense
/// embedding, and metadata.
async fn build_document_point(
    chunk_content: &str,
    dense_embedding: Vec<f32>,
    chunk_index: usize,
    total_chunks: usize,
    document_id: &str,
    metadata: &HashMap<String, String>,
    created_at: &str,
) -> Result<DocumentPoint, Status> {
    let sparse_vector = embedding::generate_sparse_vector(chunk_content).await?;
    let sparse_option = if sparse_vector.is_empty() {
        None
    } else {
        Some(sparse_vector)
    };

    let mut chunk_metadata: HashMap<String, serde_json::Value> = metadata
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();

    chunk_metadata.insert("document_id".to_string(), serde_json::json!(document_id));
    chunk_metadata.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
    chunk_metadata.insert("total_chunks".to_string(), serde_json::json!(total_chunks));
    chunk_metadata.insert("created_at".to_string(), serde_json::json!(created_at));
    chunk_metadata.insert("content".to_string(), serde_json::json!(chunk_content));

    let namespace = Uuid::parse_str(document_id).unwrap_or_else(|_| Uuid::new_v4());
    let point_id = Uuid::new_v5(&namespace, chunk_index.to_string().as_bytes()).to_string();

    Ok(DocumentPoint {
        id: point_id,
        dense_vector: dense_embedding,
        sparse_vector: sparse_option,
        payload: chunk_metadata,
    })
}

/// Generate one dense embedding per chunk via the injected provider.
///
/// All chunks go to the provider in a single batched `embed` call; the
/// provider applies its own internal batching/rate limiting. Returns vectors
/// in chunk order.
pub(crate) async fn embed_chunks(
    dense_provider: &Arc<dyn DenseProvider>,
    chunks: &[(String, usize)],
) -> Result<Vec<Vec<f32>>, Status> {
    let texts: Vec<&str> = chunks.iter().map(|(content, _)| content.as_str()).collect();
    let embeddings = dense_provider.embed(&texts).await.map_err(|e| {
        error!("Dense provider embed failed: {:?}", e);
        Status::internal(format!("Embedding generation failed: {}", e))
    })?;

    if embeddings.len() != chunks.len() {
        return Err(Status::internal(format!(
            "Provider returned {} embeddings for {} chunks",
            embeddings.len(),
            chunks.len()
        )));
    }

    Ok(embeddings.into_iter().map(|e| e.vector).collect())
}

/// Process text ingestion: chunk, embed, and store.
pub(crate) async fn ingest_text_internal(
    storage_client: &Arc<StorageClient>,
    dense_provider: &Arc<dyn DenseProvider>,
    content: String,
    collection_name: String,
    document_id: String,
    metadata: HashMap<String, String>,
    do_chunk: bool,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<IngestTextResponse, Status> {
    if content.trim().is_empty() {
        return Err(Status::invalid_argument("Content cannot be empty"));
    }

    ensure_collection_exists(
        storage_client,
        &collection_name,
        dense_provider.output_dim() as u64,
    )
    .await?;

    let chunks = chunk_text(&content, do_chunk, chunk_size, chunk_overlap);
    let total_chunks = chunks.len();

    debug!(
        "Chunked text into {} chunks (chunking_enabled={})",
        total_chunks, do_chunk
    );

    let dense_embeddings = embed_chunks(dense_provider, &chunks).await?;

    let created_at = timestamps::now_utc();
    let mut document_points = Vec::new();

    for ((chunk_content, chunk_index), dense_embedding) in chunks.into_iter().zip(dense_embeddings)
    {
        let point = build_document_point(
            &chunk_content,
            dense_embedding,
            chunk_index,
            total_chunks,
            &document_id,
            &metadata,
            &created_at,
        )
        .await?;
        document_points.push(point);
    }

    info!(
        "Inserting {} chunks for document {} into collection {}",
        document_points.len(),
        document_id,
        collection_name
    );

    match storage_client
        .insert_points_batch(&collection_name, document_points, Some(100))
        .await
    {
        Ok(stats) => {
            info!(
                "Successfully inserted {} chunks ({} successful, {} failed)",
                stats.total_points, stats.successful, stats.failed
            );
            if stats.failed > 0 {
                warn!("{} chunks failed to insert", stats.failed);
            }
            Ok(IngestTextResponse {
                document_id: document_id.clone(),
                success: stats.failed == 0,
                chunks_created: stats.successful as i32,
                error_message: if stats.failed > 0 {
                    format!("{} chunks failed to insert", stats.failed)
                } else {
                    String::new()
                },
            })
        }
        Err(e) => {
            error!("Failed to insert chunks: {:?}", e);
            Err(map_storage_error(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use workspace_qdrant_core::embedding::{DenseEmbedding, EmbeddingError};

    /// Mock provider that records embed calls. Asserts ingestion uses the
    /// injected provider (issue #95): no hardcoded FastEmbed path remains, so
    /// a counting mock is sufficient to prove which backend ran.
    #[derive(Debug, Default)]
    struct CountingProvider {
        embed_calls: AtomicUsize,
        texts_embedded: AtomicUsize,
    }

    #[async_trait]
    impl DenseProvider for CountingProvider {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            self.texts_embedded.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, text)| DenseEmbedding {
                    // Distinctive per-chunk vector so order can be asserted.
                    vector: vec![i as f32, text.len() as f32],
                    model_name: "counting-mock".to_string(),
                    sequence_length: text.len(),
                })
                .collect())
        }
        fn output_dim(&self) -> usize {
            2
        }
        fn provider_label(&self) -> &str {
            "counting-mock"
        }
        fn metrics_label(&self) -> &'static str {
            "fastembed"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_embed_chunks_uses_injected_provider() {
        let provider = Arc::new(CountingProvider::default());
        let dyn_provider: Arc<dyn DenseProvider> = provider.clone();

        let chunks = vec![
            ("first chunk".to_string(), 0),
            ("second chunk longer".to_string(), 1),
            ("third".to_string(), 2),
        ];

        let vectors = embed_chunks(&dyn_provider, &chunks)
            .await
            .expect("embed_chunks failed");

        // One batched provider call covering every chunk, vectors in order.
        assert_eq!(provider.embed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.texts_embedded.load(Ordering::SeqCst), 3);
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vec![0.0, "first chunk".len() as f32]);
        assert_eq!(vectors[1], vec![1.0, "second chunk longer".len() as f32]);
        assert_eq!(vectors[2], vec![2.0, "third".len() as f32]);
    }

    /// Provider returning a wrong-length batch must be rejected, not zipped
    /// silently (would mis-assign vectors to chunks).
    #[derive(Debug)]
    struct ShortBatchProvider;

    #[async_trait]
    impl DenseProvider for ShortBatchProvider {
        async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(vec![DenseEmbedding {
                vector: vec![1.0],
                model_name: "short".to_string(),
                sequence_length: 1,
            }])
        }
        fn output_dim(&self) -> usize {
            1
        }
        fn provider_label(&self) -> &str {
            "short"
        }
        fn metrics_label(&self) -> &'static str {
            "fastembed"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_embed_chunks_rejects_count_mismatch() {
        let provider: Arc<dyn DenseProvider> = Arc::new(ShortBatchProvider);
        let chunks = vec![("one".to_string(), 0), ("two".to_string(), 1)];

        let err = embed_chunks(&provider, &chunks)
            .await
            .expect_err("count mismatch must error");
        assert_eq!(err.code(), tonic::Code::Internal);
        assert!(err.message().contains("1 embeddings for 2 chunks"));
    }

    #[test]
    fn test_chunk_text_single_chunk() {
        let text = "Short text";

        let chunks = chunk_text(text, false, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
        assert_eq!(chunks[0].1, 0);

        let chunks = chunk_text(text, true, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
    }

    #[test]
    fn test_chunk_text_multiple_chunks() {
        let text = "This is a longer text that will be split into multiple chunks. \
                    Each chunk should overlap slightly with the previous one. \
                    This helps maintain context across chunk boundaries.";

        let chunks = chunk_text(text, true, 50, 10);

        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );

        for (i, (_, index)) in chunks.iter().enumerate() {
            assert_eq!(*index, i, "Chunk index mismatch");
        }

        for (content, _) in &chunks {
            assert!(!content.trim().is_empty(), "Empty chunk found");
        }
    }
}
