//! Main ingestion engine for the document processing pipeline.
//!
//! Provides a high-level API that orchestrates content extraction, embedding
//! generation, and Qdrant storage into a single `process_document` call.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::core_types::{DocumentContent, DocumentResult, DocumentType, ProcessingError, TextChunk};
use crate::document_id::generate_document_id;
use crate::document_processor::DocumentProcessor;
use crate::embedding::{EmbeddingConfig, EmbeddingGenerator, SparseEmbedding};

/// Main ingestion engine
///
/// Provides a high-level API for processing documents through the full pipeline:
/// 1. Content extraction and chunking via DocumentProcessor
/// 2. Embedding generation (dense + sparse) via EmbeddingGenerator
/// 3. Storage in Qdrant via StorageClient
pub struct IngestionEngine {
    _config: Config,
    storage_client: Arc<crate::storage::StorageClient>,
    embedding_generator: Arc<EmbeddingGenerator>,
    document_processor: DocumentProcessor,
}

impl IngestionEngine {
    /// Create a new ingestion engine with the provided configuration
    pub fn new(config: Config) -> std::result::Result<Self, ProcessingError> {
        let storage_client = Arc::new(crate::storage::StorageClient::new());
        let embedding_config = EmbeddingConfig::default();
        let embedding_generator = Arc::new(
            EmbeddingGenerator::new(embedding_config)
                .map_err(|e| ProcessingError::Processing(format!("Failed to initialize embedding generator: {}", e)))?
        );
        let document_processor = DocumentProcessor::new();

        Ok(Self {
            _config: config,
            storage_client,
            embedding_generator,
            document_processor,
        })
    }

    /// Process a single document through the full pipeline.
    ///
    /// Steps:
    /// 1. Extract content and generate chunks (tree-sitter for code, sliding window for text)
    /// 2. Generate dense + sparse embeddings for each chunk
    /// 3. Create Qdrant points with stable document_id and point_id
    /// 4. Upsert points to the specified collection
    pub async fn process_document(
        &self,
        file_path: &Path,
        collection: &str,
        branch: &str,
    ) -> std::result::Result<DocumentResult, ProcessingError> {
        let start = Instant::now();

        let path_str = file_path.to_string_lossy();
        let document_id = generate_document_id(collection, &path_str);

        let (content, extract_ms) = self.stage1_extract(file_path, collection, &path_str).await?;
        self.stage2_ensure_collection(collection).await?;

        let file_hash = wqm_common::hashing::compute_file_hash(file_path)
            .unwrap_or_else(|_| "unknown".to_string());
        let base_point = wqm_common::hashing::compute_base_point(
            collection, branch, &path_str, &file_hash,
        );

        let (points, embed_ms) = self
            .stage3_embed_chunks(&content, &path_str, &document_id, collection, &file_hash, &base_point)
            .await?;

        let store_ms = self.stage4_store(collection, &points).await?;

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            document_id = %document_id,
            collection = %collection,
            chunks = points.len(),
            extract_ms = extract_ms,
            embed_ms = embed_ms,
            store_ms = store_ms,
            total_ms = total_ms,
            "Document processing complete"
        );

        Ok(DocumentResult {
            document_id,
            collection: collection.to_string(),
            chunks_created: Some(points.len()),
            processing_time_ms: total_ms,
        })
    }

    /// Stage 1: extract content and chunk the document.
    async fn stage1_extract(
        &self,
        file_path: &Path,
        collection: &str,
        path_str: &str,
    ) -> std::result::Result<(DocumentContent, u128), ProcessingError> {
        let extract_start = Instant::now();
        let content = self.document_processor
            .process_file_content(file_path, collection)
            .await
            .map_err(|e| ProcessingError::Processing(format!("Document processing failed: {}", e)))?;
        let extract_ms = extract_start.elapsed().as_millis();
        tracing::info!(
            file = %path_str,
            chunks = content.chunks.len(),
            doc_type = ?content.document_type,
            extract_ms = extract_ms,
            "Stage 1: extraction complete"
        );
        Ok((content, extract_ms))
    }

    /// Stage 2: ensure the target collection exists, creating it if necessary.
    async fn stage2_ensure_collection(
        &self,
        collection: &str,
    ) -> std::result::Result<(), ProcessingError> {
        if !self.storage_client
            .collection_exists(collection)
            .await
            .map_err(|e| ProcessingError::Storage(format!("Collection check failed: {}", e)))?
        {
            let config = crate::storage::MultiTenantConfig::default();
            self.storage_client
                .create_multi_tenant_collection(collection, &config)
                .await
                .map_err(|e| ProcessingError::Storage(format!("Collection creation failed: {}", e)))?;
        }
        Ok(())
    }

    /// Stage 3: generate embeddings for each chunk and build Qdrant points.
    async fn stage3_embed_chunks(
        &self,
        content: &DocumentContent,
        path_str: &str,
        document_id: &str,
        collection: &str,
        file_hash: &str,
        base_point: &str,
    ) -> std::result::Result<(Vec<crate::storage::DocumentPoint>, u128), ProcessingError> {
        let embed_start = Instant::now();
        let mut points = Vec::with_capacity(content.chunks.len());

        for (chunk_idx, chunk) in content.chunks.iter().enumerate() {
            let embedding_result = self.embedding_generator
                .generate_embedding(&chunk.content, "bge-small-en-v1.5")
                .await
                .map_err(|e| ProcessingError::Processing(format!("Embedding generation failed: {}", e)))?;

            let payload = build_chunk_payload(
                chunk, path_str, document_id, collection, file_hash, base_point,
                &content.document_type,
            );

            let sparse_vector = sparse_to_map(&embedding_result.sparse);

            points.push(crate::storage::DocumentPoint {
                id: wqm_common::hashing::compute_point_id(base_point, chunk_idx as u32),
                dense_vector: embedding_result.dense.vector,
                sparse_vector,
                payload,
            });
        }

        let embed_ms = embed_start.elapsed().as_millis();
        let per_chunk_ms = if !points.is_empty() { embed_ms / points.len() as u128 } else { 0 };
        tracing::info!(
            chunks = points.len(),
            embed_ms = embed_ms,
            per_chunk_ms = per_chunk_ms,
            "Stage 3: embedding complete"
        );
        Ok((points, embed_ms))
    }

    /// Stage 4: upsert points to Qdrant and return elapsed storage time.
    async fn stage4_store(
        &self,
        collection: &str,
        points: &[crate::storage::DocumentPoint],
    ) -> std::result::Result<u128, ProcessingError> {
        let store_start = Instant::now();
        if !points.is_empty() {
            self.storage_client
                .insert_points_batch(collection, points.to_vec(), Some(100))
                .await
                .map_err(|e| ProcessingError::Storage(format!("Qdrant upsert failed: {}", e)))?;
        }
        let store_ms = store_start.elapsed().as_millis();
        tracing::info!(points = points.len(), store_ms = store_ms, "Stage 4: storage complete");
        Ok(store_ms)
    }

    /// Get the document processor for direct access
    pub fn document_processor(&self) -> &DocumentProcessor {
        &self.document_processor
    }

    /// Get the storage client for direct access
    pub fn storage_client(&self) -> &Arc<crate::storage::StorageClient> {
        &self.storage_client
    }
}

/// Build the Qdrant payload map for a single text chunk.
fn build_chunk_payload(
    chunk: &TextChunk,
    path_str: &str,
    document_id: &str,
    collection: &str,
    file_hash: &str,
    base_point: &str,
    doc_type: &DocumentType,
) -> HashMap<String, serde_json::Value> {
    let mut payload = HashMap::new();
    payload.insert("content".to_string(), serde_json::json!(chunk.content));
    payload.insert("chunk_index".to_string(), serde_json::json!(chunk.chunk_index));
    payload.insert("file_path".to_string(), serde_json::json!(path_str));
    payload.insert("document_id".to_string(), serde_json::json!(document_id));
    payload.insert("tenant_id".to_string(), serde_json::json!(collection));
    payload.insert("document_type".to_string(), serde_json::json!(doc_type.as_str()));
    if let Some(lang) = doc_type.language() {
        payload.insert("language".to_string(), serde_json::json!(lang));
    }
    if let Some(ext) = std::path::Path::new(path_str)
        .extension()
        .and_then(|e| e.to_str())
    {
        payload.insert("file_extension".to_string(), serde_json::json!(ext));
    }
    payload.insert("item_type".to_string(), serde_json::json!("file"));
    payload.insert("base_point".to_string(), serde_json::json!(base_point));
    payload.insert("relative_path".to_string(), serde_json::json!(path_str));
    payload.insert("absolute_path".to_string(), serde_json::json!(path_str));
    payload.insert("file_hash".to_string(), serde_json::json!(file_hash));
    for (key, value) in &chunk.metadata {
        payload.insert(format!("chunk_{}", key), serde_json::json!(value));
    }
    payload
}

/// Convert a sparse embedding to the `HashMap<u32, f32>` format expected by `DocumentPoint`.
fn sparse_to_map(sparse: &SparseEmbedding) -> Option<HashMap<u32, f32>> {
    if sparse.indices.is_empty() {
        return None;
    }
    Some(
        sparse.indices.iter()
            .zip(sparse.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect(),
    )
}
