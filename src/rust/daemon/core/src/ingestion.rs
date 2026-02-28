//! Main ingestion engine for the document processing pipeline.
//!
//! Provides a high-level API that orchestrates content extraction, embedding
//! generation, and Qdrant storage into a single `process_document` call.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::core_types::{DocumentResult, ProcessingError};
use crate::document_id::generate_document_id;
use crate::document_processor::DocumentProcessor;
use crate::embedding::{EmbeddingConfig, EmbeddingGenerator};

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

        // Generate stable document ID
        let path_str = file_path.to_string_lossy();
        let document_id = generate_document_id(collection, &path_str);

        // Stage 1: Extract content and chunk using DocumentProcessor
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

        // Stage 2: Ensure collection exists
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

        // Compute file hash and base_point for the base_point model (Task 10)
        let file_hash = wqm_common::hashing::compute_file_hash(file_path)
            .unwrap_or_else(|_| "unknown".to_string());
        let base_point = wqm_common::hashing::compute_base_point(
            collection, branch, &path_str, &file_hash,
        );

        // Stage 3: Generate embeddings and build points for each chunk
        let embed_start = Instant::now();
        let mut points = Vec::with_capacity(content.chunks.len());
        for (chunk_idx, chunk) in content.chunks.iter().enumerate() {
            let embedding_result = self.embedding_generator
                .generate_embedding(&chunk.content, "bge-small-en-v1.5")
                .await
                .map_err(|e| ProcessingError::Processing(format!("Embedding generation failed: {}", e)))?;

            // Build point payload
            let mut payload = HashMap::new();
            payload.insert("content".to_string(), serde_json::json!(chunk.content));
            payload.insert("chunk_index".to_string(), serde_json::json!(chunk.chunk_index));
            payload.insert("file_path".to_string(), serde_json::json!(path_str));
            payload.insert("document_id".to_string(), serde_json::json!(document_id));
            payload.insert("tenant_id".to_string(), serde_json::json!(collection));
            payload.insert("document_type".to_string(), serde_json::json!(content.document_type.as_str()));
            if let Some(lang) = content.document_type.language() {
                payload.insert("language".to_string(), serde_json::json!(lang));
            }
            if let Some(ext) = std::path::Path::new(&*path_str)
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

            // Include chunk metadata (symbol_name, start_line, etc.)
            for (key, value) in &chunk.metadata {
                payload.insert(format!("chunk_{}", key), serde_json::json!(value));
            }

            // Convert sparse embedding to HashMap format
            let sparse_vector = if !embedding_result.sparse.indices.is_empty() {
                let map: HashMap<u32, f32> = embedding_result.sparse.indices.iter()
                    .zip(embedding_result.sparse.values.iter())
                    .map(|(&idx, &val)| (idx, val))
                    .collect();
                Some(map)
            } else {
                None
            };

            points.push(crate::storage::DocumentPoint {
                id: wqm_common::hashing::compute_point_id(&base_point, chunk_idx as u32),
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

        // Stage 4: Upsert points to Qdrant
        let store_start = Instant::now();
        if !points.is_empty() {
            self.storage_client
                .insert_points_batch(collection, points.clone(), Some(100))
                .await
                .map_err(|e| ProcessingError::Storage(format!("Qdrant upsert failed: {}", e)))?;
        }
        let store_ms = store_start.elapsed().as_millis();
        tracing::info!(
            points = points.len(),
            store_ms = store_ms,
            "Stage 4: storage complete"
        );

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

    /// Get the document processor for direct access
    pub fn document_processor(&self) -> &DocumentProcessor {
        &self.document_processor
    }

    /// Get the storage client for direct access
    pub fn storage_client(&self) -> &Arc<crate::storage::StorageClient> {
        &self.storage_client
    }
}
