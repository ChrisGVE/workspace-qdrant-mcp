//! DocumentService gRPC implementation
//!
//! Handles direct text ingestion for non-file-based content.
//! Provides 3 RPCs: IngestText, UpdateText, DeleteText
//!
//! This service is designed for user-provided text content such as:
//! - Manual notes and annotations
//! - Chat snippets and conversations
//! - Scraped web content
//! - API responses
//! - Any text not originating from files

use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};
use uuid::Uuid;
use workspace_qdrant_core::storage::{StorageClient, DocumentPoint, StorageError};

use crate::proto::{
    document_service_server::DocumentService,
    IngestTextRequest, IngestTextResponse,
    UpdateTextRequest, UpdateTextResponse,
    DeleteTextRequest,
};

/// Default chunk size in characters for text chunking
const DEFAULT_CHUNK_SIZE: usize = 1000;
/// Default overlap size in characters for sliding window chunking
const DEFAULT_CHUNK_OVERLAP: usize = 200;
/// Default vector dimension for embeddings (all-MiniLM-L6-v2)
const DEFAULT_VECTOR_SIZE: u64 = 384;

/// DocumentService implementation with text chunking and embedding generation
#[derive(Debug)]
pub struct DocumentServiceImpl {
    storage_client: Arc<StorageClient>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl DocumentServiceImpl {
    /// Create a new DocumentService with the provided storage client
    pub fn new(storage_client: Arc<StorageClient>) -> Self {
        Self {
            storage_client,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        }
    }

    /// Create with default storage client
    pub fn default() -> Self {
        Self::new(Arc::new(StorageClient::new()))
    }

    /// Create with custom chunking configuration
    pub fn with_config(
        storage_client: Arc<StorageClient>,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Self {
        Self {
            storage_client,
            chunk_size,
            chunk_overlap,
        }
    }
}

// Helper functions for validation and processing
impl DocumentServiceImpl {
    /// Validate collection name (reuse CollectionService validation rules)
    /// Rules: 3-255 chars, alphanumeric + underscore/hyphen, no leading numbers
    fn validate_collection_name(name: &str) -> Result<(), Status> {
        if name.is_empty() {
            return Err(Status::invalid_argument("Collection name cannot be empty"));
        }

        if name.len() < 3 {
            return Err(Status::invalid_argument(
                "Collection name must be at least 3 characters"
            ));
        }

        if name.len() > 255 {
            return Err(Status::invalid_argument(
                "Collection name must not exceed 255 characters"
            ));
        }

        // Check first character is not a number
        if name.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
            return Err(Status::invalid_argument(
                "Collection name cannot start with a number"
            ));
        }

        // Check all characters are valid
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(Status::invalid_argument(
                "Collection name can only contain alphanumeric characters, underscores, and hyphens"
            ));
        }

        Ok(())
    }

    /// Format collection name from basename and tenant_id
    /// Format: {collection_basename}_{tenant_id}
    fn format_collection_name(basename: &str, tenant_id: &str) -> Result<String, Status> {
        if basename.is_empty() {
            return Err(Status::invalid_argument("Collection basename cannot be empty"));
        }

        if tenant_id.is_empty() {
            return Err(Status::invalid_argument("Tenant ID cannot be empty"));
        }

        let collection_name = format!("{}_{}", basename, tenant_id);
        Self::validate_collection_name(&collection_name)?;
        Ok(collection_name)
    }

    /// Validate document ID format (should be valid UUID)
    fn validate_document_id(id: &str) -> Result<(), Status> {
        if id.is_empty() {
            return Err(Status::invalid_argument("Document ID cannot be empty"));
        }

        // Try to parse as UUID to validate format
        Uuid::parse_str(id).map_err(|_| {
            Status::invalid_argument("Document ID must be a valid UUID")
        })?;

        Ok(())
    }

    /// Chunk text into overlapping segments
    /// Returns: Vec<(chunk_content, chunk_index)>
    fn chunk_text(&self, text: &str, enable_chunking: bool) -> Vec<(String, usize)> {
        if !enable_chunking || text.len() <= self.chunk_size {
            // Return single chunk
            return vec![(text.to_string(), 0)];
        }

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < chars.len() {
            // Determine end position for this chunk
            let end = std::cmp::min(start + self.chunk_size, chars.len());

            // Extract chunk
            let chunk: String = chars[start..end].iter().collect();

            // Add chunk if non-empty
            if !chunk.trim().is_empty() {
                chunks.push((chunk, chunk_index));
                chunk_index += 1;
            }

            // Move start position with overlap
            // For last chunk or if remaining text is small, don't overlap
            if end == chars.len() {
                break;
            }

            // Calculate next start with overlap
            start = if self.chunk_overlap < self.chunk_size {
                end - self.chunk_overlap
            } else {
                end
            };

            // Ensure we make progress
            if start >= end {
                start = end;
            }
        }

        // If no chunks were created, return the original text as single chunk
        if chunks.is_empty() {
            chunks.push((text.to_string(), 0));
        }

        chunks
    }

    /// Generate mock embedding for text
    /// TODO: Replace with actual embedding generation (fastembed-rs or external service)
    /// For now, returns a simple deterministic mock based on text hash
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Simple deterministic mock: hash text and use it to seed a pattern
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate deterministic vector based on hash
        let mut embedding = Vec::with_capacity(DEFAULT_VECTOR_SIZE as usize);
        let mut seed = hash;

        for _ in 0..DEFAULT_VECTOR_SIZE {
            // Simple LCG (Linear Congruential Generator) for deterministic randomness
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = (seed >> 16) as f32 / 32768.0 - 1.0; // Range: [-1.0, 1.0]
            embedding.push(value);
        }

        // Normalize to unit vector for cosine similarity
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= magnitude);
        }

        embedding
    }

    /// Ensure collection exists, create if not
    async fn ensure_collection_exists(&self, collection_name: &str) -> Result<(), Status> {
        match self.storage_client.collection_exists(collection_name).await {
            Ok(true) => {
                debug!("Collection '{}' already exists", collection_name);
                Ok(())
            }
            Ok(false) => {
                info!("Creating collection '{}'", collection_name);
                self.storage_client
                    .create_collection(collection_name, Some(DEFAULT_VECTOR_SIZE), None)
                    .await
                    .map_err(Self::map_storage_error)?;
                info!("Successfully created collection '{}'", collection_name);
                Ok(())
            }
            Err(e) => {
                error!("Failed to check collection existence: {:?}", e);
                Err(Self::map_storage_error(e))
            }
        }
    }

    /// Map storage errors to gRPC Status
    fn map_storage_error(err: StorageError) -> Status {
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
            StorageError::Connection(msg) => {
                Status::unavailable(format!("Connection error: {}", msg))
            }
            StorageError::Timeout(msg) => {
                Status::deadline_exceeded(format!("Timeout: {}", msg))
            }
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

    /// Process text ingestion: chunk, embed, and store
    async fn ingest_text_internal(
        &self,
        content: String,
        collection_name: String,
        document_id: String,
        metadata: HashMap<String, String>,
        chunk_text: bool,
    ) -> Result<IngestTextResponse, Status> {
        // Validate content is non-empty
        if content.trim().is_empty() {
            return Err(Status::invalid_argument("Content cannot be empty"));
        }

        // Ensure collection exists
        self.ensure_collection_exists(&collection_name).await?;

        // Chunk the text
        let chunks = self.chunk_text(&content, chunk_text);
        let total_chunks = chunks.len();

        debug!(
            "Chunked text into {} chunks (chunking_enabled={})",
            total_chunks, chunk_text
        );

        // Process each chunk: generate embedding and create document point
        let mut document_points = Vec::new();
        let created_at = chrono::Utc::now().to_rfc3339();

        for (chunk_content, chunk_index) in chunks {
            // Generate embedding
            let embedding = self.generate_embedding(&chunk_content);

            // Build metadata
            let mut chunk_metadata = metadata.clone();
            chunk_metadata.insert("document_id".to_string(), serde_json::json!(document_id.clone()));
            chunk_metadata.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
            chunk_metadata.insert("total_chunks".to_string(), serde_json::json!(total_chunks));
            chunk_metadata.insert("created_at".to_string(), serde_json::json!(created_at.clone()));
            chunk_metadata.insert("content".to_string(), serde_json::json!(chunk_content.clone()));

            // Create point ID: {document_id}_{chunk_index}
            let point_id = format!("{}_{}", document_id, chunk_index);

            let point = DocumentPoint {
                id: point_id,
                dense_vector: embedding,
                sparse_vector: None, // TODO: Add sparse vector support
                payload: chunk_metadata.into_iter()
                    .map(|(k, v)| (k, v))
                    .collect(),
            };

            document_points.push(point);
        }

        // Batch insert points
        info!(
            "Inserting {} chunks for document {} into collection {}",
            document_points.len(),
            document_id,
            collection_name
        );

        match self.storage_client
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
                Err(Self::map_storage_error(e))
            }
        }
    }
}

#[tonic::async_trait]
impl DocumentService for DocumentServiceImpl {
    async fn ingest_text(
        &self,
        request: Request<IngestTextRequest>,
    ) -> Result<Response<IngestTextResponse>, Status> {
        let req = request.into_inner();

        info!(
            "Ingesting text into collection basename '{}' for tenant '{}'",
            req.collection_basename, req.tenant_id
        );

        // Validate and format collection name
        let collection_name = Self::format_collection_name(
            &req.collection_basename,
            &req.tenant_id,
        )?;

        // Generate or validate document ID
        let document_id = if let Some(provided_id) = req.document_id {
            if !provided_id.is_empty() {
                Self::validate_document_id(&provided_id)?;
                provided_id
            } else {
                Uuid::new_v4().to_string()
            }
        } else {
            Uuid::new_v4().to_string()
        };

        debug!("Using document_id: {}", document_id);

        // Process ingestion
        let response = self.ingest_text_internal(
            req.content,
            collection_name,
            document_id,
            req.metadata,
            req.chunk_text,
        ).await?;

        Ok(Response::new(response))
    }

    async fn update_text(
        &self,
        request: Request<UpdateTextRequest>,
    ) -> Result<Response<UpdateTextResponse>, Status> {
        let req = request.into_inner();

        info!("Updating document '{}'", req.document_id);

        // Validate document ID
        Self::validate_document_id(&req.document_id)?;

        // Determine collection name
        let collection_name = if let Some(coll_name) = req.collection_name {
            Self::validate_collection_name(&coll_name)?;
            coll_name
        } else {
            return Err(Status::invalid_argument(
                "Collection name is required for updates"
            ));
        };

        // TODO: Delete existing chunks for this document_id
        // This requires implementing a delete-by-metadata filter in StorageClient
        // For now, we'll log a warning
        warn!(
            "UpdateText: Cannot delete existing chunks for document {} - delete_by_filter not yet implemented",
            req.document_id
        );

        // Ingest new content
        let response = self.ingest_text_internal(
            req.content,
            collection_name,
            req.document_id.clone(),
            req.metadata,
            true, // Always chunk for updates
        ).await?;

        Ok(Response::new(UpdateTextResponse {
            success: response.success,
            error_message: response.error_message,
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        }))
    }

    async fn delete_text(
        &self,
        request: Request<DeleteTextRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "Deleting document '{}' from collection '{}'",
            req.document_id, req.collection_name
        );

        // Validate inputs
        Self::validate_document_id(&req.document_id)?;
        Self::validate_collection_name(&req.collection_name)?;

        // TODO: Implement delete by metadata filter
        // This requires adding a delete_by_filter method to StorageClient
        // For now, return unimplemented
        Err(Status::unimplemented(
            "DeleteText not yet implemented - requires delete_by_filter support in StorageClient"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_collection_name() {
        // Valid names
        assert!(DocumentServiceImpl::validate_collection_name("memory_tenant1").is_ok());
        assert!(DocumentServiceImpl::validate_collection_name("scratchbook_user123").is_ok());
        assert!(DocumentServiceImpl::validate_collection_name("notes-project").is_ok());

        // Invalid: too short
        assert!(DocumentServiceImpl::validate_collection_name("ab").is_err());

        // Invalid: starts with number
        assert!(DocumentServiceImpl::validate_collection_name("1memory").is_err());

        // Invalid: special characters
        assert!(DocumentServiceImpl::validate_collection_name("memory@tenant").is_err());
        assert!(DocumentServiceImpl::validate_collection_name("memory.tenant").is_err());

        // Invalid: empty
        assert!(DocumentServiceImpl::validate_collection_name("").is_err());
    }

    #[test]
    fn test_format_collection_name() {
        // Valid formatting
        assert_eq!(
            DocumentServiceImpl::format_collection_name("memory", "tenant1").unwrap(),
            "memory_tenant1"
        );
        assert_eq!(
            DocumentServiceImpl::format_collection_name("scratchbook", "user-123").unwrap(),
            "scratchbook_user-123"
        );

        // Invalid: empty basename
        assert!(DocumentServiceImpl::format_collection_name("", "tenant1").is_err());

        // Invalid: empty tenant_id
        assert!(DocumentServiceImpl::format_collection_name("memory", "").is_err());
    }

    #[test]
    fn test_validate_document_id() {
        // Valid UUID
        let valid_uuid = Uuid::new_v4().to_string();
        assert!(DocumentServiceImpl::validate_document_id(&valid_uuid).is_ok());

        // Invalid: not a UUID
        assert!(DocumentServiceImpl::validate_document_id("not-a-uuid").is_err());
        assert!(DocumentServiceImpl::validate_document_id("12345").is_err());

        // Invalid: empty
        assert!(DocumentServiceImpl::validate_document_id("").is_err());
    }

    #[test]
    fn test_chunk_text_single_chunk() {
        let service = DocumentServiceImpl::default();
        let text = "Short text";

        // Single chunk mode
        let chunks = service.chunk_text(text, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
        assert_eq!(chunks[0].1, 0);

        // Text shorter than chunk_size
        let chunks = service.chunk_text(text, true);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
    }

    #[test]
    fn test_chunk_text_multiple_chunks() {
        let service = DocumentServiceImpl::with_config(
            Arc::new(StorageClient::new()),
            50, // Small chunk size for testing
            10, // Small overlap
        );

        let text = "This is a longer text that will be split into multiple chunks. \
                    Each chunk should overlap slightly with the previous one. \
                    This helps maintain context across chunk boundaries.";

        let chunks = service.chunk_text(text, true);

        // Should have multiple chunks
        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());

        // Verify chunk indices are sequential
        for (i, (_, index)) in chunks.iter().enumerate() {
            assert_eq!(*index, i, "Chunk index mismatch");
        }

        // Verify all chunks are non-empty
        for (content, _) in &chunks {
            assert!(!content.trim().is_empty(), "Empty chunk found");
        }
    }

    #[test]
    fn test_generate_embedding() {
        let service = DocumentServiceImpl::default();

        let text = "Test text for embedding";
        let embedding = service.generate_embedding(text);

        // Check dimensions
        assert_eq!(embedding.len(), DEFAULT_VECTOR_SIZE as usize);

        // Check all values are finite
        assert!(embedding.iter().all(|&x| x.is_finite()));

        // Check roughly normalized (magnitude near 1.0)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01, "Embedding not normalized: {}", magnitude);

        // Verify deterministic: same text produces same embedding
        let embedding2 = service.generate_embedding(text);
        assert_eq!(embedding, embedding2);

        // Verify different text produces different embedding
        let different_embedding = service.generate_embedding("Different text");
        assert_ne!(embedding, different_embedding);
    }
}
