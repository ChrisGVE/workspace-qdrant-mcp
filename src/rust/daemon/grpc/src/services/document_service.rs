//! DocumentService gRPC implementation
//!
//! Handles direct text ingestion for non-file-based content.
//! Provides 3 RPCs: IngestText, UpdateDocument, DeleteDocument
//!
//! This service is designed for user-provided text content such as:
//! - Manual notes and annotations
//! - Chat snippets and conversations
//! - Scraped web content
//! - API responses
//! - Any text not originating from files
//!
//! ## Multi-Tenant Routing (Task 406)
//!
//! Routes content to unified collections based on collection_basename:
//! - `memory`, `agent_memory` → Direct collection names (no multi-tenant)
//! - Other basenames → Routes to canonical `projects` or `libraries`:
//!   - If tenant_id is project ID format → `projects` with project_id metadata
//!     (path hashes like "path_abc123..." or sanitized URLs like "github_com_user_repo")
//!   - Otherwise → `libraries` with library_name metadata
//!     (human-readable names like "react", "numpy", "lodash")

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};
use uuid::Uuid;
use wqm_common::timestamps;
use workspace_qdrant_core::storage::{StorageClient, DocumentPoint, StorageError};
use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES};
use workspace_qdrant_core::BM25;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::RwLock as TokioRwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Global embedding model instance (lazy-initialized, thread-safe)
/// Uses Mutex because TextEmbedding is not Send+Sync
static EMBEDDING_MODEL: OnceLock<TokioMutex<TextEmbedding>> = OnceLock::new();

/// Global embedding cache (content hash → embedding vector)
/// Improves performance by caching embeddings for repeated content
static EMBEDDING_CACHE: OnceLock<TokioMutex<LruCache<u64, Vec<f32>>>> = OnceLock::new();

/// Global BM25 instance for sparse vector generation (thread-safe, read-write lock)
/// Uses RwLock because reads (generate_sparse_vector) are frequent and concurrent,
/// while writes (add_document) are less frequent during ingestion
static BM25_MODEL: OnceLock<TokioRwLock<BM25>> = OnceLock::new();

/// Default cache size (number of entries)
const DEFAULT_CACHE_SIZE: usize = 1000;

/// Default BM25 parameters (standard values from research)
const DEFAULT_BM25_K1: f32 = 1.2;

/// Cache metrics for monitoring
pub struct EmbeddingCacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
}

impl EmbeddingCacheMetrics {
    pub const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

/// Global cache metrics instance
pub static CACHE_METRICS: EmbeddingCacheMetrics = EmbeddingCacheMetrics::new();

use crate::proto::{
    document_service_server::DocumentService,
    IngestTextRequest, IngestTextResponse,
    UpdateDocumentRequest, UpdateDocumentResponse,
    DeleteDocumentRequest,
};

/// Default chunk size in characters for text chunking
const DEFAULT_CHUNK_SIZE: usize = 1000;
/// Default overlap size in characters for sliding window chunking
const DEFAULT_CHUNK_OVERLAP: usize = 200;
/// Default vector dimension for embeddings (all-MiniLM-L6-v2)
const DEFAULT_VECTOR_SIZE: u64 = 384;

/// DocumentService implementation with text chunking and embedding generation
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

    /// Check if tenant_id is a project ID format
    ///
    /// Project IDs are generated from:
    /// 1. Git remote URLs (sanitized): e.g., "github_com_user_repo"
    /// 2. Path hashes: e.g., "path_abc123def456789a" (21 chars: "path_" + 16 hex)
    ///
    /// Library names are human-readable without these patterns: e.g., "react", "numpy"
    fn is_project_id(tenant_id: &str) -> bool {
        // Path hash format: "path_" + 16 hex characters = 21 chars total
        if tenant_id.starts_with("path_") && tenant_id.len() == 21 {
            let hash_part = &tenant_id[5..];
            if hash_part.chars().all(|c| c.is_ascii_hexdigit()) {
                return true;
            }
        }

        // Sanitized git remote URLs contain domain patterns
        // Common patterns: github_com_, gitlab_com_, bitbucket_org_, codeberg_org_, etc.
        let domain_patterns = [
            "github_com_",
            "gitlab_com_",
            "bitbucket_org_",
            "codeberg_org_",
            "sr_ht_",  // sourcehut
            "git_",    // generic git server pattern
        ];

        for pattern in domain_patterns {
            if tenant_id.starts_with(pattern) {
                return true;
            }
        }

        // Also match pattern: domain_tld_user_repo (contains at least 3 underscores)
        // This handles custom git servers like "myserver_com_user_repo"
        let underscore_count = tenant_id.chars().filter(|c| *c == '_').count();
        if underscore_count >= 3 && tenant_id.contains("_com_") {
            return true;
        }

        false
    }

    /// Determine the target collection and tenant metadata for multi-tenant routing
    ///
    /// Routing logic:
    /// - `memory`, `agent_memory` → Direct collection name (no multi-tenant)
    /// - Other basenames:
    ///   - If tenant_id looks like project ID → `projects` with project_id
    ///     (path hash or sanitized git URL pattern)
    ///   - Otherwise → `libraries` with library_name
    ///     (human-readable names like "react", "numpy")
    ///
    /// Returns: (collection_name, tenant_type, tenant_value)
    /// - tenant_type: "project_id" or "library_name"
    fn validate_collection_basename(basename: &str) -> Result<(), Status> {
        if basename.is_empty() {
            return Err(Status::invalid_argument("Collection basename cannot be empty"));
        }
        if basename.len() < 3 {
            return Err(Status::invalid_argument(
                "Collection basename must be at least 3 characters",
            ));
        }
        if basename.len() > 255 {
            return Err(Status::invalid_argument(
                "Collection basename must not exceed 255 characters",
            ));
        }
        if basename.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
            return Err(Status::invalid_argument(
                "Collection basename cannot start with a number",
            ));
        }
        if !basename.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(Status::invalid_argument(
                "Collection basename can only contain alphanumeric characters, underscores, and hyphens",
            ));
        }
        Ok(())
    }

    fn determine_collection_routing(
        basename: &str,
        tenant_id: &str,
    ) -> Result<(String, String, String), Status> {
        // Validate basename format
        Self::validate_collection_basename(basename)?;

        if tenant_id.is_empty() {
            return Err(Status::invalid_argument("Tenant ID cannot be empty"));
        }

        // Memory collection uses single canonical name with tenant isolation via metadata
        // Per ADR-001: All memory items go to `memory` collection, filtered by project_id
        // NOTE: `agent_memory` is deprecated - route to canonical `memory` collection
        if basename == "memory" || basename == "agent_memory" {
            // Route to canonical `memory` collection (not per-tenant collections)
            // Tenant isolation achieved via project_id in metadata payload
            return Ok(("memory".to_string(), "project_id".to_string(), tenant_id.to_string()));
        }

        // Multi-tenant routing based on tenant_id format
        if Self::is_project_id(tenant_id) {
            // 12-char hex = project_id → route to canonical `projects` collection
            Ok((
                COLLECTION_PROJECTS.to_string(),
                "project_id".to_string(),
                tenant_id.to_string(),
            ))
        } else {
            // Non-hex = library_name → route to canonical `libraries` collection
            Ok((
                COLLECTION_LIBRARIES.to_string(),
                "library_name".to_string(),
                tenant_id.to_string(),
            ))
        }
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

    /// Initialize the global embedding model, cache, and BM25 if not already initialized
    fn init_embedding_model() -> Result<(), Status> {
        // Initialize the embedding model
        EMBEDDING_MODEL.get_or_init(|| {
            info!("Initializing FastEmbed model (all-MiniLM-L6-v2)...");
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(true)
            ).expect("Failed to initialize FastEmbed model");
            info!("FastEmbed model initialized successfully");
            TokioMutex::new(model)
        });

        // Initialize the embedding cache
        EMBEDDING_CACHE.get_or_init(|| {
            let cache_size = NonZeroUsize::new(DEFAULT_CACHE_SIZE)
                .expect("Cache size must be non-zero");
            info!("Initializing embedding cache with {} entries", DEFAULT_CACHE_SIZE);
            TokioMutex::new(LruCache::new(cache_size))
        });

        // Initialize BM25 for sparse vector generation
        BM25_MODEL.get_or_init(|| {
            info!("Initializing BM25 model (k1={})...", DEFAULT_BM25_K1);
            TokioRwLock::new(BM25::new(DEFAULT_BM25_K1))
        });

        Ok(())
    }

    /// Compute a hash of the input text for cache lookup
    fn content_hash(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Simple tokenization for BM25 sparse vector generation
    fn tokenize(text: &str) -> Vec<String> {
        wqm_common::nlp::tokenize(text)
    }

    /// Generate sparse vector using BM25 algorithm
    /// First adds document to corpus, then generates sparse vector
    async fn generate_sparse_vector(&self, text: &str) -> Result<HashMap<u32, f32>, Status> {
        Self::init_embedding_model()?;

        let bm25 = BM25_MODEL.get()
            .ok_or_else(|| Status::internal("BM25 model not initialized"))?;

        let tokens = Self::tokenize(text);

        if tokens.is_empty() {
            debug!("No tokens for sparse vector generation, returning empty");
            return Ok(HashMap::new());
        }

        // Add document to corpus and generate sparse vector
        let sparse_map: HashMap<u32, f32> = {
            let mut bm25_guard = bm25.write().await;

            // Add document to corpus for IDF calculation
            bm25_guard.add_document(&tokens);

            // Generate sparse vector
            let sparse = bm25_guard.generate_sparse_vector(&tokens);

            // Convert to HashMap<u32, f32>
            sparse.indices.into_iter()
                .zip(sparse.values.into_iter())
                .collect()
        };

        debug!("Generated sparse vector with {} non-zero entries", sparse_map.len());
        Ok(sparse_map)
    }

    /// Generate embedding for text using FastEmbed (all-MiniLM-L6-v2)
    /// Returns 384-dimensional dense vector for semantic search
    /// Uses LRU cache to avoid redundant computation for repeated content
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, Status> {
        // Ensure model and cache are initialized
        Self::init_embedding_model()?;

        // Check cache first
        let content_hash = Self::content_hash(text);
        if let Some(cache) = EMBEDDING_CACHE.get() {
            let mut cache_guard = cache.lock().await;
            if let Some(cached_embedding) = cache_guard.get(&content_hash) {
                CACHE_METRICS.hits.fetch_add(1, Ordering::Relaxed);
                debug!("Cache hit for content hash {}", content_hash);
                return Ok(cached_embedding.clone());
            }
        }

        // Cache miss - generate embedding
        CACHE_METRICS.misses.fetch_add(1, Ordering::Relaxed);

        let model = EMBEDDING_MODEL.get()
            .ok_or_else(|| Status::internal("Embedding model not initialized"))?;

        // Clone text for blocking task
        let text_owned = text.to_string();

        // Acquire lock and generate embedding
        // FastEmbed is CPU-bound, so we use spawn_blocking
        let embedding = {
            let mut model_guard = model.lock().await;

            // Prepare document for embedding
            let documents = vec![text_owned.as_str()];

            // Generate embedding (synchronous operation)
            // Using tokio's spawn_blocking would require moving the MutexGuard which isn't possible
            // So we perform the CPU work directly and rely on the tokio::sync::Mutex
            match model_guard.embed(documents, None) {
                Ok(embeddings) => {
                    if embeddings.is_empty() {
                        return Err(Status::internal("FastEmbed returned empty embeddings"));
                    }
                    // FastEmbed returns Vec<Vec<f32>>, we want the first one
                    embeddings.into_iter().next()
                        .ok_or_else(|| Status::internal("FastEmbed returned no embeddings"))?
                }
                Err(e) => {
                    error!("FastEmbed embedding generation failed: {:?}", e);
                    return Err(Status::internal(format!(
                        "Embedding generation failed: {}", e
                    )));
                }
            }
        };

        // Verify dimension matches expected
        if embedding.len() != DEFAULT_VECTOR_SIZE as usize {
            warn!(
                "Embedding dimension mismatch: expected {}, got {}",
                DEFAULT_VECTOR_SIZE, embedding.len()
            );
        }

        // Store in cache
        if let Some(cache) = EMBEDDING_CACHE.get() {
            let mut cache_guard = cache.lock().await;
            // Check if cache is at capacity (LRU will auto-evict, but we track it)
            if cache_guard.len() >= DEFAULT_CACHE_SIZE {
                CACHE_METRICS.evictions.fetch_add(1, Ordering::Relaxed);
            }
            cache_guard.put(content_hash, embedding.clone());
        }

        debug!("Generated {}-dimensional embedding (cached)", embedding.len());
        Ok(embedding)
    }

    /// Get embedding cache metrics for monitoring
    pub fn get_cache_metrics() -> (u64, u64, u64, f64) {
        let hits = CACHE_METRICS.hits.load(Ordering::Relaxed);
        let misses = CACHE_METRICS.misses.load(Ordering::Relaxed);
        let evictions = CACHE_METRICS.evictions.load(Ordering::Relaxed);
        let hit_rate = CACHE_METRICS.hit_rate();
        (hits, misses, evictions, hit_rate)
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
        let created_at = timestamps::now_utc();

        for (chunk_content, chunk_index) in chunks {
            // Generate dense embedding using FastEmbed
            let dense_embedding = self.generate_embedding(&chunk_content).await?;

            // Generate sparse vector using BM25
            let sparse_vector = self.generate_sparse_vector(&chunk_content).await?;
            let sparse_option = if sparse_vector.is_empty() {
                None
            } else {
                Some(sparse_vector)
            };

            // Build metadata - convert HashMap<String, String> to HashMap<String, Value>
            let mut chunk_metadata: HashMap<String, serde_json::Value> = metadata.iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect();

            chunk_metadata.insert("document_id".to_string(), serde_json::json!(document_id.clone()));
            chunk_metadata.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
            chunk_metadata.insert("total_chunks".to_string(), serde_json::json!(total_chunks));
            chunk_metadata.insert("created_at".to_string(), serde_json::json!(created_at.clone()));
            chunk_metadata.insert("content".to_string(), serde_json::json!(chunk_content.clone()));

            // Create deterministic point ID using UUID v5
            // Namespace: document_id parsed as UUID
            // Name: chunk_index as string
            // Result: Valid UUID that's unique per document+chunk combination
            let namespace = Uuid::parse_str(&document_id)
                .unwrap_or_else(|_| Uuid::new_v4()); // Fallback if document_id isn't valid UUID
            let point_id = Uuid::new_v5(&namespace, chunk_index.to_string().as_bytes()).to_string();

            let point = DocumentPoint {
                id: point_id,
                dense_vector: dense_embedding,
                sparse_vector: sparse_option,
                payload: chunk_metadata,
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

        // Determine multi-tenant collection routing
        let (collection_name, tenant_type, tenant_value) = Self::determine_collection_routing(
            &req.collection_basename,
            &req.tenant_id,
        )?;

        info!(
            "Multi-tenant routing: collection='{}', {}='{}'",
            collection_name, tenant_type, tenant_value
        );

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

        // Enrich metadata with tenant information
        let mut enriched_metadata = req.metadata.clone();
        enriched_metadata.insert(tenant_type.clone(), tenant_value.clone());

        // Add collection_basename for filtering within unified collections
        enriched_metadata.insert("collection_basename".to_string(), req.collection_basename.clone());

        // Process ingestion with enriched metadata
        let response = self.ingest_text_internal(
            req.content,
            collection_name,
            document_id,
            enriched_metadata,
            req.chunk_text,
        ).await?;

        Ok(Response::new(response))
    }

    async fn update_document(
        &self,
        request: Request<UpdateDocumentRequest>,
    ) -> Result<Response<UpdateDocumentResponse>, Status> {
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

        info!(
            "UpdateDocument: collection='{}', document='{}'",
            collection_name, req.document_id
        );

        // Delete existing chunks for this document_id before re-ingesting
        match self.storage_client
            .delete_points_by_document_id(&collection_name, &req.document_id)
            .await
        {
            Ok(_) => {
                info!(
                    "Deleted existing chunks for document '{}' in '{}'",
                    req.document_id, collection_name
                );
            }
            Err(err) => {
                // Log but continue - the document may not have existed before
                warn!(
                    "Failed to delete existing chunks for document '{}': {:?}",
                    req.document_id, err
                );
            }
        }

        // Add updated_at to metadata
        let mut enriched_metadata = req.metadata.clone();
        enriched_metadata.insert("updated_at".to_string(), timestamps::now_utc());

        // Re-ingest new content
        let response = self.ingest_text_internal(
            req.content,
            collection_name,
            req.document_id.clone(),
            enriched_metadata,
            true, // Always chunk for updates
        ).await?;

        Ok(Response::new(UpdateDocumentResponse {
            success: response.success,
            error_message: response.error_message,
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        }))
    }

    async fn delete_document(
        &self,
        request: Request<DeleteDocumentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "DeleteDocument: document='{}', collection='{}'",
            req.document_id, req.collection_name
        );

        // Validate inputs
        Self::validate_document_id(&req.document_id)?;
        Self::validate_collection_name(&req.collection_name)?;

        // Log if targeting a unified collection
        if req.collection_name == COLLECTION_PROJECTS
            || req.collection_name == COLLECTION_LIBRARIES
        {
            debug!(
                "Deleting from unified collection '{}' - document_id filter will be used",
                req.collection_name
            );
        }

        // Check collection exists
        match self.storage_client.collection_exists(&req.collection_name).await {
            Ok(false) => {
                return Err(Status::not_found(format!(
                    "Collection '{}' does not exist", req.collection_name
                )));
            }
            Err(err) => {
                error!("Failed to check collection existence: {:?}", err);
                return Err(Status::unavailable(format!(
                    "Failed to check collection: {}", err
                )));
            }
            _ => {}
        }

        // Delete all points with matching document_id
        match self.storage_client
            .delete_points_by_document_id(&req.collection_name, &req.document_id)
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully deleted document '{}' from '{}'",
                    req.document_id, req.collection_name
                );
                Ok(Response::new(()))
            }
            Err(err) => {
                error!(
                    "Failed to delete document '{}' from '{}': {:?}",
                    req.document_id, req.collection_name, err
                );
                Err(Status::internal(format!(
                    "Failed to delete document: {}", err
                )))
            }
        }
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
        // Legacy underscore-prefixed names are syntactically valid (migration compatibility)
        assert!(DocumentServiceImpl::validate_collection_name("_projects").is_ok());
        assert!(DocumentServiceImpl::validate_collection_name("_libraries").is_ok());

        // Canonical collection names (ADR-001)
        assert!(DocumentServiceImpl::validate_collection_name("projects").is_ok());
        assert!(DocumentServiceImpl::validate_collection_name("libraries").is_ok());
        assert!(DocumentServiceImpl::validate_collection_name("memory").is_ok());

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
    fn test_is_project_id() {
        // Path hash format: "path_" + 16 hex chars = 21 chars
        assert!(DocumentServiceImpl::is_project_id("path_a1b2c3d4e5f6789a")); // exact 21 chars
        assert!(DocumentServiceImpl::is_project_id("path_0000000000000000"));
        assert!(DocumentServiceImpl::is_project_id("path_ffffffffffffffff"));

        // Sanitized git remote URLs (common patterns)
        assert!(DocumentServiceImpl::is_project_id("github_com_user_repo"));
        assert!(DocumentServiceImpl::is_project_id("github_com_anthropics_claude_code"));
        assert!(DocumentServiceImpl::is_project_id("gitlab_com_org_project"));
        assert!(DocumentServiceImpl::is_project_id("bitbucket_org_team_repo"));
        assert!(DocumentServiceImpl::is_project_id("codeberg_org_user_project"));
        assert!(DocumentServiceImpl::is_project_id("sr_ht_user_repo"));  // sourcehut
        assert!(DocumentServiceImpl::is_project_id("git_myserver_com_repo"));  // custom git server

        // Custom domains with _com_ pattern
        assert!(DocumentServiceImpl::is_project_id("mycompany_com_team_project"));

        // Invalid path hash formats
        assert!(!DocumentServiceImpl::is_project_id("path_a1b2c3d4e5f6789")); // 20 chars (too short)
        assert!(!DocumentServiceImpl::is_project_id("path_a1b2c3d4e5f6789ab")); // 22 chars (too long)
        assert!(!DocumentServiceImpl::is_project_id("path_ghijklmnopqrstuv")); // non-hex
        assert!(!DocumentServiceImpl::is_project_id("paths_a1b2c3d4e5f6789a")); // wrong prefix

        // Library names (should not match)
        assert!(!DocumentServiceImpl::is_project_id("langchain"));
        assert!(!DocumentServiceImpl::is_project_id("react"));
        assert!(!DocumentServiceImpl::is_project_id("react-docs"));
        assert!(!DocumentServiceImpl::is_project_id("numpy"));
        assert!(!DocumentServiceImpl::is_project_id("lodash"));
        assert!(!DocumentServiceImpl::is_project_id("tensorflow_keras")); // only 1 underscore

        // Edge cases
        assert!(!DocumentServiceImpl::is_project_id("")); // empty
        assert!(!DocumentServiceImpl::is_project_id("path_")); // just prefix
    }

    #[test]
    fn test_determine_collection_routing_memory() {
        // Memory routes to canonical `memory` collection with tenant in metadata
        let result = DocumentServiceImpl::determine_collection_routing("memory", "github_com_user_repo");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        // All memory items go to single canonical `memory` collection
        assert_eq!(collection, "memory");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "github_com_user_repo");

        // agent_memory is deprecated but routes to canonical `memory` collection
        let result = DocumentServiceImpl::determine_collection_routing("agent_memory", "path_a1b2c3d4e5f6789a");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        // Deprecated agent_memory also routes to canonical `memory`
        assert_eq!(collection, "memory");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "path_a1b2c3d4e5f6789a");
    }

    #[test]
    fn test_determine_collection_routing_projects() {
        // Project routing: path hash tenant_id → projects
        let result = DocumentServiceImpl::determine_collection_routing("notes", "path_a1b2c3d4e5f6789a");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "path_a1b2c3d4e5f6789a");

        // Project routing: sanitized git URL tenant_id → projects
        let result = DocumentServiceImpl::determine_collection_routing("code", "github_com_anthropics_claude_code");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "github_com_anthropics_claude_code");

        // Project routing: GitLab URL → projects
        let result = DocumentServiceImpl::determine_collection_routing("src", "gitlab_com_org_project");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "gitlab_com_org_project");
    }

    #[test]
    fn test_determine_collection_routing_libraries() {
        // Library routing: human-readable library name → libraries
        let result = DocumentServiceImpl::determine_collection_routing("docs", "langchain");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "langchain");

        // Another library example
        let result = DocumentServiceImpl::determine_collection_routing("reference", "react");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "react");

        // Library with hyphen
        let result = DocumentServiceImpl::determine_collection_routing("api", "react-native");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "react-native");
    }

    #[test]
    fn test_determine_collection_routing_validation() {
        // Empty basename
        let result = DocumentServiceImpl::determine_collection_routing("", "tenant123");
        assert!(result.is_err());

        // Empty tenant_id
        let result = DocumentServiceImpl::determine_collection_routing("notes", "");
        assert!(result.is_err());

        // Both empty
        let result = DocumentServiceImpl::determine_collection_routing("", "");
        assert!(result.is_err());
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

    #[tokio::test]
    async fn test_generate_embedding() {
        let service = DocumentServiceImpl::default();

        let text = "Test text for embedding";
        let embedding = service.generate_embedding(text).await
            .expect("Failed to generate embedding");

        // Check dimensions (all-MiniLM-L6-v2 produces 384-dimensional vectors)
        assert_eq!(embedding.len(), DEFAULT_VECTOR_SIZE as usize);

        // Check all values are finite
        assert!(embedding.iter().all(|&x| x.is_finite()));

        // FastEmbed embeddings are normalized by the model
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Allow some tolerance since model normalization may vary slightly
        assert!((magnitude - 1.0).abs() < 0.1, "Embedding not normalized: {}", magnitude);

        // Note: FastEmbed is deterministic for the same input
        let embedding2 = service.generate_embedding(text).await
            .expect("Failed to generate second embedding");
        assert_eq!(embedding, embedding2, "Same text should produce same embedding");

        // Verify different text produces different embedding
        let different_embedding = service.generate_embedding("Different text for comparison").await
            .expect("Failed to generate different embedding");
        assert_ne!(embedding, different_embedding, "Different text should produce different embedding");
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let service = DocumentServiceImpl::default();

        // Reset cache metrics for this test
        CACHE_METRICS.hits.store(0, Ordering::Relaxed);
        CACHE_METRICS.misses.store(0, Ordering::Relaxed);
        CACHE_METRICS.evictions.store(0, Ordering::Relaxed);

        // First call should be a cache miss
        let text = "Text for cache testing";
        let embedding1 = service.generate_embedding(text).await
            .expect("Failed to generate embedding");

        // Second call with same text should be a cache hit
        let embedding2 = service.generate_embedding(text).await
            .expect("Failed to generate cached embedding");

        // Embeddings should be identical
        assert_eq!(embedding1, embedding2, "Cached embedding should match original");

        // Verify cache metrics
        let (hits, misses, _evictions, _hit_rate) = DocumentServiceImpl::get_cache_metrics();
        assert!(misses >= 1, "Expected at least 1 cache miss, got {}", misses);
        assert!(hits >= 1, "Expected at least 1 cache hit, got {}", hits);

        // Different text should produce cache miss
        let _embedding3 = service.generate_embedding("Different unique text").await
            .expect("Failed to generate different embedding");

        let (_hits2, misses2, _, _) = DocumentServiceImpl::get_cache_metrics();
        assert!(misses2 > misses, "Expected additional cache miss for different text");
    }

    #[test]
    fn test_content_hash() {
        // Same content should produce same hash
        let hash1 = DocumentServiceImpl::content_hash("test content");
        let hash2 = DocumentServiceImpl::content_hash("test content");
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        let hash3 = DocumentServiceImpl::content_hash("different content");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_tokenize() {
        // Basic tokenization
        let tokens = DocumentServiceImpl::tokenize("Hello world test");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));

        // Stopwords should be filtered
        let tokens2 = DocumentServiceImpl::tokenize("the quick brown fox and the lazy dog");
        assert!(!tokens2.contains(&"the".to_string()));
        assert!(!tokens2.contains(&"and".to_string()));
        assert!(tokens2.contains(&"quick".to_string()));
        assert!(tokens2.contains(&"brown".to_string()));
        assert!(tokens2.contains(&"fox".to_string()));
        assert!(tokens2.contains(&"lazy".to_string()));
        assert!(tokens2.contains(&"dog".to_string()));

        // Single-character tokens should be filtered
        let tokens3 = DocumentServiceImpl::tokenize("a b c test word");
        assert!(!tokens3.contains(&"a".to_string()));
        assert!(!tokens3.contains(&"b".to_string()));
        assert!(tokens3.contains(&"test".to_string()));
        assert!(tokens3.contains(&"word".to_string()));

        // Punctuation handling
        let tokens4 = DocumentServiceImpl::tokenize("hello, world! test-case");
        assert!(tokens4.contains(&"hello".to_string()));
        assert!(tokens4.contains(&"world".to_string()));
        assert!(tokens4.contains(&"test".to_string()));
        assert!(tokens4.contains(&"case".to_string()));
    }

    #[tokio::test]
    async fn test_generate_sparse_vector() {
        let service = DocumentServiceImpl::default();

        // BM25 needs multiple documents to calculate meaningful IDF scores
        // First document: add to corpus (IDF will be 0 for all terms)
        let doc1 = "machine learning algorithms for natural language processing";
        let _sparse1 = service.generate_sparse_vector(doc1).await
            .expect("Failed to add first document");

        // Second document: should now have meaningful sparse vectors
        // Some terms overlap (triggers IDF calculation), some are unique
        let doc2 = "deep learning neural networks for image classification";
        let sparse2 = service.generate_sparse_vector(doc2).await
            .expect("Failed to generate sparse vector");

        // With 2 documents, terms that appear in only one document should have positive IDF
        // The term "learning" appears in both, so it should have lower weight
        // Terms like "deep", "neural", "image" only appear in doc2

        // Third document: should have non-empty sparse vector
        let doc3 = "reinforcement learning algorithms for robotics control";
        let sparse3 = service.generate_sparse_vector(doc3).await
            .expect("Failed to generate third sparse vector");

        // After 3 documents, we should see meaningful sparse vectors
        // Terms unique to doc3 like "reinforcement", "robotics", "control" should have weight
        // Note: With small corpus, some terms may still have 0 IDF

        // Different texts should produce different sparse vectors (when non-empty)
        if !sparse2.is_empty() && !sparse3.is_empty() {
            assert_ne!(sparse2, sparse3, "Different documents should produce different sparse vectors");
        }

        // Verify values are non-negative when present
        for &value in sparse2.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
        for &value in sparse3.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
    }

    #[tokio::test]
    async fn test_sparse_vector_empty_input() {
        let service = DocumentServiceImpl::default();

        // Empty text should return empty sparse vector
        let sparse_vector = service.generate_sparse_vector("").await
            .expect("Failed with empty input");
        assert!(sparse_vector.is_empty(), "Empty text should produce empty sparse vector");

        // Text with only stopwords should return empty sparse vector
        let stopword_only = service.generate_sparse_vector("the and is a").await
            .expect("Failed with stopwords only");
        assert!(stopword_only.is_empty(), "Stopwords-only text should produce empty sparse vector");
    }
}
