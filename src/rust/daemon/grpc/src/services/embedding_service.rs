//! EmbeddingService gRPC implementation
//!
//! Provides embedding generation for TypeScript MCP server.
//! Exposes 2 RPCs: EmbedText (dense embedding), GenerateSparseVector (BM25 sparse vector).
//!
//! This service centralizes embedding generation in the daemon, allowing the TypeScript
//! MCP server to use the same FastEmbed model as the Rust processing pipeline.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tonic::{Request, Response, Status};
use tracing::{debug, info, error};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::RwLock as TokioRwLock;
use lru::LruCache;
use std::num::NonZeroUsize;
use workspace_qdrant_core::BM25;

use crate::proto::{
    embedding_service_server::EmbeddingService,
    EmbedTextRequest, EmbedTextResponse,
    SparseVectorRequest, SparseVectorResponse,
};

/// Global embedding model instance (lazy-initialized, thread-safe)
static EMBEDDING_MODEL: OnceLock<TokioMutex<TextEmbedding>> = OnceLock::new();

/// Global embedding cache (content hash → embedding vector)
static EMBEDDING_CACHE: OnceLock<TokioMutex<LruCache<u64, Vec<f32>>>> = OnceLock::new();

/// Global BM25 instance for sparse vector generation
static BM25_MODEL: OnceLock<TokioRwLock<BM25>> = OnceLock::new();

/// Default cache size (number of entries)
const DEFAULT_CACHE_SIZE: usize = 1000;

/// Default BM25 parameters
const DEFAULT_BM25_K1: f32 = 1.2;

/// Default vector dimension for all-MiniLM-L6-v2
const DEFAULT_VECTOR_SIZE: i32 = 384;

/// Default model name
const DEFAULT_MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// Cache metrics for monitoring
pub struct EmbeddingCacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl EmbeddingCacheMetrics {
    pub const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

/// Global cache metrics instance
pub static CACHE_METRICS: EmbeddingCacheMetrics = EmbeddingCacheMetrics::new();

/// EmbeddingService implementation
pub struct EmbeddingServiceImpl;

impl EmbeddingServiceImpl {
    /// Create a new EmbeddingService
    pub fn new() -> Self {
        Self
    }

    /// Initialize the global embedding model, cache, and BM25 if not already initialized
    fn init_embedding_model() -> Result<(), Status> {
        // Initialize the embedding model
        EMBEDDING_MODEL.get_or_init(|| {
            info!("Initializing FastEmbed model ({})...", DEFAULT_MODEL_NAME);
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
        // Common English stopwords
        const STOPWORDS: &[&str] = &[
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
            "the", "to", "was", "were", "will", "with", "this", "but", "they",
            "have", "had", "what", "when", "where", "who", "which", "why", "how"
        ];

        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty() && s.len() > 1)
            .filter(|s| !STOPWORDS.contains(s))
            .map(|s| s.to_string())
            .collect()
    }

    /// Generate dense embedding using FastEmbed
    async fn generate_embedding_internal(&self, text: &str) -> Result<Vec<f32>, Status> {
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

        let text_owned = text.to_string();

        // Acquire lock and generate embedding
        let embedding = {
            let mut model_guard = model.lock().await;
            let documents = vec![text_owned.as_str()];

            match model_guard.embed(documents, None) {
                Ok(embeddings) => {
                    if embeddings.is_empty() {
                        return Err(Status::internal("FastEmbed returned empty embeddings"));
                    }
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

        // Store in cache
        if let Some(cache) = EMBEDDING_CACHE.get() {
            let mut cache_guard = cache.lock().await;
            cache_guard.put(content_hash, embedding.clone());
        }

        debug!("Generated {}-dimensional embedding", embedding.len());
        Ok(embedding)
    }

    /// Generate sparse vector using BM25
    async fn generate_sparse_vector_internal(&self, text: &str) -> Result<HashMap<u32, f32>, Status> {
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
}

impl Default for EmbeddingServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed_text(
        &self,
        request: Request<EmbedTextRequest>,
    ) -> Result<Response<EmbedTextResponse>, Status> {
        let req = request.into_inner();

        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("Text cannot be empty"));
        }

        // Model parameter is ignored for now - only all-MiniLM-L6-v2 supported
        let model_name = req.model.unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
        if model_name != DEFAULT_MODEL_NAME {
            debug!("Requested model '{}' not available, using default '{}'", model_name, DEFAULT_MODEL_NAME);
        }

        info!("EmbedText: generating embedding for text of {} chars", req.text.len());

        match self.generate_embedding_internal(&req.text).await {
            Ok(embedding) => {
                let dimensions = embedding.len() as i32;

                // Verify dimension matches expected model output
                if dimensions != DEFAULT_VECTOR_SIZE {
                    tracing::warn!(
                        "Embedding dimension mismatch: expected {}, got {}",
                        DEFAULT_VECTOR_SIZE, dimensions
                    );
                }

                Ok(Response::new(EmbedTextResponse {
                    embedding,
                    dimensions,
                    model_name: DEFAULT_MODEL_NAME.to_string(),
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!("EmbedText failed: {:?}", e);
                Ok(Response::new(EmbedTextResponse {
                    embedding: vec![],
                    dimensions: 0,
                    model_name: DEFAULT_MODEL_NAME.to_string(),
                    success: false,
                    error_message: e.message().to_string(),
                }))
            }
        }
    }

    async fn generate_sparse_vector(
        &self,
        request: Request<SparseVectorRequest>,
    ) -> Result<Response<SparseVectorResponse>, Status> {
        let req = request.into_inner();

        if req.text.trim().is_empty() {
            return Ok(Response::new(SparseVectorResponse {
                indices_values: HashMap::new(),
                vocab_size: 0,
                success: true,
                error_message: String::new(),
            }));
        }

        info!("GenerateSparseVector: processing text of {} chars", req.text.len());

        match self.generate_sparse_vector_internal(&req.text).await {
            Ok(sparse_map) => {
                // Get vocabulary size
                let vocab_size = if let Some(bm25) = BM25_MODEL.get() {
                    let bm25_guard = bm25.read().await;
                    bm25_guard.vocab_size() as i32
                } else {
                    0
                };

                Ok(Response::new(SparseVectorResponse {
                    indices_values: sparse_map,
                    vocab_size,
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!("GenerateSparseVector failed: {:?}", e);
                Ok(Response::new(SparseVectorResponse {
                    indices_values: HashMap::new(),
                    vocab_size: 0,
                    success: false,
                    error_message: e.message().to_string(),
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        // Basic tokenization
        let tokens = EmbeddingServiceImpl::tokenize("Hello world test");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));

        // Stopwords should be filtered
        let tokens2 = EmbeddingServiceImpl::tokenize("the quick brown fox and the lazy dog");
        assert!(!tokens2.contains(&"the".to_string()));
        assert!(!tokens2.contains(&"and".to_string()));
        assert!(tokens2.contains(&"quick".to_string()));
        assert!(tokens2.contains(&"brown".to_string()));
    }

    #[test]
    fn test_content_hash() {
        // Same content should produce same hash
        let hash1 = EmbeddingServiceImpl::content_hash("test content");
        let hash2 = EmbeddingServiceImpl::content_hash("test content");
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        let hash3 = EmbeddingServiceImpl::content_hash("different content");
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_embed_text() {
        let service = EmbeddingServiceImpl::new();

        let request = Request::new(EmbedTextRequest {
            text: "Test embedding generation".to_string(),
            model: None,
        });

        let response = service.embed_text(request).await.expect("Failed to embed text");
        let resp = response.into_inner();

        assert!(resp.success);
        assert_eq!(resp.dimensions, DEFAULT_VECTOR_SIZE);
        assert_eq!(resp.embedding.len(), DEFAULT_VECTOR_SIZE as usize);
        assert_eq!(resp.model_name, DEFAULT_MODEL_NAME);
        assert!(resp.error_message.is_empty());

        // Check embeddings are normalized (FastEmbed normalizes by default)
        let magnitude: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1, "Embedding not normalized: {}", magnitude);
    }

    #[tokio::test]
    async fn test_embed_text_empty_input() {
        let service = EmbeddingServiceImpl::new();

        let request = Request::new(EmbedTextRequest {
            text: "".to_string(),
            model: None,
        });

        let result = service.embed_text(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().message().contains("empty"));
    }

    #[tokio::test]
    async fn test_generate_sparse_vector() {
        let service = EmbeddingServiceImpl::new();

        // First add some documents to build vocabulary
        let request1 = Request::new(SparseVectorRequest {
            text: "machine learning algorithms for natural language processing".to_string(),
        });
        let _ = service.generate_sparse_vector(request1).await.expect("Failed");

        // Now generate sparse vector for another document
        let request2 = Request::new(SparseVectorRequest {
            text: "deep learning neural networks for image classification".to_string(),
        });
        let response = service.generate_sparse_vector(request2).await.expect("Failed");
        let resp = response.into_inner();

        assert!(resp.success);
        assert!(resp.vocab_size > 0);
        assert!(resp.error_message.is_empty());

        // Verify all values are non-negative
        for &value in resp.indices_values.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
    }

    #[tokio::test]
    async fn test_generate_sparse_vector_empty_input() {
        let service = EmbeddingServiceImpl::new();

        let request = Request::new(SparseVectorRequest {
            text: "".to_string(),
        });

        let response = service.generate_sparse_vector(request).await.expect("Should succeed");
        let resp = response.into_inner();

        assert!(resp.success);
        assert!(resp.indices_values.is_empty());
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let service = EmbeddingServiceImpl::new();

        // Reset cache metrics
        CACHE_METRICS.hits.store(0, Ordering::Relaxed);
        CACHE_METRICS.misses.store(0, Ordering::Relaxed);

        let text = "Cache test embedding";

        // First call should be a cache miss
        let request1 = Request::new(EmbedTextRequest {
            text: text.to_string(),
            model: None,
        });
        let resp1 = service.embed_text(request1).await.expect("Failed").into_inner();

        // Second call with same text should be a cache hit
        let request2 = Request::new(EmbedTextRequest {
            text: text.to_string(),
            model: None,
        });
        let resp2 = service.embed_text(request2).await.expect("Failed").into_inner();

        // Embeddings should be identical
        assert_eq!(resp1.embedding, resp2.embedding);

        // Verify cache metrics
        let hits = CACHE_METRICS.hits.load(Ordering::Relaxed);
        let misses = CACHE_METRICS.misses.load(Ordering::Relaxed);
        assert!(misses >= 1, "Expected at least 1 cache miss, got {}", misses);
        assert!(hits >= 1, "Expected at least 1 cache hit, got {}", hits);
    }
}
