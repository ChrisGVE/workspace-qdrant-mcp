//! EmbeddingService gRPC implementation
//!
//! Provides embedding generation for the TypeScript MCP server.
//! Exposes 2 RPCs: EmbedText (dense embedding), GenerateSparseVector (BM25 sparse vector).
//!
//! Dense generation is delegated to the injected `Arc<dyn DenseProvider>`,
//! so this service inherits whichever provider the daemon was configured
//! with (FastEmbed local, OpenAI-compatible remote, etc.). Sparse vectors
//! still use a service-local BM25 corpus.

use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::RwLock as TokioRwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use workspace_qdrant_core::embedding::provider::DenseProvider;
use workspace_qdrant_core::BM25;

use crate::proto::{
    embedding_service_server::EmbeddingService, EmbedTextRequest, EmbedTextResponse,
    RerankRequest, RerankResponse, RerankResult, SparseVectorRequest, SparseVectorResponse,
};

/// Default cache size (number of entries)
const DEFAULT_CACHE_SIZE: usize = 1000;

/// Default BM25 parameters
const DEFAULT_BM25_K1: f32 = 1.2;

/// Cache metrics for monitoring
pub struct EmbeddingCacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl Default for EmbeddingCacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCacheMetrics {
    pub const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

/// Global cache metrics instance (read by health/metrics endpoints).
pub static CACHE_METRICS: EmbeddingCacheMetrics = EmbeddingCacheMetrics::new();

/// EmbeddingService implementation backed by an injected dense provider.
pub struct EmbeddingServiceImpl {
    dense_provider: Arc<dyn DenseProvider>,
    cache: Arc<TokioMutex<LruCache<u64, Vec<f32>>>>,
    bm25: Arc<TokioRwLock<BM25>>,
    /// Cross-encoder reranker (lazy-initialised on first Rerank call).
    reranker: Arc<TokioMutex<Option<TextRerank>>>,
    /// Writable cache directory for the reranker ONNX model download.
    /// Mirrors the dense/sparse providers' `model_cache_dir`; without it
    /// fastembed defaults to `./.fastembed_cache` (CWD-relative), which is
    /// unwritable when the daemon runs as a non-root user with CWD=`/`.
    model_cache_dir: Option<PathBuf>,
}

impl EmbeddingServiceImpl {
    /// Create a new EmbeddingService bound to the given dense provider.
    ///
    /// `model_cache_dir` is the writable directory the reranker downloads its
    /// ONNX model into (typically the same path the dense provider uses). When
    /// `None`, fastembed falls back to its CWD-relative default.
    pub fn new(
        dense_provider: Arc<dyn DenseProvider>,
        model_cache_dir: Option<PathBuf>,
    ) -> Self {
        let cache_size =
            NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("Cache size must be non-zero");
        Self {
            dense_provider,
            cache: Arc::new(TokioMutex::new(LruCache::new(cache_size))),
            bm25: Arc::new(TokioRwLock::new(BM25::new(DEFAULT_BM25_K1))),
            reranker: Arc::new(TokioMutex::new(None)),
            model_cache_dir,
        }
    }

    /// Rerank `documents` against `query` with a cross-encoder, returning
    /// `(index, score)` pairs sorted by score descending. Lazy-loads the Jina
    /// turbo reranker on first call (~150MB ONNX download into `model_cache_dir`).
    async fn rerank_internal(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<Vec<(usize, f32)>, Status> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut guard = self.reranker.lock().await;
        if guard.is_none() {
            info!(
                cache_dir = ?self.model_cache_dir,
                "Initializing cross-encoder reranker (first call, ~150MB ONNX download)..."
            );
            // jina-reranker-v1-turbo-en: 38M-param English cross-encoder. ~7×
            // faster on CPU than bge-reranker-base (278M) — bge measured ~3s per
            // 30-doc query, which is too slow for interactive search and broke the
            // benchmark's timeout. Turbo keeps per-query rerank well under 1s.
            let mut init_opts = RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
                .with_show_download_progress(true);
            // Redirect the model download to a writable cache dir. Without this
            // fastembed uses `./.fastembed_cache` (CWD-relative); the daemon runs
            // as a non-root user with CWD=`/`, so that path is unwritable and the
            // download fails immediately with "Failed to retrieve model file".
            if let Some(ref dir) = self.model_cache_dir {
                init_opts = init_opts.with_cache_dir(dir.clone());
            }
            let model = TextRerank::try_new(init_opts).map_err(|e| {
                error!("Reranker init failed: {:?}", e);
                Status::internal(format!("Reranker init failed: {}", e))
            })?;
            *guard = Some(model);
            info!("Cross-encoder reranker initialized");
        }
        let model = guard.as_mut().unwrap();
        let results = model
            .rerank(query.to_string(), documents, false, None)
            .map_err(|e| {
                error!("Rerank failed: {:?}", e);
                Status::internal(format!("Rerank failed: {}", e))
            })?;
        Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
    }

    /// Compute a hash of the input text for cache lookup.
    fn content_hash(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Simple tokenization for BM25 sparse vector generation.
    fn tokenize(text: &str) -> Vec<String> {
        wqm_common::nlp::tokenize(text)
    }

    /// Generate a dense embedding via the injected provider, with an LRU cache.
    async fn generate_embedding_internal(&self, text: &str) -> Result<Vec<f32>, Status> {
        let content_hash = Self::content_hash(text);
        {
            let mut cache_guard = self.cache.lock().await;
            if let Some(cached_embedding) = cache_guard.get(&content_hash) {
                CACHE_METRICS.hits.fetch_add(1, Ordering::Relaxed);
                debug!("Cache hit for content hash {}", content_hash);
                return Ok(cached_embedding.clone());
            }
        }

        CACHE_METRICS.misses.fetch_add(1, Ordering::Relaxed);

        let mut embeddings = self.dense_provider.embed(&[text]).await.map_err(|e| {
            error!("Dense provider embed failed: {:?}", e);
            Status::internal(format!("Embedding generation failed: {}", e))
        })?;

        let dense = embeddings
            .pop()
            .ok_or_else(|| Status::internal("Provider returned no embedding"))?;

        let embedding = dense.vector;

        {
            let mut cache_guard = self.cache.lock().await;
            cache_guard.put(content_hash, embedding.clone());
        }

        debug!("Generated {}-dimensional embedding", embedding.len());
        Ok(embedding)
    }

    /// Generate a sparse vector using BM25 (service-local corpus).
    async fn generate_sparse_vector_internal(
        &self,
        text: &str,
    ) -> Result<HashMap<u32, f32>, Status> {
        let tokens = Self::tokenize(text);

        if tokens.is_empty() {
            debug!("No tokens for sparse vector generation, returning empty");
            return Ok(HashMap::new());
        }

        let sparse_map: HashMap<u32, f32> = {
            let mut bm25_guard = self.bm25.write().await;
            bm25_guard.add_document(&tokens);
            let sparse = bm25_guard.generate_sparse_vector(&tokens);
            sparse.indices.into_iter().zip(sparse.values).collect()
        };

        debug!(
            "Generated sparse vector with {} non-zero entries",
            sparse_map.len()
        );
        Ok(sparse_map)
    }

    /// BM25 corpus vocabulary size (used in the SparseVectorResponse).
    async fn vocab_size(&self) -> i32 {
        let bm25_guard = self.bm25.read().await;
        bm25_guard.vocab_size() as i32
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    #[tracing::instrument(skip_all, fields(method = "EmbeddingService.embed_text"))]
    async fn embed_text(
        &self,
        request: Request<EmbedTextRequest>,
    ) -> Result<Response<EmbedTextResponse>, Status> {
        let req = request.into_inner();

        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("Text cannot be empty"));
        }

        let provider_label = self.dense_provider.provider_label().to_string();
        let requested_model = req.model.clone();
        if let Some(ref m) = requested_model {
            if m != &provider_label {
                debug!(
                    "Requested model '{}' does not match active provider '{}', using active provider",
                    m, provider_label
                );
            }
        }

        info!(
            "EmbedText: generating embedding for text of {} chars",
            req.text.len()
        );

        match self.generate_embedding_internal(&req.text).await {
            Ok(embedding) => {
                let dimensions = embedding.len() as i32;
                let expected_dim = self.dense_provider.output_dim() as i32;
                if dimensions != expected_dim {
                    tracing::warn!(
                        "Embedding dimension mismatch: expected {}, got {}",
                        expected_dim,
                        dimensions
                    );
                }

                Ok(Response::new(EmbedTextResponse {
                    embedding,
                    dimensions,
                    model_name: provider_label,
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!("EmbedText failed: {:?}", e);
                Ok(Response::new(EmbedTextResponse {
                    embedding: vec![],
                    dimensions: 0,
                    model_name: provider_label,
                    success: false,
                    error_message: e.message().to_string(),
                }))
            }
        }
    }

    #[tracing::instrument(skip_all, fields(method = "EmbeddingService.generate_sparse_vector"))]
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

        info!(
            "GenerateSparseVector: processing text of {} chars",
            req.text.len()
        );

        match self.generate_sparse_vector_internal(&req.text).await {
            Ok(sparse_map) => {
                let vocab_size = self.vocab_size().await;
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

    #[tracing::instrument(skip_all, fields(method = "EmbeddingService.rerank"))]
    async fn rerank(
        &self,
        request: Request<RerankRequest>,
    ) -> Result<Response<RerankResponse>, Status> {
        let req = request.into_inner();
        if req.query.trim().is_empty() {
            return Err(Status::invalid_argument("query cannot be empty"));
        }
        let doc_count = req.documents.len();
        let top_k = req.top_k;
        match self.rerank_internal(&req.query, req.documents).await {
            Ok(mut ranked) => {
                if let Some(k) = top_k {
                    if k > 0 {
                        ranked.truncate(k as usize);
                    }
                }
                let results = ranked
                    .into_iter()
                    .map(|(index, score)| RerankResult {
                        index: index as u32,
                        score,
                    })
                    .collect();
                debug!("Rerank: scored {} documents", doc_count);
                Ok(Response::new(RerankResponse {
                    results,
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!("Rerank failed: {:?}", e);
                Ok(Response::new(RerankResponse {
                    results: Vec::new(),
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
    use workspace_qdrant_core::embedding::provider::FastEmbedProvider;

    fn make_service() -> EmbeddingServiceImpl {
        let provider: Arc<dyn DenseProvider> = Arc::new(FastEmbedProvider::new(32, None, None));
        EmbeddingServiceImpl::new(provider, None)
    }

    #[test]
    fn test_tokenize() {
        let tokens = EmbeddingServiceImpl::tokenize("Hello world test");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));

        let tokens2 = EmbeddingServiceImpl::tokenize("the quick brown fox and the lazy dog");
        assert!(!tokens2.contains(&"the".to_string()));
        assert!(!tokens2.contains(&"and".to_string()));
        assert!(tokens2.contains(&"quick".to_string()));
        assert!(tokens2.contains(&"brown".to_string()));
    }

    #[test]
    fn test_content_hash() {
        let hash1 = EmbeddingServiceImpl::content_hash("test content");
        let hash2 = EmbeddingServiceImpl::content_hash("test content");
        assert_eq!(hash1, hash2);

        let hash3 = EmbeddingServiceImpl::content_hash("different content");
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_embed_text() {
        let service = make_service();

        let request = Request::new(EmbedTextRequest {
            text: "Test embedding generation".to_string(),
            model: None,
        });

        let response = service
            .embed_text(request)
            .await
            .expect("Failed to embed text");
        let resp = response.into_inner();

        assert!(resp.success);
        assert_eq!(resp.dimensions, 384);
        assert_eq!(resp.embedding.len(), 384);
        assert!(resp.error_message.is_empty());

        let magnitude: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.1,
            "Embedding not normalized: {}",
            magnitude
        );
    }

    #[tokio::test]
    async fn test_embed_text_empty_input() {
        let service = make_service();

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
        let service = make_service();

        let request1 = Request::new(SparseVectorRequest {
            text: "machine learning algorithms for natural language processing".to_string(),
        });
        let _ = service
            .generate_sparse_vector(request1)
            .await
            .expect("Failed");

        let request2 = Request::new(SparseVectorRequest {
            text: "deep learning neural networks for image classification".to_string(),
        });
        let response = service
            .generate_sparse_vector(request2)
            .await
            .expect("Failed");
        let resp = response.into_inner();

        assert!(resp.success);
        assert!(resp.vocab_size > 0);
        assert!(resp.error_message.is_empty());

        for &value in resp.indices_values.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
    }

    #[tokio::test]
    async fn test_generate_sparse_vector_empty_input() {
        let service = make_service();

        let request = Request::new(SparseVectorRequest {
            text: "".to_string(),
        });

        let response = service
            .generate_sparse_vector(request)
            .await
            .expect("Should succeed");
        let resp = response.into_inner();

        assert!(resp.success);
        assert!(resp.indices_values.is_empty());
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let service = make_service();

        CACHE_METRICS.hits.store(0, Ordering::Relaxed);
        CACHE_METRICS.misses.store(0, Ordering::Relaxed);

        let text = "Cache test embedding";

        let request1 = Request::new(EmbedTextRequest {
            text: text.to_string(),
            model: None,
        });
        let resp1 = service
            .embed_text(request1)
            .await
            .expect("Failed")
            .into_inner();

        let request2 = Request::new(EmbedTextRequest {
            text: text.to_string(),
            model: None,
        });
        let resp2 = service
            .embed_text(request2)
            .await
            .expect("Failed")
            .into_inner();

        assert_eq!(resp1.embedding, resp2.embedding);

        let hits = CACHE_METRICS.hits.load(Ordering::Relaxed);
        let misses = CACHE_METRICS.misses.load(Ordering::Relaxed);
        assert!(
            misses >= 1,
            "Expected at least 1 cache miss, got {}",
            misses
        );
        assert!(hits >= 1, "Expected at least 1 cache hit, got {}", hits);
    }
}
