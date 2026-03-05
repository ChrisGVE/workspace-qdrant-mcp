//! Embedding generation and caching for DocumentService
//!
//! Manages the global FastEmbed model, LRU embedding cache, and BM25 sparse
//! vector generation. All global state is lazily initialized and thread-safe.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lru::LruCache;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::RwLock as TokioRwLock;
use tonic::Status;
use tracing::{debug, error, info, warn};
use workspace_qdrant_core::BM25;

/// Global embedding model instance (lazy-initialized, thread-safe)
/// Uses Mutex because TextEmbedding is not Send+Sync
static EMBEDDING_MODEL: OnceLock<TokioMutex<TextEmbedding>> = OnceLock::new();

/// Global embedding cache (content hash -> embedding vector)
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

/// Default vector dimension for embeddings (all-MiniLM-L6-v2)
pub(crate) const DEFAULT_VECTOR_SIZE: u64 = 384;

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

/// Initialize the global embedding model, cache, and BM25 if not already initialized
pub(crate) fn init_embedding_model() -> Result<(), Status> {
    EMBEDDING_MODEL.get_or_init(|| {
        info!("Initializing FastEmbed model (all-MiniLM-L6-v2)...");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )
        .expect("Failed to initialize FastEmbed model");
        info!("FastEmbed model initialized successfully");
        TokioMutex::new(model)
    });

    EMBEDDING_CACHE.get_or_init(|| {
        let cache_size =
            NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("Cache size must be non-zero");
        info!(
            "Initializing embedding cache with {} entries",
            DEFAULT_CACHE_SIZE
        );
        TokioMutex::new(LruCache::new(cache_size))
    });

    BM25_MODEL.get_or_init(|| {
        info!("Initializing BM25 model (k1={})...", DEFAULT_BM25_K1);
        TokioRwLock::new(BM25::new(DEFAULT_BM25_K1))
    });

    Ok(())
}

/// Compute a hash of the input text for cache lookup
pub(crate) fn content_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Simple tokenization for BM25 sparse vector generation
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    wqm_common::nlp::tokenize(text)
}

/// Generate sparse vector using BM25 algorithm.
/// First adds document to corpus, then generates sparse vector.
pub(crate) async fn generate_sparse_vector(text: &str) -> Result<HashMap<u32, f32>, Status> {
    init_embedding_model()?;

    let bm25 = BM25_MODEL
        .get()
        .ok_or_else(|| Status::internal("BM25 model not initialized"))?;

    let tokens = tokenize(text);

    if tokens.is_empty() {
        debug!("No tokens for sparse vector generation, returning empty");
        return Ok(HashMap::new());
    }

    let sparse_map: HashMap<u32, f32> = {
        let mut bm25_guard = bm25.write().await;

        bm25_guard.add_document(&tokens);

        let sparse = bm25_guard.generate_sparse_vector(&tokens);

        sparse
            .indices
            .into_iter()
            .zip(sparse.values.into_iter())
            .collect()
    };

    debug!(
        "Generated sparse vector with {} non-zero entries",
        sparse_map.len()
    );
    Ok(sparse_map)
}

/// Generate embedding for text using FastEmbed (all-MiniLM-L6-v2).
/// Returns 384-dimensional dense vector for semantic search.
/// Uses LRU cache to avoid redundant computation for repeated content.
pub(crate) async fn generate_embedding(text: &str) -> Result<Vec<f32>, Status> {
    init_embedding_model()?;

    let hash = content_hash(text);
    if let Some(cache) = EMBEDDING_CACHE.get() {
        let mut cache_guard = cache.lock().await;
        if let Some(cached_embedding) = cache_guard.get(&hash) {
            CACHE_METRICS.hits.fetch_add(1, Ordering::Relaxed);
            debug!("Cache hit for content hash {}", hash);
            return Ok(cached_embedding.clone());
        }
    }

    CACHE_METRICS.misses.fetch_add(1, Ordering::Relaxed);

    let model = EMBEDDING_MODEL
        .get()
        .ok_or_else(|| Status::internal("Embedding model not initialized"))?;

    let text_owned = text.to_string();

    let embedding = {
        let mut model_guard = model.lock().await;

        let documents = vec![text_owned.as_str()];

        match model_guard.embed(documents, None) {
            Ok(embeddings) => {
                if embeddings.is_empty() {
                    return Err(Status::internal("FastEmbed returned empty embeddings"));
                }
                embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| Status::internal("FastEmbed returned no embeddings"))?
            }
            Err(e) => {
                error!("FastEmbed embedding generation failed: {:?}", e);
                return Err(Status::internal(format!(
                    "Embedding generation failed: {}",
                    e
                )));
            }
        }
    };

    if embedding.len() != DEFAULT_VECTOR_SIZE as usize {
        warn!(
            "Embedding dimension mismatch: expected {}, got {}",
            DEFAULT_VECTOR_SIZE,
            embedding.len()
        );
    }

    if let Some(cache) = EMBEDDING_CACHE.get() {
        let mut cache_guard = cache.lock().await;
        if cache_guard.len() >= DEFAULT_CACHE_SIZE {
            CACHE_METRICS.evictions.fetch_add(1, Ordering::Relaxed);
        }
        cache_guard.put(hash, embedding.clone());
    }

    debug!(
        "Generated {}-dimensional embedding (cached)",
        embedding.len()
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash() {
        let hash1 = content_hash("test content");
        let hash2 = content_hash("test content");
        assert_eq!(hash1, hash2);

        let hash3 = content_hash("different content");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello world test");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));

        let tokens2 = tokenize("the quick brown fox and the lazy dog");
        assert!(!tokens2.contains(&"the".to_string()));
        assert!(!tokens2.contains(&"and".to_string()));
        assert!(tokens2.contains(&"quick".to_string()));
        assert!(tokens2.contains(&"brown".to_string()));
        assert!(tokens2.contains(&"fox".to_string()));
        assert!(tokens2.contains(&"lazy".to_string()));
        assert!(tokens2.contains(&"dog".to_string()));

        let tokens3 = tokenize("a b c test word");
        assert!(!tokens3.contains(&"a".to_string()));
        assert!(!tokens3.contains(&"b".to_string()));
        assert!(tokens3.contains(&"test".to_string()));
        assert!(tokens3.contains(&"word".to_string()));

        let tokens4 = tokenize("hello, world! test-case");
        assert!(tokens4.contains(&"hello".to_string()));
        assert!(tokens4.contains(&"world".to_string()));
        assert!(tokens4.contains(&"test".to_string()));
        assert!(tokens4.contains(&"case".to_string()));
    }

    #[tokio::test]
    async fn test_generate_embedding() {
        let text = "Test text for embedding";
        let embedding = generate_embedding(text)
            .await
            .expect("Failed to generate embedding");

        assert_eq!(embedding.len(), DEFAULT_VECTOR_SIZE as usize);
        assert!(embedding.iter().all(|&x| x.is_finite()));

        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.1,
            "Embedding not normalized: {}",
            magnitude
        );

        let embedding2 = generate_embedding(text)
            .await
            .expect("Failed to generate second embedding");
        assert_eq!(
            embedding, embedding2,
            "Same text should produce same embedding"
        );

        let different_embedding = generate_embedding("Different text for comparison")
            .await
            .expect("Failed to generate different embedding");
        assert_ne!(
            embedding, different_embedding,
            "Different text should produce different embedding"
        );
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        CACHE_METRICS.hits.store(0, Ordering::Relaxed);
        CACHE_METRICS.misses.store(0, Ordering::Relaxed);
        CACHE_METRICS.evictions.store(0, Ordering::Relaxed);

        let text = "Text for cache testing";
        let embedding1 = generate_embedding(text)
            .await
            .expect("Failed to generate embedding");

        let embedding2 = generate_embedding(text)
            .await
            .expect("Failed to generate cached embedding");

        assert_eq!(
            embedding1, embedding2,
            "Cached embedding should match original"
        );

        let (hits, misses, _evictions, _hit_rate) = get_cache_metrics();
        assert!(
            misses >= 1,
            "Expected at least 1 cache miss, got {}",
            misses
        );
        assert!(hits >= 1, "Expected at least 1 cache hit, got {}", hits);

        let _embedding3 = generate_embedding("Different unique text")
            .await
            .expect("Failed to generate different embedding");

        let (_hits2, misses2, _, _) = get_cache_metrics();
        assert!(
            misses2 > misses,
            "Expected additional cache miss for different text"
        );
    }

    #[tokio::test]
    async fn test_generate_sparse_vector() {
        let doc1 = "machine learning algorithms for natural language processing";
        let _sparse1 = generate_sparse_vector(doc1)
            .await
            .expect("Failed to add first document");

        let doc2 = "deep learning neural networks for image classification";
        let sparse2 = generate_sparse_vector(doc2)
            .await
            .expect("Failed to generate sparse vector");

        let doc3 = "reinforcement learning algorithms for robotics control";
        let sparse3 = generate_sparse_vector(doc3)
            .await
            .expect("Failed to generate third sparse vector");

        if !sparse2.is_empty() && !sparse3.is_empty() {
            assert_ne!(
                sparse2, sparse3,
                "Different documents should produce different sparse vectors"
            );
        }

        for &value in sparse2.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
        for &value in sparse3.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
    }

    #[tokio::test]
    async fn test_sparse_vector_empty_input() {
        let sparse_vector = generate_sparse_vector("")
            .await
            .expect("Failed with empty input");
        assert!(
            sparse_vector.is_empty(),
            "Empty text should produce empty sparse vector"
        );

        let stopword_only = generate_sparse_vector("the and is a")
            .await
            .expect("Failed with stopwords only");
        assert!(
            stopword_only.is_empty(),
            "Stopwords-only text should produce empty sparse vector"
        );
    }
}
