//! Embedding generator and text preprocessor using FastEmbed.

use fastembed::{
    EmbeddingModel, InitOptions, SparseInitOptions, SparseModel, SparseTextEmbedding, TextEmbedding,
};
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Exponential backoff schedule (seconds) for failed init attempts.
/// Indexed by `init_failure_count - 1`, capped at the last entry.
const INIT_BACKOFF_SECS: &[u64] = &[30, 60, 120, 300, 600];

use super::bm25::{tokenize_for_bm25, BM25};
use super::phrase_cache::PhraseCache;
use super::types::{
    DenseEmbedding, EmbeddingConfig, EmbeddingError, EmbeddingResult, PreprocessedText,
    SparseEmbedding,
};

/// Embedding generator using FastEmbed
pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
    model: Arc<Mutex<Option<TextEmbedding>>>,
    bm25: Arc<Mutex<BM25>>,
    initialized: Arc<std::sync::atomic::AtomicBool>,
    /// Optional directory for model cache
    model_cache_dir: Option<PathBuf>,
    /// SPLADE++ sparse embedding model (lazy-initialized)
    splade_model: Arc<Mutex<Option<SparseTextEmbedding>>>,
    /// Timestamp of the last failed initialization attempt (None = never failed)
    last_failed_init: Arc<Mutex<Option<Instant>>>,
    /// Number of consecutive initialization failures
    init_failure_count: Arc<AtomicU32>,
    /// LRU cache for short-phrase dense embeddings (cross-document reuse)
    phrase_cache: PhraseCache,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("config", &self.config)
            .field(
                "initialized",
                &self.initialized.load(std::sync::atomic::Ordering::SeqCst),
            )
            .field("sparse_mode", &self.config.sparse_vector_mode)
            .finish()
    }
}

impl EmbeddingGenerator {
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let model_cache_dir = config.model_cache_dir.clone();
        let phrase_cache = PhraseCache::new(config.max_cache_size);
        Ok(Self {
            config: config.clone(),
            model: Arc::new(Mutex::new(None)),
            bm25: Arc::new(Mutex::new(BM25::new(config.bm25_k1))),
            initialized: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            model_cache_dir,
            splade_model: Arc::new(Mutex::new(None)),
            last_failed_init: Arc::new(Mutex::new(None)),
            init_failure_count: Arc::new(AtomicU32::new(0)),
            phrase_cache,
        })
    }

    /// Initialize the embedding model (lazy initialization with backoff).
    ///
    /// If a previous initialization attempt failed, subsequent calls are rate-limited
    /// using an exponential backoff schedule (`INIT_BACKOFF_SECS`). While within the
    /// backoff window, returns `EmbeddingError::TemporarilyUnavailable` so the caller
    /// can re-lease the queue item without burning its retry budget.
    async fn ensure_initialized(&self) -> Result<(), EmbeddingError> {
        if self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        // Check backoff window before attempting initialization.
        let failure_count = self
            .init_failure_count
            .load(std::sync::atomic::Ordering::SeqCst);
        if failure_count > 0 {
            let last_failed = self.last_failed_init.lock().await;
            if let Some(last_attempt) = *last_failed {
                let backoff_idx = (failure_count as usize - 1).min(INIT_BACKOFF_SECS.len() - 1);
                let backoff = INIT_BACKOFF_SECS[backoff_idx];
                let elapsed = last_attempt.elapsed().as_secs();
                if elapsed < backoff {
                    let retry_after = backoff - elapsed;
                    return Err(EmbeddingError::TemporarilyUnavailable {
                        retry_after_secs: retry_after,
                    });
                }
            }
        }

        let mut model_guard = self.model.lock().await;
        if model_guard.is_some() {
            return Ok(());
        }

        // Build InitOptions with optional cache directory and thread count
        let mut init_options =
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true);

        if let Some(threads) = self.config.num_threads {
            info!("ONNX intra-op threads: {}", threads);
            init_options = init_options.with_num_threads(threads);
        }

        if let Some(ref cache_dir) = self.model_cache_dir {
            info!(
                "Initializing FastEmbed model (all-MiniLM-L6-v2) with cache dir: {}",
                cache_dir.display()
            );
            init_options = init_options.with_cache_dir(cache_dir.clone());
        } else {
            info!("Initializing FastEmbed model (all-MiniLM-L6-v2) with default cache dir...");
        }

        match TextEmbedding::try_new(init_options) {
            Ok(model) => {
                *model_guard = Some(model);
                self.initialized
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                // Reset failure tracking on success
                self.init_failure_count
                    .store(0, std::sync::atomic::Ordering::SeqCst);
                *self.last_failed_init.lock().await = None;
                info!("FastEmbed model initialized successfully");
                Ok(())
            }
            Err(e) => {
                let new_count = self
                    .init_failure_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    + 1;
                *self.last_failed_init.lock().await = Some(Instant::now());
                let backoff_idx = (new_count as usize - 1).min(INIT_BACKOFF_SECS.len() - 1);
                let next_retry = INIT_BACKOFF_SECS[backoff_idx];
                warn!(
                    failure_count = new_count,
                    next_retry_secs = next_retry,
                    "FastEmbed initialization failed (attempt {}), next retry in {}s: {}",
                    new_count,
                    next_retry,
                    e
                );
                Err(EmbeddingError::InitializationError {
                    message: format!("Failed to initialize FastEmbed: {}", e),
                })
            }
        }
    }

    /// Returns true if the embedding subsystem is currently available.
    pub fn is_available(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns the number of consecutive initialization failures.
    pub fn init_failure_count(&self) -> u32 {
        self.init_failure_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn initialize_model(&self, _model_name: &str) -> Result<(), EmbeddingError> {
        self.ensure_initialized().await
    }

    pub async fn generate_embedding(
        &self,
        text: &str,
        _model_name: &str,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        // Check phrase cache before initializing the model (avoids lock contention
        // on hot phrases like common keywords and stdlib names)
        let cached_dense = self.phrase_cache.get(text).await;

        let dense_vector =
            if let Some(v) = cached_dense {
                v
            } else {
                self.ensure_initialized().await?;

                let mut model_guard = self.model.lock().await;
                let model =
                    model_guard
                        .as_mut()
                        .ok_or_else(|| EmbeddingError::InitializationError {
                            message: "Model not initialized".to_string(),
                        })?;

                // Generate dense embedding
                let documents = vec![text];
                let embed_start = Instant::now();
                let embeddings =
                    model
                        .embed(documents, None)
                        .map_err(|e| EmbeddingError::GenerationError {
                            message: format!("Embedding generation failed: {}", e),
                        })?;
                let embed_ms = embed_start.elapsed().as_millis();

                let v = embeddings.into_iter().next().ok_or_else(|| {
                    EmbeddingError::GenerationError {
                        message: "No embedding returned".to_string(),
                    }
                })?;

                info!(
                    text_len = text.len(),
                    dim = v.len(),
                    embed_ms = embed_ms,
                    "dense embedding generated"
                );

                // Populate cache for eligible phrases
                self.phrase_cache.put(text, v.clone()).await;
                v
            };

        // Generate sparse embedding using BM25
        let bm25_start = Instant::now();
        let tokens = tokenize_for_bm25(text);

        let sparse = {
            let mut bm25 = self.bm25.lock().await;
            bm25.add_document(&tokens);
            bm25.generate_sparse_vector(&tokens)
        };
        let bm25_ms = bm25_start.elapsed().as_millis();

        info!(
            token_count = tokens.len(),
            sparse_nnz = sparse.indices.len(),
            bm25_ms = bm25_ms,
            "BM25 sparse vector generated"
        );

        let text_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };

        Ok(EmbeddingResult {
            text_hash,
            dense: DenseEmbedding {
                vector: dense_vector,
                model_name: "all-MiniLM-L6-v2".to_string(),
                sequence_length: tokens.len(),
            },
            sparse,
            generated_at: chrono::Utc::now(),
        })
    }

    pub async fn generate_embeddings_batch(
        &self,
        texts: &[String],
        model_name: &str,
    ) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let batch_start = Instant::now();
        let batch_size = texts.len();
        let mut results = Vec::with_capacity(batch_size);
        for text in texts {
            results.push(self.generate_embedding(text, model_name).await?);
        }
        let batch_ms = batch_start.elapsed().as_millis();
        let per_item_ms = if batch_size > 0 {
            batch_ms / batch_size as u128
        } else {
            0
        };
        info!(
            batch_size = batch_size,
            batch_ms = batch_ms,
            per_item_ms = per_item_ms,
            "embedding batch completed"
        );
        Ok(results)
    }

    pub async fn add_document_to_corpus(&self, text: &str) {
        let tokens = tokenize_for_bm25(text);
        let mut bm25 = self.bm25.lock().await;
        bm25.add_document(&tokens);
    }

    /// Returns `(hits, misses)` for the phrase embedding cache.
    pub async fn cache_stats(&self) -> (usize, usize) {
        let (hits, misses, _) = self.phrase_cache.stats().await;
        (hits as usize, misses as usize)
    }

    /// Clear the phrase embedding cache.
    pub async fn clear_cache(&self) {
        self.phrase_cache.clear().await;
    }

    pub fn available_models(&self) -> Vec<String> {
        vec!["all-MiniLM-L6-v2".to_string()]
    }

    pub async fn is_model_ready(&self, _model_name: &str) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the configured sparse vector mode ("bm25" or "splade").
    pub fn sparse_vector_mode(&self) -> &str {
        &self.config.sparse_vector_mode
    }

    /// Generate a sparse vector using the SPLADE++ model.
    ///
    /// Lazy-initializes the SPLADE++ model on first call (~150MB download).
    /// Converts fastembed's `SparseEmbedding` (indices: `Vec<usize>`) to our
    /// `SparseEmbedding` (indices: `Vec<u32>`).
    pub async fn generate_splade_sparse_vector(
        &self,
        text: &str,
    ) -> Result<SparseEmbedding, EmbeddingError> {
        let mut guard = self.splade_model.lock().await;
        if guard.is_none() {
            info!("Initializing SPLADE++ model (first call, ~150MB download)...");
            let mut init_opts =
                SparseInitOptions::new(SparseModel::SPLADEPPV1).with_show_download_progress(true);
            if let Some(ref cache_dir) = self.model_cache_dir {
                init_opts = init_opts.with_cache_dir(cache_dir.clone());
            }
            let model = SparseTextEmbedding::try_new(init_opts).map_err(|e| {
                EmbeddingError::InitializationError {
                    message: format!("SPLADE++ init failed: {}", e),
                }
            })?;
            *guard = Some(model);
            info!("SPLADE++ model initialized");
        }

        let model = guard.as_mut().unwrap();
        let splade_start = Instant::now();
        let results = model.embed(vec![text.to_string()], None).map_err(|e| {
            EmbeddingError::GenerationError {
                message: format!("SPLADE++ embedding failed: {}", e),
            }
        })?;
        let splade_ms = splade_start.elapsed().as_millis();

        let fe_sparse =
            results
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::GenerationError {
                    message: "SPLADE++ returned no embeddings".to_string(),
                })?;

        info!(
            text_len = text.len(),
            nnz = fe_sparse.indices.len(),
            splade_ms = splade_ms,
            "SPLADE++ sparse vector generated"
        );

        // Convert fastembed usize indices to our u32 indices
        Ok(SparseEmbedding {
            indices: fe_sparse.indices.into_iter().map(|i| i as u32).collect(),
            values: fe_sparse.values,
            vocab_size: 30522, // BERT vocab size for SPLADE++
        })
    }
}

/// Text preprocessor
#[derive(Debug)]
pub struct TextPreprocessor {
    enable_preprocessing: bool,
}

impl TextPreprocessor {
    pub fn new(enable_preprocessing: bool) -> Self {
        Self {
            enable_preprocessing,
        }
    }

    pub fn preprocess(&self, text: &str) -> PreprocessedText {
        let cleaned = if self.enable_preprocessing {
            text.split_whitespace().collect::<Vec<_>>().join(" ")
        } else {
            text.to_string()
        };

        let tokens = tokenize_for_bm25(&cleaned);

        PreprocessedText {
            original: text.to_string(),
            cleaned,
            tokens,
            token_ids: vec![],
        }
    }
}
