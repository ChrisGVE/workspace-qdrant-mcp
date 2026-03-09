//! Embedding generator and text preprocessor using FastEmbed.

use fastembed::{
    EmbeddingModel, InitOptions, SparseInitOptions, SparseModel, SparseTextEmbedding, TextEmbedding,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;

use super::bm25::{tokenize_for_bm25, BM25};
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
        Ok(Self {
            config: config.clone(),
            model: Arc::new(Mutex::new(None)),
            bm25: Arc::new(Mutex::new(BM25::new(config.bm25_k1))),
            initialized: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            model_cache_dir,
            splade_model: Arc::new(Mutex::new(None)),
        })
    }

    /// Initialize the embedding model (lazy initialization)
    async fn ensure_initialized(&self) -> Result<(), EmbeddingError> {
        if self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
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

        let model = TextEmbedding::try_new(init_options).map_err(|e| {
            EmbeddingError::InitializationError {
                message: format!("Failed to initialize FastEmbed: {}", e),
            }
        })?;

        *model_guard = Some(model);
        self.initialized
            .store(true, std::sync::atomic::Ordering::SeqCst);
        info!("FastEmbed model initialized successfully");
        Ok(())
    }

    pub async fn initialize_model(&self, _model_name: &str) -> Result<(), EmbeddingError> {
        self.ensure_initialized().await
    }

    pub async fn generate_embedding(
        &self,
        text: &str,
        _model_name: &str,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        self.ensure_initialized().await?;

        let mut model_guard = self.model.lock().await;
        let model = model_guard
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

        let dense_vector =
            embeddings
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::GenerationError {
                    message: "No embedding returned".to_string(),
                })?;

        info!(
            text_len = text.len(),
            dim = dense_vector.len(),
            embed_ms = embed_ms,
            "dense embedding generated"
        );

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

    pub async fn cache_stats(&self) -> (usize, usize) {
        (0, self.config.max_cache_size)
    }

    pub async fn clear_cache(&self) {
        // No-op for now
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
