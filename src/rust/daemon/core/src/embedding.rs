//! Embedding generation using FastEmbed
//!
//! This module provides embedding generation capabilities using the fastembed crate.
//! It generates both dense (semantic) and sparse (BM25) vectors for hybrid search.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

/// Errors that can occur during embedding generation
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model initialization failed: {message}")]
    InitializationError { message: String },

    #[error("Embedding generation failed: {message}")]
    GenerationError { message: String },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub max_cache_size: usize,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub enable_preprocessing: bool,
    pub bm25_k1: f32,
    /// Directory for storing downloaded model files
    /// Default: Uses system-appropriate cache directory (~/.cache/fastembed/)
    #[serde(default)]
    pub model_cache_dir: Option<PathBuf>,
    /// Number of ONNX intra-op threads per embedding session.
    /// Default: 2 (sufficient for all-MiniLM-L6-v2, leaves CPU for other work)
    #[serde(default)]
    pub num_threads: Option<usize>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 10000,
            batch_size: 32,
            max_sequence_length: 512,
            enable_preprocessing: true,
            bm25_k1: 1.2,
            model_cache_dir: None,
            num_threads: Some(2),
        }
    }
}

/// Dense vector embedding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseEmbedding {
    pub vector: Vec<f32>,
    pub model_name: String,
    pub sequence_length: usize,
}

/// Sparse vector embedding result using BM25
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEmbedding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub vocab_size: usize,
}

/// Combined embedding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub text_hash: u64,
    pub dense: DenseEmbedding,
    pub sparse: SparseEmbedding,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Text preprocessing result
#[derive(Debug, Clone)]
pub struct PreprocessedText {
    pub original: String,
    pub cleaned: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
}

/// BM25 for sparse vector generation
#[derive(Debug)]
pub struct BM25 {
    k1: f32,
    vocab: std::collections::HashMap<String, u32>,
    next_vocab_id: u32,
}

impl BM25 {
    pub fn new(k1: f32) -> Self {
        Self {
            k1,
            vocab: std::collections::HashMap::new(),
            next_vocab_id: 0,
        }
    }

    pub fn add_document(&mut self, tokens: &[String]) {
        for token in tokens {
            if !self.vocab.contains_key(token) {
                self.vocab.insert(token.clone(), self.next_vocab_id);
                self.next_vocab_id += 1;
            }
        }
    }

    pub fn generate_sparse_vector(&self, tokens: &[String]) -> SparseEmbedding {
        let mut term_freq: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (term, tf) in term_freq {
            if let Some(&vocab_id) = self.vocab.get(&term) {
                // Simple TF-IDF-like score
                let score = (1.0 + (tf as f32).ln()) * self.k1;
                if score > 0.0 {
                    indices.push(vocab_id);
                    values.push(score);
                }
            }
        }

        SparseEmbedding {
            indices,
            values,
            vocab_size: self.next_vocab_id as usize,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Embedding generator using FastEmbed
pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
    model: Arc<Mutex<Option<TextEmbedding>>>,
    bm25: Arc<Mutex<BM25>>,
    initialized: Arc<std::sync::atomic::AtomicBool>,
    /// Optional directory for model cache
    model_cache_dir: Option<PathBuf>,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("config", &self.config)
            .field("initialized", &self.initialized.load(std::sync::atomic::Ordering::SeqCst))
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
        let mut init_options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true);

        if let Some(threads) = self.config.num_threads {
            info!("ONNX intra-op threads: {}", threads);
            init_options = init_options.with_num_threads(threads);
        }

        if let Some(ref cache_dir) = self.model_cache_dir {
            info!("Initializing FastEmbed model (all-MiniLM-L6-v2) with cache dir: {}", cache_dir.display());
            init_options = init_options.with_cache_dir(cache_dir.clone());
        } else {
            info!("Initializing FastEmbed model (all-MiniLM-L6-v2) with default cache dir...");
        }

        let model = TextEmbedding::try_new(init_options)
            .map_err(|e| EmbeddingError::InitializationError {
                message: format!("Failed to initialize FastEmbed: {}", e),
            })?;

        *model_guard = Some(model);
        self.initialized.store(true, std::sync::atomic::Ordering::SeqCst);
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
        let model = model_guard.as_mut().ok_or_else(|| EmbeddingError::InitializationError {
            message: "Model not initialized".to_string(),
        })?;

        // Generate dense embedding
        let documents = vec![text];
        let embeddings = model.embed(documents, None).map_err(|e| {
            EmbeddingError::GenerationError {
                message: format!("Embedding generation failed: {}", e),
            }
        })?;

        let dense_vector = embeddings.into_iter().next().ok_or_else(|| {
            EmbeddingError::GenerationError {
                message: "No embedding returned".to_string(),
            }
        })?;

        // Generate sparse embedding using BM25
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        let sparse = {
            let mut bm25 = self.bm25.lock().await;
            bm25.add_document(&tokens);
            bm25.generate_sparse_vector(&tokens)
        };

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
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.generate_embedding(text, model_name).await?);
        }
        Ok(results)
    }

    pub async fn add_document_to_corpus(&self, text: &str) {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
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
}

/// Text preprocessor
#[derive(Debug)]
pub struct TextPreprocessor {
    enable_preprocessing: bool,
}

impl TextPreprocessor {
    pub fn new(enable_preprocessing: bool) -> Self {
        Self { enable_preprocessing }
    }

    pub fn preprocess(&self, text: &str) -> PreprocessedText {
        let cleaned = if self.enable_preprocessing {
            text.split_whitespace().collect::<Vec<_>>().join(" ")
        } else {
            text.to_string()
        };

        let tokens: Vec<String> = cleaned
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        PreprocessedText {
            original: text.to_string(),
            cleaned,
            tokens,
            token_ids: vec![],
        }
    }
}
