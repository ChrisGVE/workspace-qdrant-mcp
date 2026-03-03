//! Types for embedding generation: errors, configuration, and result structs.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
    /// Sparse vector generation mode: "bm25" (default) or "splade".
    /// SPLADE++ uses a learned sparse model (~150MB download on first use).
    /// Switching modes requires re-indexing sparse vectors.
    #[serde(default = "default_sparse_mode")]
    pub sparse_vector_mode: String,
}

fn default_sparse_mode() -> String {
    "bm25".to_string()
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
            sparse_vector_mode: "bm25".to_string(),
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
