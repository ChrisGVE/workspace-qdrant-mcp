//! Embedding types stub
//!
//! This module provides stub types for the embedding system. The actual embedding
//! generation is handled by the `fastembed` crate in the gRPC document service.
//!
//! TODO: Refactor queue processors to use fastembed directly instead of these stubs.
//! See task-master task for "Migrate queue processors to use fastembed directly".

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during embedding generation
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Embedding not implemented: {message}")]
    NotImplemented { message: String },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_cache_dir: PathBuf,
    pub max_cache_size: usize,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub enable_preprocessing: bool,
    pub bm25_k1: f32,
    pub bm25_b: f32,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_cache_dir: PathBuf::from("./models"),
            max_cache_size: 10000,
            batch_size: 32,
            max_sequence_length: 512,
            enable_preprocessing: true,
            bm25_k1: 1.2,
            bm25_b: 0.75,
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

/// BM25 stub for sparse vector generation
#[derive(Debug)]
pub struct BM25 {
    k1: f32,
    b: f32,
}

impl BM25 {
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    pub fn add_document(&mut self, _tokens: &[String]) {
        // Stub - actual BM25 implementation is in document_service.rs
    }

    pub fn generate_sparse_vector(&self, _tokens: &[String]) -> SparseEmbedding {
        SparseEmbedding {
            indices: vec![],
            values: vec![],
            vocab_size: 0,
        }
    }

    pub fn vocab_size(&self) -> usize {
        0
    }
}

/// Stub embedding generator
///
/// NOTE: This is a stub. Actual embedding generation uses fastembed in
/// document_service.rs. Queue processors should be migrated to use fastembed directly.
#[derive(Debug)]
pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
}

impl EmbeddingGenerator {
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        Ok(Self { config })
    }

    pub async fn initialize_model(&self, _model_name: &str) -> Result<(), EmbeddingError> {
        Err(EmbeddingError::NotImplemented {
            message: "Use fastembed in document_service.rs for embeddings".to_string(),
        })
    }

    pub async fn generate_embedding(
        &self,
        _text: &str,
        _model_name: &str,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        Err(EmbeddingError::NotImplemented {
            message: "Use fastembed in document_service.rs for embeddings".to_string(),
        })
    }

    pub async fn generate_embeddings_batch(
        &self,
        _texts: &[String],
        _model_name: &str,
    ) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        Err(EmbeddingError::NotImplemented {
            message: "Use fastembed in document_service.rs for embeddings".to_string(),
        })
    }

    pub async fn add_document_to_corpus(&self, _text: &str) {
        // Stub
    }

    pub async fn cache_stats(&self) -> (usize, usize) {
        (0, self.config.max_cache_size)
    }

    pub async fn clear_cache(&self) {
        // Stub
    }

    pub fn available_models(&self) -> Vec<String> {
        vec!["all-MiniLM-L6-v2".to_string()]
    }

    pub async fn is_model_ready(&self, _model_name: &str) -> bool {
        false
    }
}

/// Text preprocessor stub
#[derive(Debug)]
pub struct TextPreprocessor {
    enable_preprocessing: bool,
}

impl TextPreprocessor {
    pub fn new(enable_preprocessing: bool) -> Self {
        Self { enable_preprocessing }
    }

    pub fn preprocess(&self, text: &str) -> PreprocessedText {
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
        PreprocessedText {
            original: text.to_string(),
            cleaned: text.to_string(),
            tokens,
            token_ids: vec![],
        }
    }
}
