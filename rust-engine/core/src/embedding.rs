//! Embedding generation system for text documents
//!
//! This module provides comprehensive embedding generation capabilities including:
//! - Dense vector embeddings using ONNX Runtime
//! - Sparse vector generation using BM25 algorithm
//! - Model downloading and caching
//! - Text preprocessing and normalization
//! - Embedding result caching for performance

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ort::{Environment, Session, SessionBuilder, Value};
use tokenizers::Tokenizer;
use uuid::Uuid;
use ahash::AHashMap;
use std::hash::{Hash, Hasher};
use ahash::AHasher;

/// Errors that can occur during embedding generation
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },
    
    #[error("Model download failed: {source}")]
    ModelDownloadError { source: reqwest::Error },
    
    #[error("ONNX Runtime error: {message}")]
    OnnxError { message: String },
    
    #[error("Tokenization error: {source}")]
    TokenizationError { source: Box<tokenizers::Error> },
    
    #[error("Text preprocessing failed: {message}")]
    PreprocessingError { message: String },
    
    #[error("Cache operation failed: {message}")]
    CacheError { message: String },
    
    #[error("IO error: {source}")]
    IoError { source: std::io::Error },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Directory to cache downloaded models
    pub model_cache_dir: PathBuf,
    
    /// Maximum number of embeddings to cache in memory
    pub max_cache_size: usize,
    
    /// Batch size for processing multiple texts
    pub batch_size: usize,
    
    /// Maximum length of input text (in tokens)
    pub max_sequence_length: usize,
    
    /// Enable text preprocessing
    pub enable_preprocessing: bool,
    
    /// BM25 parameters
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

/// Information about an embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub url: String,
    pub tokenizer_url: Option<String>,
    pub md5_hash: Option<String>,
    pub embedding_dim: usize,
    pub max_sequence_length: usize,
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

/// BM25 implementation for sparse vector generation
#[derive(Debug)]
pub struct BM25 {
    k1: f32,
    b: f32,
    doc_freq: AHashMap<String, u32>,
    doc_count: u32,
    avg_doc_length: f32,
    vocab: AHashMap<String, u32>,
    next_vocab_id: u32,
}

impl BM25 {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            doc_freq: AHashMap::new(),
            doc_count: 0,
            avg_doc_length: 0.0,
            vocab: AHashMap::new(),
            next_vocab_id: 0,
        }
    }
    
    /// Add a document to the corpus for IDF calculation
    pub fn add_document(&mut self, tokens: &[String]) {
        let unique_tokens: HashSet<_> = tokens.iter().collect();
        
        for token in unique_tokens {
            *self.doc_freq.entry(token.clone()).or_insert(0) += 1;
            if !self.vocab.contains_key(token) {
                self.vocab.insert(token.clone(), self.next_vocab_id);
                self.next_vocab_id += 1;
            }
        }
        
        self.doc_count += 1;
        
        // Update average document length
        let total_length = self.avg_doc_length * (self.doc_count - 1) as f32 + tokens.len() as f32;
        self.avg_doc_length = total_length / self.doc_count as f32;
    }
    
    /// Generate sparse vector for a document
    pub fn generate_sparse_vector(&self, tokens: &[String]) -> SparseEmbedding {
        let mut term_freq: AHashMap<String, u32> = AHashMap::new();
        
        // Count term frequencies
        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }
        
        let doc_length = tokens.len() as f32;
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (term, tf) in term_freq {
            if let Some(&vocab_id) = self.vocab.get(&term) {
                let df = self.doc_freq.get(&term).copied().unwrap_or(1);
                
                // Calculate BM25 score for this term
                let idf = ((self.doc_count as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln().max(0.0);
                let tf_component = (tf as f32 * (self.k1 + 1.0)) / 
                    (tf as f32 + self.k1 * (1.0 - self.b + self.b * (doc_length / self.avg_doc_length)));
                
                let score = idf * tf_component;
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
}

/// Model manager for downloading and caching models
#[derive(Debug)]
pub struct ModelManager {
    config: EmbeddingConfig,
    client: reqwest::Client,
    models: AHashMap<String, ModelInfo>,
}

impl ModelManager {
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
            .build()
            .expect("Failed to create HTTP client");
            
        let mut models = AHashMap::new();
        
        // Add default BAAI/bge-small-en-v1.5 model
        models.insert("bge-small-en-v1.5".to_string(), ModelInfo {
            name: "bge-small-en-v1.5".to_string(),
            url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx".to_string(),
            tokenizer_url: Some("https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json".to_string()),
            md5_hash: None, // Would be populated with actual hash
            embedding_dim: 384,
            max_sequence_length: 512,
        });
        
        Self {
            config,
            client,
            models,
        }
    }
    
    /// Get the local path for a model file
    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        self.config.model_cache_dir.join(format!("{}.onnx", model_name))
    }
    
    /// Get the local path for a tokenizer file
    pub fn get_tokenizer_path(&self, model_name: &str) -> PathBuf {
        self.config.model_cache_dir.join(format!("{}_tokenizer.json", model_name))
    }
    
    /// Check if a model is cached locally
    pub fn is_model_cached(&self, model_name: &str) -> bool {
        let model_path = self.get_model_path(model_name);
        let tokenizer_path = self.get_tokenizer_path(model_name);
        model_path.exists() && tokenizer_path.exists()
    }
    
    /// Download and cache a model
    pub async fn download_model(&self, model_name: &str) -> Result<(), EmbeddingError> {
        let model_info = self.models.get(model_name)
            .ok_or_else(|| EmbeddingError::ModelNotFound {
                model_name: model_name.to_string(),
            })?;
        
        // Create cache directory if it doesn't exist
        tokio::fs::create_dir_all(&self.config.model_cache_dir).await
            .map_err(|e| EmbeddingError::IoError { source: e })?;
        
        // Download model file
        let model_path = self.get_model_path(model_name);
        if !model_path.exists() {
            println!("Downloading model: {} from {}", model_name, model_info.url);
            self.download_file(&model_info.url, &model_path).await?;
        }
        
        // Download tokenizer if URL is provided
        if let Some(tokenizer_url) = &model_info.tokenizer_url {
            let tokenizer_path = self.get_tokenizer_path(model_name);
            if !tokenizer_path.exists() {
                println!("Downloading tokenizer for: {}", model_name);
                self.download_file(tokenizer_url, &tokenizer_path).await?;
            }
        }
        
        Ok(())
    }
    
    /// Download a file from URL to local path
    async fn download_file(&self, url: &str, path: &Path) -> Result<(), EmbeddingError> {
        let response = self.client.get(url).send().await
            .map_err(|e| EmbeddingError::ModelDownloadError { source: e })?;
        
        let bytes = response.bytes().await
            .map_err(|e| EmbeddingError::ModelDownloadError { source: e })?;
        
        tokio::fs::write(path, bytes).await
            .map_err(|e| EmbeddingError::IoError { source: e })?;
        
        Ok(())
    }
}

/// Text preprocessor for cleaning and normalizing input text
#[derive(Debug)]
pub struct TextPreprocessor {
    enable_preprocessing: bool,
}

impl TextPreprocessor {
    pub fn new(enable_preprocessing: bool) -> Self {
        Self { enable_preprocessing }
    }
    
    /// Preprocess text for embedding generation
    pub fn preprocess(&self, text: &str) -> PreprocessedText {
        if !self.enable_preprocessing {
            let tokens = self.simple_tokenize(text);
            return PreprocessedText {
                original: text.to_string(),
                cleaned: text.to_string(),
                tokens: tokens.clone(),
                token_ids: Vec::new(), // Will be filled by tokenizer
            };
        }
        
        // Clean and normalize text
        let mut cleaned = text.to_string();
        
        // Remove excessive whitespace
        cleaned = regex::Regex::new(r"\s+").unwrap().replace_all(&cleaned, " ").to_string();
        
        // Trim whitespace
        cleaned = cleaned.trim().to_string();
        
        // Convert to lowercase for BM25 (but preserve original case for embedding)
        let tokens = self.simple_tokenize(&cleaned.to_lowercase());
        
        PreprocessedText {
            original: text.to_string(),
            cleaned,
            tokens,
            token_ids: Vec::new(),
        }
    }
    
    /// Simple tokenization (can be enhanced with more sophisticated methods)
    fn simple_tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }
}