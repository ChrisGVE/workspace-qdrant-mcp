//! Embedding generation system for text documents
//!
//! This module provides comprehensive embedding generation capabilities including:
//! - Dense vector embeddings using ONNX Runtime
//! - Sparse vector generation using BM25 algorithm
//! - Model downloading and caching
//! - Text preprocessing and normalization
//! - Embedding result caching for performance

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[cfg(feature = "ml")]
use ort::{session::Session, value::Value};
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;
// use uuid::Uuid;  // Currently unused
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
    #[cfg(feature = "ml")]
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

/// Embedding cache for storing computed embeddings
#[derive(Debug)]
pub struct EmbeddingCache {
    cache: Arc<RwLock<AHashMap<u64, EmbeddingResult>>>,
    max_size: usize,
    access_order: Arc<RwLock<Vec<u64>>>, // LRU tracking
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(AHashMap::new())),
            max_size,
            access_order: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Calculate hash for text (used as cache key)
    fn hash_text(text: &str) -> u64 {
        let mut hasher = AHasher::default();
        text.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Get embedding from cache
    pub async fn get(&self, text: &str) -> Option<EmbeddingResult> {
        let hash = Self::hash_text(text);
        let cache = self.cache.read().await;
        
        if let Some(result) = cache.get(&hash) {
            // Update access order for LRU
            let mut access_order = self.access_order.write().await;
            if let Some(pos) = access_order.iter().position(|&x| x == hash) {
                access_order.remove(pos);
            }
            access_order.push(hash);
            
            return Some(result.clone());
        }
        
        None
    }
    
    /// Store embedding in cache
    pub async fn put(&self, text: &str, result: EmbeddingResult) {
        let hash = Self::hash_text(text);
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        
        // Check if we need to evict old entries
        while cache.len() >= self.max_size && !cache.is_empty() {
            if let Some(old_hash) = access_order.first().copied() {
                cache.remove(&old_hash);
                access_order.remove(0);
            } else {
                break;
            }
        }
        
        // Insert new entry
        cache.insert(hash, result);
        access_order.push(hash);
    }
    
    /// Clear all cached embeddings
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        cache.clear();
        access_order.clear();
    }
    
    /// Get cache statistics
    pub async fn stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        (cache.len(), self.max_size)
    }
}

/// Main embedding generator that orchestrates all components
#[derive(Debug)]
pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
    model_manager: ModelManager,
    preprocessor: TextPreprocessor,
    cache: EmbeddingCache,
    bm25: Arc<RwLock<BM25>>,
    // ONNX Runtime environment removed for simplicity
    sessions: Arc<RwLock<AHashMap<String, Session>>>,
    tokenizers: Arc<RwLock<AHashMap<String, Tokenizer>>>,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let model_manager = ModelManager::new(config.clone());
        let preprocessor = TextPreprocessor::new(config.enable_preprocessing);
        let cache = EmbeddingCache::new(config.max_cache_size);
        let bm25 = Arc::new(RwLock::new(BM25::new(config.bm25_k1, config.bm25_b)));
        
        // ONNX Runtime environment initialization simplified
        
        Ok(Self {
            config,
            model_manager,
            preprocessor,
            cache,
            bm25,
            sessions: Arc::new(RwLock::new(AHashMap::new())),
            tokenizers: Arc::new(RwLock::new(AHashMap::new())),
        })
    }
    
    /// Initialize a model (download if needed and load into memory)
    pub async fn initialize_model(&self, model_name: &str) -> Result<(), EmbeddingError> {
        // Download model if not cached
        if !self.model_manager.is_model_cached(model_name) {
            self.model_manager.download_model(model_name).await?;
        }
        
        // TODO: ONNX session loading temporarily disabled due to API changes
        // This will be re-enabled once the correct ORT v2.0 API is determined
        Err(EmbeddingError::OnnxError {
            message: "ONNX session loading temporarily disabled for API compatibility".to_string()
        })
    }
    
    /// Generate embeddings for a single text
    pub async fn generate_embedding(
        &self,
        text: &str,
        model_name: &str,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        // Check cache first
        if let Some(cached) = self.cache.get(text).await {
            return Ok(cached);
        }
        
        // Preprocess text
        let preprocessed = self.preprocessor.preprocess(text);
        
        // Generate dense embedding
        let dense = self.generate_dense_embedding(&preprocessed, model_name).await?;
        
        // Generate sparse embedding using BM25
        let sparse = {
            let bm25 = self.bm25.read().await;
            bm25.generate_sparse_vector(&preprocessed.tokens)
        };
        
        let result = EmbeddingResult {
            text_hash: EmbeddingCache::hash_text(text),
            dense,
            sparse,
            generated_at: chrono::Utc::now(),
        };
        
        // Cache the result
        self.cache.put(text, result.clone()).await;
        
        Ok(result)
    }
    
    /// Generate embeddings for multiple texts in batch
    pub async fn generate_embeddings_batch(
        &self,
        texts: &[String],
        model_name: &str,
    ) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        
        // Process in batches to manage memory
        for chunk in texts.chunks(self.config.batch_size) {
            for text in chunk {
                let result = self.generate_embedding(text, model_name).await?;
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// Generate dense embedding using ONNX model
    async fn generate_dense_embedding(
        &self,
        preprocessed: &PreprocessedText,
        model_name: &str,
    ) -> Result<DenseEmbedding, EmbeddingError> {
        // Get tokenizer and session
        let tokenizer = {
            let tokenizers = self.tokenizers.read().await;
            tokenizers.get(model_name)
                .ok_or_else(|| EmbeddingError::ModelNotFound {
                    model_name: model_name.to_string(),
                })?
                .clone()
        };
        
        // Get the session - we'll need to handle mutability properly
        let session_exists = {
            let sessions = self.sessions.read().await;
            sessions.contains_key(model_name)
        };
        
        if !session_exists {
            return Err(EmbeddingError::ModelNotFound {
                model_name: model_name.to_string(),
            });
        }
        
        // Tokenize the text
        let encoding = tokenizer.encode(preprocessed.cleaned.as_str(), false)
            .map_err(|e| EmbeddingError::TokenizationError { source: Box::new(e) })?;
        
        let input_ids = encoding.get_ids();
        let attention_mask: Vec<i64> = vec![1i64; input_ids.len()];
        
        // Truncate if needed
        let max_len = self.config.max_sequence_length.min(input_ids.len());
        let input_ids: Vec<i64> = input_ids[..max_len].iter().map(|&x| x as i64).collect();
        let attention_mask = attention_mask[..max_len].to_vec();
        
        // Create ONNX input tensors
        let input_ids_tensor = Value::from_array(([1, input_ids.len()], input_ids.into_boxed_slice()))
            .map_err(|e| EmbeddingError::OnnxError {
                message: format!("Failed to create input_ids tensor: {}", e)
            })?;
        
        let attention_mask_tensor = Value::from_array(([1, attention_mask.len()], attention_mask.into_boxed_slice()))
            .map_err(|e| EmbeddingError::OnnxError {
                message: format!("Failed to create attention_mask tensor: {}", e)
            })?;
        
        // Run inference with write lock on session
        let embedding_vec = {
            let mut sessions = self.sessions.write().await;
            let session = sessions.get_mut(model_name).unwrap();
            
            let inputs = vec![
                ("input_ids", input_ids_tensor),
                ("attention_mask", attention_mask_tensor),
            ];
            let outputs = session.run(inputs)
                .map_err(|e| EmbeddingError::OnnxError {
                    message: format!("ONNX inference failed: {}", e)
                })?;
            
            // Extract embedding from output (use named output)
            let embedding_tensor = outputs.get("last_hidden_state").ok_or_else(|| EmbeddingError::OnnxError {
                message: "No output tensor found".to_string()
            })?;
            let embedding_data = embedding_tensor.try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::OnnxError {
                    message: format!("Failed to extract embedding tensor: {}", e)
                })?;
            
            // Convert to vector (assuming output is [1, embedding_dim])
            embedding_data.1.to_vec()
        };
        
        Ok(DenseEmbedding {
            vector: embedding_vec,
            model_name: model_name.to_string(),
            sequence_length: max_len,
        })
    }
    
    /// Add a document to the BM25 corpus for better sparse vector generation
    pub async fn add_document_to_corpus(&self, text: &str) {
        let preprocessed = self.preprocessor.preprocess(text);
        let mut bm25 = self.bm25.write().await;
        bm25.add_document(&preprocessed.tokens);
    }
    
    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        self.cache.stats().await
    }
    
    /// Clear embedding cache
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
    }
    
    /// Get list of available models
    pub fn available_models(&self) -> Vec<String> {
        self.model_manager.models.keys().cloned().collect()
    }
    
    /// Check if a model is ready to use
    pub async fn is_model_ready(&self, model_name: &str) -> bool {
        let sessions = self.sessions.read().await;
        let tokenizers = self.tokenizers.read().await;
        sessions.contains_key(model_name) && tokenizers.contains_key(model_name)
    }
}