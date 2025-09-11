//! Comprehensive tests for the embedding generation system
//!
//! These tests validate all aspects of the embedding functionality including
//! model management, text preprocessing, embedding generation, caching, and BM25.

use std::path::PathBuf;
use std::time::Instant;
use tempfile::TempDir;
use tokio;
use workspace_qdrant_core::embedding::{
    EmbeddingConfig, EmbeddingGenerator, EmbeddingError,
    TextPreprocessor, BM25, ModelManager, EmbeddingCache
};

/// Helper function to create a temporary directory for tests
fn create_temp_dir() -> TempDir {
    tempfile::TempDir::new().expect("Failed to create temp directory")
}

/// Helper function to create test configuration
fn create_test_config(cache_dir: PathBuf) -> EmbeddingConfig {
    EmbeddingConfig {
        model_cache_dir: cache_dir,
        max_cache_size: 100,
        batch_size: 4,
        max_sequence_length: 128,
        enable_preprocessing: true,
        bm25_k1: 1.2,
        bm25_b: 0.75,
    }
}

#[test]
fn test_embedding_config_default() {
    let config = EmbeddingConfig::default();
    assert_eq!(config.max_cache_size, 10000);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.max_sequence_length, 512);
    assert!(config.enable_preprocessing);
    assert_eq!(config.bm25_k1, 1.2);
    assert_eq!(config.bm25_b, 0.75);
}

#[test]
fn test_text_preprocessor() {
    let preprocessor = TextPreprocessor::new(true);
    
    // Test basic preprocessing
    let text = "  Hello   World!  How are you?  ";
    let processed = preprocessor.preprocess(text);
    
    assert_eq!(processed.original, text);
    assert_eq!(processed.cleaned.trim(), "Hello World! How are you?");
    assert!(!processed.tokens.is_empty());
    
    // Test disabled preprocessing
    let preprocessor_disabled = TextPreprocessor::new(false);
    let processed_disabled = preprocessor_disabled.preprocess(text);
    assert_eq!(processed_disabled.original, text);
    assert_eq!(processed_disabled.cleaned, text);
}

#[test]
fn test_text_preprocessor_tokenization() {
    let preprocessor = TextPreprocessor::new(true);
    let text = "Hello, world! This is a test.";
    let processed = preprocessor.preprocess(text);
    
    // Check that tokens are properly extracted
    assert!(processed.tokens.contains(&"hello".to_string()));
    assert!(processed.tokens.contains(&"world".to_string()));
    assert!(processed.tokens.contains(&"test".to_string()));
    
    // Check that punctuation is filtered out
    assert!(!processed.tokens.iter().any(|t| t.contains(',')));
    assert!(!processed.tokens.iter().any(|t| t.contains('!')));
    assert!(!processed.tokens.iter().any(|t| t.contains('.')));
}

#[test]
fn test_bm25_basic_functionality() {
    let mut bm25 = BM25::new(1.2, 0.75);
    
    // Add some documents to build vocabulary and IDF
    let doc1_tokens = vec!["hello".to_string(), "world".to_string()];
    let doc2_tokens = vec!["hello".to_string(), "rust".to_string()];
    let doc3_tokens = vec!["world".to_string(), "programming".to_string()];
    
    bm25.add_document(&doc1_tokens);
    bm25.add_document(&doc2_tokens);
    bm25.add_document(&doc3_tokens);
    
    // Generate sparse vector for a query
    let query_tokens = vec!["hello".to_string(), "world".to_string()];
    let sparse_vector = bm25.generate_sparse_vector(&query_tokens);
    
    // Check that we have non-empty indices and values
    assert!(!sparse_vector.indices.is_empty());
    assert!(!sparse_vector.values.is_empty());
    assert_eq!(sparse_vector.indices.len(), sparse_vector.values.len());
    assert!(sparse_vector.vocab_size > 0);
    
    // Check that values are positive (BM25 scores should be positive)
    for &value in &sparse_vector.values {
        assert!(value >= 0.0);
    }
}

#[test]
fn test_bm25_vocabulary_building() {
    let mut bm25 = BM25::new(1.2, 0.75);
    
    let doc1_tokens = vec!["apple".to_string(), "banana".to_string()];
    let doc2_tokens = vec!["cherry".to_string(), "date".to_string()];
    
    bm25.add_document(&doc1_tokens);
    assert_eq!(bm25.vocab.len(), 2);
    
    bm25.add_document(&doc2_tokens);
    assert_eq!(bm25.vocab.len(), 4);
    
    // Adding document with duplicate terms shouldn't increase vocab size
    bm25.add_document(&doc1_tokens);
    assert_eq!(bm25.vocab.len(), 4);
}

#[tokio::test]
async fn test_embedding_cache_basic_operations() {
    let cache = EmbeddingCache::new(3); // Small cache for testing
    
    // Test cache miss
    let result = cache.get("test text").await;
    assert!(result.is_none());
    
    // Test cache stats
    let (size, max_size) = cache.stats().await;
    assert_eq!(size, 0);
    assert_eq!(max_size, 3);
    
    // Test clear operation
    cache.clear().await;
    let (size, _) = cache.stats().await;
    assert_eq!(size, 0);
}

#[test]
fn test_model_manager_creation() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let model_manager = ModelManager::new(config.clone());
    
    // Test that default model is available
    let available_models = model_manager.models.keys().collect::<Vec<_>>();
    assert!(!available_models.is_empty());
    assert!(available_models.contains(&&"bge-small-en-v1.5".to_string()));
}

#[test]
fn test_model_manager_paths() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let model_manager = ModelManager::new(config);
    
    let model_path = model_manager.get_model_path("test-model");
    let tokenizer_path = model_manager.get_tokenizer_path("test-model");
    
    assert!(model_path.to_string_lossy().contains("test-model"));
    assert!(model_path.to_string_lossy().ends_with(".onnx"));
    assert!(tokenizer_path.to_string_lossy().contains("test-model"));
    assert!(tokenizer_path.to_string_lossy().ends_with("_tokenizer.json"));
}

#[test]
fn test_model_manager_cache_check() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let model_manager = ModelManager::new(config);
    
    // Should not be cached initially
    assert!(!model_manager.is_model_cached("non-existent-model"));
    assert!(!model_manager.is_model_cached("bge-small-en-v1.5"));
}

#[test]
fn test_embedding_generator_creation() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    
    // Test successful creation
    let result = EmbeddingGenerator::new(config);
    assert!(result.is_ok());
    
    let generator = result.unwrap();
    let available_models = generator.available_models();
    assert!(!available_models.is_empty());
}

#[tokio::test]
async fn test_embedding_generator_model_readiness() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Model should not be ready initially
    assert!(!generator.is_model_ready("bge-small-en-v1.5").await);
    assert!(!generator.is_model_ready("non-existent-model").await);
}

#[tokio::test]
async fn test_embedding_generator_cache_operations() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Test cache stats
    let (size, max_size) = generator.cache_stats().await;
    assert_eq!(size, 0);
    assert_eq!(max_size, 100);
    
    // Test cache clear
    generator.clear_cache().await;
    let (size, _) = generator.cache_stats().await;
    assert_eq!(size, 0);
}

#[tokio::test]
async fn test_embedding_generator_corpus_management() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Test adding documents to corpus
    generator.add_document_to_corpus("This is a test document").await;
    generator.add_document_to_corpus("Another document for testing").await;
    generator.add_document_to_corpus("More text content").await;
    
    // This should complete without error
    // The actual corpus building is tested in BM25 tests
}

// Performance and integration tests that require network access
// These are marked with #[ignore] so they don't run in CI by default

#[ignore]
#[tokio::test]
async fn test_model_download_integration() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // This test requires network access and will download the actual model
    // It's marked as #[ignore] to prevent running in CI
    let result = generator.initialize_model("bge-small-en-v1.5").await;
    
    // Should succeed if network is available
    if result.is_ok() {
        assert!(generator.is_model_ready("bge-small-en-v1.5").await);
    } else {
        // If network is not available, we expect specific error types
        match result.unwrap_err() {
            EmbeddingError::ModelDownloadError { .. } => {},
            EmbeddingError::IoError { .. } => {},
            other => panic!("Unexpected error type: {:?}", other),
        }
    }
}

#[ignore]
#[tokio::test]
async fn test_embedding_generation_integration() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Initialize model (requires network)
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        // Skip test if model download fails (e.g., no network)
        return;
    }
    
    // Test single embedding generation
    let text = "This is a test sentence for embedding generation.";
    let result = generator.generate_embedding(text, "bge-small-en-v1.5").await;
    
    if result.is_ok() {
        let embedding = result.unwrap();
        
        // Validate dense embedding
        assert!(!embedding.dense.vector.is_empty());
        assert_eq!(embedding.dense.model_name, "bge-small-en-v1.5");
        assert!(embedding.dense.sequence_length > 0);
        
        // Validate sparse embedding
        assert_eq!(embedding.sparse.indices.len(), embedding.sparse.values.len());
        assert!(embedding.sparse.vocab_size > 0);
        
        // Validate metadata
        assert!(embedding.text_hash > 0);
        assert!(embedding.generated_at <= chrono::Utc::now());
    }
}

#[ignore]
#[tokio::test]
async fn test_batch_embedding_generation() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        return; // Skip if no network
    }
    
    let texts = vec![
        "First test sentence.".to_string(),
        "Second test sentence with different content.".to_string(),
        "Third sentence for batch testing.".to_string(),
        "Fourth and final test sentence.".to_string(),
    ];
    
    let start_time = Instant::now();
    let results = generator.generate_embeddings_batch(&texts, "bge-small-en-v1.5").await;
    let elapsed = start_time.elapsed();
    
    if results.is_ok() {
        let embeddings = results.unwrap();
        assert_eq!(embeddings.len(), texts.len());
        
        // Verify that all embeddings are valid
        for embedding in &embeddings {
            assert!(!embedding.dense.vector.is_empty());
            assert!(!embedding.sparse.indices.is_empty());
        }
        
        // Check that batch processing is reasonably fast
        // (This is a loose check - actual performance will vary)
        assert!(elapsed.as_secs() < 30, "Batch processing took too long: {:?}", elapsed);
        
        println!("Batch processing of {} texts took: {:?}", texts.len(), elapsed);
    }
}

#[ignore]
#[tokio::test]
async fn test_embedding_consistency() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        return; // Skip if no network
    }
    
    let text = "Consistency test sentence.";
    
    // Generate embedding twice
    let embedding1 = generator.generate_embedding(text, "bge-small-en-v1.5").await;
    let embedding2 = generator.generate_embedding(text, "bge-small-en-v1.5").await;
    
    if embedding1.is_ok() && embedding2.is_ok() {
        let emb1 = embedding1.unwrap();
        let emb2 = embedding2.unwrap();
        
        // Dense embeddings should be identical (deterministic)
        assert_eq!(emb1.dense.vector, emb2.dense.vector);
        
        // Sparse embeddings should be identical
        assert_eq!(emb1.sparse.indices, emb2.sparse.indices);
        assert_eq!(emb1.sparse.values, emb2.sparse.values);
        
        // Text hashes should be identical
        assert_eq!(emb1.text_hash, emb2.text_hash);
    }
}

#[ignore]
#[tokio::test]
async fn test_caching_effectiveness() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        return; // Skip if no network
    }
    
    let text = "Caching test sentence.";
    
    // First generation (cache miss)
    let start1 = Instant::now();
    let result1 = generator.generate_embedding(text, "bge-small-en-v1.5").await;
    let time1 = start1.elapsed();
    
    if result1.is_err() {
        return; // Skip if generation fails
    }
    
    // Check cache stats
    let (cache_size, _) = generator.cache_stats().await;
    assert_eq!(cache_size, 1);
    
    // Second generation (cache hit)
    let start2 = Instant::now();
    let result2 = generator.generate_embedding(text, "bge-small-en-v1.5").await;
    let time2 = start2.elapsed();
    
    if result2.is_ok() {
        // Cache hit should be significantly faster
        // Note: This is a loose check as performance can vary
        println!("First generation: {:?}, Second generation: {:?}", time1, time2);
        
        // At minimum, cache hit should be less than 10ms for simple lookup
        assert!(time2.as_millis() < 10, "Cache hit was too slow: {:?}", time2);
        
        // Results should be identical
        let emb1 = result1.unwrap();
        let emb2 = result2.unwrap();
        assert_eq!(emb1.text_hash, emb2.text_hash);
        assert_eq!(emb1.dense.vector, emb2.dense.vector);
    }
}

// Error handling tests

#[tokio::test]
async fn test_invalid_model_error() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Try to initialize non-existent model
    let result = generator.initialize_model("non-existent-model").await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        EmbeddingError::ModelNotFound { model_name } => {
            assert_eq!(model_name, "non-existent-model");
        },
        other => panic!("Expected ModelNotFound error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_embedding_without_model_initialization() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    // Try to generate embedding without initializing model
    let result = generator.generate_embedding("test", "bge-small-en-v1.5").await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        EmbeddingError::ModelNotFound { .. } => {
            // Expected error
        },
        other => panic!("Expected ModelNotFound error, got: {:?}", other),
    }
}

// Stress tests

#[ignore]
#[tokio::test]
async fn test_large_text_processing() {
    let temp_dir = create_temp_dir();
    let config = create_test_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        return; // Skip if no network
    }
    
    // Create a large text (beyond normal sequence length)
    let large_text = "This is a test sentence. ".repeat(100); // ~2500 chars
    
    let result = generator.generate_embedding(&large_text, "bge-small-en-v1.5").await;
    
    if result.is_ok() {
        let embedding = result.unwrap();
        
        // Should handle truncation gracefully
        assert!(!embedding.dense.vector.is_empty());
        assert!(embedding.dense.sequence_length <= 128); // Our test config limit
    }
}

#[ignore]
#[tokio::test] 
async fn test_memory_usage_with_large_cache() {
    let temp_dir = create_temp_dir();
    let mut config = create_test_config(temp_dir.path().to_path_buf());
    config.max_cache_size = 1000; // Larger cache
    
    let generator = EmbeddingGenerator::new(config).expect("Failed to create generator");
    
    if generator.initialize_model("bge-small-en-v1.5").await.is_err() {
        return; // Skip if no network
    }
    
    // Generate embeddings for many different texts
    for i in 0..50 {
        let text = format!("Test sentence number {}.", i);
        let _ = generator.generate_embedding(&text, "bge-small-en-v1.5").await;
    }
    
    let (cache_size, max_cache_size) = generator.cache_stats().await;
    assert!(cache_size <= max_cache_size);
    assert!(cache_size > 0);
    
    println!("Cache utilization: {}/{}", cache_size, max_cache_size);
}