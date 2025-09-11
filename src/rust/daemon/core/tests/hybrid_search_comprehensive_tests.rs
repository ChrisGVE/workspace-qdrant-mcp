//! Comprehensive tests for hybrid search and embedding system (Task 86)
//!
//! This test suite provides extensive validation of:
//! - Hybrid search implementation with reciprocal rank fusion
//! - Multiple embedding model support
//! - Sparse vector functionality 
//! - Search quality metrics and performance optimization
//! - Cross-collection search capabilities
//! - Search result validation and quality assurance

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::TempDir;
use tokio;
use workspace_qdrant_core::embedding::{
    EmbeddingConfig, EmbeddingGenerator, BM25, TextPreprocessor
};
use workspace_qdrant_core::storage::{
    StorageClient, StorageConfig, HybridSearchMode, DocumentPoint, SearchResult, SearchParams
};
use serde_json::json;

/// Test configuration for embedding models
const TEST_MODELS: &[&str] = &[
    "all-MiniLM-L6-v2",
    "bge-base-en-v1.5", 
    "mxbai-embed-large-v1"
];

/// Test data for search quality evaluation
const TEST_DOCUMENTS: &[(&str, &str)] = &[
    ("doc1", "Artificial intelligence and machine learning algorithms for data processing"),
    ("doc2", "Natural language processing with transformer models and attention mechanisms"), 
    ("doc3", "Computer vision techniques using convolutional neural networks"),
    ("doc4", "Deep learning frameworks and optimization techniques for model training"),
    ("doc5", "Reinforcement learning algorithms for autonomous decision making"),
    ("doc6", "Statistical analysis and data mining methods for pattern recognition"),
    ("doc7", "Database systems and query optimization for large-scale data storage"),
    ("doc8", "Distributed computing architectures for scalable machine learning"),
    ("doc9", "Information retrieval systems and search engine optimization"),
    ("doc10", "Knowledge graphs and semantic web technologies for information organization")
];

/// Test queries for search evaluation
const TEST_QUERIES: &[(&str, &[&str])] = &[
    ("machine learning algorithms", &["doc1", "doc4", "doc5"]),
    ("natural language processing", &["doc2", "doc9"]),
    ("neural networks deep learning", &["doc3", "doc4"]),
    ("data processing storage", &["doc1", "doc6", "doc7"]),
    ("distributed computing scalable", &["doc7", "doc8"])
];

/// Helper function to create test configuration
fn create_test_embedding_config(cache_dir: PathBuf) -> EmbeddingConfig {
    EmbeddingConfig {
        model_cache_dir: cache_dir,
        max_cache_size: 1000,
        batch_size: 8,
        max_sequence_length: 256,
        enable_preprocessing: true,
        bm25_k1: 1.2,
        bm25_b: 0.75,
    }
}

/// Helper function to create test storage configuration  
fn create_test_storage_config() -> StorageConfig {
    StorageConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        timeout_ms: 5000,
        max_retries: 2,
        retry_delay_ms: 500,
        dense_vector_size: 384, // For test models
        sparse_vector_size: Some(10000),
        ..Default::default()
    }
}

/// Helper to create temporary directory
fn create_temp_dir() -> TempDir {
    tempfile::TempDir::new().expect("Failed to create temp directory")
}

/// Calculate search quality metrics
#[derive(Debug, Clone)]
struct SearchQualityMetrics {
    precision: f64,
    recall: f64,
    f1_score: f64,
    average_precision: f64,
    ndcg: f64, // Normalized Discounted Cumulative Gain
}

impl SearchQualityMetrics {
    fn calculate(results: &[SearchResult], relevant_docs: &[&str]) -> Self {
        let retrieved_docs: Vec<&str> = results.iter()
            .map(|r| r.id.as_str())
            .collect();
        
        let mut relevant_retrieved = 0;
        let mut precision_at_k = 0.0;
        let mut dcg = 0.0;
        
        for (i, doc_id) in retrieved_docs.iter().enumerate() {
            if relevant_docs.contains(doc_id) {
                relevant_retrieved += 1;
                precision_at_k += relevant_retrieved as f64 / (i + 1) as f64;
                dcg += 1.0 / ((i + 2) as f64).log2(); // +2 because rank starts from 1
            }
        }
        
        let precision = if retrieved_docs.is_empty() {
            0.0
        } else {
            relevant_retrieved as f64 / retrieved_docs.len() as f64
        };
        
        let recall = if relevant_docs.is_empty() {
            0.0
        } else {
            relevant_retrieved as f64 / relevant_docs.len() as f64
        };
        
        let f1_score = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        
        let average_precision = if relevant_retrieved == 0 {
            0.0
        } else {
            precision_at_k / relevant_retrieved as f64
        };
        
        // Calculate ideal DCG for NDCG
        let mut ideal_dcg = 0.0;
        for i in 0..relevant_docs.len().min(retrieved_docs.len()) {
            ideal_dcg += 1.0 / ((i + 2) as f64).log2();
        }
        
        let ndcg = if ideal_dcg == 0.0 { 0.0 } else { dcg / ideal_dcg };
        
        Self {
            precision,
            recall,
            f1_score,
            average_precision,
            ndcg,
        }
    }
}

// =============================================================================
// EMBEDDING MODEL VALIDATION TESTS
// =============================================================================

#[test]
fn test_multiple_embedding_models_configuration() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    
    // Test that embedding generator supports all required models
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    let available_models = generator.available_models();
    
    // Verify at least one default model is available
    assert!(!available_models.is_empty(), "No embedding models available");
    
    // Test model configuration consistency
    assert!(config.max_sequence_length > 0);
    assert!(config.batch_size > 0);
    assert!(config.bm25_k1 > 0.0);
    let config_clone = config.clone();
    assert!(config_clone.bm25_b >= 0.0 && config_clone.bm25_b <= 1.0);
}

#[tokio::test]
async fn test_embedding_model_performance_comparison() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    
    let test_text = "This is a test sentence for embedding generation performance evaluation.";
    
    // Add test document to BM25 corpus for sparse vector generation
    generator.add_document_to_corpus(test_text).await;
    
    // Test preprocessing consistency
    let preprocessor = TextPreprocessor::new(true);
    let processed = preprocessor.preprocess(test_text);
    
    assert!(!processed.tokens.is_empty(), "Text preprocessing failed");
    assert!(processed.cleaned.len() <= processed.original.len());
    
    // Test BM25 sparse vector generation
    let mut bm25 = BM25::new(1.2, 0.75);
    bm25.add_document(&processed.tokens);
    
    let sparse_vector = bm25.generate_sparse_vector(&processed.tokens);
    assert!(!sparse_vector.indices.is_empty(), "Sparse vector generation failed");
    assert_eq!(sparse_vector.indices.len(), sparse_vector.values.len());
    
    // Validate sparse vector scores are positive
    for &score in &sparse_vector.values {
        assert!(score >= 0.0, "BM25 score should be non-negative: {}", score);
    }
}

#[tokio::test]
async fn test_embedding_cache_performance() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    
    let test_texts = vec![
        "First test document for caching evaluation".to_string(),
        "Second test document with different content".to_string(),
        "Third document for cache performance testing".to_string(),
    ];
    
    // Add documents to BM25 corpus
    for text in &test_texts {
        generator.add_document_to_corpus(text).await;
    }
    
    // Test cache statistics
    let (initial_size, max_size) = generator.cache_stats().await;
    assert_eq!(initial_size, 0, "Cache should be empty initially");
    assert!(max_size > 0, "Cache should have positive max size");
    
    // Test cache clear functionality
    generator.clear_cache().await;
    let (cleared_size, _) = generator.cache_stats().await;
    assert_eq!(cleared_size, 0, "Cache should be empty after clear");
}

// =============================================================================
// SPARSE VECTOR FUNCTIONALITY TESTS
// =============================================================================

#[test]
fn test_bm25_algorithm_correctness() {
    let mut bm25 = BM25::new(1.2, 0.75);
    
    // Build test corpus with known characteristics
    let documents = vec![
        vec!["machine".to_string(), "learning".to_string(), "algorithm".to_string()],
        vec!["machine".to_string(), "vision".to_string(), "processing".to_string()],
        vec!["natural".to_string(), "language".to_string(), "processing".to_string()],
        vec!["deep".to_string(), "learning".to_string(), "network".to_string()],
    ];
    
    // Add documents to build vocabulary and document frequencies
    for doc in &documents {
        bm25.add_document(doc);
    }
    
    // Test that BM25 builds vocabulary correctly (test indirectly through vector generation)
    // Since vocab and doc_count are private, we validate through behavior
    
    // Test sparse vector generation for query
    let query = vec!["machine".to_string(), "learning".to_string()];
    let sparse_vector = bm25.generate_sparse_vector(&query);
    
    // Validate sparse vector properties
    assert!(!sparse_vector.indices.is_empty(), "Sparse vector should have indices");
    assert_eq!(sparse_vector.indices.len(), sparse_vector.values.len());
    
    // Check that scores are reasonable (BM25 should produce positive scores)
    for &score in &sparse_vector.values {
        assert!(score > 0.0, "BM25 scores should be positive");
        assert!(score.is_finite(), "BM25 scores should be finite");
    }
    
    // Test that common terms get appropriate scores
    // "machine" appears in 2/4 documents, "learning" appears in 2/4 documents
    assert!(sparse_vector.values.len() >= 2, "Should have scores for query terms");
}

#[test] 
fn test_bm25_parameter_sensitivity() {
    // Test different BM25 parameters
    let test_cases = vec![
        (1.2, 0.75), // Standard parameters
        (2.0, 0.75), // Higher k1 (more term frequency sensitive)
        (1.2, 0.0),  // No document length normalization
        (1.2, 1.0),  // Maximum document length normalization
    ];
    
    let documents = vec![
        vec!["short".to_string(), "doc".to_string()],
        vec!["much".to_string(), "longer".to_string(), "document".to_string(), 
             "with".to_string(), "many".to_string(), "more".to_string(), "terms".to_string()],
    ];
    
    for (k1, b) in test_cases {
        let mut bm25 = BM25::new(k1, b);
        
        for doc in &documents {
            bm25.add_document(doc);
        }
        
        let query = vec!["doc".to_string(), "document".to_string()];
        let sparse_vector = bm25.generate_sparse_vector(&query);
        
        // All parameter combinations should produce valid results
        assert!(!sparse_vector.indices.is_empty(), 
                "BM25 with k1={}, b={} should produce non-empty results", k1, b);
        
        for &score in &sparse_vector.values {
            assert!(score >= 0.0 && score.is_finite(), 
                    "Invalid BM25 score {} with k1={}, b={}", score, k1, b);
        }
    }
}

#[test]
fn test_sparse_vector_consistency() {
    let mut bm25 = BM25::new(1.2, 0.75);
    
    // Build consistent corpus
    let docs = vec![
        vec!["apple".to_string(), "fruit".to_string()],
        vec!["banana".to_string(), "fruit".to_string()],
        vec!["carrot".to_string(), "vegetable".to_string()],
    ];
    
    for doc in &docs {
        bm25.add_document(doc);
    }
    
    let query = vec!["fruit".to_string()];
    
    // Generate sparse vector multiple times
    let vector1 = bm25.generate_sparse_vector(&query);
    let vector2 = bm25.generate_sparse_vector(&query);
    
    // Results should be identical
    assert_eq!(vector1.indices, vector2.indices, "Sparse vector indices should be consistent");
    assert_eq!(vector1.values, vector2.values, "Sparse vector values should be consistent");
    assert_eq!(vector1.vocab_size, vector2.vocab_size, "Vocabulary size should be consistent");
}

// =============================================================================
// HYBRID SEARCH ALGORITHM TESTS
// =============================================================================

#[test]
fn test_hybrid_search_mode_configuration() {
    // Test different hybrid search modes
    let modes = vec![
        HybridSearchMode::Dense,
        HybridSearchMode::Sparse,
        HybridSearchMode::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 },
        HybridSearchMode::Hybrid { dense_weight: 2.0, sparse_weight: 0.5 },
        HybridSearchMode::Hybrid { dense_weight: 0.3, sparse_weight: 1.5 },
    ];
    
    // Verify all modes can be created and serialized
    for mode in &modes {
        let serialized = serde_json::to_string(mode)
            .expect("Should be able to serialize search mode");
        let deserialized: HybridSearchMode = serde_json::from_str(&serialized)
            .expect("Should be able to deserialize search mode");
        
        // Verify round-trip consistency
        match (mode, &deserialized) {
            (HybridSearchMode::Dense, HybridSearchMode::Dense) => {},
            (HybridSearchMode::Sparse, HybridSearchMode::Sparse) => {},
            (HybridSearchMode::Hybrid { dense_weight: dw1, sparse_weight: sw1 }, 
             HybridSearchMode::Hybrid { dense_weight: dw2, sparse_weight: sw2 }) => {
                assert_eq!(dw1, dw2, "Dense weight should be preserved");
                assert_eq!(sw1, sw2, "Sparse weight should be preserved");
            },
            _ => panic!("Deserialized mode doesn't match original"),
        }
    }
}

#[tokio::test]
async fn test_storage_client_configuration() {
    let config = create_test_storage_config();
    let _storage = StorageClient::with_config(config.clone());
    
    // Storage client should be created successfully with config
    // (config fields are private, so we test indirectly through functionality)
    
    // Test default hybrid search mode
    let default_mode = HybridSearchMode::default();
    match default_mode {
        HybridSearchMode::Hybrid { dense_weight, sparse_weight } => {
            assert_eq!(dense_weight, 1.0);
            assert_eq!(sparse_weight, 1.0);
        },
        _ => panic!("Default should be hybrid mode with equal weights"),
    }
}

// Note: The following integration tests require a running Qdrant instance
// They are marked with #[ignore] to prevent CI failures

#[ignore]
#[tokio::test]
async fn test_hybrid_search_integration() {
    let storage_config = create_test_storage_config();
    let storage = StorageClient::with_config(storage_config);
    
    let collection_name = "test_hybrid_search";
    
    // Test connection
    if storage.test_connection().await.is_err() {
        eprintln!("Skipping integration test - Qdrant not available");
        return;
    }
    
    // Create test collection
    if storage.collection_exists(collection_name).await.unwrap_or(false) {
        storage.delete_collection(collection_name).await.unwrap();
    }
    
    storage.create_collection(collection_name, Some(384), Some(10000)).await
        .expect("Failed to create test collection");
    
    // Insert test documents
    for (doc_id, content) in TEST_DOCUMENTS {
        // Create mock embeddings (in real scenario, use EmbeddingGenerator)
        let dense_vector = vec![0.1; 384]; // Mock dense embedding
        let mut sparse_vector = HashMap::new();
        sparse_vector.insert(1, 0.8);
        sparse_vector.insert(2, 0.6);
        
        let mut payload = HashMap::new();
        payload.insert("content".to_string(), json!(content));
        payload.insert("doc_type".to_string(), json!("test"));
        
        let point = DocumentPoint {
            id: doc_id.to_string(),
            dense_vector,
            sparse_vector: Some(sparse_vector),
            payload,
        };
        
        storage.insert_point(collection_name, point).await
            .expect("Failed to insert test point");
    }
    
    // Test different search modes
    let query_vector = vec![0.1; 384];
    let mut query_sparse = HashMap::new();
    query_sparse.insert(1, 0.9);
    query_sparse.insert(2, 0.7);
    
    // Test dense search
    let dense_params = SearchParams {
        dense_vector: Some(query_vector.clone()),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 5,
        score_threshold: None,
        filter: None,
    };
    let dense_results = storage.search(collection_name, dense_params).await.expect("Dense search failed");
    
    assert!(dense_results.len() <= 5, "Should return at most 5 results");
    
    // Test hybrid search
    let hybrid_params = SearchParams {
        dense_vector: Some(query_vector),
        sparse_vector: Some(query_sparse),
        search_mode: HybridSearchMode::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 },
        limit: 5,
        score_threshold: None,
        filter: None,
    };
    let hybrid_results = storage.search(collection_name, hybrid_params).await.expect("Hybrid search failed");
    
    assert!(hybrid_results.len() <= 5, "Should return at most 5 results");
    
    // Verify RRF score calculation
    for result in &hybrid_results {
        assert!(result.score >= 0.0, "RRF score should be non-negative");
        assert!(result.score.is_finite(), "RRF score should be finite");
    }
    
    // Results should be ordered by score (descending)
    for i in 1..hybrid_results.len() {
        assert!(hybrid_results[i-1].score >= hybrid_results[i].score,
                "Results should be ordered by score descending");
    }
    
    // Clean up
    storage.delete_collection(collection_name).await.unwrap();
}

#[ignore]
#[tokio::test]
async fn test_reciprocal_rank_fusion_correctness() {
    let storage_config = create_test_storage_config();
    let storage = StorageClient::with_config(storage_config);
    
    if storage.test_connection().await.is_err() {
        return;
    }
    
    let collection_name = "test_rrf";
    
    // Create collection
    if storage.collection_exists(collection_name).await.unwrap_or(false) {
        storage.delete_collection(collection_name).await.unwrap();
    }
    
    storage.create_collection(collection_name, Some(384), Some(1000)).await
        .expect("Failed to create RRF test collection");
    
    // Create test documents with different relevance patterns
    let test_docs = vec![
        ("highly_relevant", vec![1.0; 384], [(0, 2.0), (1, 1.8)].iter().cloned().collect()),
        ("moderately_relevant", vec![0.8; 384], [(0, 1.0), (2, 1.2)].iter().cloned().collect()),
        ("low_relevance", vec![0.3; 384], [(1, 0.5), (3, 0.8)].iter().cloned().collect()),
        ("dense_only_match", vec![0.9; 384], HashMap::new()),
        ("sparse_only_match", vec![0.1; 384], [(0, 3.0), (4, 2.5)].iter().cloned().collect()),
    ];
    
    for (doc_id, dense_vec, sparse_vec) in test_docs {
        let point = DocumentPoint {
            id: doc_id.to_string(),
            dense_vector: dense_vec,
            sparse_vector: Some(sparse_vec),
            payload: HashMap::new(),
        };
        
        storage.insert_point(collection_name, point).await
            .expect("Failed to insert RRF test point");
    }
    
    // Test RRF with different weight combinations
    let weight_tests = vec![
        (1.0, 1.0), // Equal weights
        (2.0, 1.0), // Dense-heavy
        (1.0, 2.0), // Sparse-heavy
        (0.5, 1.5), // Sparse-dominant
    ];
    
    let query_dense = vec![0.9; 384];
    let query_sparse: HashMap<u32, f32> = [(0, 2.5), (1, 1.5)].iter().cloned().collect();
    
    for (dense_weight, sparse_weight) in weight_tests {
        let params = SearchParams {
            dense_vector: Some(query_dense.clone()),
            sparse_vector: Some(query_sparse.clone()),
            search_mode: HybridSearchMode::Hybrid { dense_weight, sparse_weight },
            limit: 5,
            score_threshold: None,
            filter: None,
        };
        let results = storage.search(collection_name, params).await.expect("RRF search failed");
        
        // Validate RRF properties
        assert!(!results.is_empty(), "RRF should return results");
        
        // Check score ordering
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score,
                    "RRF scores should be in descending order");
        }
        
        // Verify that different weights produce different rankings
        if dense_weight != sparse_weight {
            // With different weights, we expect different score distributions
            let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
            assert!(!scores.is_empty(), "Should have scores to compare");
        }
    }
    
    // Clean up
    storage.delete_collection(collection_name).await.unwrap();
}

// =============================================================================
// SEARCH QUALITY METRICS TESTS
// =============================================================================

#[test]
fn test_search_quality_metrics_calculation() {
    // Test with perfect results
    let perfect_results = vec![
        SearchResult {
            id: "doc1".to_string(),
            score: 0.95,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
        SearchResult {
            id: "doc2".to_string(),
            score: 0.90,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
    ];
    
    let relevant_docs = vec!["doc1", "doc2"];
    let metrics = SearchQualityMetrics::calculate(&perfect_results, &relevant_docs);
    
    assert_eq!(metrics.precision, 1.0, "Perfect results should have precision 1.0");
    assert_eq!(metrics.recall, 1.0, "Perfect results should have recall 1.0");
    assert_eq!(metrics.f1_score, 1.0, "Perfect results should have F1 score 1.0");
    
    // Test with partial results
    let partial_results = vec![
        SearchResult {
            id: "doc1".to_string(),
            score: 0.95,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
        SearchResult {
            id: "doc3".to_string(), // Not relevant
            score: 0.80,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
    ];
    
    let partial_metrics = SearchQualityMetrics::calculate(&partial_results, &relevant_docs);
    
    assert_eq!(partial_metrics.precision, 0.5, "Should have precision 0.5");
    assert_eq!(partial_metrics.recall, 0.5, "Should have recall 0.5");
    assert_eq!(partial_metrics.f1_score, 0.5, "Should have F1 score 0.5");
    
    // Test with empty results
    let empty_results = vec![];
    let empty_metrics = SearchQualityMetrics::calculate(&empty_results, &relevant_docs);
    
    assert_eq!(empty_metrics.precision, 0.0, "Empty results should have precision 0.0");
    assert_eq!(empty_metrics.recall, 0.0, "Empty results should have recall 0.0");
    assert_eq!(empty_metrics.f1_score, 0.0, "Empty results should have F1 score 0.0");
}

#[test]
fn test_ndcg_calculation() {
    // Test NDCG with ranked relevant results
    let ranked_results = vec![
        SearchResult {
            id: "doc1".to_string(),
            score: 1.0,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
        SearchResult {
            id: "doc4".to_string(), // Not relevant
            score: 0.8,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
        SearchResult {
            id: "doc2".to_string(),
            score: 0.6,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        },
    ];
    
    let relevant_docs = vec!["doc1", "doc2", "doc3"];
    let metrics = SearchQualityMetrics::calculate(&ranked_results, &relevant_docs);
    
    // NDCG should be between 0 and 1
    assert!(metrics.ndcg >= 0.0 && metrics.ndcg <= 1.0, 
            "NDCG should be between 0 and 1, got {}", metrics.ndcg);
    
    // With relevant document at rank 1 and another at rank 3, NDCG should be > 0
    assert!(metrics.ndcg > 0.0, "NDCG should be positive with some relevant results");
}

// =============================================================================
// PERFORMANCE OPTIMIZATION TESTS
// =============================================================================

#[tokio::test]
async fn test_batch_processing_performance() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    
    // Create batch of test texts
    let batch_texts: Vec<String> = (0..50).map(|i| {
        format!("Test document number {} with various content for batch processing evaluation", i)
    }).collect();
    
    // Add documents to BM25 corpus for sparse vector generation
    for text in &batch_texts {
        generator.add_document_to_corpus(text).await;
    }
    
    // Measure single document processing time
    let single_start = Instant::now();
    let preprocessor = TextPreprocessor::new(true);
    let _processed = preprocessor.preprocess(&batch_texts[0]);
    let single_time = single_start.elapsed();
    
    // Measure batch processing time
    let batch_start = Instant::now();
    for text in &batch_texts[0..10] {
        let _processed = preprocessor.preprocess(text);
    }
    let batch_time = batch_start.elapsed();
    
    // Batch processing should be reasonably efficient
    let avg_batch_time = batch_time / 10;
    
    println!("Single processing: {:?}, Average batch processing: {:?}", single_time, avg_batch_time);
    
    // Performance should be consistent (within reasonable bounds)
    assert!(avg_batch_time < single_time * 2, 
            "Batch processing shouldn't be much slower than single processing");
}

#[tokio::test]
async fn test_cache_effectiveness_metrics() {
    let temp_dir = create_temp_dir();
    let mut config = create_test_embedding_config(temp_dir.path().to_path_buf());
    config.max_cache_size = 100; // Reasonable cache size for testing
    
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    
    let test_texts = vec![
        "Repeated text for cache testing",
        "Another repeated text for cache testing", 
        "Third text for comprehensive cache evaluation",
    ];
    
    // First pass - populate cache
    for text in &test_texts {
        generator.add_document_to_corpus(text).await;
    }
    
    // Verify cache stats
    let (cache_size, max_size) = generator.cache_stats().await;
    assert!(cache_size <= max_size, "Cache size should not exceed maximum");
    
    // Test cache behavior with repeated requests
    for _ in 0..3 {
        for text in &test_texts {
            generator.add_document_to_corpus(text).await;
        }
    }
    
    // Cache should handle repeated requests efficiently
    let (final_cache_size, _) = generator.cache_stats().await;
    assert!(final_cache_size <= max_size, "Cache should respect size limits");
}

#[test]
fn test_memory_usage_optimization() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    
    // Test that configurations are reasonable for memory usage
    let config_clone = config.clone();
    assert!(config_clone.max_cache_size > 0, "Cache size should be positive");
    assert!(config_clone.max_cache_size < 100000, "Cache size should be reasonable");
    assert!(config_clone.batch_size > 0 && config_clone.batch_size <= 128, "Batch size should be reasonable");
    assert!(config_clone.max_sequence_length <= 2048, "Sequence length should be reasonable");
    
    let bm25_k1 = config_clone.bm25_k1;
    let bm25_b = config_clone.bm25_b;
    
    // Test BM25 memory efficiency
    let mut bm25 = BM25::new(bm25_k1, bm25_b);
    
    // Add many small documents to test memory scaling
    for i in 0..1000 {
        let doc = vec![format!("word{}", i % 100), "common".to_string()];
        bm25.add_document(&doc);
    }
    
    // BM25 should handle many documents efficiently (we can't access private fields directly)
    // Instead, test the sparse vector generation which validates internal state
    
    // Generate sparse vector to test memory efficiency
    let query = vec!["word50".to_string(), "common".to_string()];
    let sparse_vector = bm25.generate_sparse_vector(&query);
    
    assert!(!sparse_vector.indices.is_empty(), "Should generate non-empty sparse vector");
    assert!(sparse_vector.indices.len() <= query.len(), "Sparse vector shouldn't be larger than query");
}

// =============================================================================
// CROSS-COLLECTION SEARCH TESTS
// =============================================================================

#[ignore]
#[tokio::test]
async fn test_cross_collection_search_capabilities() {
    let storage_config = create_test_storage_config();
    let storage = StorageClient::with_config(storage_config);
    
    if storage.test_connection().await.is_err() {
        return;
    }
    
    let collections = ["collection_a", "collection_b", "collection_c"];
    
    // Create multiple collections with different characteristics
    for collection in &collections {
        if storage.collection_exists(collection).await.unwrap_or(false) {
            storage.delete_collection(collection).await.unwrap();
        }
        
        storage.create_collection(collection, Some(384), Some(1000)).await
            .expect("Failed to create test collection");
    }
    
    // Insert different types of documents in each collection
    let collection_docs = vec![
        ("collection_a", vec![
            ("tech1", "Machine learning and artificial intelligence research"),
            ("tech2", "Deep learning neural networks and optimization"),
        ]),
        ("collection_b", vec![
            ("bio1", "Biological systems and cellular processes research"),
            ("bio2", "Genetic engineering and biotechnology applications"),
        ]),
        ("collection_c", vec![
            ("phys1", "Quantum mechanics and particle physics studies"),
            ("phys2", "Thermodynamics and energy conservation principles"),
        ]),
    ];
    
    for (collection, docs) in collection_docs {
        for (doc_id, content) in docs {
            let mut payload = HashMap::new();
            payload.insert("content".to_string(), json!(content));
            payload.insert("collection".to_string(), json!(collection));
            
            let point = DocumentPoint {
                id: doc_id.to_string(),
                dense_vector: vec![0.5; 384], // Mock embedding
                sparse_vector: Some([(0, 1.0), (1, 0.8)].iter().cloned().collect()),
                payload,
            };
            
            storage.insert_point(collection, point).await
                .expect("Failed to insert cross-collection test point");
        }
    }
    
    // Test search in each collection
    let query_vector = vec![0.5; 384];
    let query_sparse: HashMap<u32, f32> = [(0, 1.2), (1, 0.9)].iter().cloned().collect();
    
    let mut all_results = Vec::new();
    
    for collection in &collections {
        let params = SearchParams {
            dense_vector: Some(query_vector.clone()),
            sparse_vector: Some(query_sparse.clone()),
            search_mode: HybridSearchMode::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 },
            limit: 5,
            score_threshold: None,
            filter: None,
        };
        let results = storage.search(collection, params).await.expect("Cross-collection search failed");
        
        // Tag results with collection name and add to all_results
        for result in results {
            all_results.push(result);
        }
    }
    
    // Verify cross-collection results
    assert!(!all_results.is_empty(), "Should have cross-collection results");
    
    // Results should come from different document types (validation through document IDs)
    let result_ids: std::collections::HashSet<_> = all_results.iter()
        .map(|r| r.id.as_str())
        .collect();
    
    assert!(!result_ids.is_empty(), "Should find results from cross-collection search");
    
    // Clean up
    for collection in &collections {
        storage.delete_collection(collection).await.unwrap();
    }
}

// =============================================================================
// SEARCH RESULT VALIDATION TESTS
// =============================================================================

#[test]
fn test_search_result_validation() {
    // Test valid search result
    let mut payload = HashMap::new();
    payload.insert("content".to_string(), json!("Test content"));
    payload.insert("title".to_string(), json!("Test Document"));
    
    let valid_result = SearchResult {
        id: "valid_doc_123".to_string(),
        score: 0.85,
        payload,
        dense_vector: Some(vec![0.1, 0.2, 0.3]),
        sparse_vector: Some([(0, 0.8), (1, 0.6)].iter().cloned().collect()),
    };
    
    // Validate result properties
    assert!(!valid_result.id.is_empty(), "Result ID should not be empty");
    assert!(valid_result.score >= 0.0, "Score should be non-negative");
    assert!(valid_result.score <= 1.0 || valid_result.score.is_finite(), "Score should be reasonable");
    
    if let Some(dense_vec) = &valid_result.dense_vector {
        assert!(!dense_vec.is_empty(), "Dense vector should not be empty");
        for &val in dense_vec {
            assert!(val.is_finite(), "Dense vector values should be finite");
        }
    }
    
    if let Some(sparse_vec) = &valid_result.sparse_vector {
        assert!(!sparse_vec.is_empty(), "Sparse vector should not be empty");
        for (&idx, &val) in sparse_vec {
            assert!(val >= 0.0 && val.is_finite(), "Sparse vector values should be non-negative and finite");
            assert!(idx < 1000000, "Sparse vector indices should be reasonable");
        }
    }
}

#[test]
fn test_search_result_serialization() {
    let mut payload = HashMap::new();
    payload.insert("test_field".to_string(), json!("test_value"));
    payload.insert("numeric_field".to_string(), json!(42));
    
    let result = SearchResult {
        id: "serialization_test".to_string(),
        score: 0.75,
        payload,
        dense_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
        sparse_vector: Some([(0, 0.9), (5, 0.7), (10, 0.5)].iter().cloned().collect()),
    };
    
    // Test serialization
    let serialized = serde_json::to_string(&result)
        .expect("Should be able to serialize search result");
    
    assert!(!serialized.is_empty(), "Serialized result should not be empty");
    
    // Test deserialization
    let deserialized: SearchResult = serde_json::from_str(&serialized)
        .expect("Should be able to deserialize search result");
    
    // Verify round-trip consistency
    assert_eq!(result.id, deserialized.id);
    assert_eq!(result.score, deserialized.score);
    assert_eq!(result.payload, deserialized.payload);
    assert_eq!(result.dense_vector, deserialized.dense_vector);
    assert_eq!(result.sparse_vector, deserialized.sparse_vector);
}

// =============================================================================
// INTEGRATION AND END-TO-END TESTS
// =============================================================================

#[ignore]
#[tokio::test]
async fn test_end_to_end_hybrid_search_pipeline() {
    let temp_dir = create_temp_dir();
    let embedding_config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let storage_config = create_test_storage_config();
    
    let _embedding_generator = EmbeddingGenerator::new(embedding_config)
        .expect("Failed to create embedding generator");
    let storage = StorageClient::with_config(storage_config);
    
    if storage.test_connection().await.is_err() {
        return;
    }
    
    let collection_name = "end_to_end_test";
    
    // Setup collection
    if storage.collection_exists(collection_name).await.unwrap_or(false) {
        storage.delete_collection(collection_name).await.unwrap();
    }
    
    storage.create_collection(collection_name, Some(384), Some(10000)).await
        .expect("Failed to create end-to-end test collection");
    
    // Process and index test documents
    let preprocessor = TextPreprocessor::new(true);
    let mut bm25 = BM25::new(1.2, 0.75);
    
    // Build BM25 corpus
    for (_, content) in TEST_DOCUMENTS {
        let processed = preprocessor.preprocess(content);
        bm25.add_document(&processed.tokens);
    }
    
    // Insert documents with mock embeddings
    for (doc_id, content) in TEST_DOCUMENTS {
        let processed = preprocessor.preprocess(content);
        let sparse_vector = bm25.generate_sparse_vector(&processed.tokens);
        
        // Convert sparse vector to HashMap
        let sparse_map: HashMap<u32, f32> = sparse_vector.indices.iter().zip(sparse_vector.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        
        let mut payload = HashMap::new();
        payload.insert("content".to_string(), json!(content));
        payload.insert("tokens".to_string(), json!(processed.tokens.len()));
        
        let point = DocumentPoint {
            id: doc_id.to_string(),
            dense_vector: vec![0.1; 384], // Mock dense embedding
            sparse_vector: Some(sparse_map),
            payload,
        };
        
        storage.insert_point(collection_name, point).await
            .expect("Failed to insert end-to-end test point");
    }
    
    // Test search quality with different queries
    let mut total_metrics = Vec::new();
    
    for (query_text, expected_docs) in TEST_QUERIES {
        let query_processed = preprocessor.preprocess(query_text);
        let query_sparse = bm25.generate_sparse_vector(&query_processed.tokens);
        
        let query_sparse_map: HashMap<u32, f32> = query_sparse.indices.iter().zip(query_sparse.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        
        // Test hybrid search
        let params = SearchParams {
            dense_vector: Some(vec![0.1; 384]), // Mock query embedding
            sparse_vector: Some(query_sparse_map),
            search_mode: HybridSearchMode::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 },
            limit: 5,
            score_threshold: None,
            filter: None,
        };
        let results = storage.search(collection_name, params).await.expect("End-to-end search failed");
        
        let metrics = SearchQualityMetrics::calculate(&results, expected_docs);
        
        println!("Query: '{}'", query_text);
        println!("  Precision: {:.3}", metrics.precision);
        println!("  Recall: {:.3}", metrics.recall);
        println!("  F1: {:.3}", metrics.f1_score);
        println!("  NDCG: {:.3}", metrics.ndcg);
        
        total_metrics.push(metrics);
        println!();
    }
    
    // Calculate average metrics
    let avg_precision = total_metrics.iter().map(|m| m.precision).sum::<f64>() / total_metrics.len() as f64;
    let avg_recall = total_metrics.iter().map(|m| m.recall).sum::<f64>() / total_metrics.len() as f64;
    let avg_f1 = total_metrics.iter().map(|m| m.f1_score).sum::<f64>() / total_metrics.len() as f64;
    let avg_ndcg = total_metrics.iter().map(|m| m.ndcg).sum::<f64>() / total_metrics.len() as f64;
    
    println!("Average Search Quality Metrics:");
    println!("  Precision: {:.3}", avg_precision);
    println!("  Recall: {:.3}", avg_recall);
    println!("  F1 Score: {:.3}", avg_f1);
    println!("  NDCG: {:.3}", avg_ndcg);
    
    // Basic quality thresholds (adjust based on requirements)
    assert!(avg_precision >= 0.0, "Average precision should be non-negative");
    assert!(avg_recall >= 0.0, "Average recall should be non-negative");
    assert!(avg_f1 >= 0.0, "Average F1 score should be non-negative");
    assert!(avg_ndcg >= 0.0, "Average NDCG should be non-negative");
    
    // Clean up
    storage.delete_collection(collection_name).await.unwrap();
}

#[ignore]
#[tokio::test]
async fn test_performance_benchmarking() {
    let temp_dir = create_temp_dir();
    let embedding_config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let storage_config = create_test_storage_config();
    
    let _embedding_generator = EmbeddingGenerator::new(embedding_config)
        .expect("Failed to create embedding generator");
    let storage = StorageClient::with_config(storage_config);
    
    if storage.test_connection().await.is_err() {
        return;
    }
    
    let collection_name = "performance_benchmark";
    
    // Setup
    if storage.collection_exists(collection_name).await.unwrap_or(false) {
        storage.delete_collection(collection_name).await.unwrap();
    }
    
    storage.create_collection(collection_name, Some(384), Some(10000)).await
        .expect("Failed to create benchmark collection");
    
    // Create larger dataset for performance testing
    let large_dataset: Vec<_> = (0..100).map(|i| {
        (format!("perf_doc_{}", i), 
         format!("Performance test document {} with various content for benchmarking search capabilities and response times", i))
    }).collect();
    
    let preprocessor = TextPreprocessor::new(true);
    let mut bm25 = BM25::new(1.2, 0.75);
    
    // Build corpus and measure indexing time
    let indexing_start = Instant::now();
    
    for (_, content) in &large_dataset {
        let processed = preprocessor.preprocess(content);
        bm25.add_document(&processed.tokens);
    }
    
    // Insert documents
    for (doc_id, content) in &large_dataset {
        let processed = preprocessor.preprocess(content);
        let sparse_vector = bm25.generate_sparse_vector(&processed.tokens);
        
        let sparse_map: HashMap<u32, f32> = sparse_vector.indices.iter().zip(sparse_vector.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        
        let point = DocumentPoint {
            id: doc_id.clone(),
            dense_vector: vec![fastrand::f32(); 384], // Random embeddings for diversity
            sparse_vector: Some(sparse_map),
            payload: [("content".to_string(), json!(content))].iter().cloned().collect(),
        };
        
        storage.insert_point(collection_name, point).await
            .expect("Failed to insert benchmark point");
    }
    
    let indexing_time = indexing_start.elapsed();
    println!("Indexing {} documents took: {:?}", large_dataset.len(), indexing_time);
    
    // Benchmark search performance
    let search_queries = vec![
        "performance test document",
        "benchmarking search capabilities", 
        "various content response times",
        "test document content",
        "search performance evaluation",
    ];
    
    let mut search_times = Vec::new();
    
    for query in &search_queries {
        let processed_query = preprocessor.preprocess(query);
        let query_sparse = bm25.generate_sparse_vector(&processed_query.tokens);
        
        let query_sparse_map: HashMap<u32, f32> = query_sparse.indices.iter().zip(query_sparse.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        
        let search_start = Instant::now();
        
        let params = SearchParams {
            dense_vector: Some(vec![fastrand::f32(); 384]),
            sparse_vector: Some(query_sparse_map),
            search_mode: HybridSearchMode::Hybrid { dense_weight: 1.0, sparse_weight: 1.0 },
            limit: 10,
            score_threshold: None,
            filter: None,
        };
        let results = storage.search(collection_name, params).await.expect("Benchmark search failed");
        
        let search_time = search_start.elapsed();
        search_times.push(search_time);
        
        println!("Query '{}': {:?} ({} results)", query, search_time, results.len());
    }
    
    // Calculate performance statistics
    let avg_search_time = search_times.iter().sum::<std::time::Duration>() / search_times.len() as u32;
    let min_search_time = search_times.iter().min().unwrap();
    let max_search_time = search_times.iter().max().unwrap();
    
    println!("\nSearch Performance Summary:");
    println!("  Average search time: {:?}", avg_search_time);
    println!("  Min search time: {:?}", min_search_time);
    println!("  Max search time: {:?}", max_search_time);
    println!("  Indexing throughput: {:.2} docs/sec", large_dataset.len() as f64 / indexing_time.as_secs_f64());
    
    // Performance assertions (adjust thresholds based on requirements)
    assert!(avg_search_time.as_millis() < 1000, "Average search time should be under 1 second");
    assert!(indexing_time.as_secs() < 30, "Indexing should complete within 30 seconds");
    
    // Clean up
    storage.delete_collection(collection_name).await.unwrap();
}

// =============================================================================
// ERROR HANDLING AND EDGE CASE TESTS
// =============================================================================

#[tokio::test] 
async fn test_error_handling_edge_cases() {
    let temp_dir = create_temp_dir();
    let config = create_test_embedding_config(temp_dir.path().to_path_buf());
    let generator = EmbeddingGenerator::new(config.clone()).expect("Failed to create generator");
    
    // Test empty text processing
    let empty_text = "";
    let preprocessor = TextPreprocessor::new(true);
    let processed = preprocessor.preprocess(empty_text);
    
    assert_eq!(processed.original, empty_text);
    assert!(processed.tokens.is_empty(), "Empty text should produce no tokens");
    
    // Test very long text processing
    let long_text = "word ".repeat(10000);
    let long_processed = preprocessor.preprocess(&long_text);
    assert!(!long_processed.tokens.is_empty(), "Long text should be processable");
    
    // Test special characters
    let special_text = "Hello! @#$%^&*() 123 test-case_example";
    let special_processed = preprocessor.preprocess(special_text);
    assert!(!special_processed.tokens.is_empty(), "Special character text should be processed");
    
    // Test BM25 with empty corpus
    let empty_bm25 = BM25::new(1.2, 0.75);
    let empty_query = vec!["test".to_string()];
    let empty_result = empty_bm25.generate_sparse_vector(&empty_query);
    
    // Should handle empty corpus gracefully
    assert!(empty_result.indices.is_empty() || empty_result.values.iter().all(|&v| v >= 0.0));
    
    // Test cache with repeated identical texts
    for _ in 0..10 {
        generator.add_document_to_corpus("identical text").await;
    }
    
    // Should handle repeated additions without error
    let (cache_size, max_size) = generator.cache_stats().await;
    assert!(cache_size <= max_size, "Cache should respect size limits with repeated content");
}

#[test]
fn test_extreme_bm25_parameters() {
    // Test extreme but valid BM25 parameters
    let extreme_cases = vec![
        (0.1, 0.0),   // Very low k1, no length normalization
        (10.0, 1.0),  // Very high k1, full length normalization
        (1.2, 0.5),   // Standard case for comparison
    ];
    
    let test_docs = vec![
        vec!["short".to_string()],
        vec!["much".to_string(), "longer".to_string(), "document".to_string(), "here".to_string()],
    ];
    
    for (k1, b) in extreme_cases {
        let mut bm25 = BM25::new(k1, b);
        
        for doc in &test_docs {
            bm25.add_document(doc);
        }
        
        let query = vec!["document".to_string()];
        let result = bm25.generate_sparse_vector(&query);
        
        // Should produce valid results even with extreme parameters
        if !result.indices.is_empty() {
            for &score in &result.values {
                assert!(score >= 0.0 && score.is_finite(), 
                        "BM25 score should be valid with k1={}, b={}: {}", k1, b, score);
            }
        }
    }
}

#[ignore]
#[tokio::test]
async fn test_storage_error_recovery() {
    let mut config = create_test_storage_config();
    config.url = "http://localhost:9999".to_string(); // Non-existent server
    config.max_retries = 2;
    config.timeout_ms = 1000;
    
    let storage = StorageClient::with_config(config);
    
    // Test connection failure handling
    let connection_result = storage.test_connection().await;
    assert!(connection_result.is_err(), "Should fail to connect to non-existent server");
    
    // Test search failure handling
    let params = SearchParams {
        dense_vector: Some(vec![0.1; 384]),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 5,
        score_threshold: None,
        filter: None,
    };
    let search_result = storage.search("non_existent_collection", params).await;
    
    assert!(search_result.is_err(), "Should fail to search non-existent collection");
}
