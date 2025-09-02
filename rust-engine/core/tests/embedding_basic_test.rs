//! Basic compilation test for embedding module

use workspace_qdrant_core::embedding::*;

#[test]
fn test_embedding_types_compile() {
    // Test that all our basic types compile and can be instantiated
    let config = EmbeddingConfig::default();
    assert!(config.max_cache_size > 0);
    
    // Test error types
    let error = EmbeddingError::InvalidInput { message: "test".to_string() };
    assert!(!format!("{}", error).is_empty());
}

#[test]
fn test_text_preprocessor_basic() {
    let preprocessor = TextPreprocessor::new(true);
    let result = preprocessor.preprocess("Hello world");
    assert_eq!(result.original, "Hello world");
    assert!(!result.tokens.is_empty());
}

#[test]
fn test_bm25_basic() {
    let mut bm25 = BM25::new(1.2, 0.75);
    let tokens = vec!["hello".to_string(), "world".to_string()];
    bm25.add_document(&tokens);
    
    let sparse = bm25.generate_sparse_vector(&tokens);
    assert!(!sparse.indices.is_empty());
    assert!(!sparse.values.is_empty());
    assert_eq!(sparse.indices.len(), sparse.values.len());
}