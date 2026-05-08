//! Tests for embedding generation, BM25 scoring, and tokenization.

use super::bm25::{tokenize_for_bm25, BM25};
use super::generator::EmbeddingGenerator;
use super::types::{EmbeddingConfig, EmbeddingError, SparseEmbedding};

#[test]
fn test_tokenize_for_bm25_splits_punctuation() {
    let tokens = tokenize_for_bm25("spatial.kdtree(data, metric='euclidean')");
    assert!(tokens.contains(&"spatial".to_string()));
    assert!(tokens.contains(&"kdtree".to_string()));
    assert!(tokens.contains(&"data".to_string()));
    assert!(tokens.contains(&"metric".to_string()));
    assert!(tokens.contains(&"euclidean".to_string()));
    // Should not contain concatenated garbage
    assert!(!tokens.iter().any(|t| t.contains('(')));
    assert!(!tokens.iter().any(|t| t.contains(')')));
}

#[test]
fn test_tokenize_for_bm25_filters_junk() {
    let tokens =
        tokenize_for_bm25("version 2.0.0 hash abc123def456 num 120 path /usr/bin/test 0xff");
    assert!(tokens.contains(&"version".to_string()));
    assert!(tokens.contains(&"hash".to_string()));
    assert!(tokens.contains(&"num".to_string()));
    assert!(tokens.contains(&"path".to_string()));
    // Junk filtered
    assert!(
        !tokens.contains(&"2.0.0".to_string()),
        "version strings should be filtered"
    );
    assert!(
        !tokens.contains(&"abc123def456".to_string()),
        "hex hashes should be filtered"
    );
    assert!(
        !tokens.contains(&"120".to_string()),
        "pure digits should be filtered"
    );
    assert!(
        !tokens.contains(&"0xff".to_string()),
        "hex literals should be filtered"
    );
}

#[test]
fn test_tokenize_for_bm25_filters_single_chars() {
    let tokens = tokenize_for_bm25("a b c hello world");
    assert!(!tokens.contains(&"a".to_string()));
    assert!(!tokens.contains(&"b".to_string()));
    assert!(!tokens.contains(&"c".to_string()));
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
}

#[test]
fn test_tokenize_for_bm25_code_tokens() {
    let tokens = tokenize_for_bm25("fn process_item(queue_manager: &QueueManager) -> Result<()>");
    assert!(tokens.contains(&"fn".to_string()));
    assert!(
        tokens.contains(&"process_item".to_string()) || tokens.contains(&"process".to_string())
    );
    assert!(
        tokens.contains(&"queue_manager".to_string())
            || tokens.contains(&"queuemanager".to_string())
    );
    assert!(tokens.contains(&"result".to_string()));
    // Should not contain angle brackets or ampersands
    assert!(!tokens
        .iter()
        .any(|t| t.contains('<') || t.contains('>') || t.contains('&')));
}

#[test]
fn test_bm25_idf_common_vs_rare_terms() {
    let mut bm25 = BM25::new(1.2);

    // Add 10 documents. "function" appears in all, "quantum" in only 1
    for i in 0..10 {
        let mut tokens = vec!["function".to_string(), "code".to_string()];
        if i == 0 {
            tokens.push("quantum".to_string());
        }
        bm25.add_document(&tokens);
    }

    assert_eq!(bm25.total_docs(), 10);

    // Generate sparse vector for a query containing both terms
    let query_tokens = vec!["function".to_string(), "quantum".to_string()];
    let sparse = bm25.generate_sparse_vector(&query_tokens);

    // Find scores for each term
    let function_id = bm25.vocab().get("function").unwrap();
    let quantum_id = bm25.vocab().get("quantum").unwrap();

    let function_score = sparse
        .indices
        .iter()
        .zip(&sparse.values)
        .find(|(&idx, _)| idx == *function_id)
        .map(|(_, &v)| v);
    let quantum_score = sparse
        .indices
        .iter()
        .zip(&sparse.values)
        .find(|(&idx, _)| idx == *quantum_id)
        .map(|(_, &v)| v);

    // "quantum" (rare, df=1) should score higher than "function" (common, df=10)
    assert!(
        quantum_score.unwrap() > function_score.unwrap_or(0.0),
        "Rare term 'quantum' ({:?}) should score higher than common term 'function' ({:?})",
        quantum_score,
        function_score
    );
}

#[test]
fn test_bm25_idf_zero_for_universal_terms() {
    let mut bm25 = BM25::new(1.2);

    // Add 5 documents all containing "the"
    for _ in 0..5 {
        bm25.add_document(&["the".to_string(), "code".to_string()]);
    }

    let sparse = bm25.generate_sparse_vector(&["the".to_string()]);
    let the_id = bm25.vocab().get("the").unwrap();
    let the_score = sparse
        .indices
        .iter()
        .zip(&sparse.values)
        .find(|(&idx, _)| idx == *the_id)
        .map(|(_, &v)| v);

    // IDF for a term in all documents: ln((5 - 5 + 0.5)/(5 + 0.5)) = ln(0.5/5.5) < 0 → clamped to 0
    // So score should be 0 (term filtered out)
    assert!(
        the_score.is_none() || the_score.unwrap() == 0.0,
        "Universal term should have zero score, got {:?}",
        the_score
    );
}

#[test]
fn test_bm25_doc_freq_tracking() {
    let mut bm25 = BM25::new(1.2);

    bm25.add_document(&["hello".to_string(), "world".to_string()]);
    bm25.add_document(&["hello".to_string(), "rust".to_string()]);
    bm25.add_document(&["goodbye".to_string(), "world".to_string()]);

    assert_eq!(bm25.total_docs(), 3);

    let hello_id = *bm25.vocab().get("hello").unwrap();
    let world_id = *bm25.vocab().get("world").unwrap();
    let rust_id = *bm25.vocab().get("rust").unwrap();

    assert_eq!(*bm25.doc_freq().get(&hello_id).unwrap(), 2);
    assert_eq!(*bm25.doc_freq().get(&world_id).unwrap(), 2);
    assert_eq!(*bm25.doc_freq().get(&rust_id).unwrap(), 1);
}

#[test]
fn test_bm25_from_persisted() {
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("test".to_string(), 0u32);
    vocab.insert("data".to_string(), 1u32);

    let mut doc_freq = std::collections::HashMap::new();
    doc_freq.insert(0u32, 5u32);
    doc_freq.insert(1u32, 2u32);

    let bm25 = BM25::from_persisted(1.2, vocab, doc_freq, 10);

    assert_eq!(bm25.total_docs(), 10);
    assert_eq!(bm25.vocab_size(), 2);
    assert_eq!(bm25.next_vocab_id, 2);

    // "data" (df=2) should score higher than "test" (df=5) for same TF
    let sparse = bm25.generate_sparse_vector(&["test".to_string(), "data".to_string()]);
    let test_id = *bm25.vocab().get("test").unwrap();
    let data_id = *bm25.vocab().get("data").unwrap();

    let test_score = sparse
        .indices
        .iter()
        .zip(&sparse.values)
        .find(|(&idx, _)| idx == test_id)
        .map(|(_, &v)| v)
        .unwrap_or(0.0);
    let data_score = sparse
        .indices
        .iter()
        .zip(&sparse.values)
        .find(|(&idx, _)| idx == data_id)
        .map(|(_, &v)| v)
        .unwrap_or(0.0);

    assert!(
        data_score > test_score,
        "Rarer term 'data' (df=2) should score higher than 'test' (df=5)"
    );
}

#[test]
fn test_bm25_empty_corpus_fallback() {
    let bm25 = BM25::new(1.2);
    // With no documents added, generate should still work (TF-only fallback)
    // but vocab is empty so no matching terms → empty result
    let sparse = bm25.generate_sparse_vector(&["hello".to_string()]);
    assert!(sparse.indices.is_empty());
}

#[test]
fn test_splade_embedding_conversion() {
    // Simulate fastembed SparseEmbedding with usize indices
    let fe_indices: Vec<usize> = vec![42, 1337, 30000];
    let fe_values: Vec<f32> = vec![0.5, 1.2, 0.8];

    // Convert to our SparseEmbedding with u32 indices
    let sparse = SparseEmbedding {
        indices: fe_indices.into_iter().map(|i| i as u32).collect(),
        values: fe_values.clone(),
        vocab_size: 30522,
    };

    assert_eq!(sparse.indices, vec![42u32, 1337, 30000]);
    assert_eq!(sparse.values, vec![0.5, 1.2, 0.8]);
    assert_eq!(sparse.vocab_size, 30522);
}

#[test]
fn test_sparse_vector_mode_config() {
    // Default mode is bm25
    let config = EmbeddingConfig::default();
    assert_eq!(config.sparse_vector_mode, "bm25");

    // Can be set to splade
    let config = EmbeddingConfig {
        sparse_vector_mode: "splade".to_string(),
        ..EmbeddingConfig::default()
    };
    assert_eq!(config.sparse_vector_mode, "splade");
}

#[test]
fn test_embedding_generator_sparse_mode_accessor() {
    use crate::embedding::provider::FastEmbedProvider;
    use std::sync::Arc;

    let config = EmbeddingConfig {
        sparse_vector_mode: "splade".to_string(),
        ..EmbeddingConfig::default()
    };
    let provider = Arc::new(FastEmbedProvider::new(32, None, None));
    let gen = EmbeddingGenerator::new(config, provider).unwrap();
    assert_eq!(gen.sparse_vector_mode(), "splade");
}

#[cfg(test)]
mod delegation_tests {
    //! §7.5 delegation tests: confirm `EmbeddingGenerator` forwards dense work
    //! to its injected `DenseProvider` and that BM25 sparse generation still
    //! runs alongside an arbitrary provider.

    use super::*;
    use crate::embedding::provider::DenseProvider;
    use crate::embedding::types::DenseEmbedding;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Mock provider returning a fixed-dim, fixed-content vector. Tracks
    /// number of `embed` calls so tests can verify delegation.
    #[derive(Debug)]
    struct MockProvider {
        dim: usize,
        fill: f32,
        embed_calls: AtomicUsize,
    }

    impl MockProvider {
        fn new(dim: usize, fill: f32) -> Arc<Self> {
            Arc::new(Self {
                dim,
                fill,
                embed_calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl DenseProvider for MockProvider {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| DenseEmbedding {
                    vector: vec![self.fill; self.dim],
                    model_name: "mock".to_string(),
                    sequence_length: t.len(),
                })
                .collect())
        }

        fn output_dim(&self) -> usize {
            self.dim
        }

        fn provider_label(&self) -> &str {
            "mock"
        }

        fn metrics_label(&self) -> &'static str {
            "fastembed"
        }

        async fn probe(&self) -> Result<(), EmbeddingError> {
            Ok(())
        }
    }

    #[test]
    fn test_generator_dense_dim_delegates_to_provider() {
        let provider = MockProvider::new(999, 0.0);
        let gen = EmbeddingGenerator::new(EmbeddingConfig::default(), provider.clone()).unwrap();
        assert_eq!(gen.dense_dim(), 999);
    }

    #[tokio::test]
    async fn test_generate_embedding_uses_provider() {
        let provider = MockProvider::new(4, 0.25);
        let gen = EmbeddingGenerator::new(EmbeddingConfig::default(), provider.clone()).unwrap();

        let result = gen
            .generate_embedding("the quick brown fox", "ignored")
            .await
            .expect("generate_embedding must succeed with mock provider");

        assert_eq!(result.dense.vector, vec![0.25_f32; 4]);
        assert_eq!(provider.embed_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_bm25_still_works_with_openai_provider() {
        // Drive BM25 path with a non-FastEmbed mock provider; the sparse
        // branch must remain populated regardless of which dense backend
        // is wired in.
        let provider = MockProvider::new(4, 0.5);
        let gen = EmbeddingGenerator::new(EmbeddingConfig::default(), provider).unwrap();

        // Seed corpus with enough distinct documents so the queried term
        // has a positive IDF (df=1, N>=6 ⇒ idf > 0). BM25 zeros-out
        // entries with non-positive idf, which would otherwise leave the
        // sparse vector empty.
        gen.add_document_to_corpus("machine learning algorithms")
            .await;
        gen.add_document_to_corpus("alpha bravo charlie").await;
        gen.add_document_to_corpus("delta echo foxtrot").await;
        gen.add_document_to_corpus("golf hotel india").await;
        gen.add_document_to_corpus("juliet kilo lima").await;
        gen.add_document_to_corpus("mike november oscar").await;

        let result = gen
            .generate_embedding("learning algorithms", "ignored")
            .await
            .expect("BM25 path must succeed");
        assert!(!result.sparse.indices.is_empty());
        assert_eq!(result.sparse.indices.len(), result.sparse.values.len());
    }
}

#[test]
fn test_temporarily_unavailable_error_display() {
    let err = EmbeddingError::TemporarilyUnavailable {
        retry_after_secs: 120,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("120"),
        "Error message should contain retry_after_secs: {msg}"
    );
    assert!(
        msg.to_lowercase().contains("unavailable"),
        "Error message should mention 'unavailable': {msg}"
    );
}

#[test]
fn test_bm25_duplicate_tokens_in_doc() {
    let mut bm25 = BM25::new(1.2);

    // Document with repeated token
    bm25.add_document(&["test".to_string(), "test".to_string(), "test".to_string()]);
    assert_eq!(bm25.total_docs(), 1);

    // doc_freq should be 1 (appears in 1 document, not 3)
    let test_id = *bm25.vocab().get("test").unwrap();
    assert_eq!(*bm25.doc_freq().get(&test_id).unwrap(), 1);
}
