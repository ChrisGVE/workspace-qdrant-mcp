//! BM25 sparse-vector generation for DocumentService
//!
//! Dense embeddings are produced by the daemon-wide injected
//! `Arc<dyn DenseProvider>` (see `DocumentServiceImpl`), so this module only
//! manages the service-local BM25 corpus used for sparse vectors. The BM25
//! state is lazily initialized and thread-safe.

use std::collections::HashMap;
use std::sync::OnceLock;

use tokio::sync::RwLock as TokioRwLock;
use tonic::Status;
use tracing::{debug, info};
use workspace_qdrant_core::BM25;

/// Global BM25 instance for sparse vector generation (thread-safe, read-write lock)
/// Uses RwLock because reads (generate_sparse_vector) are frequent and concurrent,
/// while writes (add_document) are less frequent during ingestion
static BM25_MODEL: OnceLock<TokioRwLock<BM25>> = OnceLock::new();

/// Default BM25 parameters (standard values from research)
const DEFAULT_BM25_K1: f32 = 1.2;

/// Initialize the BM25 sparse-vector model (infallible).
fn init_bm25() -> &'static TokioRwLock<BM25> {
    BM25_MODEL.get_or_init(|| {
        info!("Initializing BM25 model (k1={})...", DEFAULT_BM25_K1);
        TokioRwLock::new(BM25::new(DEFAULT_BM25_K1))
    })
}

/// Simple tokenization for BM25 sparse vector generation
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    wqm_common::nlp::tokenize(text)
}

/// Generate sparse vector using BM25 algorithm.
///
/// First adds document to corpus, then generates sparse vector.
pub(crate) async fn generate_sparse_vector(text: &str) -> Result<HashMap<u32, f32>, Status> {
    let bm25 = init_bm25();

    let tokens = tokenize(text);

    if tokens.is_empty() {
        debug!("No tokens for sparse vector generation, returning empty");
        return Ok(HashMap::new());
    }

    let sparse_map: HashMap<u32, f32> = {
        let mut bm25_guard = bm25.write().await;

        bm25_guard.add_document(&tokens);

        let sparse = bm25_guard.generate_sparse_vector(&tokens);

        sparse.indices.into_iter().zip(sparse.values).collect()
    };

    debug!(
        "Generated sparse vector with {} non-zero entries",
        sparse_map.len()
    );
    Ok(sparse_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello world test");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));

        let tokens2 = tokenize("the quick brown fox and the lazy dog");
        assert!(!tokens2.contains(&"the".to_string()));
        assert!(!tokens2.contains(&"and".to_string()));
        assert!(tokens2.contains(&"quick".to_string()));
        assert!(tokens2.contains(&"brown".to_string()));
        assert!(tokens2.contains(&"fox".to_string()));
        assert!(tokens2.contains(&"lazy".to_string()));
        assert!(tokens2.contains(&"dog".to_string()));

        let tokens3 = tokenize("a b c test word");
        assert!(!tokens3.contains(&"a".to_string()));
        assert!(!tokens3.contains(&"b".to_string()));
        assert!(tokens3.contains(&"test".to_string()));
        assert!(tokens3.contains(&"word".to_string()));

        let tokens4 = tokenize("hello, world! test-case");
        assert!(tokens4.contains(&"hello".to_string()));
        assert!(tokens4.contains(&"world".to_string()));
        assert!(tokens4.contains(&"test".to_string()));
        assert!(tokens4.contains(&"case".to_string()));
    }

    #[tokio::test]
    async fn test_generate_sparse_vector() {
        let doc1 = "machine learning algorithms for natural language processing";
        let _sparse1 = generate_sparse_vector(doc1)
            .await
            .expect("Failed to add first document");

        let doc2 = "deep learning neural networks for image classification";
        let sparse2 = generate_sparse_vector(doc2)
            .await
            .expect("Failed to generate sparse vector");

        let doc3 = "reinforcement learning algorithms for robotics control";
        let sparse3 = generate_sparse_vector(doc3)
            .await
            .expect("Failed to generate third sparse vector");

        if !sparse2.is_empty() && !sparse3.is_empty() {
            assert_ne!(
                sparse2, sparse3,
                "Different documents should produce different sparse vectors"
            );
        }

        for &value in sparse2.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
        for &value in sparse3.values() {
            assert!(value >= 0.0, "BM25 scores should be non-negative");
        }
    }

    #[tokio::test]
    async fn test_sparse_vector_empty_input() {
        let sparse_vector = generate_sparse_vector("")
            .await
            .expect("Failed with empty input");
        assert!(
            sparse_vector.is_empty(),
            "Empty text should produce empty sparse vector"
        );

        let stopword_only = generate_sparse_vector("the and is a")
            .await
            .expect("Failed with stopwords only");
        assert!(
            stopword_only.is_empty(),
            "Stopwords-only text should produce empty sparse vector"
        );
    }
}
