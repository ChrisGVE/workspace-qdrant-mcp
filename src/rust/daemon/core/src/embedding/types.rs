//! Types for embedding generation: errors, configuration, and result structs.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during embedding generation
#[derive(Error, Debug, Clone)]
pub enum EmbeddingError {
    #[error("Model initialization failed: {message}")]
    InitializationError { message: String },

    #[error("Embedding generation failed: {message}")]
    GenerationError { message: String },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Embedding subsystem is temporarily unavailable (within backoff window after
    /// a failed init). The caller should not count this against the item's retry
    /// budget — just re-lease the item for later.
    #[error("Embedding subsystem temporarily unavailable, retry after {retry_after_secs}s")]
    TemporarilyUnavailable { retry_after_secs: u64 },

    /// Remote embedding endpoint returned a non-success HTTP status. The
    /// `status_code` is the upstream value; `message` carries any structured
    /// error body the provider supplied (truncated by the caller).
    #[error("Remote embedding error: HTTP {status_code}: {message}")]
    RemoteError { status_code: u16, message: String },

    /// Adaptive rate limiter exhausted its retry budget. `consecutive_429s`
    /// counts the unbroken streak of throttle responses; `retry_after_secs`
    /// reflects the longest server-supplied wait observed in that streak.
    #[error("Rate limit exhausted after {consecutive_429s} consecutive 429s; retry after {retry_after_secs}s")]
    RateLimitExhausted {
        consecutive_429s: u32,
        retry_after_secs: u64,
    },

    /// Active provider dimensionality disagrees with the dimensionality
    /// stored on existing Qdrant collections. The daemon refuses to start
    /// in this state; the operator must run `wqm admin reembed --confirm`.
    #[error(
        "Embedding dimension mismatch: provider returns {actual_dim}, collections store {stored_dim}"
    )]
    DimensionMismatch {
        actual_dim: usize,
        stored_dim: usize,
    },
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
    /// Prefix prepended to texts before DENSE embedding (ingestion side).
    /// Instruction-tuned retrieval models require it (multilingual-e5:
    /// "passage: ", including the trailing space). Applied to the dense leg
    /// only — BM25/sparse tokenization always sees the raw text. Empty (the
    /// default) means no prefix.
    #[serde(default)]
    pub dense_document_prefix: String,
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
            dense_document_prefix: String::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_error_is_clone() {
        let err = EmbeddingError::TemporarilyUnavailable {
            retry_after_secs: 30,
        };
        let cloned = err.clone();
        assert!(matches!(
            cloned,
            EmbeddingError::TemporarilyUnavailable {
                retry_after_secs: 30
            }
        ));
    }

    #[test]
    fn remote_error_displays_status_and_message() {
        let err = EmbeddingError::RemoteError {
            status_code: 503,
            message: "service unavailable".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("HTTP 503"), "missing status: {s}");
        assert!(s.contains("service unavailable"), "missing body: {s}");
    }

    #[test]
    fn rate_limit_exhausted_displays_streak_and_retry_after() {
        let err = EmbeddingError::RateLimitExhausted {
            consecutive_429s: 6,
            retry_after_secs: 60,
        };
        let s = err.to_string();
        assert!(s.contains("6 consecutive 429s"), "missing streak: {s}");
        assert!(s.contains("retry after 60s"), "missing retry: {s}");
    }

    #[test]
    fn dimension_mismatch_displays_both_dims() {
        let err = EmbeddingError::DimensionMismatch {
            actual_dim: 1536,
            stored_dim: 384,
        };
        let s = err.to_string();
        assert!(s.contains("1536"));
        assert!(s.contains("384"));
    }

    #[test]
    fn dimension_mismatch_round_trip_clone() {
        let err = EmbeddingError::DimensionMismatch {
            actual_dim: 1536,
            stored_dim: 384,
        };
        let cloned = err.clone();
        match cloned {
            EmbeddingError::DimensionMismatch {
                actual_dim,
                stored_dim,
            } => {
                assert_eq!(actual_dim, 1536);
                assert_eq!(stored_dim, 384);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
