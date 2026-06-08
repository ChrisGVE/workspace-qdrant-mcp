//! Error types for the unified queue processor.

use crate::embedding::EmbeddingError;
use thiserror::Error;

/// Unified queue processor errors
#[derive(Error, Debug)]
pub enum UnifiedProcessorError {
    #[error("Queue operation failed: {0}")]
    QueueOperation(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Embedding subsystem is within its backoff window after a failed init.
    /// The item should be re-leased without incrementing its retry count.
    #[error("Embedding subsystem temporarily unavailable: {0}")]
    EmbeddingUnavailable(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid payload: {0}")]
    InvalidPayload(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Shutdown requested")]
    ShutdownRequested,
}

impl From<EmbeddingError> for UnifiedProcessorError {
    /// Map a provider embedding failure to the queue-processor error whose
    /// classification matches how recoverable the failure actually is:
    ///
    /// - `TemporarilyUnavailable` → `EmbeddingUnavailable`: the subsystem is in
    ///   its backoff window; re-lease the item without burning its retry budget.
    /// - A permanent payload rejection (HTTP 400 / 413 / 422) → `InvalidPayload`:
    ///   the embedder refuses this specific input, so retrying can never
    ///   succeed. Routing it to `InvalidPayload` classifies it as `permanent_data`
    ///   — it lands in the DLQ once instead of being retried and resurrected
    ///   forever (the cause of the #113 flood).
    /// - Everything else (429, 5xx, transport, init) → transient `Embedding`.
    ///   429 still classifies as `rate_limit` downstream via the message text.
    fn from(e: EmbeddingError) -> Self {
        match e {
            EmbeddingError::TemporarilyUnavailable { .. } => {
                UnifiedProcessorError::EmbeddingUnavailable(e.to_string())
            }
            EmbeddingError::RemoteError { status_code, .. }
                if matches!(status_code, 400 | 413 | 422) =>
            {
                UnifiedProcessorError::InvalidPayload(e.to_string())
            }
            _ => UnifiedProcessorError::Embedding(e.to_string()),
        }
    }
}

/// Result type for unified processor operations
pub type UnifiedProcessorResult<T> = Result<T, UnifiedProcessorError>;
