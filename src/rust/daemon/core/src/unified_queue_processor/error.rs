//! Error types for the unified queue processor.

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

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid payload: {0}")]
    InvalidPayload(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Shutdown requested")]
    ShutdownRequested,
}

/// Result type for unified processor operations
pub type UnifiedProcessorResult<T> = Result<T, UnifiedProcessorError>;
