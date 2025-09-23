//! Error types for Qdrant operations

use thiserror::Error;

/// Result type for Qdrant operations
pub type QdrantResult<T> = Result<T, QdrantError>;

/// Comprehensive error types for Qdrant client operations
#[derive(Error, Debug)]
pub enum QdrantError {
    /// Connection errors
    #[error("Connection failed: {message}")]
    Connection { message: String },

    /// Authentication errors
    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    /// Collection operation errors
    #[error("Collection operation failed: {operation} - {message}")]
    CollectionOperation { operation: String, message: String },

    /// Vector operation errors
    #[error("Vector operation failed: {operation} - {message}")]
    VectorOperation { operation: String, message: String },

    /// Search operation errors
    #[error("Search operation failed: {message}")]
    SearchOperation { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Timeout errors
    #[error("Operation timed out after {timeout_secs} seconds: {operation}")]
    Timeout { operation: String, timeout_secs: u64 },

    /// Network errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Invalid vector dimension errors
    #[error("Invalid vector dimensions: expected {expected}, got {actual}")]
    InvalidVectorDimensions { expected: usize, actual: usize },

    /// Collection not found errors
    #[error("Collection not found: {collection_name}")]
    CollectionNotFound { collection_name: String },

    /// Collection already exists errors
    #[error("Collection already exists: {collection_name}")]
    CollectionAlreadyExists { collection_name: String },

    /// Point not found errors
    #[error("Point not found: {point_id}")]
    PointNotFound { point_id: String },

    /// Invalid operation parameters
    #[error("Invalid operation parameters: {message}")]
    InvalidParameters { message: String },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted { resource: String, message: String },

    /// Internal server errors
    #[error("Internal server error: {message}")]
    InternalServer { message: String },

    /// Circuit breaker open
    #[error("Circuit breaker is open for operation: {operation}")]
    CircuitBreakerOpen { operation: String },

    /// Batch operation errors
    #[error("Batch operation failed: {operation} - failed items: {failed_count}/{total_count}")]
    BatchOperation {
        operation: String,
        failed_count: usize,
        total_count: usize,
        errors: Vec<String>
    },

    /// Generic client errors
    #[error("Qdrant client error: {message}")]
    Client { message: String },

    /// Wrapped qdrant-client errors
    #[error("Qdrant client library error: {0}")]
    QdrantClientError(#[from] qdrant_client::QdrantError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Any other errors
    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

impl QdrantError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            QdrantError::Connection { .. }
            | QdrantError::Network { .. }
            | QdrantError::Timeout { .. }
            | QdrantError::ResourceExhausted { .. }
            | QdrantError::InternalServer { .. } => true,

            QdrantError::Authentication { .. }
            | QdrantError::Configuration { .. }
            | QdrantError::InvalidVectorDimensions { .. }
            | QdrantError::CollectionNotFound { .. }
            | QdrantError::CollectionAlreadyExists { .. }
            | QdrantError::PointNotFound { .. }
            | QdrantError::InvalidParameters { .. }
            | QdrantError::CircuitBreakerOpen { .. }
            | QdrantError::Serialization { .. } => false,

            QdrantError::CollectionOperation { .. }
            | QdrantError::VectorOperation { .. }
            | QdrantError::SearchOperation { .. }
            | QdrantError::BatchOperation { .. } => true, // May be retryable depending on cause

            QdrantError::QdrantClientError(e) => {
                // Check underlying qdrant client error for retryability
                match e {
                    qdrant_client::QdrantError::ResponseError { status } => {
                        // 5xx errors are typically retryable, 4xx are not
                        let code = status.code() as u16;
                        code >= 500
                    },
                    _ => true, // Other client errors might be retryable
                }
            },

            QdrantError::Client { .. }
            | QdrantError::Io(..)
            | QdrantError::Json(..)
            | QdrantError::Unknown { .. } => false,
        }
    }

    /// Get error category for metrics/logging
    pub fn category(&self) -> &'static str {
        match self {
            QdrantError::Connection { .. } => "connection",
            QdrantError::Authentication { .. } => "authentication",
            QdrantError::CollectionOperation { .. } => "collection",
            QdrantError::VectorOperation { .. } => "vector",
            QdrantError::SearchOperation { .. } => "search",
            QdrantError::Configuration { .. } => "configuration",
            QdrantError::Timeout { .. } => "timeout",
            QdrantError::Network { .. } => "network",
            QdrantError::Serialization { .. } => "serialization",
            QdrantError::InvalidVectorDimensions { .. } => "validation",
            QdrantError::CollectionNotFound { .. } => "not_found",
            QdrantError::CollectionAlreadyExists { .. } => "conflict",
            QdrantError::PointNotFound { .. } => "not_found",
            QdrantError::InvalidParameters { .. } => "validation",
            QdrantError::ResourceExhausted { .. } => "resource",
            QdrantError::InternalServer { .. } => "server",
            QdrantError::CircuitBreakerOpen { .. } => "circuit_breaker",
            QdrantError::BatchOperation { .. } => "batch",
            QdrantError::Client { .. } => "client",
            QdrantError::QdrantClientError(..) => "qdrant_client",
            QdrantError::Io(..) => "io",
            QdrantError::Json(..) => "json",
            QdrantError::Unknown { .. } => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryability() {
        // Retryable errors
        assert!(QdrantError::Connection { message: "test".to_string() }.is_retryable());
        assert!(QdrantError::Network { message: "test".to_string() }.is_retryable());
        assert!(QdrantError::Timeout { operation: "test".to_string(), timeout_secs: 30 }.is_retryable());
        assert!(QdrantError::ResourceExhausted { resource: "test".to_string(), message: "test".to_string() }.is_retryable());

        // Non-retryable errors
        assert!(!QdrantError::Authentication { message: "test".to_string() }.is_retryable());
        assert!(!QdrantError::Configuration { message: "test".to_string() }.is_retryable());
        assert!(!QdrantError::InvalidVectorDimensions { expected: 384, actual: 512 }.is_retryable());
        assert!(!QdrantError::CollectionNotFound { collection_name: "test".to_string() }.is_retryable());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(QdrantError::Connection { message: "test".to_string() }.category(), "connection");
        assert_eq!(QdrantError::CollectionOperation { operation: "create".to_string(), message: "test".to_string() }.category(), "collection");
        assert_eq!(QdrantError::VectorOperation { operation: "upsert".to_string(), message: "test".to_string() }.category(), "vector");
        assert_eq!(QdrantError::SearchOperation { message: "test".to_string() }.category(), "search");
        assert_eq!(QdrantError::InvalidVectorDimensions { expected: 384, actual: 512 }.category(), "validation");
    }

    #[test]
    fn test_error_display() {
        let error = QdrantError::InvalidVectorDimensions { expected: 384, actual: 512 };
        assert_eq!(format!("{}", error), "Invalid vector dimensions: expected 384, got 512");

        let error = QdrantError::BatchOperation {
            operation: "upsert".to_string(),
            failed_count: 5,
            total_count: 100,
            errors: vec!["error1".to_string(), "error2".to_string()]
        };
        assert_eq!(format!("{}", error), "Batch operation failed: upsert - failed items: 5/100");
    }
}