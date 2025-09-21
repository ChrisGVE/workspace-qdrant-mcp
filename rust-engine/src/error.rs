//! Error types for the Workspace Qdrant Daemon

use thiserror::Error;
use tonic::{Code, Status};

/// Main error type for daemon operations
#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::transport::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("File watching error: {0}")]
    FileWatcher(#[from] notify::Error),

    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Document processing error: {message}")]
    DocumentProcessing { message: String },

    #[error("Search error: {message}")]
    Search { message: String },

    #[error("Memory management error: {message}")]
    Memory { message: String },

    #[error("System error: {message}")]
    System { message: String },

    #[error("Project detection error: {message}")]
    ProjectDetection { message: String },

    #[error("Connection pool error: {message}")]
    ConnectionPool { message: String },

    #[error("Timeout error: operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl From<DaemonError> for Status {
    fn from(err: DaemonError) -> Self {
        match err {
            DaemonError::Config(_) | DaemonError::InvalidInput { .. } => {
                Status::new(Code::InvalidArgument, err.to_string())
            },
            DaemonError::NotFound { .. } => {
                Status::new(Code::NotFound, err.to_string())
            },
            DaemonError::Timeout { .. } => {
                Status::new(Code::DeadlineExceeded, err.to_string())
            },
            DaemonError::Database(_) | DaemonError::Io(_) | DaemonError::FileWatcher(_) => {
                Status::new(Code::Internal, "Internal server error")
            },
            _ => Status::new(Code::Internal, "Internal server error"),
        }
    }
}

/// Result type alias for daemon operations
pub type DaemonResult<T> = Result<T, DaemonError>;