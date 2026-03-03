// Error type classification for queue error handling

use serde::{Deserialize, Serialize};

/// Error category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Temporary errors, retry possible
    Transient,
    /// Permanent errors, no retry
    Permanent,
    /// Rate limiting, retry with backoff
    RateLimit,
    /// Resource exhaustion, retry with delay
    Resource,
}

/// Specific error types with categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    // Transient errors (retry with backoff)
    NetworkTimeout,
    ConnectionRefused,
    TemporaryFailure,
    DatabaseLocked,

    // Rate limit errors (retry with longer backoff)
    RateLimitExceeded,
    TooManyRequests,

    // Resource errors (retry with delay)
    OutOfMemory,
    DiskFull,
    QuotaExceeded,

    // Permanent errors (no retry)
    FileNotFound,
    InvalidFormat,
    PermissionDenied,
    InvalidConfiguration,
    ValidationError,
    MalformedData,
}

impl ErrorType {
    pub fn category(&self) -> ErrorCategory {
        match self {
            ErrorType::NetworkTimeout
            | ErrorType::ConnectionRefused
            | ErrorType::TemporaryFailure
            | ErrorType::DatabaseLocked => ErrorCategory::Transient,

            ErrorType::RateLimitExceeded | ErrorType::TooManyRequests => {
                ErrorCategory::RateLimit
            }

            ErrorType::OutOfMemory | ErrorType::DiskFull | ErrorType::QuotaExceeded => {
                ErrorCategory::Resource
            }

            ErrorType::FileNotFound
            | ErrorType::InvalidFormat
            | ErrorType::PermissionDenied
            | ErrorType::InvalidConfiguration
            | ErrorType::ValidationError
            | ErrorType::MalformedData => ErrorCategory::Permanent,
        }
    }

    pub fn max_retries(&self) -> i32 {
        match self {
            ErrorType::NetworkTimeout => 5,
            ErrorType::ConnectionRefused => 5,
            ErrorType::TemporaryFailure => 3,
            ErrorType::DatabaseLocked => 10,
            ErrorType::RateLimitExceeded => 10,
            ErrorType::TooManyRequests => 8,
            ErrorType::OutOfMemory => 3,
            ErrorType::DiskFull => 3,
            ErrorType::QuotaExceeded => 5,
            _ => 0, // Permanent errors
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorType::NetworkTimeout => "network_timeout",
            ErrorType::ConnectionRefused => "connection_refused",
            ErrorType::TemporaryFailure => "temporary_failure",
            ErrorType::DatabaseLocked => "database_locked",
            ErrorType::RateLimitExceeded => "rate_limit_exceeded",
            ErrorType::TooManyRequests => "too_many_requests",
            ErrorType::OutOfMemory => "out_of_memory",
            ErrorType::DiskFull => "disk_full",
            ErrorType::QuotaExceeded => "quota_exceeded",
            ErrorType::FileNotFound => "file_not_found",
            ErrorType::InvalidFormat => "invalid_format",
            ErrorType::PermissionDenied => "permission_denied",
            ErrorType::InvalidConfiguration => "invalid_configuration",
            ErrorType::ValidationError => "validation_error",
            ErrorType::MalformedData => "malformed_data",
        }
    }
}
