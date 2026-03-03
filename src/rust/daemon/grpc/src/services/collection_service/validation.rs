//! Validation helpers for CollectionService
//!
//! Provides input validation for collection names, vector sizes,
//! alias names, and distance metric mapping.

use tonic::Status;
use tracing::warn;

/// Canonical collection names that cannot be used as aliases (per ADR-001)
pub(super) const CANONICAL_COLLECTIONS: &[&str] = &["projects", "libraries", "rules", "scratchpad"];

/// Validate collection name.
/// Rules: 3-255 chars, alphanumeric + underscore/hyphen, no leading numbers.
pub(super) fn validate_collection_name(name: &str) -> Result<(), Status> {
    if name.len() < 3 {
        return Err(Status::invalid_argument(
            "Collection name must be at least 3 characters",
        ));
    }

    if name.len() > 255 {
        return Err(Status::invalid_argument(
            "Collection name must not exceed 255 characters",
        ));
    }

    // Check first character is not a number
    if name.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
        return Err(Status::invalid_argument(
            "Collection name cannot start with a number",
        ));
    }

    // Check all characters are valid
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(Status::invalid_argument(
            "Collection name can only contain alphanumeric characters, underscores, and hyphens",
        ));
    }

    Ok(())
}

/// Validate vector size.
/// Default embedding model (all-MiniLM-L6-v2) uses 384 dimensions.
pub(super) fn validate_vector_size(size: i32) -> Result<(), Status> {
    if size <= 0 {
        return Err(Status::invalid_argument("Vector size must be positive"));
    }

    if size > 10000 {
        return Err(Status::invalid_argument(
            "Vector size exceeds maximum allowed (10000)",
        ));
    }

    // Warn if not standard size (but don't fail)
    if size != 384 && size != 768 && size != 1536 {
        warn!(
            "Non-standard vector size: {}. Ensure this matches your embedding model.",
            size
        );
    }

    Ok(())
}

/// Validate that an alias name does not conflict with canonical collection names.
pub(super) fn validate_alias_name(alias_name: &str) -> Result<(), Status> {
    if CANONICAL_COLLECTIONS.contains(&alias_name) {
        return Err(Status::invalid_argument(format!(
            "Cannot use '{}' as alias: conflicts with canonical collection name",
            alias_name
        )));
    }
    Ok(())
}

/// Map distance metric string to Qdrant Distance enum string.
pub(super) fn map_distance_metric(metric: &str) -> Result<String, Status> {
    match metric {
        "Cosine" => Ok("Cosine".to_string()),
        "Euclidean" => Ok("Euclid".to_string()),
        "Dot" => Ok("Dot".to_string()),
        _ => Err(Status::invalid_argument(format!(
            "Invalid distance metric: {}. Must be one of: Cosine, Euclidean, Dot",
            metric
        ))),
    }
}

/// Map storage errors to gRPC Status.
pub(super) fn map_storage_error(
    err: workspace_qdrant_core::storage::StorageError,
) -> Status {
    use workspace_qdrant_core::storage::StorageError;

    match err {
        StorageError::Collection(msg) if msg.contains("already exists") => {
            Status::already_exists(format!("Collection already exists: {}", msg))
        }
        StorageError::Collection(msg) if msg.contains("not found") => {
            Status::not_found(format!("Collection not found: {}", msg))
        }
        StorageError::Collection(msg) => {
            Status::failed_precondition(format!("Collection error: {}", msg))
        }
        StorageError::Connection(msg) => {
            Status::unavailable(format!("Connection error: {}", msg))
        }
        StorageError::Timeout(msg) => {
            Status::deadline_exceeded(format!("Timeout: {}", msg))
        }
        StorageError::Qdrant(err) => {
            let err_msg = format!("{:?}", err);
            if err_msg.contains("rate limit") || err_msg.contains("too many requests") {
                Status::resource_exhausted("Rate limit exceeded")
            } else if err_msg.contains("not found") {
                Status::not_found(err_msg)
            } else {
                Status::internal(format!("Qdrant error: {}", err_msg))
            }
        }
        _ => Status::internal(format!("Storage error: {}", err)),
    }
}
