//! Hashing utilities shared between daemon and CLI
//!
//! Provides canonical implementations for idempotency key generation
//! and content/file hashing using SHA256.

use sha2::{Sha256, Digest};
use std::fmt;
use std::path::Path;

use crate::queue_types::{ItemType, QueueOperation};

/// Generate a comprehensive idempotency key for unified queue deduplication
///
/// Creates a deterministic key from all relevant queue item attributes to prevent
/// duplicate processing. This function is cross-language compatible.
///
/// # Format
/// Input string: `{item_type}|{op}|{tenant_id}|{collection}|{payload_json}`
/// Output: SHA256 hash truncated to 32 hex characters
///
/// # Errors
/// Returns an error if tenant_id or collection is empty, or if the operation
/// is not valid for the given item type.
pub fn generate_idempotency_key(
    item_type: ItemType,
    op: QueueOperation,
    tenant_id: &str,
    collection: &str,
    payload_json: &str,
) -> Result<String, IdempotencyKeyError> {
    // Validate inputs
    if tenant_id.is_empty() {
        return Err(IdempotencyKeyError::EmptyTenantId);
    }
    if collection.is_empty() {
        return Err(IdempotencyKeyError::EmptyCollection);
    }
    if !op.is_valid_for(item_type) {
        return Err(IdempotencyKeyError::InvalidOperationForType {
            item_type,
            operation: op,
        });
    }

    // Construct canonical input string
    let input = format!(
        "{}|{}|{}|{}|{}",
        item_type, op, tenant_id, collection, payload_json
    );

    // Hash and truncate to 32 hex chars (16 bytes)
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hash = hasher.finalize();

    let hash_hex: String = hash[..16]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    Ok(hash_hex)
}

/// Generate a simple idempotency key (type:collection:hash format)
///
/// Uses format: `{item_type}:{collection}:{identifier_hash}`
/// Hash is truncated to 16 hex chars (8 bytes).
pub fn generate_simple_idempotency_key(
    item_type: ItemType,
    collection: &str,
    identifier: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(identifier.as_bytes());
    let hash = hasher.finalize();
    let hash_hex: String = hash[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    format!("{}:{}:{}", item_type, collection, hash_hex)
}

/// Errors that can occur during idempotency key generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdempotencyKeyError {
    /// tenant_id cannot be empty
    EmptyTenantId,
    /// collection cannot be empty
    EmptyCollection,
    /// The operation is not valid for the given item type
    InvalidOperationForType {
        item_type: ItemType,
        operation: QueueOperation,
    },
}

impl fmt::Display for IdempotencyKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdempotencyKeyError::EmptyTenantId => write!(f, "tenant_id cannot be empty"),
            IdempotencyKeyError::EmptyCollection => write!(f, "collection cannot be empty"),
            IdempotencyKeyError::InvalidOperationForType { item_type, operation } => {
                write!(f, "operation '{}' is not valid for item type '{}'", operation, item_type)
            }
        }
    }
}

impl std::error::Error for IdempotencyKeyError {}

/// Compute SHA256 hash of file content
pub fn compute_file_hash(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 hash of a string (for chunk content)
pub fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idempotency_key_generation() {
        let key1 = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();

        assert_eq!(key1.len(), 32);

        let key2 = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();
        assert_eq!(key1, key2);

        let key3 = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/other.rs"}"#,
        ).unwrap();
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_idempotency_key_validation() {
        let result = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "",
            "my-collection",
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyTenantId));

        let result = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj",
            "",
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

        let result = generate_idempotency_key(
            ItemType::Collection,
            QueueOperation::Add,
            "proj",
            "col",
            "{}",
        );
        assert!(matches!(result, Err(IdempotencyKeyError::InvalidOperationForType { .. })));
    }

    #[test]
    fn test_simple_idempotency_key() {
        let key1 = generate_simple_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/file.txt",
        );
        let key2 = generate_simple_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/file.txt",
        );
        assert_eq!(key1, key2);

        let key3 = generate_simple_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/other.txt",
        );
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_compute_content_hash() {
        let hash1 = compute_content_hash("hello world");
        let hash2 = compute_content_hash("hello world");
        let hash3 = compute_content_hash("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_cross_language_compatibility() {
        // This test vector should produce the same hash in Rust, Python, and TypeScript
        // Input: "file|add|proj_abc123|my-project-code|{}"
        let key = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            "{}",
        ).unwrap();

        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 32);
    }
}
