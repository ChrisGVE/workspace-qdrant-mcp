//! Idempotency key generation for unified queue items.
//!
//! Provides two key-generation strategies:
//! - [`generate_idempotency_key`]: simple format using only item type, collection, and identifier hash.
//! - [`generate_unified_idempotency_key`]: comprehensive format including operation and tenant ID
//!   (re-exported from `wqm_common`).

use wqm_common::queue_types::ItemType;

// Re-export comprehensive idempotency key generation from wqm-common
pub use wqm_common::hashing::{
    generate_idempotency_key as generate_unified_idempotency_key, IdempotencyKeyError,
};

/// Generate an idempotency key for a queue item (simple format).
///
/// Uses format: `{item_type}:{collection}:{identifier_hash}`
/// where the hash is truncated to 16 hex chars (8 bytes).
///
/// For the comprehensive format with operation and tenant_id,
/// use [`generate_unified_idempotency_key`].
pub fn generate_idempotency_key(
    item_type: ItemType,
    collection: &str,
    identifier: &str,
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(identifier.as_bytes());
    let hash = hasher.finalize();
    // Encode first 8 bytes as hex manually
    let hash_hex: String = hash[..8].iter().map(|b| format!("{:02x}", b)).collect();
    format!("{}:{}:{}", item_type, collection, hash_hex)
}
