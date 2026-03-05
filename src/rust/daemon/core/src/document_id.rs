//! Stable document and point ID generation.
//!
//! Provides deterministic ID generation for documents and Qdrant points,
//! ensuring the same input always produces the same ID for deduplication
//! and surgical updates.

use sha2::{Digest, Sha256};

/// Namespace UUID for document IDs (UUID v5).
/// Generated deterministically from the DNS namespace + "workspace-qdrant-mcp.document".
const DOCUMENT_ID_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes([
    0x7a, 0x3b, 0x9c, 0x4d, 0xe5, 0xf6, 0x47, 0x8a, 0xb1, 0xc2, 0xd3, 0xe4, 0xf5, 0x06, 0x17, 0x28,
]);

/// Generate a stable document ID from tenant_id and file path.
///
/// The document ID is deterministic: the same tenant + file always produces
/// the same ID, enabling surgical updates and deduplication.
///
/// Algorithm: `UUID v5(DOCUMENT_ID_NAMESPACE, "tenant_id|normalized_path")`
///
/// Path normalization:
/// - Converts to absolute path (if possible)
/// - Uses forward slashes for cross-platform consistency
/// - Strips trailing slashes
pub fn generate_document_id(tenant_id: &str, file_path: &str) -> String {
    let normalized = normalize_path_for_id(file_path);
    let input = format!("{}|{}", tenant_id, normalized);
    uuid::Uuid::new_v5(&DOCUMENT_ID_NAMESPACE, input.as_bytes()).to_string()
}

/// Generate a stable, branch-scoped Qdrant point ID.
///
/// Formula: `SHA256(tenant_id|branch|file_path|chunk_index)[:32]`
///
/// Branch scoping ensures the same file on different branches gets distinct
/// point IDs, enabling proper branch isolation in Qdrant.
///
/// Path normalization is applied to `file_path` for cross-platform consistency.
pub fn generate_point_id(
    tenant_id: &str,
    branch: &str,
    file_path: &str,
    chunk_index: usize,
) -> String {
    let normalized = normalize_path_for_id(file_path);
    let input = format!("{}|{}|{}|{}", tenant_id, branch, normalized, chunk_index);
    let hash = Sha256::digest(input.as_bytes());
    format!("{:x}", hash)[..32].to_string()
}

/// Generate a stable document ID for content items (no file path).
///
/// Content items use a hash of tenant_id + content to produce stable IDs.
/// This means identical content from the same tenant always gets the same ID.
pub fn generate_content_document_id(tenant_id: &str, content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}|{}", tenant_id, content).as_bytes());
    let hash = hasher.finalize();
    // Use first 32 hex chars as document_id for content
    hash[..16].iter().map(|b| format!("{:02x}", b)).collect()
}

/// Normalize a file path for stable ID generation.
///
/// Ensures the same physical file always produces the same path string:
/// - Uses forward slashes
/// - Strips trailing slashes
pub(crate) fn normalize_path_for_id(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    normalized.trim_end_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_document_id_stability() {
        let id1 = generate_document_id("tenant-abc", "/home/user/project/src/main.rs");
        let id2 = generate_document_id("tenant-abc", "/home/user/project/src/main.rs");
        assert_eq!(id1, id2, "Same inputs must produce identical document_id");
    }

    #[test]
    fn test_generate_document_id_uniqueness() {
        let id1 = generate_document_id("tenant-abc", "/src/main.rs");
        let id2 = generate_document_id("tenant-abc", "/src/lib.rs");
        assert_ne!(id1, id2, "Different files must produce different IDs");
    }

    #[test]
    fn test_generate_document_id_tenant_isolation() {
        let id1 = generate_document_id("tenant-a", "/src/main.rs");
        let id2 = generate_document_id("tenant-b", "/src/main.rs");
        assert_ne!(id1, id2, "Different tenants must produce different IDs");
    }

    #[test]
    fn test_generate_document_id_is_valid_uuid() {
        let id = generate_document_id("tenant", "/some/file.rs");
        assert!(
            uuid::Uuid::parse_str(&id).is_ok(),
            "Must be valid UUID: {}",
            id
        );
        let parsed = uuid::Uuid::parse_str(&id).unwrap();
        assert_eq!(parsed.get_version_num(), 5, "Must be UUID v5");
    }

    #[test]
    fn test_generate_point_id_stability() {
        let p1 = generate_point_id("tenant", "main", "/file.rs", 0);
        let p2 = generate_point_id("tenant", "main", "/file.rs", 0);
        assert_eq!(p1, p2, "Same inputs must produce same point_id");
    }

    #[test]
    fn test_generate_point_id_uniqueness_across_chunks() {
        let p0 = generate_point_id("tenant", "main", "/file.rs", 0);
        let p1 = generate_point_id("tenant", "main", "/file.rs", 1);
        let p2 = generate_point_id("tenant", "main", "/file.rs", 2);
        assert_ne!(p0, p1, "Different chunks must produce different point IDs");
        assert_ne!(p1, p2, "Different chunks must produce different point IDs");
        assert_ne!(p0, p2, "Different chunks must produce different point IDs");
    }

    #[test]
    fn test_generate_point_id_uniqueness_across_files() {
        let p1 = generate_point_id("tenant", "main", "/file1.rs", 0);
        let p2 = generate_point_id("tenant", "main", "/file2.rs", 0);
        assert_ne!(
            p1, p2,
            "Same chunk index but different files must produce different point IDs"
        );
    }

    #[test]
    fn test_generate_point_id_branch_isolation() {
        let p_main = generate_point_id("tenant", "main", "/file.rs", 0);
        let p_dev = generate_point_id("tenant", "dev", "/file.rs", 0);
        assert_ne!(
            p_main, p_dev,
            "Same file on different branches must produce different point IDs"
        );
    }

    #[test]
    fn test_generate_point_id_is_hex_string() {
        let point_id = generate_point_id("tenant", "main", "/file.rs", 42);
        assert_eq!(point_id.len(), 32, "Point ID should be 32-char hex string");
        assert!(
            point_id.chars().all(|c| c.is_ascii_hexdigit()),
            "Point ID must be hex: {}",
            point_id
        );
    }

    #[test]
    fn test_generate_content_document_id_stability() {
        let id1 = generate_content_document_id("tenant", "Some content to index");
        let id2 = generate_content_document_id("tenant", "Some content to index");
        assert_eq!(id1, id2, "Same content must produce same ID");
    }

    #[test]
    fn test_generate_content_document_id_uniqueness() {
        let id1 = generate_content_document_id("tenant", "Content A");
        let id2 = generate_content_document_id("tenant", "Content B");
        assert_ne!(id1, id2, "Different content must produce different IDs");
    }

    #[test]
    fn test_generate_content_document_id_length() {
        let id = generate_content_document_id("tenant", "content");
        assert_eq!(id.len(), 32, "Content document_id must be 32 hex chars");
    }

    #[test]
    fn test_normalize_path_for_id() {
        assert_eq!(
            normalize_path_for_id("/home/user/file.rs"),
            "/home/user/file.rs"
        );
        assert_eq!(
            normalize_path_for_id("C:\\Users\\user\\file.rs"),
            "C:/Users/user/file.rs"
        );
        assert_eq!(normalize_path_for_id("/home/user/dir/"), "/home/user/dir");
        assert_eq!(normalize_path_for_id(""), "");
    }

    #[test]
    fn test_document_id_path_normalization() {
        let id1 = generate_document_id("tenant", "/home/user/file.rs");
        let id2 = generate_document_id("tenant", "\\home\\user\\file.rs");
        assert_eq!(
            id1, id2,
            "Forward and back slashes should produce same document_id"
        );
    }

    #[test]
    fn test_document_id_trailing_slash_invariance() {
        let id1 = generate_document_id("tenant", "/home/user/dir");
        let id2 = generate_document_id("tenant", "/home/user/dir/");
        assert_eq!(id1, id2, "Trailing slash should not affect document_id");
    }
}
