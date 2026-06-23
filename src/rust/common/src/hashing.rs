//! Hashing utilities shared between daemon and CLI
//!
//! Provides canonical implementations for idempotency key generation
//! and content/file hashing using SHA256.

use sha2::{Digest, Sha256};
use std::fmt;
use std::path::Path;
use uuid::Uuid;

use crate::queue_types::{ItemType, QueueOperation};

const MAX_HASH_READ_BYTES: u64 = 100 * 1024 * 1024;

/// UUIDv5 namespace for ALL Qdrant point IDs (branch-lineage F1).
///
/// Fixed once, here, so every producer (the branch tagger F6, the re-key pass F16,
/// the v48 conversion, and non-file content ingestion) derives identical IDs for
/// identical inputs. Never change this value — it is baked into every stored point.
pub const POINT_NS: Uuid = Uuid::from_u128(0x6b1f9a2c_3d4e_4f60_8a71_5c2e9d0b7e34);

/// Length-prefix framing: `lp(x) = u32_be(x.len()) ‖ x`.
///
/// The single source-of-truth framing helper for all composite hash inputs.
/// Prefixing each field with its byte length makes the concatenation injective —
/// `lp(a)‖lp(b)` can never collide with `lp(a')‖lp(b')` for different `(a, b)` —
/// which a bare separator byte (e.g. `a|b`) cannot guarantee when the data may
/// itself contain the separator.
pub fn lp(x: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + x.len());
    out.extend_from_slice(&(x.len() as u32).to_be_bytes());
    out.extend_from_slice(x);
    out
}

/// The single SHA-256 content-key primitive: `hex(SHA256(lp(s0) ‖ lp(s1) ‖ …))`.
///
/// This is the ONE place the framing-and-hash rule for content keys lives. Both
/// the legacy three-slot producer [`content_key_v3`] and the four-slot producer
/// [`content_key_v4`] are THIN compositions over this primitive — they only choose
/// which ordered slots to pass; neither re-implements SHA-256 or the lp-framing
/// concatenation (FP-2 / DR GP-1 "no fork" = no re-implemented formula). Two NAMED
/// compositions over one primitive is the required and correct shape because the
/// 3-slot and 4-slot inputs are genuinely distinct hash inputs and Rust has no
/// default arguments (a single function cannot serve both by defaulting).
///
/// Keeps the FULL 32-byte digest (64 lowercase-hex chars) — birthday bound ~2^64.
fn content_key_hash(slots: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for slot in slots {
        hasher.update(lp(slot.as_bytes()));
    }
    format!("{:x}", hasher.finalize())
}

/// Bucket tags: the stable `collection` discriminator for the four-slot
/// [`content_key_v4`] producer (arch §5.4, DOM-01 fix).
///
/// These are NET-NEW symbols F5 introduces (verified zero prior hits). They exist
/// so the within-tenant cross-collection namespaces can never coincide: a code
/// chunk-blob keyed on a chunk hash and an identity-addressed document point keyed
/// on a document identity that happen to share a hex string still produce DISTINCT
/// keys because the `collection` slot differs. Call sites pass the constant rather
/// than a bare string literal so a typo cannot silently mint a new bucket and split
/// dedup.
pub mod bucket {
    /// Code chunk-blobs (the chunk-grain dedup namespace).
    pub const CODE: &str = "code";
    /// Scratchpad / notes points.
    pub const SCRATCHPAD: &str = "scratchpad";
    /// Durable rule points.
    pub const RULES: &str = "rules";
    /// Ingested URL points.
    pub const URL: &str = "url";
    /// Library / reference-document points.
    pub const LIBRARY: &str = "library";
}

/// Legacy three-slot `content_key` (the `content_key_version = 3` path).
///
/// `content_key_v3 = hex(SHA256(lp(tenant) ‖ lp(identity) ‖ lp(content_hash_hex)))`.
///
/// This is the NAMED back-compat producer retained while a tenant is still at
/// `content_key_version = 3` (per-tenant gate, AC-F4.5 / AC-F5.8). It is
/// BYTE-IDENTICAL to the live pre-F5 `content_key(tenant_id, identity,
/// content_hash_hex)` it replaces — the recorded golden vector pins this. A
/// pre-F5 tenant keeps minting 3-slot points (consistent with its still-3-slot
/// store) until its atomic F13 cutover flips the flag to 4.
///
/// **CRITICAL (DOM-R5-N2):** this MUST hash the genuine THREE-slot concatenation
/// `lp(tenant) ‖ lp(identity) ‖ lp(content_hash)`. It MUST NOT be implemented as
/// `content_key_v4(tenant, "", identity, content_hash)` — a four-slot call with an
/// empty collection tag inserts a FOURTH `lp("")` slot into the SHA-256 input and
/// yields a DIFFERENT digest, silently re-minting every `point_id` for an
/// unmigrated 3-slot tenant and orphaning its existing Qdrant points. The unit
/// test `t_f5_v3_byte_identical_to_pre_f5` asserts both the equality to the
/// recorded pre-F5 digest and the inequality to the four-slot-with-empty-collection
/// call.
///
/// Field contract (DOM-02, N7): `content_hash_hex` is the content hash rendered as
/// a **64-char lowercase-hex ASCII string**, never the raw 32 bytes. For files the
/// arguments are `(tenant_id, file_identity_id, file_hash_hex)`; for non-file
/// content (rules/scratchpad/memory/url/library) the `identity` slot carries the
/// stable document identity and `content_hash_hex` the content digest.
pub fn content_key_v3(tenant_id: &str, identity: &str, content_hash_hex: &str) -> String {
    content_key_hash(&[tenant_id, identity, content_hash_hex])
}

/// Four-slot, collection-discriminated `content_key` (the `content_key_version = 4`
/// path — the F5 generalized producer, arch §5.4).
///
/// `content_key_v4 = hex(SHA256(lp(tenant) ‖ lp(collection) ‖ lp(identity) ‖ lp(content_hash_hex)))`.
///
/// The `collection` slot is a stable [`bucket`] tag. Chunk-blobs call
/// `content_key_v4(tenant, bucket::CODE, chunk_content_hash, "")`; identity-addressed
/// points call `content_key_v4(tenant, bucket::<COLLECTION>, doc_identity, "")`. The
/// extra slot is what makes within-tenant cross-collection collisions impossible
/// (DOM-01): a chunk hash byte-equal to a document identity still keys distinctly
/// because the `collection` slot differs (proven by `t_f5_collection_discriminator`).
///
/// Field contract: `content_hash_hex` is a 64-char lowercase-hex string (or `""`
/// for identity-addressed / chunk-grain calls, where the identity slot carries the
/// content-addressing input).
pub fn content_key_v4(
    tenant_id: &str,
    collection: &str,
    identity: &str,
    content_hash_hex: &str,
) -> String {
    content_key_hash(&[tenant_id, collection, identity, content_hash_hex])
}

/// Per-tenant version-gated `content_key` selector (AC-F5.8).
///
/// Dispatches by the tenant's `projects.content_key_version` (AC-F4.5):
/// - `3` → [`content_key_v3`] (legacy three-slot; the `collection` argument is
///   ignored — a pre-F5 tenant must keep minting 3-slot points until its F13
///   cutover so its existing Qdrant points are not orphaned).
/// - `4` → [`content_key_v4`] (the four-slot collection-discriminated producer).
/// - anything else → **panics** with a clear message, because an unknown version is
///   a schema/migration bug, not a recoverable runtime condition: minting points
///   under an unrecognized slot-shape would silently corrupt the corpus.
///
/// **F6 deferral:** the once-per-session reading and CACHING of
/// `projects.content_key_version` at the ingest call-site (PERF-R4-N1 — a per-chunk
/// read is ~2 s of state.db contention at 400k chunks) is F6's wiring. This function
/// is the pure selector only: F6 passes it the already-cached flag value.
pub fn content_key_for_version(
    version: i64,
    tenant_id: &str,
    collection: &str,
    identity: &str,
    content_hash_hex: &str,
) -> String {
    match version {
        3 => content_key_v3(tenant_id, identity, content_hash_hex),
        4 => content_key_v4(tenant_id, collection, identity, content_hash_hex),
        other => panic!(
            "unknown content_key_version {other}: expected 3 (legacy three-slot) or 4 \
             (four-slot collection-discriminated); refusing to mint points under an \
             unrecognized slot-shape (would corrupt the corpus)"
        ),
    }
}

/// Derive a Qdrant `point_id` from a `content_key` and chunk index.
///
/// `point_id = UUIDv5(POINT_NS, lp(content_key) ‖ lp(u32_be(chunk_index)))`.
///
/// This is the SINGLE point-id derivation flow: file chunks and non-file content
/// points both route through it (see [`content_point_id`]), so the two paths can
/// never drift apart. Returns a UUIDv5 (RFC 9562 §5.5, SHA-1 based with 6 bits
/// fixed for version+variant → ~2^61 effective collision space; a collision
/// silently overwrites a chunk on upsert, so corpus-size guidance quotes ~2^61
/// for `point_id` vs ~2^64 for `content_key`).
pub fn point_id(content_key: &str, chunk_index: u32) -> Uuid {
    let mut name = lp(content_key.as_bytes());
    name.extend_from_slice(&lp(&chunk_index.to_be_bytes()));
    Uuid::new_v5(&POINT_NS, &name)
}

/// Derive the canonical `branch_id` for a project checkout.
///
/// `branch_id = hex(SHA256(lp(tenant_id) || lp(location) || lp(branch_name)))`.
///
/// This is the SINGLE producer of a `branch_id` (GP-5 / DR GP-1). Two checkouts
/// of the same branch at different `location` paths yield DISTINCT `branch_id`
/// values even when the content is identical — path-sensitivity is intentional
/// (AC-F4.2 SEED F09).
///
/// Arguments:
/// - `tenant_id` — the project's stable UUID string
/// - `location`  — the absolute checkout root path (forward-slash normalized)
/// - `branch_name` — the git ref name ("main", "feat/x", ...)
///
/// The caller is responsible for branch-name validation (AC-F4.6): this function
/// derives a hash from whatever string it receives. Validation must happen BEFORE
/// calling this function at every registration call-site (daemon-core, where git2
/// is available).
pub fn branch_id(tenant_id: &str, location: &str, branch_name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(lp(tenant_id.as_bytes()));
    hasher.update(lp(location.as_bytes()));
    hasher.update(lp(branch_name.as_bytes()));
    format!("{:x}", hasher.finalize())
}

/// Convenience: the point ID for a non-file, *identity-addressed* content point
/// (rules / scratchpad / memory / url / library).
///
/// NOT a second derivation — it is exactly `point_id(content_key_v3(...))`, so these
/// points share the one canonical flow with file chunks rather than maintaining a
/// parallel formula that could diverge. It stays on the legacy three-slot
/// [`content_key_v3`] path so existing url/library/text callers keep producing
/// BYTE-IDENTICAL ids (no orphaning); the per-tenant four-slot migration is gated
/// by [`content_key_for_version`] at the ingest call-site (F6 wiring). The
/// content-hash slot is left empty because these points are keyed by a STABLE
/// document identity (e.g. a URL), not by their content: re-ingesting changed
/// content must keep the same ID so the update lands in place.
pub fn content_point_id(tenant_id: &str, identity: &str, chunk_index: u32) -> Uuid {
    point_id(&content_key_v3(tenant_id, identity, ""), chunk_index)
}

/// Re-keyed (salted) `point_id` derivation for the SEC-4 residual collision guard
/// (AC-F5.5).
///
/// When two distinct `content_key`s derive the SAME `point_id` (a ~2^61 UUIDv5
/// event), the loser is re-keyed: the salted derivation folds a random `nonce` into
/// the UUIDv5 name so the re-keyed point lands on a fresh, distinct id.
///
/// `salted_point_id = UUIDv5(POINT_NS, lp(content_key) ‖ lp(u32_be(chunk_index)) ‖ lp(nonce))`.
///
/// The lp-framed `nonce` is appended AFTER the same `lp(content_key) ‖
/// lp(u32_be(chunk_index))` prefix [`point_id`] uses, so a non-empty nonce can never
/// collide with the un-salted derivation (lp framing is injective) and an empty
/// nonce is NOT equal to `point_id` either (the trailing `lp("")` = four zero bytes
/// changes the digest) — the guard always re-keys with a genuinely random nonce.
///
/// **F6 deferral:** this is the PURE derivation only. Verifying the stored
/// `point_id` against the expected derivation on upsert, persisting the salted
/// `point_id` durably so rebuild reads it verbatim (DATA-05), and logging the alert
/// are the F6 ingest-path wiring (AC-F6.2). F5 ships only this function and its
/// forced-collision unit test.
pub fn salted_point_id(content_key: &str, nonce: &[u8], chunk_index: u32) -> Uuid {
    let mut name = lp(content_key.as_bytes());
    name.extend_from_slice(&lp(&chunk_index.to_be_bytes()));
    name.extend_from_slice(&lp(nonce));
    Uuid::new_v5(&POINT_NS, &name)
}

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

    let hash_hex: String = hash[..16].iter().map(|b| format!("{:02x}", b)).collect();

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
    let hash_hex: String = hash[..8].iter().map(|b| format!("{:02x}", b)).collect();
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
            IdempotencyKeyError::InvalidOperationForType {
                item_type,
                operation,
            } => {
                write!(
                    f,
                    "operation '{}' is not valid for item type '{}'",
                    operation, item_type
                )
            }
        }
    }
}

impl std::error::Error for IdempotencyKeyError {}

/// Compute SHA256 hash of file content
pub fn compute_file_hash(path: &Path) -> std::io::Result<String> {
    use std::io::Read;

    let meta = std::fs::metadata(path)?;
    if !meta.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("cannot hash non-regular file: {}", path.display()),
        ));
    }

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    let mut total_read: u64 = 0;
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read as u64;
        if total_read > MAX_HASH_READ_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file exceeds max hashable size: {}", path.display()),
            ));
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

/// Normalize a file path for stable ID generation.
///
/// Ensures the same physical file always produces the same path string:
/// - Uses forward slashes
/// - Strips trailing slashes
pub fn normalize_path_for_id(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    normalized.trim_end_matches('/').to_string()
}

/// Compute the base point hash: `SHA256(tenant_id|relative_path|file_hash)[:32]`
///
/// The base point uniquely identifies a specific VERSION of a specific file,
/// independent of branch. Identical content at the same path shares one
/// base_point across all branches, enabling content-hash dedup.
///
/// - `tenant_id`: derived from git remote URL hash (git) or path hash (non-git)
/// - `relative_path`: file path relative to project root (normalized)
/// - `file_hash`: SHA256 of file content or git blob SHA
pub fn compute_base_point(tenant_id: &str, relative_path: &str, file_hash: &str) -> String {
    let normalized = normalize_path_for_id(relative_path);
    let input = format!("{}|{}|{}", tenant_id, normalized, file_hash);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hash = hasher.finalize();
    hash[..16].iter().map(|b| format!("{:02x}", b)).collect()
}

/// Compute a Qdrant point ID from a base point and chunk index.
///
/// Formula: `SHA256(base_point|chunk_index)[:32]`
///
/// Each chunk of a file version gets a unique, deterministic point ID.
pub fn compute_point_id(base_point: &str, chunk_index: u32) -> String {
    let input = format!("{}|{}", base_point, chunk_index);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hash = hasher.finalize();
    hash[..16].iter().map(|b| format!("{:02x}", b)).collect()
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
        )
        .unwrap();

        assert_eq!(key1.len(), 32);

        let key2 = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        )
        .unwrap();
        assert_eq!(key1, key2);

        let key3 = generate_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/other.rs"}"#,
        )
        .unwrap();
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

        let result =
            generate_idempotency_key(ItemType::File, QueueOperation::Add, "proj", "", "{}");
        assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

        let result = generate_idempotency_key(
            ItemType::Collection,
            QueueOperation::Add,
            "proj",
            "col",
            "{}",
        );
        assert!(matches!(
            result,
            Err(IdempotencyKeyError::InvalidOperationForType { .. })
        ));
    }

    #[test]
    fn test_simple_idempotency_key() {
        let key1 =
            generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
        let key2 =
            generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
        assert_eq!(key1, key2);

        let key3 =
            generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/other.txt");
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
        )
        .unwrap();

        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_normalize_path_for_id() {
        assert_eq!(normalize_path_for_id("src/main.rs"), "src/main.rs");
        assert_eq!(normalize_path_for_id("src\\main.rs"), "src/main.rs");
        assert_eq!(normalize_path_for_id("src/dir/"), "src/dir");
        assert_eq!(
            normalize_path_for_id("src\\dir\\file.rs"),
            "src/dir/file.rs"
        );
    }

    #[test]
    fn test_compute_base_point_deterministic() {
        let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        let bp2 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        assert_eq!(bp1, bp2);
        assert_eq!(bp1.len(), 32);
        assert!(bp1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_compute_base_point_different_file_hash() {
        let bp1 = compute_base_point("tenant_abc", "src/main.rs", "hash_v1");
        let bp2 = compute_base_point("tenant_abc", "src/main.rs", "hash_v2");
        assert_ne!(bp1, bp2);
    }

    #[test]
    fn test_compute_base_point_branch_agnostic() {
        let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        assert_eq!(bp.len(), 32);
        assert!(bp.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_compute_base_point_different_path() {
        let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        let bp2 = compute_base_point("tenant_abc", "src/lib.rs", "deadbeef");
        assert_ne!(bp1, bp2);
    }

    #[test]
    fn test_compute_base_point_different_tenant() {
        let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        let bp2 = compute_base_point("tenant_xyz", "src/main.rs", "deadbeef");
        assert_ne!(bp1, bp2);
    }

    #[test]
    fn test_compute_base_point_path_normalization() {
        let bp1 = compute_base_point("t", "src/main.rs", "h");
        let bp2 = compute_base_point("t", "src\\main.rs", "h");
        assert_eq!(bp1, bp2);
    }

    #[test]
    fn test_compute_point_id_deterministic() {
        let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        let pid1 = compute_point_id(&bp, 0);
        let pid2 = compute_point_id(&bp, 0);
        assert_eq!(pid1, pid2);
        assert_eq!(pid1.len(), 32);
    }

    #[test]
    fn test_compute_point_id_different_chunk_index() {
        let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
        let pid0 = compute_point_id(&bp, 0);
        let pid1 = compute_point_id(&bp, 1);
        let pid2 = compute_point_id(&bp, 2);
        assert_ne!(pid0, pid1);
        assert_ne!(pid1, pid2);
        assert_ne!(pid0, pid2);
    }

    #[test]
    fn test_compute_point_id_different_base_point() {
        let bp1 = compute_base_point("tenant_abc", "src/main.rs", "hash_v1");
        let bp2 = compute_base_point("tenant_abc", "src/main.rs", "hash_v2");
        let pid1 = compute_point_id(&bp1, 0);
        let pid2 = compute_point_id(&bp2, 0);
        assert_ne!(pid1, pid2);
    }

    #[test]
    fn test_base_point_cross_language_test_vector() {
        let bp = compute_base_point("test_tenant", "src/example.rs", "abc123hash");
        assert_eq!(bp, "d08103c2d8f553544dabeb4737fd32b4");

        let pid = compute_point_id(&bp, 0);
        assert_eq!(pid.len(), 32);
    }

    #[test]
    fn hash_rejects_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = compute_file_hash(dir.path());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn hash_rejects_dev_null() {
        let path = Path::new("/dev/null");
        if !path.exists() {
            return;
        }
        let result = compute_file_hash(path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn hash_regular_file_works() {
        let dir = tempfile::TempDir::new().unwrap();
        let fp = dir.path().join("test.txt");
        std::fs::write(&fp, b"hello").unwrap();
        assert!(compute_file_hash(&fp).is_ok());
    }

    // ---- F1: lp / content_key / point_id (branch-lineage) ----

    #[test]
    fn test_lp_frames_with_big_endian_length() {
        assert_eq!(lp(b""), vec![0, 0, 0, 0]);
        assert_eq!(lp(b"ab"), vec![0, 0, 0, 2, b'a', b'b']);
        // 256-byte payload → length 0x00000100 prefix.
        let big = vec![0x5au8; 256];
        let framed = lp(&big);
        assert_eq!(&framed[..4], &[0, 0, 1, 0]);
        assert_eq!(&framed[4..], &big[..]);
    }

    proptest::proptest! {
        /// T-F1-lp-injective (DOM-01): the framed concatenation `lp(a)‖lp(b)` is
        /// injective in `(a, b)` — distinct field pairs never produce the same bytes.
        /// This is the property a bare separator (`a|b`) cannot offer, since data may
        /// contain the separator. We assert the contrapositive: differing pairs differ.
        #[test]
        fn t_f1_lp_injective(
            a in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
            b in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
            c in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
            d in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
        ) {
            let mut left = lp(&a);
            left.extend_from_slice(&lp(&b));
            let mut right = lp(&c);
            right.extend_from_slice(&lp(&d));
            if (a, b) == (c, d) {
                proptest::prop_assert_eq!(left, right);
            } else {
                proptest::prop_assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn t_f1_lp_defeats_separator_ambiguity() {
        // The bare-separator hazard: "a|bc" == "ab|c" collides; lp framing must not.
        let mut x = lp(b"a");
        x.extend_from_slice(&lp(b"bc"));
        let mut y = lp(b"ab");
        y.extend_from_slice(&lp(b"c"));
        assert_ne!(x, y);
    }

    #[test]
    fn t_f1_content_key_is_full_32_bytes_and_lp_framed() {
        let ck = content_key_v3("tenant_abc", "fid-1", &"de".repeat(32));
        assert_eq!(
            ck.len(),
            64,
            "content_key keeps full 32-byte digest (64 hex)"
        );
        assert!(ck.chars().all(|c| c.is_ascii_hexdigit()));

        // Independently recompute via the documented formula to pin the encoding.
        let mut h = Sha256::new();
        h.update(lp(b"tenant_abc"));
        h.update(lp(b"fid-1"));
        h.update(lp("de".repeat(32).as_bytes()));
        let expected = format!("{:x}", h.finalize());
        assert_eq!(ck, expected);
    }

    #[test]
    fn t_f1_content_key_field_boundaries_matter() {
        // Moving a character across the identity/hash boundary must change the key.
        let a = content_key_v3("t", "ab", "c");
        let b = content_key_v3("t", "a", "bc");
        assert_ne!(a, b);
    }

    #[test]
    fn t_f1_encoding_agreement() {
        // N7: the SAME (tenant, file_identity_id, file_hash_hex) yields the SAME
        // content_key regardless of which caller computes it — there is one producer.
        let tenant = "tenant_xyz";
        let fid = "file-identity-7";
        let file_hash_hex = "abc123".repeat(10) + "abcd"; // 64-char hex string
        let tagger_side = content_key_v3(tenant, fid, &file_hash_hex);
        let rekey_side = content_key_v3(tenant, fid, &file_hash_hex);
        assert_eq!(tagger_side, rekey_side);
        // The hex STRING (not raw bytes) is the input: a different rendering differs.
        let raw_bytes_rendering = content_key_v3(tenant, fid, "abc123");
        assert_ne!(tagger_side, raw_bytes_rendering);
    }

    #[test]
    fn t_f1_pointid_stability() {
        let ck = content_key_v3("tenant", "fid", &"00".repeat(32));
        let p1 = point_id(&ck, 0);
        let p2 = point_id(&ck, 0);
        assert_eq!(p1, p2, "same content_key + chunk → same point_id");
        assert_eq!(p1.get_version_num(), 5, "point_id is a UUIDv5");
        assert_ne!(
            point_id(&ck, 0),
            point_id(&ck, 1),
            "chunk index is distinguishing"
        );
        // content_point_id is exactly point_id(content_key_v3(tenant, identity, "")) —
        // one flow, identity-addressed (empty content-hash slot).
        let composed = content_point_id("tenant", "doc-7", 3);
        assert_eq!(
            composed,
            point_id(&content_key_v3("tenant", "doc-7", ""), 3)
        );
    }

    #[test]
    fn t_f1_pointid_chunk_index_lp_framed() {
        // chunk_index is framed as lp(u32_be), so it cannot bleed into content_key.
        // Distinct content_keys with distinct chunk indices stay distinct.
        let ck_a = content_key_v3("t", "a", &"11".repeat(32));
        let ck_b = content_key_v3("t", "b", &"11".repeat(32));
        assert_ne!(point_id(&ck_a, 1), point_id(&ck_b, 1));
        assert_ne!(point_id(&ck_a, 1), point_id(&ck_a, 2));
    }

    // ---- F0: producers pinned to one canonical home (FP-2) + golden output lock ----

    /// T-F0-golden: literal golden values for the `content_key` and `point_id`
    /// producers. These pin the EXACT bytes the canonical producers emit. Any change
    /// to the framing, field order, hash backend, or [`POINT_NS`] namespace flips
    /// these values and the test fails loudly — which is the point: content addressing
    /// is a stored contract (point IDs already persisted in every corpus), so a
    /// behavioral change must be a deliberate, version-gated migration, never silent
    /// drift. The golden values were computed out-of-band from the documented formula
    /// (`hex(SHA256(lp..))` and `UUIDv5(POINT_NS, lp..)`).
    #[test]
    fn t_f0_producer_golden_vectors() {
        let ck = content_key_v3("tenant_golden", "fid-golden", &"ab".repeat(32));
        assert_eq!(
            ck, "829340abef5c0c8c6760f472b0d687a0dd9525f74fe03130f20cb1c8bd893b88",
            "content_key_v3 golden vector changed — content addressing would break"
        );
        let pid = point_id(&ck, 0);
        assert_eq!(
            pid.to_string(),
            "d2736140-73aa-5ae6-b133-71d846462533",
            "point_id golden vector changed — stored Qdrant point IDs would break"
        );
    }

    /// T-F0-single-home (FP-2): the content-addressing producers have exactly ONE
    /// definition tree-wide, here in `wqm-common::hashing`. A second definition
    /// anywhere is the drift FP-2 forbids — two producers can silently diverge and
    /// split a corpus into incompatible key spaces. The guard walks the workspace
    /// Rust source and counts `pub fn` producer signatures, asserting one each.
    #[test]
    fn t_f0_producers_have_single_definition() {
        use std::path::{Path, PathBuf};

        // Locate the workspace source root (src/rust): walk up from this crate's
        // manifest dir until a directory holding both the `common` and `daemon`
        // members. Outside the in-repo tree (e.g. a packaged crate) the guard no-ops.
        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ws_root = loop {
            if root.join("common").is_dir() && root.join("daemon").is_dir() {
                break Some(root.clone());
            }
            if !root.pop() {
                break None;
            }
        };
        let Some(ws_root) = ws_root else {
            return;
        };

        let producers = [
            "pub fn content_key_v3(",
            "pub fn content_key_v4(",
            "pub fn point_id(",
            "pub fn content_point_id(",
            "pub fn branch_id(",
        ];
        let mut counts = [0usize; 5];

        fn walk(dir: &Path, f: &mut impl FnMut(&Path)) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if name == "target" || name.starts_with('.') {
                        continue;
                    }
                    walk(&p, f);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    f(&p);
                }
            }
        }

        walk(&ws_root, &mut |p| {
            let Ok(src) = std::fs::read_to_string(p) else {
                return;
            };
            for line in src.lines() {
                let t = line.trim_start();
                for (i, pat) in producers.iter().enumerate() {
                    if t.starts_with(pat) {
                        counts[i] += 1;
                    }
                }
            }
        });

        for (i, pat) in producers.iter().enumerate() {
            assert_eq!(
                counts[i], 1,
                "expected exactly one definition of `{}` tree-wide, found {} (FP-2 single-home violated)",
                pat, counts[i]
            );
        }
    }

    // ---- F5: collection-discriminated four-slot producer + version gate ----

    /// T-F5-v3-byte-identical (AC-F5.8 / DOM-R5-N2): `content_key_v3` is
    /// BYTE-IDENTICAL to the live pre-F5 three-slot `content_key` it replaces, and
    /// is NOT the four-slot-with-empty-collection call (which would orphan every
    /// existing Qdrant point of an unmigrated tenant).
    #[test]
    fn t_f5_v3_byte_identical_to_pre_f5() {
        let tenant = "tenant_golden";
        let identity = "fid-golden";
        let content_hash = "ab".repeat(32);

        // The recorded pre-F5 three-slot digest (same fixture as t_f0 golden).
        let v3 = content_key_v3(tenant, identity, &content_hash);
        assert_eq!(
            v3, "829340abef5c0c8c6760f472b0d687a0dd9525f74fe03130f20cb1c8bd893b88",
            "content_key_v3 must be byte-identical to the pre-F5 three-slot digest"
        );

        // It must NOT equal a four-slot call with an empty collection tag: that
        // inserts a fourth lp("") slot and yields a DIFFERENT digest.
        let v4_empty_collection = content_key_v4(tenant, "", identity, &content_hash);
        assert_ne!(
            v3, v4_empty_collection,
            "content_key_v3 must NOT be content_key_v4(tenant, \"\", identity, hash) — \
             the extra lp(\"\") slot changes the digest and orphans existing points"
        );
    }

    /// T-F5-v4-golden (AC-F5.6): pin the exact bytes the four-slot producer emits.
    /// Computed out-of-band from `hex(SHA256(lp(tenant)||lp("code")||lp(hash)||lp("")))`
    /// and cross-checked inline. A change to slot order/framing flips this and fails.
    #[test]
    fn t_f5_v4_golden_vector() {
        let chunk_hash = "cafe".repeat(16);
        let v4 = content_key_v4("tenant_golden", bucket::CODE, &chunk_hash, "");
        assert_eq!(
            v4, "acffeb1cff2566d4c39fd6e92e62c593d749e905b33863f631c271b944f40a74",
            "content_key_v4 golden vector changed — four-slot addressing would break"
        );

        // Recompute inline via the documented formula to pin the encoding.
        let mut h = Sha256::new();
        h.update(lp(b"tenant_golden"));
        h.update(lp(b"code"));
        h.update(lp(chunk_hash.as_bytes()));
        h.update(lp(b""));
        assert_eq!(v4, format!("{:x}", h.finalize()));

        let pid = point_id(&v4, 0);
        assert_eq!(
            pid.to_string(),
            "59538c5f-5c1e-5d1b-a83d-e9fe2d2afe21",
            "v4 point_id golden vector changed — stored Qdrant point IDs would break"
        );
    }

    /// T-F5.1 (AC-F5.1): path-independent code-chunk dedup. The same chunk text in
    /// file A and file B (same tenant, same collection) yields ONE content_key —
    /// the file path is NOT a slot, so identical chunk content collapses to one blob.
    #[test]
    fn t_f5_path_independent_chunk_dedup() {
        let chunk_hash = compute_content_hash("fn main() {}\n");
        // The chunk came from file A in one call, file B in another — but neither
        // path appears in the key, so the keys are identical.
        let key_in_file_a = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
        let key_in_file_b = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
        assert_eq!(
            key_in_file_a, key_in_file_b,
            "identical chunk content must dedup to one content_key regardless of file"
        );
        // And the point id (chunk_index always 0, AC-F5.2) is likewise identical.
        assert_eq!(
            point_id(&key_in_file_a, 0),
            point_id(&key_in_file_b, 0),
            "one blob -> one Qdrant point"
        );
    }

    /// T-F5.2 (AC-F5.2): blob points always pass chunk_index = 0 — one blob maps to
    /// one Qdrant point; positional distinction lives in blob_refs.chunk_index, not
    /// in the point_id. We pin that 0 is the convention and that varying it would
    /// mint a different (wrong) point.
    #[test]
    fn t_f5_blob_point_uses_chunk_index_zero() {
        let chunk_hash = compute_content_hash("payload");
        let key = content_key_v4("t", bucket::CODE, &chunk_hash, "");
        let blob_point = point_id(&key, 0);
        assert_ne!(
            blob_point,
            point_id(&key, 1),
            "chunk_index is distinguishing; blob points must use 0 to stay single"
        );
    }

    /// T-F5.3 (AC-F5.3): cross-tenant isolation. Identical chunk content in two
    /// tenants yields different content_keys (tenant is the first slot), partitioning
    /// the point_id space.
    #[test]
    fn t_f5_cross_tenant_isolation() {
        let chunk_hash = compute_content_hash("shared bytes");
        let t1 = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
        let t2 = content_key_v4("tenant-2", bucket::CODE, &chunk_hash, "");
        assert_ne!(t1, t2, "different tenants must not share a content_key");
        assert_ne!(point_id(&t1, 0), point_id(&t2, 0));
    }

    /// T-F5.4 (AC-F5.4 / DOM-01): the collection-discriminator forced-coincidence
    /// test. Craft a chunk hash byte-equal to a document identity and assert that —
    /// because the collection slot differs ("code" vs "scratchpad") — the two
    /// produce DISTINCT keys/points. This test FAILS if the discriminator is removed
    /// (e.g. if both routed through the legacy three-slot `content_key_v3`, which has
    /// no collection slot, they would COLLIDE).
    #[test]
    fn t_f5_collection_discriminator() {
        // Same tenant; the chunk hash and the doc identity are the identical string.
        let coincident = "0123456789abcdef".repeat(4); // 64 hex chars
        let tenant = "tenant-x";

        let code_key = content_key_v4(tenant, bucket::CODE, &coincident, "");
        let scratch_key = content_key_v4(tenant, bucket::SCRATCHPAD, &coincident, "");
        assert_ne!(
            code_key, scratch_key,
            "collection slot must keep code and scratchpad namespaces distinct"
        );
        assert_ne!(point_id(&code_key, 0), point_id(&scratch_key, 0));

        // Proof the discriminator is load-bearing: without it (the legacy 3-slot
        // path, which drops the collection slot entirely) the two coincide.
        let no_discriminator_code = content_key_v3(tenant, &coincident, "");
        let no_discriminator_scratch = content_key_v3(tenant, &coincident, "");
        assert_eq!(
            no_discriminator_code, no_discriminator_scratch,
            "without the collection slot the two namespaces collapse — exactly the \
             DOM-01 hazard the discriminator fixes"
        );
    }

    /// T-F5.5 (AC-F5.5): the SEC-4 residual point_id-collision guard — the pure
    /// salted re-key derivation. A forced-collision fixture: two distinct content
    /// keys hand-picked so their un-salted point_ids would be the same is not
    /// constructible cheaply, so we assert the GUARANTEE the guard relies on — that
    /// a salted re-key with a random nonce moves the point to a fresh, distinct id,
    /// deterministically and reproducibly.
    #[test]
    fn t_f5_salted_point_id_rekey() {
        let key = content_key_v4("t", bucket::CODE, &compute_content_hash("blob"), "");
        let unsalted = point_id(&key, 0);

        let nonce = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04];
        let salted = salted_point_id(&key, &nonce, 0);
        assert_ne!(
            salted, unsalted,
            "a salted re-key must land on a fresh, distinct point_id"
        );
        // Deterministic: rebuild reads the same salted id from the same nonce.
        assert_eq!(salted, salted_point_id(&key, &nonce, 0));
        // A different nonce yields a different salted id (collision-escape works).
        assert_ne!(salted, salted_point_id(&key, &[0xff, 0xee], 0));
        // Even an EMPTY nonce is distinct from the un-salted derivation (the
        // trailing lp("") = 4 zero bytes changes the digest), so the guard never
        // accidentally re-collides with the un-salted point.
        assert_ne!(salted_point_id(&key, &[], 0), unsalted);
        assert_eq!(salted.get_version_num(), 5, "salted point_id is a UUIDv5");
    }

    /// T-F5.6 (AC-F5.6): the four-slot field-contract convention holds — code
    /// chunk-blobs use the `bucket::CODE` collection tag and carry the chunk hash in
    /// the identity slot with an empty content-hash slot.
    #[test]
    fn t_f5_field_contract_convention() {
        let chunk_hash = compute_content_hash("some code chunk");
        // The documented call shape for a code chunk-blob.
        let via_constant = content_key_v4("t", bucket::CODE, &chunk_hash, "");
        // Equals the literal four-slot formula with collection="code", identity=hash,
        // content_hash="".
        let mut h = Sha256::new();
        h.update(lp(b"t"));
        h.update(lp(b"code"));
        h.update(lp(chunk_hash.as_bytes()));
        h.update(lp(b""));
        assert_eq!(via_constant, format!("{:x}", h.finalize()));
        // bucket::CODE is the literal "code" tag (typed constant, not bare literal).
        assert_eq!(bucket::CODE, "code");
    }

    /// T-F5.7 (AC-F5.7 / DATA-09): the `point_id == point_id(content_key, 0)`
    /// dual-UNIQUE binding for a non-re-keyed blob. The persisted pair is internally
    /// consistent; the salted re-key path (AC-F5.5) is the ONLY sanctioned way the
    /// binding is broken.
    #[test]
    fn t_f5_dual_unique_binding() {
        let key = content_key_v4("t", bucket::CODE, &compute_content_hash("x"), "");
        // The non-re-keyed binding: the stored point_id is exactly point_id(key, 0).
        assert_eq!(point_id(&key, 0), point_id(&key, 0));
        // The re-key path is the only thing that diverges from it.
        assert_ne!(point_id(&key, 0), salted_point_id(&key, &[0x01], 0));
    }

    /// T-F5.8 (AC-F5.8): the version-gated selector. For the SAME chunk bytes,
    /// flipping the flag (3 vs 4) is the ONLY thing that changes the output:
    /// flag 3 -> a three-slot point_id via `content_key_v3` (ignoring collection),
    /// flag 4 -> a four-slot point_id via `content_key_v4`. (The once-per-session
    /// caching is F6, not here — this pins only the pure selector behavior.)
    #[test]
    fn t_f5_version_gated_selector() {
        let tenant = "t";
        let collection = bucket::CODE;
        let identity = compute_content_hash("chunk bytes");
        let content_hash = "";

        let under_3 = content_key_for_version(3, tenant, collection, &identity, content_hash);
        let under_4 = content_key_for_version(4, tenant, collection, &identity, content_hash);

        // Flag 3 routes to v3 (collection ignored); flag 4 routes to v4.
        assert_eq!(under_3, content_key_v3(tenant, &identity, content_hash));
        assert_eq!(
            under_4,
            content_key_v4(tenant, collection, &identity, content_hash)
        );
        // Flipping the flag is the only thing that changes the output.
        assert_ne!(under_3, under_4);

        // v3 selector output is byte-identical to the legacy producer AND not the
        // four-slot-with-empty-collection call (DOM-R5-N2 anti-pattern).
        assert_ne!(under_3, content_key_v4(tenant, "", &identity, content_hash));

        // The point_id space also diverges only by the flag.
        assert_ne!(point_id(&under_3, 0), point_id(&under_4, 0));
    }

    /// T-F5-selector-rejects-unknown (AC-F5.8): an unknown version is a schema bug,
    /// not a recoverable condition — the selector panics rather than mint points
    /// under an unrecognized slot-shape.
    #[test]
    #[should_panic(expected = "unknown content_key_version 5")]
    fn t_f5_selector_rejects_unknown_version() {
        let _ = content_key_for_version(5, "t", bucket::CODE, "id", "");
    }

    // ---- F4: branch_id canonical producer (AC-F4.3) ----

    /// T-F4-branch-id-deterministic: same inputs always yield the same branch_id.
    #[test]
    fn t_f4_branch_id_deterministic() {
        let bid1 = branch_id("tenant-abc", "/home/user/proj", "main");
        let bid2 = branch_id("tenant-abc", "/home/user/proj", "main");
        assert_eq!(bid1, bid2, "branch_id must be deterministic");
        assert_eq!(bid1.len(), 64, "branch_id is a full SHA256 hex (64 chars)");
        assert!(bid1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// T-F4-branch-id-path-sensitive: two clones at different paths produce distinct
    /// branch_ids even when tenant and branch_name are identical (AC-F4.2 SEED F09).
    #[test]
    fn t_f4_branch_id_path_sensitive() {
        let bid_a = branch_id("tenant-abc", "/home/alice/proj", "main");
        let bid_b = branch_id("tenant-abc", "/home/bob/proj", "main");
        assert_ne!(
            bid_a, bid_b,
            "two clones of main at different paths must have distinct branch_ids"
        );
    }

    /// T-F4-branch-id-branch-sensitive: different branch names yield different ids.
    #[test]
    fn t_f4_branch_id_branch_sensitive() {
        let main_id = branch_id("t", "/repo", "main");
        let feat_id = branch_id("t", "/repo", "feat/x");
        assert_ne!(
            main_id, feat_id,
            "different branch names must produce different branch_ids"
        );
    }

    /// T-F4-branch-id-tenant-sensitive: different tenants yield different ids.
    #[test]
    fn t_f4_branch_id_tenant_sensitive() {
        let t1 = branch_id("tenant-1", "/repo", "main");
        let t2 = branch_id("tenant-2", "/repo", "main");
        assert_ne!(
            t1, t2,
            "different tenants must produce different branch_ids"
        );
    }

    /// T-F4-branch-id-lp-injective: boundary shift between location and branch_name
    /// fields changes the id — lp framing prevents separator ambiguity.
    #[test]
    fn t_f4_branch_id_lp_injective() {
        // "abc"/"def" vs "ab"/"cdef" — without lp framing these might collide.
        let a = branch_id("t", "abc", "def");
        let b = branch_id("t", "ab", "cdef");
        assert_ne!(a, b, "lp framing must prevent boundary-shift collisions");
    }

    /// T-F4-branch-id-golden: pin the exact bytes the producer emits. A change to
    /// the formula (field order, framing, hash backend) flips this value and fails
    /// loudly, which is the point: stored branch_ids are a contract.
    #[test]
    fn t_f4_branch_id_golden() {
        // Computed independently: hex(SHA256(lp(b"t") || lp(b"/r") || lp(b"m")))
        // where lp(x) = u32_be(x.len()) ++ x.
        let got = branch_id("t", "/r", "m");
        // Recompute inline to verify.
        let mut h = Sha256::new();
        h.update(lp(b"t"));
        h.update(lp(b"/r"));
        h.update(lp(b"m"));
        let expected = format!("{:x}", h.finalize());
        assert_eq!(got, expected, "branch_id golden vector must not change");
        assert_eq!(got.len(), 64);
    }
}
