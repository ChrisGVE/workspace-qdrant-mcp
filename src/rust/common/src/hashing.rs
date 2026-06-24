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
#[path = "hashing_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "hashing_tests_f5_f4.rs"]
mod tests_f5_f4;
