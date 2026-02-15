//! Node.js native addon exposing wqm-common functions via napi-rs
//!
//! Provides single-source-of-truth implementations of:
//! - Project ID calculation and git URL normalization
//! - Idempotency key and content hashing
//! - NLP tokenization
//! - Constants (collection names, URLs, ports)
//! - Queue type and operation constants and validation

use napi_derive::napi;

// ============================================================================
// Project ID
// ============================================================================

/// Calculate a unique project ID from a project root path and optional git remote URL.
///
/// Uses SHA256 hashing on the normalized git URL (or path for local projects).
/// Returns a 12-character hex string (or "local_" + 12 chars for non-git projects).
#[napi]
pub fn calculate_project_id(
    project_root: String,
    git_remote: Option<String>,
) -> String {
    let calc = wqm_common::project_id::ProjectIdCalculator::new();
    calc.calculate(
        std::path::Path::new(&project_root),
        git_remote.as_deref(),
        None,
    )
}

/// Calculate a unique project ID with disambiguation path.
///
/// Used when multiple clones of the same repo exist on the same machine.
#[napi]
pub fn calculate_project_id_with_disambiguation(
    project_root: String,
    git_remote: Option<String>,
    disambiguation_path: Option<String>,
) -> String {
    let calc = wqm_common::project_id::ProjectIdCalculator::new();
    calc.calculate(
        std::path::Path::new(&project_root),
        git_remote.as_deref(),
        disambiguation_path.as_deref(),
    )
}

/// Normalize a git URL to a canonical form.
///
/// All of these normalize to "github.com/user/repo":
/// - `https://github.com/user/repo.git`
/// - `git@github.com:user/repo.git`
/// - `ssh://git@github.com/user/repo`
/// - `http://github.com/user/repo`
#[napi]
pub fn normalize_git_url(url: String) -> String {
    wqm_common::project_id::ProjectIdCalculator::normalize_git_url(&url)
}

/// Detect the git remote URL for a project path.
///
/// Tries `origin` first, falls back to `upstream`. Returns null if not in a git repo.
#[napi]
pub fn detect_git_remote(project_root: String) -> Option<String> {
    wqm_common::project_id::detect_git_remote(std::path::Path::new(&project_root))
}

/// Convenience: calculate tenant ID for a project path.
///
/// Combines detect_git_remote() + calculate_project_id().
#[napi]
pub fn calculate_tenant_id(project_root: String) -> String {
    wqm_common::project_id::calculate_tenant_id(std::path::Path::new(&project_root))
}

// ============================================================================
// Hashing
// ============================================================================

/// Generate an idempotency key for queue deduplication.
///
/// Format: SHA256("{item_type}|{op}|{tenant_id}|{collection}|{payload_json}")[:32]
///
/// Returns null if validation fails (empty fields, invalid operation for type).
#[napi]
pub fn generate_idempotency_key(
    item_type: String,
    op: String,
    tenant_id: String,
    collection: String,
    payload_json: String,
) -> Option<String> {
    let it = wqm_common::queue_types::ItemType::from_str(&item_type)?;
    let qop = wqm_common::queue_types::QueueOperation::from_str(&op)?;
    wqm_common::hashing::generate_idempotency_key(it, qop, &tenant_id, &collection, &payload_json)
        .ok()
}

/// Compute SHA256 hash of a string content.
///
/// Returns the full 64-character hex hash.
#[napi]
pub fn compute_content_hash(content: String) -> String {
    wqm_common::hashing::compute_content_hash(&content)
}

// ============================================================================
// NLP
// ============================================================================

/// Tokenize text for BM25 sparse vector generation.
///
/// Splits on whitespace/punctuation, lowercases, removes stopwords and single-char tokens.
#[napi]
pub fn tokenize(text: String) -> Vec<String> {
    wqm_common::nlp::tokenize(&text)
}

// ============================================================================
// Constants — Collections
// ============================================================================

/// Get the canonical projects collection name ("projects")
#[napi]
pub fn collection_projects() -> String {
    wqm_common::constants::COLLECTION_PROJECTS.to_string()
}

/// Get the canonical libraries collection name ("libraries")
#[napi]
pub fn collection_libraries() -> String {
    wqm_common::constants::COLLECTION_LIBRARIES.to_string()
}

/// Get the canonical memory collection name ("memory")
#[napi]
pub fn collection_memory() -> String {
    wqm_common::constants::COLLECTION_MEMORY.to_string()
}

/// Get the canonical scratchpad collection name ("scratchpad")
#[napi]
pub fn collection_scratchpad() -> String {
    wqm_common::constants::COLLECTION_SCRATCHPAD.to_string()
}

/// Get the default Qdrant URL ("http://localhost:6333")
#[napi]
pub fn default_qdrant_url() -> String {
    wqm_common::constants::DEFAULT_QDRANT_URL.to_string()
}

/// Get the default gRPC port (50051)
#[napi]
pub fn default_grpc_port() -> u32 {
    wqm_common::constants::DEFAULT_GRPC_PORT as u32
}

/// Get the default Git branch name ("main")
#[napi]
pub fn default_branch() -> String {
    wqm_common::constants::DEFAULT_BRANCH.to_string()
}

/// Get the HIGH queue priority value (1 - processed first)
#[napi]
pub fn priority_high() -> i32 {
    wqm_common::constants::priority::HIGH
}

/// Get the NORMAL queue priority value (3 - default for registered projects)
#[napi]
pub fn priority_normal() -> i32 {
    wqm_common::constants::priority::NORMAL
}

/// Get the LOW queue priority value (5 - background processing)
#[napi]
pub fn priority_low() -> i32 {
    wqm_common::constants::priority::LOW
}

// ============================================================================
// Constants — Item types
// ============================================================================

/// Get the "text" item type constant
#[napi]
pub fn item_type_text() -> String {
    wqm_common::constants::item_type::TEXT.to_string()
}

/// Get the "file" item type constant
#[napi]
pub fn item_type_file() -> String {
    wqm_common::constants::item_type::FILE.to_string()
}

/// Get the "url" item type constant
#[napi]
pub fn item_type_url() -> String {
    wqm_common::constants::item_type::URL.to_string()
}

/// Get the "website" item type constant
#[napi]
pub fn item_type_website() -> String {
    wqm_common::constants::item_type::WEBSITE.to_string()
}

/// Get the "doc" item type constant
#[napi]
pub fn item_type_doc() -> String {
    wqm_common::constants::item_type::DOC.to_string()
}

/// Get the "folder" item type constant
#[napi]
pub fn item_type_folder() -> String {
    wqm_common::constants::item_type::FOLDER.to_string()
}

/// Get the "tenant" item type constant
#[napi]
pub fn item_type_tenant() -> String {
    wqm_common::constants::item_type::TENANT.to_string()
}

/// Get the "collection" item type constant
#[napi]
pub fn item_type_collection() -> String {
    wqm_common::constants::item_type::COLLECTION.to_string()
}

/// Get all valid item type strings
#[napi]
pub fn all_item_types() -> Vec<String> {
    wqm_common::queue_types::ItemType::all()
        .iter()
        .map(|it| it.as_str().to_string())
        .collect()
}

// ============================================================================
// Constants — Operations
// ============================================================================

/// Get the "add" operation constant
#[napi]
pub fn operation_add() -> String {
    wqm_common::constants::operation::ADD.to_string()
}

/// Get the "update" operation constant
#[napi]
pub fn operation_update() -> String {
    wqm_common::constants::operation::UPDATE.to_string()
}

/// Get the "delete" operation constant
#[napi]
pub fn operation_delete() -> String {
    wqm_common::constants::operation::DELETE.to_string()
}

/// Get the "scan" operation constant
#[napi]
pub fn operation_scan() -> String {
    wqm_common::constants::operation::SCAN.to_string()
}

/// Get the "rename" operation constant
#[napi]
pub fn operation_rename() -> String {
    wqm_common::constants::operation::RENAME.to_string()
}

/// Get the "uplift" operation constant
#[napi]
pub fn operation_uplift() -> String {
    wqm_common::constants::operation::UPLIFT.to_string()
}

/// Get the "reset" operation constant
#[napi]
pub fn operation_reset() -> String {
    wqm_common::constants::operation::RESET.to_string()
}

/// Get all valid operation strings
#[napi]
pub fn all_operations() -> Vec<String> {
    wqm_common::queue_types::QueueOperation::all()
        .iter()
        .map(|op| op.as_str().to_string())
        .collect()
}

// ============================================================================
// Queue type validation
// ============================================================================

/// Check if a string is a valid ItemType
#[napi]
pub fn is_valid_item_type(s: String) -> bool {
    wqm_common::queue_types::ItemType::from_str(&s).is_some()
}

/// Check if a string is a valid QueueOperation
#[napi]
pub fn is_valid_queue_operation(s: String) -> bool {
    wqm_common::queue_types::QueueOperation::from_str(&s).is_some()
}

/// Check if a string is a valid QueueStatus
#[napi]
pub fn is_valid_queue_status(s: String) -> bool {
    wqm_common::queue_types::QueueStatus::from_str(&s).is_some()
}

/// Check if an operation is valid for a given item type
#[napi]
pub fn is_valid_operation_for_type(item_type: String, op: String) -> bool {
    let it = match wqm_common::queue_types::ItemType::from_str(&item_type) {
        Some(v) => v,
        None => return false,
    };
    let qop = match wqm_common::queue_types::QueueOperation::from_str(&op) {
        Some(v) => v,
        None => return false,
    };
    qop.is_valid_for(it)
}
