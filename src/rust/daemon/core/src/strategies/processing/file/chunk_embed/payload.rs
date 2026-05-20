//! Qdrant payload construction for individual chunks.
//!
//! Builds the full payload map for each chunk, including file metadata,
//! language/extension tags, chunk-level metadata forwarding, and test-file
//! detection for tag filtering.

use std::collections::HashMap;
use std::path::Path;

use crate::embedding::SparseEmbedding;
use crate::file_classification::is_test_file;
use crate::tagging::tier1;
use crate::unified_queue_schema::UnifiedQueueItem;
use crate::DocumentContent;
use wqm_common::constants::field;

/// Build the Qdrant payload map for a single chunk.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_chunk_payload(
    content: &str,
    chunk_index: usize,
    item: &UnifiedQueueItem,
    document_content: &DocumentContent,
    file_path: &Path,
    file_document_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_type: Option<&str>,
    chunk_metadata: &HashMap<String, String>,
) -> HashMap<String, serde_json::Value> {
    let file_path_str = file_path.to_string_lossy();
    let mut payload = HashMap::new();

    payload.insert("content".to_string(), serde_json::json!(content));
    payload.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
    payload.insert("file_path".to_string(), serde_json::json!(file_path_str));
    payload.insert(
        "document_id".to_string(),
        serde_json::json!(file_document_id),
    );
    payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
    payload.insert("branch".to_string(), serde_json::json!(item.branch));
    payload.insert("base_point".to_string(), serde_json::json!(base_point));
    payload.insert(
        "relative_path".to_string(),
        serde_json::json!(relative_path),
    );
    payload.insert(
        "absolute_path".to_string(),
        serde_json::json!(file_path_str),
    );
    payload.insert("file_hash".to_string(), serde_json::json!(file_hash));
    payload.insert(
        "document_type".to_string(),
        serde_json::json!(document_content.document_type.as_str()),
    );
    if let Some(lang) = document_content.document_type.language() {
        payload.insert("language".to_string(), serde_json::json!(lang));
    }
    // Add file extension as metadata (e.g., "rs", "py", "md") -- lowercase for consistency
    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        payload.insert(
            "file_extension".to_string(),
            serde_json::json!(ext.to_lowercase()),
        );
    }
    payload.insert("item_type".to_string(), serde_json::json!("file"));

    if let Some(ft) = file_type {
        payload.insert(
            "file_type".to_string(),
            serde_json::json!(ft.to_lowercase()),
        );
    }

    // Build tags array: static metadata + Tier 1 rule-based tags
    build_tags_array(&mut payload, file_type, document_content, file_path);

    // Add chunk-level metadata (symbol_name, start_line, etc.)
    for (key, value) in chunk_metadata {
        payload.insert(format!("chunk_{}", key), serde_json::json!(value));
    }

    payload
}

/// Build the `tags` payload array combining static metadata tags and
/// Tier 1 rule-based tags (path-derived, PDF metadata, dependency concepts).
fn build_tags_array(
    payload: &mut HashMap<String, serde_json::Value>,
    file_type: Option<&str>,
    document_content: &DocumentContent,
    file_path: &Path,
) {
    let mut tags = Vec::new();

    // Static metadata tags
    if let Some(ft) = file_type {
        tags.push(ft.to_lowercase());
    }
    if let Some(lang) = document_content.document_type.language() {
        tags.push(lang.to_string());
    }
    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        tags.push(ext.to_lowercase());
    }
    if is_test_file(file_path) {
        tags.push("test".to_string());
    }

    // Tier 1 rule-based tags (path-derived, PDF metadata, dependency concepts)
    let tier1_tags = tier1::extract_tier1_tags(file_path, None);
    for selected_tag in &tier1_tags {
        tags.push(selected_tag.phrase.clone());
    }

    // Deduplicate while preserving insertion order
    let mut seen = std::collections::HashSet::new();
    tags.retain(|t| seen.insert(t.clone()));

    if !tags.is_empty() {
        payload.insert(field::TAGS.to_string(), serde_json::json!(tags));
    }
}

/// Convert a `SparseEmbedding` to the `HashMap` format expected by `DocumentPoint`.
pub(super) fn sparse_embedding_to_map(sparse: &SparseEmbedding) -> Option<HashMap<u32, f32>> {
    crate::shared::embedding_pipeline::sparse_embedding_to_map(sparse)
}
