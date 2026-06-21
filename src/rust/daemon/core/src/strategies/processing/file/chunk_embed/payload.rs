//! Qdrant payload construction for individual chunks.
//!
//! Builds the full payload map for each chunk, including file metadata,
//! language/extension tags, chunk-level metadata forwarding, test-file
//! detection for tag filtering, and library hierarchy fields when
//! processing library collection items.

use std::collections::HashMap;
use std::path::Path;

use crate::embedding::SparseEmbedding;
use crate::file_classification::is_test_file;
use crate::library_hierarchy;
use crate::tagging::tier1;
use crate::unified_queue_schema::UnifiedQueueItem;
use crate::DocumentContent;
use wqm_common::constants::field;

/// Optional library context for injecting hierarchy metadata into payloads.
///
/// When a file is ingested into the `libraries` collection, this struct
/// carries the library name (tenant) and root directory so that
/// `library_name` and `library_path` fields are populated on every
/// Qdrant point.
#[derive(Debug, Clone)]
pub(crate) struct LibraryContext<'a> {
    /// Library name (also the tenant_id for libraries collection).
    pub library_name: &'a str,
    /// Absolute path to the library root directory.
    pub library_root: &'a Path,
}

/// Build the Qdrant payload map for a single chunk.
///
/// `branch` is the detected current branch (from `BranchCache`), which may
/// differ from `item.branch` when a branch switch happened after the queue
/// item was enqueued.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_chunk_payload(
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
    library_ctx: Option<&LibraryContext<'_>>,
    branch: &str,
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
    payload.insert("branches".to_string(), serde_json::json!([branch]));
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

    // Inject library hierarchy metadata when processing library collection items.
    if let Some(lib_ctx) = library_ctx {
        inject_library_hierarchy(&mut payload, lib_ctx, file_path);
    }

    insert_provenance_fields(&mut payload, item);

    // Build tags array: static metadata + Tier 1 rule-based tags
    build_tags_array(&mut payload, file_type, document_content, file_path);

    // Add chunk-level metadata (symbol_name, start_line, etc.)
    for (key, value) in chunk_metadata {
        payload.insert(format!("chunk_{}", key), serde_json::json!(value));
    }

    payload
}

/// Inject `library_name` and `library_path` fields into a Qdrant payload.
///
/// Uses `library_hierarchy::extract_library_path` to derive the relative
/// path within the library from the absolute file path and library root.
fn inject_library_hierarchy(
    payload: &mut HashMap<String, serde_json::Value>,
    lib_ctx: &LibraryContext<'_>,
    file_path: &Path,
) {
    payload.insert(
        field::LIBRARY_NAME.to_string(),
        serde_json::json!(lib_ctx.library_name),
    );

    match library_hierarchy::extract_library_path(lib_ctx.library_root, file_path) {
        Some((lib_path, document_name)) => {
            payload.insert(field::LIBRARY_PATH.to_string(), serde_json::json!(lib_path));
            payload.insert(
                "document_name".to_string(),
                serde_json::json!(document_name),
            );
        }
        None => {
            payload.insert(field::LIBRARY_PATH.to_string(), serde_json::json!(""));
            let doc_name = file_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            payload.insert("document_name".to_string(), serde_json::json!(doc_name));
        }
    }
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

fn insert_provenance_fields(
    payload: &mut HashMap<String, serde_json::Value>,
    item: &UnifiedQueueItem,
) {
    payload.insert(
        field::SOURCE_COLLECTION.to_string(),
        serde_json::json!(item.collection),
    );
    let source_type = if item.collection == wqm_common::constants::COLLECTION_LIBRARIES {
        "library_doc"
    } else {
        "code_file"
    };
    payload.insert(
        field::SOURCE_TYPE.to_string(),
        serde_json::json!(source_type),
    );
    if let Some(reason) = extract_routing_reason(item) {
        payload.insert(field::ROUTING_REASON.to_string(), serde_json::json!(reason));
    }
}

fn extract_routing_reason(item: &UnifiedQueueItem) -> Option<String> {
    item.metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .and_then(|v| v.get("routing_reason")?.as_str().map(String::from))
}

/// Convert a `SparseEmbedding` to the `HashMap` format expected by `DocumentPoint`.
pub(super) fn sparse_embedding_to_map(sparse: &SparseEmbedding) -> Option<HashMap<u32, f32>> {
    crate::shared::embedding_pipeline::sparse_embedding_to_map(sparse)
}
