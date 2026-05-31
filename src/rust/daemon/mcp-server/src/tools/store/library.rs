//! Library/document sub-handler for the `store` tool.

use rmcp::model::CallToolResult;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::canonicalize::payload_builders::build_store_payload;

use super::{StoreDaemon, StoreInput, StoreLibraryResult};
use crate::tools::envelope::ok_text;

/// Generate document ID using SHA256 hash for idempotency.
///
/// Mirrors `generateDocumentId` in store.ts:207-213:
/// `sha256(tenant_id + content).hex[:32]`
pub fn generate_document_id(content: &str, tenant_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tenant_id.as_bytes());
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    // First 32 hex chars — store.ts:212
    format!("{result:x}")[..32].to_string()
}

/// Resolve tenant ID and library label for library store.
///
/// Mirrors `resolveTenant` in store.ts:158-188.
pub fn resolve_library_tenant(
    for_project: bool,
    project_id: Option<&str>,
    library_name: Option<&str>,
) -> Result<(String, String), String> {
    if for_project {
        let pid = project_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                "No active project detected. forProject requires an active project session."
                    .to_string()
            })?;
        let label = library_name
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("project-refs")
            .to_string();
        return Ok((pid.to_string(), label));
    }
    let lib = library_name
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "libraryName is required - this tool stores to the libraries collection only. \
             For project content, use file watching (daemon handles this automatically)."
                .to_string()
        })?;
    Ok((lib.to_string(), lib.to_string()))
}

/// Build the store metadata map — mirrors `buildStoreMetadata` in store.ts:190-203.
///
/// Returns a `serde_json::Map` matching the `metadata` field of the payload.
fn build_store_metadata(
    base: &Map<String, Value>,
    source_type: &str,
    title: Option<&str>,
    url: Option<&str>,
    file_path: Option<&str>,
) -> Map<String, Value> {
    let mut full = base.clone();
    full.insert(
        "source_type".to_string(),
        Value::String(source_type.to_string()),
    );
    if let Some(t) = title {
        if !t.is_empty() {
            full.insert("title".to_string(), Value::String(t.to_string()));
        }
    }
    if let Some(u) = url {
        if !u.is_empty() {
            full.insert("url".to_string(), Value::String(u.to_string()));
        }
    }
    if let Some(fp) = file_path {
        if !fp.is_empty() {
            full.insert("file_path".to_string(), Value::String(fp.to_string()));
        }
    }
    full
}

/// Handle store type=library (default) — mirrors `StoreTool.store` in store.ts:119-156.
pub(super) async fn store_library<D>(input: StoreInput, daemon: &mut D) -> CallToolResult
where
    D: StoreDaemon,
{
    use wqm_common::constants::COLLECTION_LIBRARIES;

    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            return ok_text(&StoreLibraryResult {
                success: false,
                document_id: None,
                collection: COLLECTION_LIBRARIES.to_string(),
                message: "Content is required for storing".to_string(),
                fallback_mode: "unified_queue".to_string(),
                queue_id: None,
            });
        }
    };

    let tenant_result = resolve_library_tenant(
        input.for_project,
        input.project_id.as_deref(),
        input.library_name.as_deref(),
    );
    let (tenant_id, library_label) = match tenant_result {
        Err(e) => {
            return ok_text(&StoreLibraryResult {
                success: false,
                document_id: None,
                collection: COLLECTION_LIBRARIES.to_string(),
                message: e,
                fallback_mode: "unified_queue".to_string(),
                queue_id: None,
            });
        }
        Ok(t) => t,
    };

    let document_id = generate_document_id(content, &tenant_id);
    let full_metadata = build_store_metadata(
        &input.metadata,
        &input.source_type,
        input.title.as_deref(),
        input.url.as_deref(),
        input.file_path.as_deref(),
    );

    let payload_json = build_store_payload(
        content,
        &document_id,
        &input.source_type,
        &full_metadata,
        &library_label,
    );

    match daemon
        .enqueue_item(
            "tenant",
            "add",
            &tenant_id,
            COLLECTION_LIBRARIES,
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_store_tool\"}"),
        )
        .await
    {
        Err(e) => ok_text(&StoreLibraryResult {
            success: false,
            document_id: None,
            collection: COLLECTION_LIBRARIES.to_string(),
            message: format!("Failed to queue content: {e}"),
            fallback_mode: "unified_queue".to_string(),
            queue_id: None,
        }),
        Ok(queue_id) => ok_text(&StoreLibraryResult {
            success: true,
            document_id: Some(document_id),
            collection: COLLECTION_LIBRARIES.to_string(),
            message: format!("Content queued for processing by daemon (libraries/{tenant_id})"),
            fallback_mode: "unified_queue".to_string(),
            queue_id: Some(queue_id),
        }),
    }
}
