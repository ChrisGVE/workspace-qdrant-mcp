//! `store` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/store.ts`,
//! `src/typescript/mcp-server/src/store-handlers.ts`, and
//! `src/typescript/mcp-server/src/tool-dispatcher.ts` (`dispatchStore`).
//!
//! # Dispatch table (tool-dispatcher.ts:37-43)
//!
//! | `type` arg    | Rust path                     | TS path                 |
//! |---------------|-------------------------------|-------------------------|
//! | `"project"`   | `store_project`               | `registerProjectFromTool` |
//! | `"url"`       | `store_url`                   | `storeUrl`              |
//! | `"scratchpad"`| `store_scratchpad`            | `storeScratchpad`       |
//! | *(default)*   | `store_library` (library/doc) | `storeTool.store`       |
//!
//! # Result shapes (per TS field order)
//!
//! **project**: `{ success, project_id, created, is_active, message }`
//! (session-lifecycle.ts:207-213)
//!
//! **url**: `{ success, message, queue_id?, collection }`
//! (store-handlers.ts:108-113)
//!
//! **scratchpad**: `{ success, message, queue_id?, collection }`
//! (store-handlers.ts:209-215)
//!
//! **library**: `{ success, documentId?, collection, message, fallback_mode, queue_id? }`
//! (store.ts:101-116)
//!
//! # Output-parity rules
//!
//! - `queue_id` is adjacent to `message` in url/scratchpad; serialized only when
//!   `Some` (`#[serde(skip_serializing_if)]`).
//! - `documentId` (camelCase) in library result; `queue_id` (snake_case) in url/scratchpad
//!   and library — per TS field names.
//! - `fallback_mode` is always `"unified_queue"` in library result.

use rmcp::model::CallToolResult;
use serde::Serialize;
use serde_json::{Map, Value};

use wqm_common::constants::{COLLECTION_SCRATCHPAD, TENANT_GLOBAL};
use wqm_common::timestamps::now_utc;

use crate::canonicalize::payload_builders::build_store_payload;
use crate::tools::envelope::{error_text, ok_text};

// ─────────────────────────────────────────────────────────────────────────────
// Public injectable trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over daemon I/O needed by the store tool.
///
/// This is injected by tests to avoid live gRPC/SQLite dependencies.
pub trait StoreDaemon {
    /// Register a project via gRPC.
    ///
    /// Returns `(project_id, newly_registered, is_active)`.
    fn register_project(
        &mut self,
        path: &str,
        name: &str,
    ) -> impl std::future::Future<Output = Result<ProjectRegisterResult, String>> + Send;

    /// Enqueue an item into the unified queue.
    ///
    /// Returns the `queue_id` assigned by the daemon.
    fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> impl std::future::Future<Output = Result<String, String>> + Send;

    /// Upsert scratchpad mirror entry — fire-and-forget.
    ///
    /// Mirrors `writeScratchpadMirror` in store-handlers.ts:124-140.
    fn upsert_scratchpad_mirror(
        &mut self,
        scratchpad_id: String,
        content: String,
        title: Option<String>,
        tags: Option<String>,
        tenant_id: String,
        created_at: String,
        updated_at: String,
    ) -> impl std::future::Future<Output = ()> + Send;
}

/// Result from project registration.
#[derive(Debug, Clone)]
pub struct ProjectRegisterResult {
    pub project_id: String,
    pub newly_registered: bool,
    pub is_active: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: DaemonClient → StoreDaemon
// ─────────────────────────────────────────────────────────────────────────────

impl StoreDaemon for crate::grpc::DaemonClient {
    async fn register_project(
        &mut self,
        path: &str,
        name: &str,
    ) -> Result<ProjectRegisterResult, String> {
        use crate::proto::RegisterProjectRequest;
        let req = RegisterProjectRequest {
            path: path.to_string(),
            project_id: String::new(),
            name: Some(name.to_string()),
            git_remote: None,
            register_if_new: true,
            priority: Some("high".to_string()),
        };
        let resp = crate::grpc::DaemonClient::register_project(self, req)
            .await
            .map_err(|e| e.to_string())?;
        Ok(ProjectRegisterResult {
            project_id: resp.project_id,
            newly_registered: resp.newly_registered,
            is_active: resp.is_active,
        })
    }

    async fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> Result<String, String> {
        let resp = crate::grpc::DaemonClient::enqueue_item(
            self,
            item_type.to_string(),
            op.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.to_string(),
            branch.to_string(),
            metadata_json.map(str::to_string),
        )
        .await
        .map_err(|e| e.to_string())?;
        Ok(resp.queue_id)
    }

    async fn upsert_scratchpad_mirror(
        &mut self,
        scratchpad_id: String,
        content: String,
        title: Option<String>,
        tags: Option<String>,
        tenant_id: String,
        created_at: String,
        updated_at: String,
    ) {
        let _ = crate::grpc::DaemonClient::upsert_scratchpad_mirror(
            self,
            scratchpad_id,
            content,
            title,
            tags,
            tenant_id,
            created_at,
            updated_at,
        )
        .await;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed input for the store tool.
#[derive(Debug)]
pub struct StoreInput {
    /// `"project"` | `"url"` | `"scratchpad"` | `"library"` (default)
    pub store_type: String,
    pub content: Option<String>,
    pub library_name: Option<String>,
    pub for_project: bool,
    pub project_id: Option<String>,
    pub title: Option<String>,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub source_type: String,
    pub metadata: Map<String, Value>,
    pub tags: Vec<String>,
    /// Only used for store type=project
    pub path: Option<String>,
    /// Only used for store type=project
    pub name: Option<String>,
}

impl StoreInput {
    /// Parse from the JSON arguments map.
    ///
    /// Mirrors `dispatchStore` in tool-dispatcher.ts:37-43 and
    /// `buildStoreOptions` in tool-builders/store.ts.
    pub fn from_args(
        args: &serde_json::Map<String, serde_json::Value>,
        session_project_id: Option<&str>,
    ) -> Self {
        // type defaults to "library" — tool-dispatcher.ts:37
        let store_type = args
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("library")
            .to_string();

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let library_name = args
            .get("libraryName")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let for_project = args
            .get("forProject")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // When forProject=true, use session project_id
        let project_id = if for_project {
            session_project_id.map(str::to_string)
        } else {
            args.get("projectId")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };

        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let url = args.get("url").and_then(|v| v.as_str()).map(str::to_string);

        let file_path = args
            .get("filePath")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // sourceType default "user_input" — store.ts:129
        let source_type = args
            .get("sourceType")
            .and_then(|v| v.as_str())
            .filter(|s| matches!(*s, "user_input" | "web" | "file" | "scratchbook" | "note"))
            .unwrap_or("user_input")
            .to_string();

        let metadata = args
            .get("metadata")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        Self {
            store_type,
            content,
            library_name,
            for_project,
            project_id,
            title,
            url,
            file_path,
            source_type,
            metadata,
            tags,
            path,
            name,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types — field order MUST match TS declaration order
// ─────────────────────────────────────────────────────────────────────────────

/// Result for store type=project (session-lifecycle.ts:207-213).
///
/// Field order: success → project_id → created → is_active → message
#[derive(Debug, Serialize)]
pub struct StoreProjectResult {
    pub success: bool,
    pub project_id: String,
    pub created: bool,
    pub is_active: bool,
    pub message: String,
}

/// Result for store type=url and type=scratchpad (store-handlers.ts:13-17).
///
/// Field order: success → message → queue_id? → collection
#[derive(Debug, Serialize)]
pub struct StoreUrlScratchpadResult {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_id: Option<String>,
    pub collection: String,
}

/// Result for store type=library/doc (store.ts:48-55).
///
/// Field order: success → documentId? → collection → message → fallback_mode → queue_id?
#[derive(Debug, Serialize)]
pub struct StoreLibraryResult {
    pub success: bool,
    #[serde(rename = "documentId", skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    pub collection: String,
    pub message: String,
    pub fallback_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_id: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Execute the `store` tool.
///
/// Dispatches by `type` arg. Errors from sub-handlers propagate as
/// `error_text` (TS re-throws from `dispatchStore` — tool-dispatcher.ts:106-109).
pub async fn store_tool<D>(
    input: StoreInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
    daemon_connected: bool,
) -> CallToolResult
where
    D: StoreDaemon,
{
    match input.store_type.as_str() {
        "project" => store_project(input, daemon, session_project_id, daemon_connected).await,
        "url" => store_url(input, daemon, session_project_id).await,
        "scratchpad" => store_scratchpad(input, daemon, session_project_id).await,
        _ => store_library(input, daemon).await,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-handlers
// ─────────────────────────────────────────────────────────────────────────────

/// Handle store type=project — mirrors `registerProjectFromTool` in
/// session-lifecycle.ts:182-213.
///
/// Throws (returns error_text) if `path` is missing or daemon is not connected,
/// matching the TS throw semantics (tool-dispatcher.ts:106-109).
async fn store_project<D>(
    input: StoreInput,
    daemon: &mut D,
    _session_project_id: Option<&str>,
    daemon_connected: bool,
) -> CallToolResult
where
    D: StoreDaemon,
{
    let path = match input.path.as_deref() {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => return error_text("path is required for store type \"project\""),
    };

    if !daemon_connected {
        return error_text("Daemon is not connected — cannot register project");
    }

    // name defaults to last segment of path — session-lifecycle.ts:200
    let name = input
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| path.rsplit('/').next().unwrap_or("unknown"))
        .to_string();

    match daemon.register_project(&path, &name).await {
        Err(e) => error_text(&e),
        Ok(resp) => {
            let message = if resp.newly_registered {
                format!("Project registered and activated: {path}")
            } else {
                format!("Project already registered and activated: {path}")
            };
            ok_text(&StoreProjectResult {
                success: true,
                project_id: resp.project_id,
                created: resp.newly_registered,
                is_active: resp.is_active,
                message,
            })
        }
    }
}

/// Validate a URL string — mirrors `validateUrlInput` in store-handlers.ts:27-52.
fn validate_url(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("url is required when type is \"url\"".to_string());
    }
    // Basic scheme check — we parse protocol manually (no URL crate dep needed)
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err(format!(
            "url must use http:// or https:// (got {})",
            if let Some(idx) = trimmed.find("://") {
                &trimmed[..idx + 3]
            } else {
                trimmed
            }
        ));
    }
    // Extract hostname from the URL
    let after_scheme = if let Some(rest) = trimmed.strip_prefix("https://") {
        rest
    } else {
        trimmed.strip_prefix("http://").unwrap_or(trimmed)
    };
    let host = after_scheme.split('/').next().unwrap_or("");
    // Remove port if present
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        return Err("url has empty hostname".to_string());
    }
    if host.chars().all(|c| c == '.' || c == ' ') {
        return Err("url has invalid hostname (dots/whitespace only)".to_string());
    }
    Ok(())
}

/// Build the URL queue payload — mirrors `buildUrlPayload` in store-handlers.ts:55-68.
fn build_url_payload(url: &str, library_name: Option<&str>, title: Option<&str>) -> String {
    let mut map = Map::new();
    map.insert("url".to_string(), Value::String(url.trim().to_string()));
    map.insert("crawl".to_string(), Value::Bool(false));
    map.insert("max_depth".to_string(), Value::Number(0.into()));
    map.insert("max_pages".to_string(), Value::Number(1.into()));
    if let Some(lib) = library_name {
        map.insert(
            "library_name".to_string(),
            Value::String(lib.trim().to_string()),
        );
    }
    if let Some(t) = title {
        map.insert("title".to_string(), Value::String(t.to_string()));
    }
    serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
}

/// Handle store type=url — mirrors `storeUrl` in store-handlers.ts:78-121.
async fn store_url<D>(
    input: StoreInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: StoreDaemon,
{
    let url = input.url.as_deref().unwrap_or("").to_string();
    if let Err(e) = validate_url(&url) {
        let result = StoreUrlScratchpadResult {
            success: false,
            message: e,
            queue_id: None,
            collection: String::new(),
        };
        return ok_text(&result);
    }

    let library_name = input.library_name.as_deref();
    let collection = if library_name.is_some() {
        wqm_common::constants::COLLECTION_LIBRARIES.to_string()
    } else {
        COLLECTION_SCRATCHPAD.to_string()
    };
    let tenant_id = library_name
        .map(|l| l.trim().to_string())
        .or_else(|| session_project_id.map(str::to_string))
        .unwrap_or_else(|| TENANT_GLOBAL.to_string());

    let payload_json = build_url_payload(&url, library_name, input.title.as_deref());

    match daemon
        .enqueue_item(
            "url",
            "add",
            &tenant_id,
            &collection,
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_store_url\"}"),
        )
        .await
    {
        Err(e) => {
            let result = StoreUrlScratchpadResult {
                success: false,
                message: format!("Failed to queue URL: {e}"),
                queue_id: None,
                collection,
            };
            ok_text(&result)
        }
        Ok(queue_id) => {
            let result = StoreUrlScratchpadResult {
                success: true,
                message: format!("URL queued for fetch and ingestion ({collection}/{tenant_id})"),
                queue_id: Some(queue_id),
                collection,
            };
            ok_text(&result)
        }
    }
}

/// Build the scratchpad queue payload — mirrors `buildScratchpadPayload`
/// in store-handlers.ts:150-158.
fn build_scratchpad_payload(content: &str, title: Option<&str>, tags: &[String]) -> String {
    let mut map = Map::new();
    map.insert(
        "content".to_string(),
        Value::String(content.trim().to_string()),
    );
    map.insert(
        "source_type".to_string(),
        Value::String("scratchpad".to_string()),
    );
    if let Some(t) = title {
        if !t.trim().is_empty() {
            map.insert("title".to_string(), Value::String(t.trim().to_string()));
        }
    }
    if !tags.is_empty() {
        let arr: Vec<Value> = tags.iter().map(|s| Value::String(s.clone())).collect();
        map.insert("tags".to_string(), Value::Array(arr));
    }
    serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
}

/// Handle store type=scratchpad — mirrors `storeScratchpad` in
/// store-handlers.ts:161-222.
async fn store_scratchpad<D>(
    input: StoreInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: StoreDaemon,
{
    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c.to_string(),
        _ => {
            return ok_text(&StoreUrlScratchpadResult {
                success: false,
                message: "content is required when type is \"scratchpad\"".to_string(),
                queue_id: None,
                collection: COLLECTION_SCRATCHPAD.to_string(),
            });
        }
    };

    let tenant_id = session_project_id
        .map(str::to_string)
        .unwrap_or_else(|| TENANT_GLOBAL.to_string());

    let payload_json = build_scratchpad_payload(&content, input.title.as_deref(), &input.tags);

    let result = daemon
        .enqueue_item(
            "text",
            "add",
            &tenant_id,
            COLLECTION_SCRATCHPAD,
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_store_scratchpad\"}"),
        )
        .await;

    match result {
        Err(e) => ok_text(&StoreUrlScratchpadResult {
            success: false,
            message: format!("Failed to queue scratchpad entry: {e}"),
            queue_id: None,
            collection: COLLECTION_SCRATCHPAD.to_string(),
        }),
        Ok(queue_id) => {
            // Fire-and-forget mirror write — store-handlers.ts:208
            let now = now_utc();
            let tags_json = if input.tags.is_empty() {
                "[]".to_string()
            } else {
                serde_json::to_string(&input.tags).unwrap_or_else(|_| "[]".to_string())
            };
            daemon
                .upsert_scratchpad_mirror(
                    uuid::Uuid::new_v4().to_string(),
                    content.trim().to_string(),
                    input.title.as_ref().and_then(|t| {
                        let trimmed = t.trim();
                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.to_string())
                        }
                    }),
                    Some(tags_json),
                    tenant_id.clone(),
                    now.clone(),
                    now,
                )
                .await;

            ok_text(&StoreUrlScratchpadResult {
                success: true,
                message: format!("Scratchpad entry queued for processing ({tenant_id})"),
                queue_id: Some(queue_id),
                collection: COLLECTION_SCRATCHPAD.to_string(),
            })
        }
    }
}

/// Generate document ID using SHA256 hash for idempotency.
///
/// Mirrors `generateDocumentId` in store.ts:207-213:
/// `sha256(tenant_id + content).hex[:32]`
fn generate_document_id(content: &str, tenant_id: &str) -> String {
    use sha2::{Digest, Sha256};
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
fn resolve_library_tenant(
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
async fn store_library<D>(input: StoreInput, daemon: &mut D) -> CallToolResult
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
