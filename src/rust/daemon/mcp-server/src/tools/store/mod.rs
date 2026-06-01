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

mod library;
mod url_scratchpad;

use std::path::Path;

use rmcp::model::CallToolResult;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::session::project_detect::{find_git_root, get_git_remote_url};
use crate::tools::envelope::{error_text, ok_text};

// Re-export helpers for test access (store_tests_part2.rs)
#[cfg(test)]
pub(crate) use library::generate_document_id;
#[cfg(test)]
pub(crate) use url_scratchpad::validate_url;

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
        git_remote: Option<&str>,
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
        git_remote: Option<&str>,
    ) -> Result<ProjectRegisterResult, String> {
        use crate::proto::RegisterProjectRequest;
        let req = RegisterProjectRequest {
            path: path.to_string(),
            project_id: String::new(),
            name: Some(name.to_string()),
            git_remote: git_remote.map(str::to_string),
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
        "url" => url_scratchpad::store_url(input, daemon, session_project_id).await,
        "scratchpad" => url_scratchpad::store_scratchpad(input, daemon, session_project_id).await,
        _ => library::store_library(input, daemon).await,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// store_project sub-handler
// ─────────────────────────────────────────────────────────────────────────────

/// Handle store type=project — mirrors `registerProjectFromTool` in
/// session-lifecycle.ts:198-211.
///
/// Key steps that mirror the TS implementation (lines 198-202, 210-211):
/// - `resolved_path = find_git_root(path) ?? path`  (line 198)
/// - `name` defaults to the basename of `resolved_path`  (line 199)
/// - `git_remote = get_git_remote_url(resolved_path)`  (line 200)
/// - Register `resolved_path` (not the raw input path)  (line 202)
/// - Build success message from `resolved_path`  (lines 210-211)
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
    let raw_path = match input.path.as_deref() {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => return error_text("path is required for store type \"project\""),
    };

    if !daemon_connected {
        return error_text("Daemon is not connected — cannot register project");
    }

    // Resolve to git root when available — session-lifecycle.ts:198
    let resolved_path = find_git_root(Path::new(&raw_path))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(raw_path);

    // name defaults to basename of resolved path — session-lifecycle.ts:199
    let name = input
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            Path::new(&resolved_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
        })
        .to_string();

    // git_remote from resolved path — session-lifecycle.ts:200
    let git_remote = get_git_remote_url(Path::new(&resolved_path));

    match daemon
        .register_project(&resolved_path, &name, git_remote.as_deref())
        .await
    {
        Err(e) => error_text(&e),
        Ok(resp) => {
            // message built from resolved_path — session-lifecycle.ts:210-211
            let message = if resp.newly_registered {
                format!("Project registered and activated: {resolved_path}")
            } else {
                format!("Project already registered and activated: {resolved_path}")
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../store_tests.rs"]
mod tests;
