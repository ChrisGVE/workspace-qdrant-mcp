//! `list` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/list-files/index.ts`.
//!
//! Reads from the daemon's `tracked_files` SQLite table (via
//! [`StateManager`]) and returns tree / summary / flat views of the
//! project structure.
//!
//! # Result shape (field order matches TS `ListResponse` declaration)
//!
//! ```json
//! {
//!   "success": true,
//!   "projectPath": "/home/user/proj",
//!   "basePath": ".",
//!   "format": "tree",
//!   "listing": "src/\n  main.rs [rs]",
//!   "stats": { "files": 1, "folders": 1, "languages": ["rust"],
//!               "truncated": false, "totalMatching": 1 }
//! }
//! ```

pub mod renderers;
pub mod tree;
pub mod types;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use rmcp::model::CallToolResult;
use std::collections::HashSet;

use crate::server_types::SessionState;
use crate::sqlite::tracked_files::{
    count_tracked_files, list_project_components, list_submodules, list_tracked_files,
    ComponentEntry, ListTrackedFilesOptions, SubmoduleEntry, TrackedFileEntry,
};
use crate::sqlite::{project_queries, StateManager};
use crate::tools::envelope::ok_text;

use renderers::{render_flat, render_summary, render_tree};
use tree::{build_tree, count_folders};
use types::{
    ComponentSummary, ListResponse, ListStats, DEFAULT_DEPTH, DEFAULT_LIMIT, MAX_DEPTH, MAX_LIMIT,
};

// ---------------------------------------------------------------------------
// Input struct
// ---------------------------------------------------------------------------

/// Parsed input for the `list` tool.
///
/// All fields are optional; defaults mirror the TS `ListOptions` type.
#[derive(Debug, Default)]
pub struct ListInput {
    pub path: Option<String>,
    pub depth: Option<u32>,
    pub format: Option<String>,
    pub file_type: Option<String>,
    pub language: Option<String>,
    pub extension: Option<String>,
    pub pattern: Option<String>,
    pub include_tests: Option<bool>,
    pub limit: Option<u32>,
    pub project_id: Option<String>,
    pub component: Option<String>,
    pub cursor: Option<String>,
    pub page_size: Option<u32>,
    pub branch: Option<String>,
}

impl ListInput {
    /// Parse from the JSON `arguments` map of a `CallToolRequestParams`.
    pub fn from_args(args: &serde_json::Map<String, serde_json::Value>) -> Self {
        Self {
            path: args
                .get("path")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            depth: args.get("depth").and_then(|v| v.as_u64()).map(|v| v as u32),
            format: args
                .get("format")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            file_type: args
                .get("fileType")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            language: args
                .get("language")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            extension: args
                .get("extension")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            pattern: args
                .get("pattern")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            include_tests: args.get("includeTests").and_then(|v| v.as_bool()),
            limit: args.get("limit").and_then(|v| v.as_u64()).map(|v| v as u32),
            project_id: args
                .get("projectId")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            component: args
                .get("component")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            cursor: args
                .get("cursor")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            page_size: args
                .get("pageSize")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            branch: args
                .get("branch")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        }
    }
}

// ---------------------------------------------------------------------------
// Component matching
// ---------------------------------------------------------------------------

/// Check whether a component ID matches a filter string (prefix / exact).
///
/// Mirrors `componentMatchesFilter` in component-detector/helpers.ts.
fn component_matches_filter(component_id: &str, filter: &str) -> bool {
    component_id == filter
        || component_id.starts_with(&format!("{filter}."))
        || component_id.starts_with(&format!("{filter}/"))
}

/// Resolve a component filter string to a list of base-path prefixes for SQL.
///
/// Returns `None` when no filter is set (all files pass).
/// Returns an empty `Vec` when a filter is set but matches nothing.
///
/// Mirrors `resolveComponentBasePaths` in list-files/index.ts lines 295-308.
fn resolve_component_base_paths(
    component: Option<&str>,
    components: &[ComponentEntry],
) -> Option<Vec<String>> {
    let filter = component?;
    let paths: Vec<String> = components
        .iter()
        .filter(|c| component_matches_filter(&c.component_name, filter))
        .map(|c| c.base_path.clone())
        .collect();
    Some(paths)
}

// ---------------------------------------------------------------------------
// Pagination helpers
// ---------------------------------------------------------------------------

/// Encode a relative path as a base64 cursor (standard alphabet, padded).
///
/// Mirrors `Buffer.from(lastFile.relativePath).toString('base64')` in index.ts.
fn encode_cursor(relative_path: &str) -> String {
    BASE64_STANDARD.encode(relative_path.as_bytes())
}

/// Decode a base64 cursor back to a relative path.
fn decode_cursor(cursor: &str) -> Option<String> {
    BASE64_STANDARD
        .decode(cursor)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
}

// ---------------------------------------------------------------------------
// Stats builder
// ---------------------------------------------------------------------------

fn build_list_stats(
    page_files: &[TrackedFileEntry],
    submodules: &[SubmoduleEntry],
    base_path: &str,
    truncated: bool,
    total_matching: i64,
    component_summaries: Option<Vec<ComponentSummary>>,
) -> ListStats {
    let mut lang_set: HashSet<String> = HashSet::new();
    for f in page_files {
        if let Some(lang) = &f.language {
            lang_set.insert(lang.clone());
        }
    }
    let mut languages: Vec<String> = lang_set.into_iter().collect();
    languages.sort();

    let tree = build_tree(page_files, submodules, base_path);
    let folders = count_folders(&tree);

    ListStats {
        files: page_files.len(),
        folders,
        languages,
        truncated,
        total_matching,
        components: component_summaries,
    }
}

// ---------------------------------------------------------------------------
// Core tool function
// ---------------------------------------------------------------------------

/// Execute the `list` tool.
///
/// Mirrors `ListFilesTool.list()` in list-files/index.ts.
/// Synchronous — reads from SQLite only, no async I/O.
pub fn list_tool(input: ListInput, sqlite: &StateManager, state: &SessionState) -> CallToolResult {
    let format = input.format.as_deref().unwrap_or("tree");
    let depth = input
        .depth
        .map(|d| d.clamp(1, MAX_DEPTH))
        .unwrap_or(DEFAULT_DEPTH);
    let limit = input
        .limit
        .map(|l| l.clamp(1, MAX_LIMIT))
        .unwrap_or(DEFAULT_LIMIT);
    let base_path = input.path.clone().unwrap_or_default();

    // Resolve project ID: input override → session state.
    let project_id = match input.project_id.as_deref().or(state.project_id.as_deref()) {
        Some(id) => id.to_string(),
        None => {
            return ok_text(&ListResponse::error(
                "Could not detect project. Use projectId parameter.",
                &base_path,
                format,
            ));
        }
    };

    let conn = sqlite.connection();

    // Resolve watch_folder_id from tenant_id.
    let watch_folder_id = match project_queries::get_watch_folder_id_by_tenant(conn, &project_id) {
        Some(id) => id,
        None => {
            return ok_text(&ListResponse::error(
                "Project not found in database. Has the daemon indexed it?",
                &base_path,
                format,
            ));
        }
    };

    // Resolve project path for the response.
    let project_path =
        project_queries::get_project_by_id(conn, &project_id).map(|p| p.project_path);

    // Load components (DB first, then skip — no live filesystem detection in
    // the Rust server; component detection from disk is a TS-only concern).
    let db_components = list_project_components(conn, &watch_folder_id);
    let component_summaries: Option<Vec<ComponentSummary>> = if db_components.is_empty() {
        None
    } else {
        Some(
            db_components
                .iter()
                .map(|c| ComponentSummary {
                    id: c.component_name.clone(),
                    base_path: c.base_path.clone(),
                    source: c.source.clone(),
                })
                .collect(),
        )
    };

    // Resolve component filter to SQL base-paths.
    let component_base_paths =
        resolve_component_base_paths(input.component.as_deref(), &db_components);

    // Decode cursor for keyset pagination.
    let after_path = input.cursor.as_deref().and_then(decode_cursor);

    // Effective page size: pageSize > limit > DEFAULT_LIMIT, capped at MAX_LIMIT.
    let page_size = input.page_size.unwrap_or(limit).clamp(1, MAX_LIMIT);

    // Build filter options.
    let mut opts = ListTrackedFilesOptions {
        watch_folder_id: watch_folder_id.clone(),
        limit: Some(page_size as usize),
        ..Default::default()
    };
    if !base_path.is_empty() {
        opts.path = Some(base_path.clone());
    }
    if let Some(ft) = &input.file_type {
        opts.file_type = Some(ft.clone());
    }
    if let Some(lang) = &input.language {
        opts.language = Some(lang.clone());
    }
    if let Some(ext) = &input.extension {
        opts.extension = Some(ext.clone());
    }
    if let Some(it) = input.include_tests {
        opts.include_tests = Some(it);
    }
    if let Some(pat) = &input.pattern {
        opts.glob = Some(pat.clone());
    }
    if let Some(paths) = component_base_paths {
        if !paths.is_empty() {
            opts.component_base_paths = Some(paths);
        }
    }
    // "*" means all branches — omit filter entirely.
    let branch = input.branch.as_deref().or(state.current_branch.as_deref());
    if let Some(br) = branch {
        if br != "*" {
            opts.branch = Some(br.to_string());
        }
    }

    // COUNT with all filters but no cursor, no limit.
    let count_opts = ListTrackedFilesOptions {
        after_path: None,
        limit: None,
        ..opts.clone()
    };
    let total_matching = count_tracked_files(conn, &count_opts);

    // Paginated fetch.
    if let Some(ap) = after_path {
        opts.after_path = Some(ap);
    }
    let page_files = list_tracked_files(conn, &opts);

    // Submodules.
    let submodules = list_submodules(conn, &watch_folder_id);

    // Render the listing string.
    let (listing, rendered_count) =
        render_files(&page_files, &submodules, &base_path, format, depth, limit);

    // truncated: render limit hit within the page.
    let truncated = rendered_count < page_files.len();

    // next_token: present when there could be more pages.
    let has_next_page = page_files.len() >= page_size as usize;
    let next_token = if has_next_page {
        page_files.last().map(|f| encode_cursor(&f.relative_path))
    } else {
        None
    };

    let final_listing = if truncated || next_token.is_some() {
        format!("{listing}\n... (truncated, {total_matching} total files match)")
    } else {
        listing
    };

    let stats = build_list_stats(
        &page_files,
        &submodules,
        &base_path,
        truncated || next_token.is_some(),
        total_matching,
        component_summaries,
    );

    let mut response = ListResponse {
        success: true,
        project_path,
        base_path: if base_path.is_empty() {
            ".".to_string()
        } else {
            base_path
        },
        format: format.to_string(),
        listing: final_listing,
        stats,
        message: None,
        next_token: None,
    };
    response.next_token = next_token;

    ok_text(&response)
}

// ---------------------------------------------------------------------------
// Internal: dispatch to the right renderer
// ---------------------------------------------------------------------------

fn render_files(
    files: &[TrackedFileEntry],
    submodules: &[SubmoduleEntry],
    base_path: &str,
    format: &str,
    depth: u32,
    limit: u32,
) -> (String, usize) {
    let root = build_tree(files, submodules, base_path);
    match format {
        "summary" => render_summary(&root, depth, limit),
        "flat" => render_flat(files, limit),
        _ => render_tree(&root, depth, limit),
    }
}

// ---------------------------------------------------------------------------
// Tests (sibling file)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../list_tests.rs"]
mod tests;
