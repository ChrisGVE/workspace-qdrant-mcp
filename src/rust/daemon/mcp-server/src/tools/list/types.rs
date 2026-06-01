//! Response types for the `list` MCP tool.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/list-files-types.ts`.
//!
//! # Field ordering (parity mandate)
//!
//! Serde serialises struct fields in declaration order. The order here matches
//! the TypeScript `ListResponse` and `ListStats` interfaces exactly so that
//! `JSON.stringify(result)` byte-for-byte matches the TS output.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default directory depth (mirrors `DEFAULT_DEPTH` in list-files-types.ts).
pub const DEFAULT_DEPTH: u32 = 3;
/// Maximum directory depth (mirrors `MAX_DEPTH`).
pub const MAX_DEPTH: u32 = 10;
/// Default entry limit (mirrors `DEFAULT_LIMIT`).
pub const DEFAULT_LIMIT: u32 = 200;
/// Maximum entry limit (mirrors `MAX_LIMIT`).
pub const MAX_LIMIT: u32 = 500;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A detected project component.
///
/// Mirrors `ComponentSummary` in list-files-types.ts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentSummary {
    pub id: String,
    #[serde(rename = "basePath")]
    pub base_path: String,
    pub source: String,
}

/// Statistics for a list result.
///
/// Mirrors `ListStats` in list-files-types.ts.
/// Field order matches the TypeScript declaration:
/// `files, folders, languages, truncated, totalMatching, components?`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ListStats {
    pub files: usize,
    pub folders: usize,
    pub languages: Vec<String>,
    pub truncated: bool,
    #[serde(rename = "totalMatching")]
    pub total_matching: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<Vec<ComponentSummary>>,
}

/// The full response returned by the `list` tool.
///
/// Mirrors `ListResponse` in list-files-types.ts.
/// Field order (TS declaration order):
/// `success, projectPath, basePath, format, listing, stats, message?, next_token?`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ListResponse {
    pub success: bool,
    #[serde(rename = "projectPath")]
    pub project_path: Option<String>,
    #[serde(rename = "basePath")]
    pub base_path: String,
    pub format: String,
    pub listing: String,
    pub stats: ListStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Opaque cursor for the next page (base64-encoded relative_path of the
    /// last file in this page). Matches the TS field name `next_token`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

impl ListResponse {
    /// Build an error response (project not found, database unavailable, etc.)
    pub fn error(message: impl Into<String>, base_path: impl Into<String>, format: &str) -> Self {
        Self {
            success: false,
            project_path: None,
            base_path: {
                let bp = base_path.into();
                if bp.is_empty() {
                    ".".to_string()
                } else {
                    bp
                }
            },
            format: format.to_string(),
            listing: String::new(),
            stats: ListStats {
                files: 0,
                folders: 0,
                languages: Vec::new(),
                truncated: false,
                total_matching: 0,
                components: None,
            },
            message: Some(message.into()),
            next_token: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal tree types
// ---------------------------------------------------------------------------

/// A leaf file node in the folder tree.
///
/// Mirrors `FileLeaf` in list-files-types.ts.
#[derive(Debug, Clone)]
pub struct FileLeaf {
    pub name: String,
    pub extension: Option<String>,
    pub language: Option<String>,
    pub is_test: bool,
}

/// A folder node in the directory tree.
///
/// Mirrors `FolderNode` in list-files-types.ts.
#[derive(Debug, Clone)]
pub struct FolderNode {
    pub name: String,
    /// Child folders, keyed by segment name.
    pub children: std::collections::BTreeMap<String, FolderNode>,
    pub files: Vec<FileLeaf>,
    /// If set, this folder is a submodule root — do not expand children.
    pub submodule: Option<SubmoduleMarker>,
    /// Total file count in this subtree (computed during build).
    pub total_files: usize,
}

/// Marks a folder as a submodule root.
///
/// Mirrors `SubmoduleMarker` in list-files-types.ts.
#[derive(Debug, Clone)]
pub struct SubmoduleMarker {
    pub repo_name: String,
}
