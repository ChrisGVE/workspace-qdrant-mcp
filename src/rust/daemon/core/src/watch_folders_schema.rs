//! Watch Folders Schema Definitions
//!
//! This module defines the types and schema for the unified watch_folders table.
//! It consolidates: registered_projects, project_submodules, watch_configurations,
//! and library_watches into a single table per WORKSPACE_QDRANT_MCP.md spec v1.6.3.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Collection types for watch folders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchCollection {
    /// Project content (code, docs, tests, configs)
    Projects,
    /// Library documentation (books, papers, API docs)
    Libraries,
}

impl fmt::Display for WatchCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatchCollection::Projects => write!(f, "projects"),
            WatchCollection::Libraries => write!(f, "libraries"),
        }
    }
}

impl WatchCollection {
    /// Parse collection from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "projects" => Some(WatchCollection::Projects),
            "libraries" => Some(WatchCollection::Libraries),
            _ => None,
        }
    }
}

/// Library mode for library watch folders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LibraryMode {
    /// Full synchronization - deletes removed files from Qdrant
    Sync,
    /// Incremental only - never deletes, only adds/updates
    Incremental,
}

impl fmt::Display for LibraryMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LibraryMode::Sync => write!(f, "sync"),
            LibraryMode::Incremental => write!(f, "incremental"),
        }
    }
}

impl LibraryMode {
    /// Parse library mode from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sync" => Some(LibraryMode::Sync),
            "incremental" => Some(LibraryMode::Incremental),
            _ => None,
        }
    }
}

/// A watch folder entry representing a project, library, or submodule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchFolder {
    /// Unique identifier for this watch
    pub watch_id: String,
    /// Absolute filesystem path
    pub path: String,
    /// Collection type (projects or libraries)
    pub collection: WatchCollection,
    /// Tenant identifier (project_id for projects, library name for libraries)
    pub tenant_id: String,

    // Hierarchy (for submodules)
    /// Parent watch ID (NULL for top-level, references parent for submodules)
    pub parent_watch_id: Option<String>,
    /// Relative path within parent (NULL if not submodule)
    pub submodule_path: Option<String>,

    // Project-specific fields (NULL for libraries)
    /// Normalized git remote URL
    pub git_remote_url: Option<String>,
    /// SHA256 hash of remote URL (first 12 chars) for duplicate grouping
    pub remote_hash: Option<String>,
    /// Path suffix for clone disambiguation
    pub disambiguation_path: Option<String>,
    /// Activity flag - inherited by all subprojects
    pub is_active: bool,
    /// Last activity timestamp - synced across parent and all subprojects
    pub last_activity_at: Option<String>,

    // Library-specific fields (NULL for projects)
    /// Library mode: sync or incremental
    pub library_mode: Option<LibraryMode>,

    // Shared configuration
    /// Whether to follow symlinks during watching
    pub follow_symlinks: bool,
    /// Whether this watch is enabled
    pub enabled: bool,
    /// Whether to remove content from Qdrant when watch is disabled
    pub cleanup_on_disable: bool,

    // Timestamps
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Last update timestamp (ISO 8601)
    pub updated_at: String,
    /// Last scan timestamp (ISO 8601), NULL if never scanned
    pub last_scan: Option<String>,
}

impl WatchFolder {
    /// Check if this is a submodule (has a parent)
    pub fn is_submodule(&self) -> bool {
        self.parent_watch_id.is_some()
    }

    /// Check if this is a library watch
    pub fn is_library(&self) -> bool {
        self.collection == WatchCollection::Libraries
    }

    /// Check if this is a project watch
    pub fn is_project(&self) -> bool {
        self.collection == WatchCollection::Projects
    }
}

/// SQL to create the watch_folders table
pub const CREATE_WATCH_FOLDERS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS watch_folders (
    -- Primary identification
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
    tenant_id TEXT NOT NULL,

    -- Hierarchy (for submodules)
    parent_watch_id TEXT,
    submodule_path TEXT,

    -- Project-specific (NULL for libraries)
    git_remote_url TEXT,
    remote_hash TEXT,
    disambiguation_path TEXT,
    is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1)),
    last_activity_at TEXT,

    -- Library-specific (NULL for projects)
    library_mode TEXT CHECK (library_mode IS NULL OR library_mode IN ('sync', 'incremental')),

    -- Shared configuration
    follow_symlinks INTEGER DEFAULT 0 CHECK (follow_symlinks IN (0, 1)),
    enabled INTEGER DEFAULT 1 CHECK (enabled IN (0, 1)),
    cleanup_on_disable INTEGER DEFAULT 0 CHECK (cleanup_on_disable IN (0, 1)),

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_scan TEXT,

    -- Foreign key for submodule hierarchy
    FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
)
"#;

/// SQL to create indexes for the watch_folders table
pub const CREATE_WATCH_FOLDERS_INDEXES_SQL: &[&str] = &[
    // Index for finding duplicates (same remote, different paths)
    r#"CREATE INDEX IF NOT EXISTS idx_watch_remote_hash
       ON watch_folders(remote_hash)"#,
    // Index for active project lookups (used in queue priority calculation)
    r#"CREATE INDEX IF NOT EXISTS idx_watch_active
       ON watch_folders(is_active) WHERE is_active = 1"#,
    // Index for daemon polling (find recently updated watches)
    r#"CREATE INDEX IF NOT EXISTS idx_watch_updated
       ON watch_folders(updated_at)"#,
    // Index for enabled watches only
    r#"CREATE INDEX IF NOT EXISTS idx_watch_enabled
       ON watch_folders(enabled) WHERE enabled = 1"#,
    // Index for subproject hierarchy (find children of a parent)
    r#"CREATE INDEX IF NOT EXISTS idx_watch_parent
       ON watch_folders(parent_watch_id)"#,
    // Index for collection + tenant queries
    r#"CREATE INDEX IF NOT EXISTS idx_watch_collection_tenant
       ON watch_folders(collection, tenant_id)"#,
    // Index for path lookups (common query pattern)
    r#"CREATE INDEX IF NOT EXISTS idx_watch_path
       ON watch_folders(path)"#,
];

/// SQL to activate a project and all its related watches (parent, siblings, children)
/// Use with parameter :watch_id
pub const ACTIVATE_PROJECT_GROUP_SQL: &str = r#"
UPDATE watch_folders
SET is_active = 1,
    last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE watch_id = :watch_id
   OR parent_watch_id = :watch_id
   OR watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
   OR parent_watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
"#;

/// SQL to deactivate a project and all its related watches
/// Use with parameter :watch_id
pub const DEACTIVATE_PROJECT_GROUP_SQL: &str = r#"
UPDATE watch_folders
SET is_active = 0,
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE watch_id = :watch_id
   OR parent_watch_id = :watch_id
   OR watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
   OR parent_watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_display() {
        assert_eq!(WatchCollection::Projects.to_string(), "projects");
        assert_eq!(WatchCollection::Libraries.to_string(), "libraries");
    }

    #[test]
    fn test_collection_from_str() {
        assert_eq!(WatchCollection::from_str("projects"), Some(WatchCollection::Projects));
        assert_eq!(WatchCollection::from_str("LIBRARIES"), Some(WatchCollection::Libraries));
        assert_eq!(WatchCollection::from_str("invalid"), None);
    }

    #[test]
    fn test_library_mode_display() {
        assert_eq!(LibraryMode::Sync.to_string(), "sync");
        assert_eq!(LibraryMode::Incremental.to_string(), "incremental");
    }

    #[test]
    fn test_library_mode_from_str() {
        assert_eq!(LibraryMode::from_str("sync"), Some(LibraryMode::Sync));
        assert_eq!(LibraryMode::from_str("INCREMENTAL"), Some(LibraryMode::Incremental));
        assert_eq!(LibraryMode::from_str("invalid"), None);
    }

    #[test]
    fn test_sql_constants_are_valid() {
        // Basic validation that SQL constants don't have obvious issues
        assert!(CREATE_WATCH_FOLDERS_SQL.contains("CREATE TABLE"));
        assert!(CREATE_WATCH_FOLDERS_SQL.contains("watch_id TEXT PRIMARY KEY"));
        assert!(CREATE_WATCH_FOLDERS_SQL.contains("parent_watch_id"));
        assert!(CREATE_WATCH_FOLDERS_SQL.contains("is_active"));

        assert!(!CREATE_WATCH_FOLDERS_INDEXES_SQL.is_empty());
        for idx_sql in CREATE_WATCH_FOLDERS_INDEXES_SQL {
            assert!(idx_sql.contains("CREATE INDEX"));
        }
    }
}
