//! Watch folder record type and row conversion.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};

use super::{DaemonStateManager, DaemonStateResult};

/// Watch folder record matching spec-defined watch_folders table
/// Per WORKSPACE_QDRANT_MCP.md v1.6.4, this unified table consolidates
/// project and library watching configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchFolderRecord {
    /// Unique watch identifier (PRIMARY KEY)
    pub watch_id: String,
    /// Absolute filesystem path
    pub path: String,
    /// Collection type: "projects" or "libraries"
    pub collection: String,
    /// Tenant identifier (project_id for projects, library name for libraries)
    pub tenant_id: String,

    // Hierarchy (for submodules)
    /// Parent watch ID (None for top-level)
    pub parent_watch_id: Option<String>,
    /// Relative path within parent (None if not submodule)
    pub submodule_path: Option<String>,

    // Project-specific fields (None for libraries)
    /// Normalized git remote URL
    pub git_remote_url: Option<String>,
    /// SHA256 hash of remote URL (first 12 chars)
    pub remote_hash: Option<String>,
    /// Path suffix for clone disambiguation
    pub disambiguation_path: Option<String>,
    /// Activity flag - inherited by all subprojects
    pub is_active: bool,
    /// Last activity timestamp
    pub last_activity_at: Option<DateTime<Utc>>,
    /// Whether this watch folder is paused (events buffered, not processed)
    pub is_paused: bool,
    /// Timestamp when pause began, None if not paused
    pub pause_start_time: Option<DateTime<Utc>>,
    /// Whether this folder is archived (no watching/ingesting, but remains searchable)
    pub is_archived: bool,
    /// Last known commit hash (for git-tracked projects)
    pub last_commit_hash: Option<String>,
    /// Whether this watch folder is git-tracked (has .git/ or .git file)
    pub is_git_tracked: bool,

    // Library-specific fields (None for projects)
    /// Library mode: "sync" or "incremental"
    pub library_mode: Option<String>,

    // Shared configuration
    /// Whether to follow symlinks during watching
    pub follow_symlinks: bool,
    /// Whether this watch is enabled
    pub enabled: bool,
    /// Whether to remove content from Qdrant when watch is disabled
    pub cleanup_on_disable: bool,

    // Timestamps
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Last scan timestamp (None if never scanned)
    pub last_scan: Option<DateTime<Utc>>,
}

impl WatchFolderRecord {
    /// Check if this is a submodule (has a parent)
    pub fn is_submodule(&self) -> bool {
        self.parent_watch_id.is_some()
    }

    /// Check if this is a library watch
    pub fn is_library(&self) -> bool {
        self.collection == COLLECTION_LIBRARIES
    }

    /// Check if this is a project watch
    pub fn is_project(&self) -> bool {
        self.collection == COLLECTION_PROJECTS
    }
}

impl DaemonStateManager {
    /// Helper to convert a database row to WatchFolderRecord
    pub(crate) fn row_to_watch_folder(&self, row: &sqlx::sqlite::SqliteRow) -> DaemonStateResult<WatchFolderRecord> {
        let parse_datetime = |s: &str| -> DateTime<Utc> {
            DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now())
        };

        let parse_optional_datetime = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc)))
        };

        Ok(WatchFolderRecord {
            watch_id: row.try_get("watch_id")?,
            path: row.try_get("path")?,
            collection: row.try_get("collection")?,
            tenant_id: row.try_get("tenant_id")?,
            parent_watch_id: row.try_get("parent_watch_id")?,
            submodule_path: row.try_get("submodule_path")?,
            git_remote_url: row.try_get("git_remote_url")?,
            remote_hash: row.try_get("remote_hash")?,
            disambiguation_path: row.try_get("disambiguation_path")?,
            is_active: row.try_get::<i32, _>("is_active")? != 0,
            last_activity_at: parse_optional_datetime(row.try_get("last_activity_at")?),
            is_paused: row.try_get::<i32, _>("is_paused").unwrap_or(0) != 0,
            pause_start_time: parse_optional_datetime(row.try_get("pause_start_time").unwrap_or(None)),
            is_archived: row.try_get::<i32, _>("is_archived").unwrap_or(0) != 0,
            last_commit_hash: row.try_get("last_commit_hash").unwrap_or(None),
            is_git_tracked: row.try_get::<i32, _>("is_git_tracked").unwrap_or(0) != 0,
            library_mode: row.try_get("library_mode")?,
            follow_symlinks: row.try_get::<i32, _>("follow_symlinks")? != 0,
            enabled: row.try_get::<i32, _>("enabled")? != 0,
            cleanup_on_disable: row.try_get::<i32, _>("cleanup_on_disable")? != 0,
            created_at: parse_datetime(&row.try_get::<String, _>("created_at")?),
            updated_at: parse_datetime(&row.try_get::<String, _>("updated_at")?),
            last_scan: parse_optional_datetime(row.try_get("last_scan")?),
        })
    }
}
