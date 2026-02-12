//! Project ID Disambiguation for Multi-Clone Scenarios
//!
//! This module handles the disambiguation of project IDs when multiple clones
//! of the same repository exist on the filesystem.
//!
//! Core types (DisambiguationConfig, ProjectIdCalculator, DisambiguationPathComputer)
//! are defined in `wqm_common::project_id` and re-exported here.
//! Daemon-specific types (ProjectRecord, RegisteredProject, DisambiguationError)
//! remain in this module.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

// Re-export canonical types from wqm-common
pub use wqm_common::project_id::{
    DisambiguationConfig, ProjectIdCalculator, DisambiguationPathComputer,
};

/// Errors that can occur during project disambiguation
#[derive(Error, Debug)]
pub enum DisambiguationError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Path resolution error: {path}")]
    PathResolution { path: String },

    #[error("Git error: {0}")]
    Git(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type DisambiguationResult<T> = Result<T, DisambiguationError>;

/// Represents a registered project with disambiguation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRecord {
    /// Unique project ID (may include disambiguation hash)
    pub project_id: String,

    /// Filesystem path where the project is located
    pub project_path: PathBuf,

    /// Original git remote URL (non-normalized)
    pub git_remote_url: Option<String>,

    /// Hash of the normalized remote URL for grouping clones
    pub remote_hash: Option<String>,

    /// Disambiguation path component (relative to common ancestor)
    pub disambiguation_path: Option<String>,

    /// When this project was first registered
    pub registered_at: DateTime<Utc>,

    /// When this project was last active
    pub last_activity: Option<DateTime<Utc>>,
}

/// Extended project registration record with disambiguation
///
/// NOTE: This struct is deprecated. Use `WatchFolder` from `watch_folders_schema` instead.
/// The unified `watch_folders` table consolidates all project/library tracking.
/// This struct remains for backward compatibility during migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredProject {
    /// Unique project ID
    pub project_id: String,

    /// Filesystem path
    pub project_path: PathBuf,

    /// Original git remote URL (non-normalized)
    pub git_remote_url: Option<String>,

    /// Hash of normalized remote (for grouping clones)
    pub remote_hash: Option<String>,

    /// Disambiguation path component
    pub disambiguation_path: Option<String>,

    /// Registration timestamp
    pub registered_at: DateTime<Utc>,

    /// Last activity timestamp
    pub last_activity_at: Option<DateTime<Utc>>,

    /// Whether this project is currently active
    pub is_active: bool,
}

// NOTE: Legacy SQL constants (CREATE_REGISTERED_PROJECTS_SQL, CREATE_REGISTERED_PROJECTS_INDEXES_SQL)
// have been removed. Use the unified `watch_folders` table from `watch_folders_schema.rs` instead.
// See WORKSPACE_QDRANT_MCP.md v1.6.2+ for the consolidated schema.

// NOTE: Tests for ProjectIdCalculator, DisambiguationPathComputer, and related functions
// have been moved to wqm_common::project_id. Only daemon-specific tests remain here.
