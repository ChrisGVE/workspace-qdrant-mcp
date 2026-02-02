//! Project ID Disambiguation for Multi-Clone Scenarios
//!
//! This module handles the disambiguation of project IDs when multiple clones
//! of the same repository exist on the filesystem. It provides:
//!
//! - Unique project IDs for each clone based on disambiguation paths
//! - Alias management for project ID transitions (local → remote)
//! - Duplicate detection when registering new projects
//!
//! # Problem Statement
//!
//! When a user clones the same repository to multiple locations:
//! - `~/work/myproject` (clone of github.com/user/repo)
//! - `~/personal/myproject` (clone of same repo)
//!
//! Without disambiguation, both would get the same project_id based on the
//! normalized git remote URL. This module ensures each clone gets a unique
//! project_id while maintaining the ability to query by original remote.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

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

/// Configuration for project ID calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguationConfig {
    /// Length of project ID hash suffix
    pub id_hash_length: usize,

    /// Whether to include disambiguation in project_id
    pub enable_disambiguation: bool,

    /// Alias retention period in days
    pub alias_retention_days: u32,
}

impl Default for DisambiguationConfig {
    fn default() -> Self {
        Self {
            id_hash_length: 12,
            enable_disambiguation: true,
            alias_retention_days: 30,
        }
    }
}

/// Calculator for unique project IDs with disambiguation support
pub struct ProjectIdCalculator {
    config: DisambiguationConfig,
}

impl ProjectIdCalculator {
    /// Create a new calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(DisambiguationConfig::default())
    }

    /// Create a new calculator with custom configuration
    pub fn with_config(config: DisambiguationConfig) -> Self {
        Self { config }
    }

    /// Calculate a unique project ID
    ///
    /// # Algorithm
    ///
    /// 1. If git_remote exists:
    ///    - Normalize the URL to a canonical form
    ///    - If disambiguation_path provided: hash(normalized_url|disambiguation_path)
    ///    - Otherwise: hash(normalized_url)
    /// 2. If no git_remote (local project):
    ///    - hash(project_root_path)
    ///
    /// # Arguments
    ///
    /// * `project_root` - Path to the project root directory
    /// * `git_remote` - Optional git remote URL
    /// * `disambiguation_path` - Optional path for disambiguation (when multiple clones exist)
    ///
    /// # Returns
    ///
    /// A unique project ID string
    pub fn calculate(
        &self,
        project_root: &Path,
        git_remote: Option<&str>,
        disambiguation_path: Option<&str>,
    ) -> String {
        if let Some(remote) = git_remote {
            let normalized = Self::normalize_git_url(remote);

            let input = if let Some(disambig) = disambiguation_path {
                format!("{}|{}", normalized, disambig)
            } else {
                normalized.clone()
            };

            self.hash_to_id(&input)
        } else {
            // Local project - hash the path
            let path_str = project_root
                .canonicalize()
                .unwrap_or_else(|_| project_root.to_path_buf())
                .to_string_lossy()
                .to_string();

            format!("local_{}", self.hash_to_id(&path_str))
        }
    }

    /// Normalize a git URL to a canonical form
    ///
    /// # Examples
    ///
    /// ```
    /// // All of these normalize to "github.com/user/repo":
    /// // - https://github.com/user/repo.git
    /// // - git@github.com:user/repo.git
    /// // - ssh://git@github.com/user/repo
    /// // - http://github.com/user/repo
    /// ```
    pub fn normalize_git_url(url: &str) -> String {
        // Lowercase first for case-insensitive matching
        let mut normalized = url.to_lowercase();

        // Remove common protocols
        for protocol in &["https://", "http://", "ssh://", "git://"] {
            if normalized.starts_with(protocol) {
                normalized = normalized[protocol.len()..].to_string();
                break;
            }
        }

        // Handle git@ prefix (SSH format)
        if normalized.starts_with("git@") {
            normalized = normalized[4..].to_string();
            // Replace the first : with / for SSH format
            if let Some(idx) = normalized.find(':') {
                normalized.replace_range(idx..idx + 1, "/");
            }
        }

        // Remove .git suffix
        if normalized.ends_with(".git") {
            normalized = normalized[..normalized.len() - 4].to_string();
        }

        // Remove trailing slashes
        normalized.trim_end_matches('/').to_string()
    }

    /// Hash an input string to create an ID
    fn hash_to_id(&self, input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        let hash = hasher.finalize();
        let hash_hex = format!("{:x}", hash);
        hash_hex[..self.config.id_hash_length].to_string()
    }

    /// Calculate the remote hash for grouping clones
    pub fn calculate_remote_hash(&self, remote_url: &str) -> String {
        let normalized = Self::normalize_git_url(remote_url);
        self.hash_to_id(&normalized)
    }
}

impl Default for ProjectIdCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the disambiguation path for a new project
///
/// Given a new project path and existing projects with the same remote,
/// finds the first differing directory component from the common ancestor.
pub struct DisambiguationPathComputer;

impl DisambiguationPathComputer {
    /// Compute disambiguation path for a new project
    ///
    /// # Algorithm
    ///
    /// 1. Find common ancestor path between new_path and existing paths
    /// 2. Return the path from the differing point to the project root
    ///
    /// # Arguments
    ///
    /// * `new_path` - Path of the new project being registered
    /// * `existing_paths` - Paths of existing projects with the same remote
    ///
    /// # Returns
    ///
    /// Disambiguation path string (relative path from common ancestor)
    pub fn compute(new_path: &Path, existing_paths: &[PathBuf]) -> String {
        if existing_paths.is_empty() {
            // No existing clones, no disambiguation needed
            return String::new();
        }

        let new_components: Vec<_> = new_path.components().collect();

        // Find the shortest common prefix across all existing paths
        let mut min_common_idx = new_components.len();

        for existing_path in existing_paths {
            let existing_components: Vec<_> = existing_path.components().collect();

            let mut common_idx = 0;
            for (i, (a, b)) in new_components.iter().zip(&existing_components).enumerate() {
                if a != b {
                    common_idx = i;
                    break;
                }
                common_idx = i + 1;
            }

            min_common_idx = min_common_idx.min(common_idx);
        }

        // Build disambiguation path from the differing point
        if min_common_idx < new_components.len() {
            new_components[min_common_idx..]
                .iter()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/")
        } else {
            // Paths are identical up to the project root, use the full path
            new_path.to_string_lossy().to_string()
        }
    }

    /// Recompute disambiguation paths for all clones of a repository
    ///
    /// This is called when a new clone is added or an existing clone is removed
    /// to ensure all disambiguation paths are minimal and consistent.
    pub fn recompute_all(paths: &[PathBuf]) -> HashMap<PathBuf, String> {
        let mut result = HashMap::new();

        if paths.len() <= 1 {
            // Single clone or no clones - no disambiguation needed
            for path in paths {
                result.insert(path.clone(), String::new());
            }
            return result;
        }

        // For each path, compute its disambiguation against all others
        for path in paths {
            let others: Vec<_> = paths
                .iter()
                .filter(|p| *p != path)
                .cloned()
                .collect();

            let disambig = Self::compute(path, &others);
            result.insert(path.clone(), disambig);
        }

        result
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_git_url_https() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("https://github.com/user/repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_ssh() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("git@github.com:user/repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_http() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("http://github.com/user/repo"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_normalize_git_url_case_insensitive() {
        assert_eq!(
            ProjectIdCalculator::normalize_git_url("https://GitHub.COM/User/Repo.git"),
            "github.com/user/repo"
        );
    }

    #[test]
    fn test_calculate_project_id_with_remote() {
        let calc = ProjectIdCalculator::new();
        let id = calc.calculate(
            Path::new("/home/user/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        assert_eq!(id.len(), 12);
        assert!(!id.starts_with("local_"));
    }

    #[test]
    fn test_calculate_project_id_local() {
        let calc = ProjectIdCalculator::new();
        let id = calc.calculate(
            Path::new("/home/user/project"),
            None,
            None,
        );

        assert!(id.starts_with("local_"));
        assert_eq!(id.len(), 6 + 12); // "local_" + 12 char hash
    }

    #[test]
    fn test_calculate_project_id_with_disambiguation() {
        let calc = ProjectIdCalculator::new();

        let id1 = calc.calculate(
            Path::new("/home/user/work/project"),
            Some("https://github.com/user/repo.git"),
            Some("work/project"),
        );

        let id2 = calc.calculate(
            Path::new("/home/user/personal/project"),
            Some("https://github.com/user/repo.git"),
            Some("personal/project"),
        );

        // Same remote but different disambiguation paths should produce different IDs
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_same_remote_same_id_without_disambiguation() {
        let calc = ProjectIdCalculator::new();

        let id1 = calc.calculate(
            Path::new("/home/user/work/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        let id2 = calc.calculate(
            Path::new("/home/user/personal/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        // Same remote without disambiguation produces same ID (the problem we're solving)
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_disambiguation_path_compute_single() {
        let new_path = Path::new("/home/user/work/project");
        let existing: Vec<PathBuf> = vec![];

        let disambig = DisambiguationPathComputer::compute(new_path, &existing);
        assert!(disambig.is_empty());
    }

    #[test]
    fn test_disambiguation_path_compute_two_clones() {
        let new_path = Path::new("/home/user/work/project");
        let existing = vec![PathBuf::from("/home/user/personal/project")];

        let disambig = DisambiguationPathComputer::compute(new_path, &existing);
        assert_eq!(disambig, "work/project");
    }

    #[test]
    fn test_disambiguation_path_compute_three_clones() {
        let new_path = Path::new("/home/user/test/project");
        let existing = vec![
            PathBuf::from("/home/user/work/project"),
            PathBuf::from("/home/user/personal/project"),
        ];

        let disambig = DisambiguationPathComputer::compute(new_path, &existing);
        assert_eq!(disambig, "test/project");
    }

    #[test]
    fn test_recompute_all_disambiguation() {
        let paths = vec![
            PathBuf::from("/home/user/work/project"),
            PathBuf::from("/home/user/personal/project"),
        ];

        let result = DisambiguationPathComputer::recompute_all(&paths);

        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&paths[0]).unwrap(), "work/project");
        assert_eq!(result.get(&paths[1]).unwrap(), "personal/project");
    }

    #[test]
    fn test_remote_hash_grouping() {
        let calc = ProjectIdCalculator::new();

        // Different URL formats for same repo should produce same hash
        let hash1 = calc.calculate_remote_hash("https://github.com/user/repo.git");
        let hash2 = calc.calculate_remote_hash("git@github.com:user/repo.git");
        let hash3 = calc.calculate_remote_hash("http://GITHUB.COM/User/Repo");

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

}
