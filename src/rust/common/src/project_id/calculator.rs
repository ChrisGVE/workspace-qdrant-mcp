//! Project ID calculator with git URL normalization and hashing

use std::path::Path;
use sha2::{Sha256, Digest};
use super::types::DisambiguationConfig;

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
    /// All of these normalize to "github.com/user/repo":
    /// - `https://github.com/user/repo.git`
    /// - `git@github.com:user/repo.git`
    /// - `ssh://git@github.com/user/repo`
    /// - `http://github.com/user/repo`
    pub fn normalize_git_url(url: &str) -> String {
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
    pub fn hash_to_id(&self, input: &str) -> String {
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
