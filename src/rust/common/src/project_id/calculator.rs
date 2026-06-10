//! Project ID calculator with git URL normalization and hashing

use super::types::DisambiguationConfig;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::paths::CanonicalPath;

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
    ///    - hash(syntactic-canonical project_root_path)
    ///
    /// Per spec §16 (path-abstraction §3.2.3) the local-project path is
    /// no longer resolved through `std::fs::canonicalize` — it is reduced
    /// to its syntactic canonical form (rules in §3.1). For projects whose
    /// root is a symlink, this changes the resulting project_id; the
    /// pre-release "NO MIGRATION EFFORT" policy makes that acceptable.
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
            // Local project — hash the syntactic-canonical form of the
            // project root. Fall back to a lossy stringification only when
            // canonical normalization is impossible (non-UTF-8, etc.) so
            // that we still emit *some* deterministic ID; this is the
            // documented fallback behavior of the legacy implementation.
            let path_str = path_to_syntactic_canonical(project_root);

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
        let mut had_scheme = false;
        for protocol in &["https://", "http://", "ssh://", "git://"] {
            if normalized.starts_with(protocol) {
                normalized = normalized[protocol.len()..].to_string();
                had_scheme = true;
                break;
            }
        }

        // Strip any userinfo (`user[:password]@`) so credentials never enter
        // the hash input and credential rotation cannot change the tenant id
        // (#126). This also covers the plain `git@` SSH prefix.
        let path_start = normalized.find('/').unwrap_or(normalized.len());
        if let Some(at) = normalized[..path_start].rfind('@') {
            normalized = normalized[at + 1..].to_string();
            // scp-like syntax (user@host:path): replace the first : with /
            if !had_scheme {
                if let Some(idx) = normalized.find(':') {
                    normalized.replace_range(idx..idx + 1, "/");
                }
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

/// Reduce a [`Path`] to its syntactic-canonical UTF-8 string per spec §3.1.
///
/// Relative inputs are absolutized against CWD purely syntactically. No
/// `std::fs::canonicalize`. Returns the lossy stringification as a last-
/// resort fallback so the caller always receives a deterministic value.
fn path_to_syntactic_canonical(path: &Path) -> String {
    if let Some(s) = path.to_str() {
        if let Ok(cp) = CanonicalPath::from_user_input(s) {
            return cp.into_string();
        }
        if let Ok(cwd) = std::env::current_dir() {
            let joined = cwd.join(path);
            if let Some(joined_str) = joined.to_str() {
                if let Ok(cp) = CanonicalPath::from_user_input(joined_str) {
                    return cp.into_string();
                }
            }
        }
    }
    path.to_string_lossy().to_string()
}
