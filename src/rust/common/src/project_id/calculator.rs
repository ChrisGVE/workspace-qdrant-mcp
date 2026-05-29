//! Project ID calculator with git URL normalization and hashing

use super::types::DisambiguationConfig;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::paths::{canonicalize_docker_mount_alias, CanonicalPath};

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
    ///    - Rewrite well-known Docker Desktop mount aliases (e.g.
    ///      `/run/desktop/mnt/host/c/...` and `/host_mnt/c/...`) back to
    ///      the canonical host form `/mnt/c/...` so the same physical
    ///      project hashes to the same tenant_id whether it is registered
    ///      from the WSL2 host or from inside a Docker Desktop container.
    ///      See `paths/docker_alias.rs` for the recognized prefix families.
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
            // Local project. Two steps:
            //
            // 1. Rewrite Docker Desktop mount aliases back to host form so
            //    a project registered via `/mnt/c/...` (WSL2 host) and the
            //    same project registered via `/run/desktop/mnt/host/c/...`
            //    or `/host_mnt/c/...` (inside a Docker Desktop container)
            //    converge on a single tenant_id. Without this step we
            //    create duplicate watch_folders rows that index every file
            //    twice — see issue #8 unit 1.
            // 2. Reduce to the syntactic-canonical UTF-8 form (spec §3.1).
            //    Fall back to lossy stringification when canonicalization
            //    is impossible so we still emit *some* deterministic ID.
            let aliased: PathBuf = path_str(project_root)
                .as_deref()
                .and_then(canonicalize_docker_mount_alias)
                .map(PathBuf::from)
                .unwrap_or_else(|| project_root.to_path_buf());

            let canonical = path_to_syntactic_canonical(&aliased);

            format!("local_{}", self.hash_to_id(&canonical))
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

/// Return the path's UTF-8 string view, or `None` when the path contains
/// non-UTF-8 bytes. This is a thin wrapper around [`Path::to_str`] kept here
/// so the call site reads as a step in the alias-then-canonicalize pipeline.
fn path_str(path: &Path) -> Option<String> {
    path.to_str().map(|s| s.to_string())
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
