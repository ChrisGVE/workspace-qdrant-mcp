//! Project ID calculation and disambiguation
//!
//! Provides the canonical project ID calculation algorithm used by both
//! the daemon and CLI to ensure consistent tenant IDs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

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

/// Computes the disambiguation path for a new project
///
/// Given a new project path and existing projects with the same remote,
/// finds the first differing directory component from the common ancestor.
pub struct DisambiguationPathComputer;

impl DisambiguationPathComputer {
    /// Compute disambiguation path for a new project
    pub fn compute(new_path: &Path, existing_paths: &[PathBuf]) -> String {
        if existing_paths.is_empty() {
            return String::new();
        }

        let new_components: Vec<_> = new_path.components().collect();
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

        if min_common_idx < new_components.len() {
            new_components[min_common_idx..]
                .iter()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/")
        } else {
            new_path.to_string_lossy().to_string()
        }
    }

    /// Recompute disambiguation paths for all clones of a repository
    pub fn recompute_all(paths: &[PathBuf]) -> HashMap<PathBuf, String> {
        let mut result = HashMap::new();

        if paths.len() <= 1 {
            for path in paths {
                result.insert(path.clone(), String::new());
            }
            return result;
        }

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

/// Detect git remote URL for a project using `git` CLI
///
/// Tries `origin` first, falls back to `upstream`. Returns `None` on failure.
pub fn detect_git_remote(project_root: &Path) -> Option<String> {
    for remote_name in &["origin", "upstream"] {
        if let Ok(output) = std::process::Command::new("git")
            .args(["-C", &project_root.to_string_lossy(), "remote", "get-url", remote_name])
            .output()
        {
            if output.status.success() {
                let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !url.is_empty() {
                    return Some(url);
                }
            }
        }
    }
    None
}

/// Calculate tenant ID for a project path (convenience function)
///
/// Combines `detect_git_remote()` + `ProjectIdCalculator::new().calculate()`.
pub fn calculate_tenant_id(project_root: &Path) -> String {
    let git_remote = detect_git_remote(project_root);
    let calculator = ProjectIdCalculator::new();
    calculator.calculate(project_root, git_remote.as_deref(), None)
}

/// Resolve a working directory to a registered project.
///
/// Looks up the `watch_folders` table for the longest matching path where
/// `cwd` equals or is a subdirectory of a registered project. Returns
/// `(tenant_id, path)` on success, or `None` if no match or on any error.
///
/// Opens the database read-only. Any failure (missing db, missing table,
/// query error) returns `None` silently — callers should degrade gracefully.
#[cfg(feature = "sqlite")]
pub fn resolve_path_to_project(db_path: &Path, cwd: &Path) -> Option<(String, String)> {
    use crate::schema::sqlite::watch_folders as wf;

    let cwd_str = cwd.to_str()?;

    let conn = rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;

    let sql = format!(
        "SELECT {tenant}, {path} FROM {table} \
         WHERE {collection} = 'projects' \
           AND (?1 = {path} OR ?1 LIKE {path} || '/' || '%') \
         ORDER BY length({path}) DESC \
         LIMIT 1",
        tenant = wf::TENANT_ID.name,
        path = wf::PATH.name,
        table = wf::TABLE.name,
        collection = wf::COLLECTION.name,
    );

    let mut stmt = conn.prepare(&sql).ok()?;
    stmt.query_row(rusqlite::params![cwd_str], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })
    .ok()
}

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

        let hash1 = calc.calculate_remote_hash("https://github.com/user/repo.git");
        let hash2 = calc.calculate_remote_hash("git@github.com:user/repo.git");
        let hash3 = calc.calculate_remote_hash("http://GITHUB.COM/User/Repo");

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    // ─── resolve_path_to_project tests ──────────────────────────────────

    /// Helper: create a SQLite database with watch_folders table and rows
    #[cfg(feature = "sqlite")]
    fn setup_test_db(rows: &[(&str, &str)]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("state.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                tenant_id TEXT NOT NULL,
                path TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'projects'
            )",
        )
        .unwrap();
        for (tenant, path) in rows {
            conn.execute(
                "INSERT INTO watch_folders (tenant_id, path, collection) VALUES (?1, ?2, 'projects')",
                rusqlite::params![tenant, path],
            )
            .unwrap();
        }
        (dir, db_path)
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_exact() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_abc", "/home/user/project-a"),
        ]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a"));
        assert_eq!(result, Some(("tid_abc".into(), "/home/user/project-a".into())));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_subdirectory() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_abc", "/home/user/project-a"),
        ]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a/src/lib"));
        assert_eq!(result, Some(("tid_abc".into(), "/home/user/project-a".into())));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_longest_wins() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_parent", "/home/user"),
            ("tid_child", "/home/user/project-a"),
        ]);
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-a/src"));
        assert_eq!(result, Some(("tid_child".into(), "/home/user/project-a".into())));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_no_match() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_abc", "/home/user/project-a"),
        ]);
        let result = resolve_path_to_project(&db_path, Path::new("/other/dir"));
        assert_eq!(result, None);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_no_false_prefix() {
        let (_dir, db_path) = setup_test_db(&[
            ("tid_abc", "/home/user/project"),
        ]);
        // "/home/user/project-extra" should NOT match "/home/user/project"
        let result = resolve_path_to_project(&db_path, Path::new("/home/user/project-extra"));
        assert_eq!(result, None);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_resolve_path_missing_db() {
        let result = resolve_path_to_project(
            Path::new("/nonexistent/state.db"),
            Path::new("/home/user/project"),
        );
        assert_eq!(result, None);
    }
}
