//! Project detection: git root, branch, remote URL, and project ID (WI-d5, #82).
//!
//! The pure filesystem detection (root walk, branch, remote) lives here so both
//! clients share one implementation. The registered-project-id lookup is
//! abstracted behind the [`ProjectLookup`] trait: the MCP server implements it
//! over its SQLite `StateManager`; a CLI could implement it over a read-only
//! connection or a daemon gRPC call. wqm-client itself stays SQLite-free.
//!
//! Mirrors the TS logic in `utils/project-detector.ts`, `utils/git-utils.ts`,
//! and `utils/git-branch.ts`. The project id is looked up (never calculated
//! locally) to avoid drift with the daemon's registration.

use std::fs;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Information about the project detected at session start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectInfo {
    /// Project root directory (canonical `.git` parent).
    pub project_path: PathBuf,
    /// Project ID registered with the daemon, if the project is known.
    pub project_id: Option<String>,
    /// Git remote URL (`origin` or first remote), if available.
    pub git_remote: Option<String>,
    /// Current git branch (`"default"` when not in a branch or not a git repo).
    pub branch: String,
}

/// Abstracts the registered-project-id lookup so the SQLite read stays in the
/// consuming client (keeping wqm-client SQLite-free).
pub trait ProjectLookup {
    /// Return the registered project/tenant id for `path` (longest-prefix
    /// match), or `None` when unknown or the store is degraded.
    fn lookup_project_id(&self, path: &Path) -> Option<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Git root detection
// ─────────────────────────────────────────────────────────────────────────────

/// Project marker files/directories — matches TS `PROJECT_MARKERS`.
const PROJECT_MARKERS: &[&str] = &[
    ".git",
    "package.json",
    "Cargo.toml",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Makefile",
    ".workspace-qdrant",
];

/// Maximum directory levels to search upward — matches TS `MAX_SEARCH_DEPTH`.
const MAX_SEARCH_DEPTH: usize = 20;

/// Find the git root by walking up from `start`.
///
/// Looks for a `.git` entry (directory or file for worktrees).
/// Returns the first ancestor that contains `.git`, or `None`.
///
/// Mirrors `findGitRoot` in `git-utils.ts`.
pub fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    for _ in 0..MAX_SEARCH_DEPTH {
        if current.join(".git").exists() {
            return Some(current);
        }
        let parent = match current.parent() {
            Some(p) if p != current => p.to_path_buf(),
            _ => break,
        };
        current = parent;
    }
    None
}

/// Find the project root by walking up from `start` looking for any marker.
///
/// Mirrors `ProjectDetector.findProjectRoot` in `project-detector.ts`.
/// Falls back to the directory itself if no markers are found within
/// `MAX_SEARCH_DEPTH`.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    for _ in 0..MAX_SEARCH_DEPTH {
        if has_project_marker(&current) {
            return Some(current);
        }
        let parent = match current.parent() {
            Some(p) if p != current => p.to_path_buf(),
            _ => break,
        };
        current = parent;
    }
    None
}

fn has_project_marker(dir: &Path) -> bool {
    PROJECT_MARKERS.iter().any(|m| dir.join(m).exists())
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the current git branch for a project root.
///
/// Algorithm:
///   1. Locate `.git/HEAD` inside `project_root`.
///   2. If `ref: refs/heads/<name>` → return `<name>`.
///   3. If a bare 40-char SHA (detached HEAD) → return first 8 chars.
///   4. Otherwise → return `"default"`.
///
/// Never panics; returns `"default"` on any I/O error.
///
/// Mirrors `detectCurrentBranch` in `git-branch.ts`.
pub fn detect_branch(project_root: &Path) -> String {
    let git_root = find_git_root(project_root).unwrap_or_else(|| project_root.to_path_buf());
    let head_path = git_root.join(".git").join("HEAD");

    let content = match fs::read_to_string(&head_path) {
        Ok(c) => c.trim().to_string(),
        Err(_) => return "default".to_string(),
    };

    // Symbolic ref — normal branch checkout
    if let Some(branch) = content.strip_prefix("ref: refs/heads/") {
        return branch.trim().to_string();
    }

    // Detached HEAD — bare 40-char hex SHA
    if content.len() == 40 && content.chars().all(|c| c.is_ascii_hexdigit()) {
        return content[..8].to_string();
    }

    "default".to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Git remote URL
// ─────────────────────────────────────────────────────────────────────────────

/// Read the `origin` remote URL from `.git/config`.
///
/// Returns `None` on any I/O error or if no `origin` remote is configured.
///
/// Mirrors `getGitRemoteUrl` in `git-utils.ts` — reads the file directly,
/// no subprocess.
pub fn get_git_remote_url(repo_root: &Path) -> Option<String> {
    let config_path = repo_root.join(".git").join("config");
    let config = fs::read_to_string(config_path).ok()?;

    // Match `[remote "origin"]` section and extract the `url = ` line.
    // Regex equivalent: /\[remote "origin"\][^\[]*url = (.+)/m
    let mut in_origin = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed == r#"[remote "origin"]"# {
            in_origin = true;
            continue;
        }
        if in_origin {
            if trimmed.starts_with('[') {
                break; // new section
            }
            if let Some(url) = trimmed.strip_prefix("url = ") {
                let url = url.trim().to_string();
                if !url.is_empty() {
                    return Some(url);
                }
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level project detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the project for the given working directory.
///
/// 1. Find project root (walk up for any marker).
/// 2. Read branch from `.git/HEAD` and remote URL from `.git/config`.
/// 3. Look up the registered project id via `lookup` — by **cwd directly**
///    (longest-prefix match), NOT by the marker-derived `project_root`. A
///    registered project path need not contain a filesystem marker, so resolving
///    via `project_root` can skip a deeper markerless registered project and
///    return the wrong (ancestor) tenant. Mirrors TS `getCurrentProject(cwd)`
///    and keeps `project_id` consistent with the search tool's cwd-direct
///    resolution (GitHub #83/#84). `project_path`/`branch` still use the marker
///    root — those drive registration/watching.
///
/// Returns `Some(ProjectInfo)` for any `cwd` (project root falls back to `cwd`).
pub fn detect_project<L: ProjectLookup + ?Sized>(cwd: &Path, lookup: &L) -> Option<ProjectInfo> {
    let project_root = find_project_root(cwd).unwrap_or_else(|| cwd.to_path_buf());

    let git_remote = get_git_remote_url(&project_root);
    let branch = detect_branch(&project_root);
    let project_id = lookup.lookup_project_id(cwd);

    Some(ProjectInfo {
        project_path: project_root,
        project_id,
        git_remote,
        branch,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory [`ProjectLookup`] for hermetic detection tests.
    struct MockLookup(HashMap<PathBuf, String>);

    impl ProjectLookup for MockLookup {
        fn lookup_project_id(&self, path: &Path) -> Option<String> {
            self.0.get(path).cloned()
        }
    }

    /// Unique temp dir per test (named after the caller) — avoids collisions
    /// when tests run in parallel, without a `tempfile` dev-dependency.
    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("wqmclient_project_{name}"))
    }

    #[test]
    fn find_git_root_finds_immediate_dir() {
        let dir = tmp("git_root_immediate");
        fs::create_dir_all(dir.join(".git")).unwrap();
        assert_eq!(find_git_root(&dir).as_deref(), Some(dir.as_path()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_branch_symbolic_ref() {
        let dir = tmp("branch_symbolic");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main\n").unwrap();
        assert_eq!(detect_branch(&dir), "main");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_branch_detached_head_first_8() {
        let dir = tmp("branch_detached");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(
            dir.join(".git").join("HEAD"),
            "0123456789abcdef0123456789abcdef01234567\n",
        )
        .unwrap();
        assert_eq!(detect_branch(&dir), "01234567");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_branch_no_git_is_default() {
        let dir = tmp("branch_no_git");
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(detect_branch(&dir), "default");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_git_remote_url_returns_origin() {
        let dir = tmp("remote_origin");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(
            dir.join(".git").join("config"),
            "[remote \"origin\"]\n\turl = https://example.com/repo.git\n",
        )
        .unwrap();
        assert_eq!(
            get_git_remote_url(&dir).as_deref(),
            Some("https://example.com/repo.git")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_project_uses_lookup_for_id_by_cwd() {
        let dir = tmp("detect_proj_lookup");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/dev\n").unwrap();
        fs::write(dir.join("Cargo.toml"), "[package]\n").unwrap();

        let mut map = HashMap::new();
        map.insert(dir.clone(), "tenant-abc".to_string());
        let lookup = MockLookup(map);

        let info = detect_project(&dir, &lookup).unwrap();
        assert_eq!(info.project_path, dir);
        assert_eq!(info.project_id.as_deref(), Some("tenant-abc"));
        assert_eq!(info.branch, "dev");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_project_none_id_when_lookup_empty() {
        let dir = tmp("detect_proj_none");
        fs::create_dir_all(dir.join(".git")).unwrap();
        let lookup = MockLookup(HashMap::new());
        let info = detect_project(&dir, &lookup).unwrap();
        assert!(info.project_id.is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
