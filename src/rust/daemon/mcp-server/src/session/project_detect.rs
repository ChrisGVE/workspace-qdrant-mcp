//! Project detection: git root, branch, remote URL, and project ID.
//!
//! Mirrors the logic in:
//! - `src/typescript/mcp-server/src/utils/project-detector.ts` — root walk + project_id from SQLite
//! - `src/typescript/mcp-server/src/utils/git-utils.ts` — `findGitRoot`, `getGitRemoteUrl`
//! - `src/typescript/mcp-server/src/utils/git-branch.ts` — `detectCurrentBranch`
//!
//! # Design
//! All functions are pure or take explicit inputs for testability.
//! The `detect_project` top-level function composes them and queries
//! the daemon's SQLite database for the registered project ID (matching
//! TS `ProjectDetector.getProjectInfo` behaviour — project_id is NOT
//! calculated locally to avoid drift).
//!
//! # TS divergence
//! - `detect_git_remote_url` reads `.git/config` directly (matching TS `getGitRemoteUrl`),
//!   not via subprocess.  This differs from `wqm_common::detect_git_remote` which shells
//!   out to `git`.

use std::fs;
use std::path::{Path, PathBuf};

use crate::sqlite::manager::StateManager;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Information about the project detected at session start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectInfo {
    /// Project root directory (canonical `.git` parent).
    pub project_path: PathBuf,
    /// Project ID registered with the daemon, if the project is known to SQLite.
    pub project_id: Option<String>,
    /// Git remote URL (`origin` or first remote), if available.
    pub git_remote: Option<String>,
    /// Current git branch (`"default"` when not in a branch or not a git repo).
    pub branch: String,
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
/// 2. Find git root (for branch + remote; may differ from project root in
///    monorepos).
/// 3. Read branch from `.git/HEAD`.
/// 4. Read git remote URL from `.git/config`.
/// 5. Look up project ID in the daemon's SQLite database.
///
/// Returns `Some(ProjectInfo)` when a project root is found; `None` when
/// `cwd` is not inside any known project directory structure.
///
/// Mirrors the logic in `detectProjectForSession` + `ProjectDetector.getProjectInfo`.
pub fn detect_project(cwd: &Path, state_manager: &StateManager) -> Option<ProjectInfo> {
    let project_root = find_project_root(cwd).unwrap_or_else(|| cwd.to_path_buf());

    let git_remote = get_git_remote_url(&project_root);
    let branch = detect_branch(&project_root);

    // Look up project ID from daemon's SQLite (longest-prefix match).
    let project_id = lookup_project_id(state_manager, &project_root);

    Some(ProjectInfo {
        project_path: project_root,
        project_id,
        git_remote,
        branch,
    })
}

/// Look up the registered project ID for a path from the daemon's SQLite.
///
/// Uses the `watch_folders` table (collection = 'projects'), longest-prefix
/// match. Returns `None` when the database is degraded or no match found.
///
/// Exposed `pub` so the search tool can resolve a tenant from cwd without the
/// git-remote / branch work that `detect_project` also performs (GitHub #83).
pub fn lookup_project_id(state_manager: &StateManager, project_root: &Path) -> Option<String> {
    let conn = state_manager.connection()?;
    let path_str = project_root.to_str()?;

    let sql = "SELECT tenant_id FROM watch_folders \
               WHERE collection = 'projects' \
                 AND (? = path OR ? LIKE path || '/' || '%') \
               ORDER BY length(path) DESC \
               LIMIT 1";

    conn.query_row(sql, rusqlite::params![path_str, path_str], |row| {
        row.get::<_, String>(0)
    })
    .ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "project_detect_tests.rs"]
mod tests;
