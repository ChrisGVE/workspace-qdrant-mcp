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
///   1. Locate `.git` inside `project_root`. When it is a FILE (submodule /
///      linked-worktree checkout) follow its `gitdir: <path>` pointer to the
///      real git directory (#99 — submodule checkouts like
///      `~/.config/main-docker` resolved to `"default"` without this).
///   2. Read `HEAD`; `ref: refs/heads/<name>` → return `<name>`.
///   3. If a bare 40-char SHA (detached HEAD) → return first 8 chars.
///   4. Otherwise → return `"default"`.
///
/// Never panics; returns `"default"` on any I/O error.
///
/// Mirrors `detectCurrentBranch` in `git-branch.ts`.
pub fn detect_branch(project_root: &Path) -> String {
    let git_root = find_git_root(project_root).unwrap_or_else(|| project_root.to_path_buf());
    let dot_git = git_root.join(".git");

    let git_dir = if dot_git.is_file() {
        match fs::read_to_string(&dot_git) {
            Ok(content) => match content.trim().strip_prefix("gitdir:") {
                Some(target) => {
                    let target = Path::new(target.trim());
                    if target.is_absolute() {
                        target.to_path_buf()
                    } else {
                        git_root.join(target)
                    }
                }
                None => return "default".to_string(),
            },
            Err(_) => return "default".to_string(),
        }
    } else {
        dot_git
    };

    let head_path = git_dir.join("HEAD");

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
                let url = url.trim();
                if !url.is_empty() {
                    // Credentials in the URL userinfo must never leave the
                    // read boundary (#126).
                    return Some(wqm_common::git_url::sanitize_git_remote_url(url));
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
    // Fold a Windows-host WSL UNC cwd (`\\wsl.localhost\<distro>\...`) to the
    // POSIX path the daemon registered; a no-op for native paths (#134 salvage).
    let cwd_str = cwd.to_string_lossy();
    let folded = wqm_common::paths::canonicalize_host_path(&cwd_str);
    let cwd: &Path = Path::new(&folded);

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

    /// #99: submodule / linked-worktree checkouts have a `.git` FILE with a
    /// `gitdir:` pointer — detect_branch must follow it (relative form).
    #[test]
    fn detect_branch_follows_gitfile_pointer() {
        let dir = tmp("branch_gitfile");
        // super/.git/modules/sub holds the real git dir.
        let modules = dir.join(".git").join("modules").join("sub");
        fs::create_dir_all(&modules).unwrap();
        fs::write(modules.join("HEAD"), "ref: refs/heads/feature-y\n").unwrap();
        // super/sub/.git is a gitfile pointing back into the superproject.
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join(".git"), "gitdir: ../.git/modules/sub\n").unwrap();
        assert_eq!(detect_branch(&sub), "feature-y");
        let _ = fs::remove_dir_all(&dir);
    }

    /// #99: absolute `gitdir:` pointers (linked worktrees) are honoured too.
    #[test]
    fn detect_branch_follows_absolute_gitfile_pointer() {
        let dir = tmp("branch_gitfile_abs");
        let real = dir.join("real-gitdir");
        fs::create_dir_all(&real).unwrap();
        fs::write(real.join("HEAD"), "ref: refs/heads/wt-branch\n").unwrap();
        let wt = dir.join("worktree");
        fs::create_dir_all(&wt).unwrap();
        fs::write(wt.join(".git"), format!("gitdir: {}\n", real.display())).unwrap();
        assert_eq!(detect_branch(&wt), "wt-branch");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A malformed `.git` file (no `gitdir:` prefix) degrades to "default".
    #[test]
    fn detect_branch_malformed_gitfile_is_default() {
        let dir = tmp("branch_gitfile_bad");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".git"), "not a gitdir pointer\n").unwrap();
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

    /// #126: URL credentials never leave the read boundary.
    #[test]
    fn get_git_remote_url_strips_credentials() {
        let dir = tmp("remote_credentials");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(
            dir.join(".git").join("config"),
            "[remote \"origin\"]\n\turl = https://x-access-token:ghp_secret@example.com/org/repo.git\n",
        )
        .unwrap();
        assert_eq!(
            get_git_remote_url(&dir).as_deref(),
            Some("https://example.com/org/repo.git")
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
