use git2::Repository;
use std::path::Path;
use thiserror::Error;

/// Git integration errors
#[derive(Error, Debug)]
pub enum GitError {
    #[error("Not a Git repository: {path}")]
    NotARepository { path: String },

    #[error("Git repository error: {message}")]
    RepositoryError {
        message: String,
        #[source]
        source: git2::Error,
    },

    #[error("Detached HEAD state in repository: {path}")]
    DetachedHead { path: String },

    #[error("Permission denied accessing repository: {path}")]
    PermissionDenied { path: String },

    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

/// Result type for Git operations
pub type GitResult<T> = Result<T, GitError>;

/// Git status for a project directory, detected at registration time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitStatus {
    /// Whether the path is inside a git repository.
    pub is_git: bool,
    /// Current branch name ("HEAD" for detached HEAD, "default" for non-git).
    pub branch: String,
    /// SHA of the HEAD commit (None for non-git or empty repos).
    pub commit_hash: Option<String>,
    /// Whether this is a git worktree (not the main working tree).
    pub is_worktree: bool,
}

impl GitStatus {
    /// Create a non-git status.
    pub fn not_git() -> Self {
        Self {
            is_git: false,
            branch: "default".to_string(),
            commit_hash: None,
            is_worktree: false,
        }
    }
}

/// Detect git status for a project directory.
///
/// Determines whether the directory is inside a git repository, retrieves the
/// current branch name and HEAD commit hash. Handles worktrees, detached HEAD,
/// bare repos, and empty repos gracefully.
///
/// This is a one-time synchronous check intended for project registration.
pub fn detect_git_status(project_root: &Path) -> GitStatus {
    let git_indicator = project_root.join(".git");
    if !git_indicator.exists() {
        return GitStatus::not_git();
    }

    // .git file (not directory) indicates a worktree
    let is_worktree = git_indicator.is_file();

    let repo = match Repository::open(project_root) {
        Ok(r) => r,
        Err(_) => return GitStatus::not_git(),
    };

    // Bare repos have no working tree -- treat as not-git for file watching
    if repo.is_bare() {
        return GitStatus::not_git();
    }

    let head = match repo.head() {
        Ok(h) => h,
        Err(e) if e.code() == git2::ErrorCode::UnbornBranch => {
            // Empty repo with no commits yet
            return GitStatus {
                is_git: true,
                branch: "main".to_string(),
                commit_hash: None,
                is_worktree,
            };
        }
        Err(_) => {
            // HEAD is unreadable -- still a git repo
            return GitStatus {
                is_git: true,
                branch: "HEAD".to_string(),
                commit_hash: None,
                is_worktree,
            };
        }
    };

    let commit_hash = head.target().map(|oid| oid.to_string());

    let branch = if head.is_branch() {
        head.shorthand().unwrap_or("HEAD").to_string()
    } else {
        // Detached HEAD -- use short SHA or "HEAD"
        commit_hash
            .as_ref()
            .map(|h| {
                if h.len() >= 8 {
                    h[..8].to_string()
                } else {
                    h.clone()
                }
            })
            .unwrap_or_else(|| "HEAD".to_string())
    };

    GitStatus {
        is_git: true,
        branch,
        commit_hash,
        is_worktree,
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cache entries
    pub total_entries: usize,
    /// Number of valid (non-expired) entries
    pub valid_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper to create a Git repository with initial commit
    fn create_test_repo(path: &Path) -> Result<Repository, git2::Error> {
        let repo = Repository::init(path)?;

        let sig = git2::Signature::now("Test User", "test@example.com")?;
        let tree_id = {
            let mut index = repo.index()?;
            index.write_tree()?
        };
        {
            let tree = repo.find_tree(tree_id)?;
            repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])?;
        }

        Ok(repo)
    }

    #[test]
    fn test_detect_git_status_git_repo() {
        let tmp = tempdir().unwrap();
        let _repo = create_test_repo(tmp.path()).unwrap();

        let status = detect_git_status(tmp.path());
        assert!(status.is_git);
        assert!(!status.branch.is_empty());
        assert!(status.commit_hash.is_some());
        assert!(!status.is_worktree);
    }

    #[test]
    fn test_detect_git_status_non_git() {
        let tmp = tempdir().unwrap();
        let status = detect_git_status(tmp.path());
        assert!(!status.is_git);
        assert_eq!(status.branch, "default");
        assert!(status.commit_hash.is_none());
        assert!(!status.is_worktree);
    }

    #[test]
    fn test_detect_git_status_not_git_struct() {
        let status = GitStatus::not_git();
        assert!(!status.is_git);
        assert_eq!(status.branch, "default");
        assert!(status.commit_hash.is_none());
    }

    #[test]
    fn test_detect_git_status_empty_repo() {
        let tmp = tempdir().unwrap();
        let _repo = Repository::init(tmp.path()).unwrap();

        let status = detect_git_status(tmp.path());
        assert!(status.is_git);
        assert_eq!(status.branch, "main");
        assert!(status.commit_hash.is_none());
    }

    #[test]
    fn test_detect_git_status_bare_repo() {
        let tmp = tempdir().unwrap();
        let _repo = Repository::init_bare(tmp.path()).unwrap();

        let status = detect_git_status(tmp.path());
        assert!(!status.is_git);
    }

    #[test]
    fn test_detect_git_status_detached_head() {
        let tmp = tempdir().unwrap();
        let repo = create_test_repo(tmp.path()).unwrap();

        let head = repo.head().unwrap();
        let oid = head.target().unwrap();
        repo.set_head_detached(oid).unwrap();

        let status = detect_git_status(tmp.path());
        assert!(status.is_git);
        assert!(status.commit_hash.is_some());
        assert!(status.branch.len() == 8 || status.branch == "HEAD");
    }
}
