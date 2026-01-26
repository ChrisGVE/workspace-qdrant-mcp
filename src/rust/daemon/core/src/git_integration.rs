//! Git Integration Module
//!
//! Provides Git branch detection and branch switching detection for the daemon.
//! Uses git2-rs for native Git operations without subprocess overhead.

use git2::Repository;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

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

/// Cached branch information with TTL
#[derive(Debug, Clone)]
struct CachedBranch {
    /// Branch name (e.g., "main", "feature/auth")
    branch_name: String,
    /// Last time this cache entry was checked
    last_checked: Instant,
    /// Time-to-live for this cache entry
    ttl: Duration,
}

impl CachedBranch {
    /// Create a new cached branch entry
    fn new(branch_name: String, ttl: Duration) -> Self {
        Self {
            branch_name,
            last_checked: Instant::now(),
            ttl,
        }
    }

    /// Check if this cache entry is still valid
    fn is_valid(&self) -> bool {
        self.last_checked.elapsed() < self.ttl
    }
}

/// Git branch detector with caching
#[derive(Clone)]
pub struct GitBranchDetector {
    /// Cache of repository path -> branch info
    cache: Arc<RwLock<HashMap<PathBuf, CachedBranch>>>,
    /// Default TTL for cache entries
    default_ttl: Duration,
}

impl GitBranchDetector {
    /// Create a new GitBranchDetector with default TTL (60 seconds)
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(60))
    }

    /// Create a new GitBranchDetector with custom TTL
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: ttl,
        }
    }

    /// Get the current branch name for a repository
    ///
    /// This function first checks the cache. If the cache entry is valid,
    /// it returns the cached value. Otherwise, it queries Git and updates the cache.
    ///
    /// # Arguments
    /// * `repo_path` - Path to a directory within a Git repository
    ///
    /// # Returns
    /// Branch name (e.g., "main", "feature/auth")
    ///
    /// # Errors
    /// - `GitError::NotARepository` if path is not in a Git repository
    /// - `GitError::DetachedHead` if repository is in detached HEAD state
    /// - `GitError::PermissionDenied` if permission error occurs
    /// - `GitError::RepositoryError` for other Git errors
    pub async fn get_current_branch(&self, repo_path: &Path) -> GitResult<String> {
        // Canonicalize path to handle symlinks and relative paths
        let canonical_path = repo_path
            .canonicalize()
            .map_err(|e| GitError::InvalidPath(format!("{}: {}", repo_path.display(), e)))?;

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&canonical_path) {
                if cached.is_valid() {
                    debug!(
                        "Cache hit for branch detection: {} -> {}",
                        canonical_path.display(),
                        cached.branch_name
                    );
                    return Ok(cached.branch_name.clone());
                }
            }
        }

        // Cache miss or expired - query Git
        debug!(
            "Cache miss for branch detection, querying Git: {}",
            canonical_path.display()
        );
        let branch_name = self.query_git_branch(&canonical_path)?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                canonical_path.clone(),
                CachedBranch::new(branch_name.clone(), self.default_ttl),
            );
        }

        Ok(branch_name)
    }

    /// Query Git directly for the current branch
    fn query_git_branch(&self, repo_path: &Path) -> GitResult<String> {
        // Open repository - discover searches parent directories
        let repo = Repository::discover(repo_path).map_err(|e| {
            if e.code() == git2::ErrorCode::NotFound {
                GitError::NotARepository {
                    path: repo_path.display().to_string(),
                }
            } else if e.code() == git2::ErrorCode::Auth {
                GitError::PermissionDenied {
                    path: repo_path.display().to_string(),
                }
            } else {
                GitError::RepositoryError {
                    message: format!("Failed to open repository at {}", repo_path.display()),
                    source: e,
                }
            }
        })?;

        // Get HEAD reference
        let head = repo.head().map_err(|e| {
            if e.code() == git2::ErrorCode::UnbornBranch {
                // Repository exists but no commits yet - return "main" as default
                return GitError::RepositoryError {
                    message: "Repository has no commits yet".to_string(),
                    source: e,
                };
            }
            GitError::RepositoryError {
                message: "Failed to read HEAD".to_string(),
                source: e,
            }
        })?;

        // Check if HEAD is detached
        if !head.is_branch() {
            return Err(GitError::DetachedHead {
                path: repo_path.display().to_string(),
            });
        }

        // Get branch name
        let branch_name = head
            .shorthand()
            .ok_or_else(|| GitError::RepositoryError {
                message: "Failed to get branch name from HEAD".to_string(),
                source: git2::Error::from_str("Invalid UTF-8 in branch name"),
            })?
            .to_string();

        debug!(
            "Detected branch '{}' for repository at {}",
            branch_name,
            repo_path.display()
        );

        Ok(branch_name)
    }

    /// Detect if the branch has changed from a known previous value
    ///
    /// # Arguments
    /// * `repo_path` - Path to a directory within a Git repository
    /// * `last_known_branch` - Previously known branch name
    ///
    /// # Returns
    /// - `Ok(Some(new_branch))` if branch has changed
    /// - `Ok(None)` if branch is unchanged
    /// - `Err(GitError)` if an error occurred
    pub async fn detect_branch_change(
        &self,
        repo_path: &Path,
        last_known_branch: &str,
    ) -> GitResult<Option<String>> {
        let current_branch = self.get_current_branch(repo_path).await?;

        if current_branch != last_known_branch {
            warn!(
                "Branch change detected in {}: {} -> {}",
                repo_path.display(),
                last_known_branch,
                current_branch
            );
            Ok(Some(current_branch))
        } else {
            Ok(None)
        }
    }

    /// Invalidate the cache entry for a specific repository
    ///
    /// Forces the next call to `get_current_branch` to query Git directly.
    /// Useful when you know a branch switch may have occurred.
    pub async fn invalidate_cache(&self, repo_path: &Path) {
        let canonical_path = match repo_path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    "Failed to canonicalize path for cache invalidation: {}: {}",
                    repo_path.display(),
                    e
                );
                return;
            }
        };

        let mut cache = self.cache.write().await;
        if cache.remove(&canonical_path).is_some() {
            debug!(
                "Invalidated cache entry for repository: {}",
                canonical_path.display()
            );
        }
    }

    /// Clear all cache entries
    ///
    /// Useful for testing or when you want to force fresh Git queries for all repositories.
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        let count = cache.len();
        cache.clear();
        debug!("Cleared {} cache entries", count);
    }

    /// Get cache statistics for monitoring
    pub async fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total_entries = cache.len();
        let valid_entries = cache.values().filter(|c| c.is_valid()).count();

        CacheStats {
            total_entries,
            valid_entries,
            expired_entries: total_entries - valid_entries,
        }
    }
}

impl Default for GitBranchDetector {
    fn default() -> Self {
        Self::new()
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
    use std::fs;
    use tempfile::tempdir;

    /// Helper to create a Git repository with initial commit
    fn create_test_repo(path: &Path) -> Result<Repository, git2::Error> {
        let repo = Repository::init(path)?;

        // Create initial commit (required for branch to exist)
        let sig = git2::Signature::now("Test User", "test@example.com")?;
        let tree_id = {
            let mut index = repo.index()?;
            index.write_tree()?
        };
        {
            let tree = repo.find_tree(tree_id)?;
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "Initial commit",
                &tree,
                &[],
            )?;
        } // tree is dropped here

        Ok(repo)
    }

    #[tokio::test]
    async fn test_detect_branch_in_git_repo() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        // Initialize Git repository
        let _repo = create_test_repo(repo_path).unwrap();

        // Default branch should be "main" or "master"
        let detector = GitBranchDetector::new();
        let branch = detector.get_current_branch(repo_path).await.unwrap();

        // Git 2.28+ defaults to "main", older versions use "master"
        assert!(
            branch == "main" || branch == "master",
            "Expected 'main' or 'master', got '{}'",
            branch
        );
    }

    #[tokio::test]
    async fn test_non_git_directory() {
        let temp_dir = tempdir().unwrap();
        let non_git_path = temp_dir.path();

        let detector = GitBranchDetector::new();
        let result = detector.get_current_branch(non_git_path).await;

        assert!(matches!(result, Err(GitError::NotARepository { .. })));
    }

    #[tokio::test]
    async fn test_detached_head() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        // Get HEAD commit
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();

        // Detach HEAD
        repo.set_head_detached(commit.id()).unwrap();

        let detector = GitBranchDetector::new();
        let result = detector.get_current_branch(repo_path).await;

        assert!(matches!(result, Err(GitError::DetachedHead { .. })));
    }

    #[tokio::test]
    async fn test_branch_switching() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        let detector = GitBranchDetector::new();
        let initial_branch = detector.get_current_branch(repo_path).await.unwrap();

        // Create and switch to new branch
        let new_branch_name = "feature/test";
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch(new_branch_name, &commit, false).unwrap();
        repo.set_head(&format!("refs/heads/{}", new_branch_name))
            .unwrap();

        // Invalidate cache to force fresh query
        detector.invalidate_cache(repo_path).await;

        let new_branch = detector.get_current_branch(repo_path).await.unwrap();
        assert_eq!(new_branch, new_branch_name);

        // Test detect_branch_change
        let change = detector
            .detect_branch_change(repo_path, &initial_branch)
            .await
            .unwrap();
        assert_eq!(change, Some(new_branch_name.to_string()));
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        let detector = GitBranchDetector::with_ttl(Duration::from_secs(10));

        // First call - cache miss
        let branch1 = detector.get_current_branch(repo_path).await.unwrap();

        // Second call - should hit cache
        let branch2 = detector.get_current_branch(repo_path).await.unwrap();
        assert_eq!(branch1, branch2);

        // Check cache stats
        let stats = detector.cache_stats().await;
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.valid_entries, 1);
        assert_eq!(stats.expired_entries, 0);

        // Clear cache
        detector.clear_cache().await;
        let stats = detector.cache_stats().await;
        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        // Create detector with very short TTL
        let detector = GitBranchDetector::with_ttl(Duration::from_millis(50));

        // Populate cache
        detector.get_current_branch(repo_path).await.unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = detector.cache_stats().await;
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.expired_entries, 1);
    }

    #[tokio::test]
    async fn test_subdirectory_detection() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        // Create subdirectory
        let subdir = repo_path.join("src").join("lib");
        fs::create_dir_all(&subdir).unwrap();

        // Should detect branch from subdirectory
        let detector = GitBranchDetector::new();
        let branch = detector.get_current_branch(&subdir).await.unwrap();

        assert!(branch == "main" || branch == "master");
    }

    #[tokio::test]
    async fn test_no_branch_change() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        let detector = GitBranchDetector::new();
        let current_branch = detector.get_current_branch(repo_path).await.unwrap();

        // No change expected
        let change = detector
            .detect_branch_change(repo_path, &current_branch)
            .await
            .unwrap();
        assert_eq!(change, None);
    }
}
