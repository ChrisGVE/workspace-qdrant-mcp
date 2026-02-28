use git2::Repository;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use super::types::{CacheStats, GitError, GitResult};

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
    pub async fn get_current_branch(&self, repo_path: &Path) -> GitResult<String> {
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

        let head = repo.head().map_err(|e| {
            if e.code() == git2::ErrorCode::UnbornBranch {
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

        if !head.is_branch() {
            return Err(GitError::DetachedHead {
                path: repo_path.display().to_string(),
            });
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "Initial commit",
                &tree,
                &[],
            )?;
        }

        Ok(repo)
    }

    #[tokio::test]
    async fn test_detect_branch_in_git_repo() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let _repo = create_test_repo(repo_path).unwrap();

        let detector = GitBranchDetector::new();
        let branch = detector.get_current_branch(repo_path).await.unwrap();

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

        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();

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

        let new_branch_name = "feature/test";
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch(new_branch_name, &commit, false).unwrap();
        repo.set_head(&format!("refs/heads/{}", new_branch_name))
            .unwrap();

        detector.invalidate_cache(repo_path).await;

        let new_branch = detector.get_current_branch(repo_path).await.unwrap();
        assert_eq!(new_branch, new_branch_name);

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

        let branch1 = detector.get_current_branch(repo_path).await.unwrap();
        let branch2 = detector.get_current_branch(repo_path).await.unwrap();
        assert_eq!(branch1, branch2);

        let stats = detector.cache_stats().await;
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.valid_entries, 1);
        assert_eq!(stats.expired_entries, 0);

        detector.clear_cache().await;
        let stats = detector.cache_stats().await;
        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        let detector = GitBranchDetector::with_ttl(Duration::from_millis(50));

        detector.get_current_branch(repo_path).await.unwrap();

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

        let subdir = repo_path.join("src").join("lib");
        fs::create_dir_all(&subdir).unwrap();

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

        let change = detector
            .detect_branch_change(repo_path, &current_branch)
            .await
            .unwrap();
        assert_eq!(change, None);
    }
}
