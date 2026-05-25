//! TTL-based branch cache for resolving the current git branch from the filesystem.
//!
//! During file processing, the queue item carries a `branch` field that may be
//! stale (e.g. enqueued before a branch switch). This module provides a
//! lightweight, synchronous cache that reads `.git/HEAD` with a short TTL to
//! detect the actual current branch at processing time.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tracing::debug;

use super::reflog::{read_current_branch, resolve_git_dir};

/// Time-to-live for cached branch entries.
const BRANCH_CACHE_TTL: Duration = Duration::from_secs(5);

/// A single cached branch resolution.
struct CacheEntry {
    branch: String,
    expires_at: Instant,
}

impl CacheEntry {
    fn new(branch: String, ttl: Duration) -> Self {
        Self {
            branch,
            expires_at: Instant::now() + ttl,
        }
    }

    fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }
}

/// Synchronous, TTL-based cache mapping project root paths to their current
/// git branch.
///
/// Designed for use inside the file processing pipeline where many queue items
/// from the same project are processed in rapid succession. The 5-second TTL
/// avoids re-reading `.git/HEAD` for every item while still detecting branch
/// switches within a few seconds.
pub struct BranchCache {
    entries: Mutex<HashMap<PathBuf, CacheEntry>>,
    ttl: Duration,
}

impl BranchCache {
    /// Create a new cache with the default 5-second TTL.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl: BRANCH_CACHE_TTL,
        }
    }

    /// Create a new cache with a custom TTL (useful for tests).
    #[cfg(test)]
    fn with_ttl(ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// Get the current branch for a project root path.
    ///
    /// 1. Returns the cached value if the entry has not expired.
    /// 2. Otherwise reads `.git/HEAD` via `resolve_git_dir` + `read_current_branch`.
    /// 3. If detection succeeds, caches and returns the branch name.
    /// 4. If detection fails (not a git repo, detached HEAD, etc.), returns `fallback`.
    pub fn get_branch(&self, project_root: &Path, fallback: &str) -> String {
        // Check cache under lock, release immediately.
        {
            let entries = self.entries.lock().expect("BranchCache lock poisoned");
            if let Some(entry) = entries.get(project_root) {
                if entry.is_valid() {
                    debug!(
                        "BranchCache hit: {} -> {}",
                        project_root.display(),
                        entry.branch
                    );
                    return entry.branch.clone();
                }
            }
        }

        // Cache miss or expired — read from filesystem.
        let detected =
            resolve_git_dir(project_root).and_then(|git_dir| read_current_branch(&git_dir));

        match detected {
            Some(branch) => {
                debug!(
                    "BranchCache miss, detected: {} -> {}",
                    project_root.display(),
                    branch
                );
                let mut entries = self.entries.lock().expect("BranchCache lock poisoned");
                entries.insert(
                    project_root.to_path_buf(),
                    CacheEntry::new(branch.clone(), self.ttl),
                );
                branch
            }
            None => {
                debug!(
                    "BranchCache miss, detection failed: {} -> fallback '{}'",
                    project_root.display(),
                    fallback
                );
                fallback.to_string()
            }
        }
    }

    /// Invalidate the cache entry for a specific project (e.g. on branch switch event).
    pub fn invalidate(&self, project_root: &Path) {
        let mut entries = self.entries.lock().expect("BranchCache lock poisoned");
        if entries.remove(project_root).is_some() {
            debug!("BranchCache invalidated: {}", project_root.display());
        }
    }
}

impl Default for BranchCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_cache_returns_fallback_for_non_git_dir() {
        let tmp = tempdir().unwrap();
        let cache = BranchCache::new();

        let result = cache.get_branch(tmp.path(), "fallback-branch");
        assert_eq!(result, "fallback-branch");
    }

    #[test]
    fn test_cache_hit_avoids_filesystem_read() {
        let tmp = tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/feature/test\n").unwrap();

        let cache = BranchCache::new();

        // First call reads from filesystem.
        let branch1 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(branch1, "feature/test");

        // Mutate the file — but the cache should still return the old value
        // because the TTL has not expired.
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/other-branch\n").unwrap();

        let branch2 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(
            branch2, "feature/test",
            "expected cache hit, not filesystem re-read"
        );
    }

    #[test]
    fn test_cache_expires_after_ttl() {
        let tmp = tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();

        // Use a very short TTL so we can test expiration without sleeping.
        let cache = BranchCache::with_ttl(Duration::from_millis(0));

        let branch1 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(branch1, "main");

        // With 0ms TTL, the entry is already expired.
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/develop\n").unwrap();

        let branch2 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(
            branch2, "develop",
            "expected cache to expire and re-read filesystem"
        );
    }

    #[test]
    fn test_invalidate_clears_entry() {
        let tmp = tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();

        let cache = BranchCache::new();

        let branch1 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(branch1, "main");

        // Change the branch on disk.
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/release/v2\n").unwrap();

        // Invalidate the cache.
        cache.invalidate(tmp.path());

        // Next call should re-read from filesystem.
        let branch2 = cache.get_branch(tmp.path(), "fallback");
        assert_eq!(branch2, "release/v2");
    }

    #[test]
    fn test_detached_head_returns_fallback() {
        let tmp = tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        // Detached HEAD contains a raw SHA, not a ref.
        fs::write(
            git_dir.join("HEAD"),
            "abc1234567890123456789012345678901234567\n",
        )
        .unwrap();

        let cache = BranchCache::new();
        let result = cache.get_branch(tmp.path(), "queue-branch");
        assert_eq!(result, "queue-branch");
    }

    #[test]
    fn test_multiple_projects_cached_independently() {
        let tmp1 = tempdir().unwrap();
        let git1 = tmp1.path().join(".git");
        fs::create_dir(&git1).unwrap();
        fs::write(git1.join("HEAD"), "ref: refs/heads/alpha\n").unwrap();

        let tmp2 = tempdir().unwrap();
        let git2 = tmp2.path().join(".git");
        fs::create_dir(&git2).unwrap();
        fs::write(git2.join("HEAD"), "ref: refs/heads/beta\n").unwrap();

        let cache = BranchCache::new();

        assert_eq!(cache.get_branch(tmp1.path(), "x"), "alpha");
        assert_eq!(cache.get_branch(tmp2.path(), "x"), "beta");

        // Invalidating one does not affect the other.
        cache.invalidate(tmp1.path());
        fs::write(git1.join("HEAD"), "ref: refs/heads/gamma\n").unwrap();

        assert_eq!(cache.get_branch(tmp1.path(), "x"), "gamma");
        assert_eq!(cache.get_branch(tmp2.path(), "x"), "beta"); // still cached
    }
}
