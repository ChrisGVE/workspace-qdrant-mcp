//! Git Integration Module
//!
//! Provides Git branch detection, branch switching detection, and branch lifecycle
//! management for the daemon. Uses git2-rs for native Git operations without
//! subprocess overhead.
//!
//! # Branch Lifecycle Management
//!
//! This module handles branch lifecycle events:
//! - **Branch Creation**: Detects new branches via .git/refs/heads/ changes
//! - **Branch Deletion**: Tracks deleted branches for cleanup
//! - **Branch Rename**: Correlates delete/create events for renames
//! - **Default Branch Change**: Monitors .git/HEAD for default branch changes

use git2::Repository;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

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

//
// ========== BRANCH LIFECYCLE TYPES ==========
//

/// Branch lifecycle event types
///
/// These events are emitted when branches are created, deleted, renamed,
/// or when the default branch changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchEvent {
    /// A new branch was created
    Created {
        /// Name of the created branch
        branch: String,
        /// Commit hash the branch points to
        commit_hash: Option<String>,
    },
    /// A branch was deleted
    Deleted {
        /// Name of the deleted branch
        branch: String,
    },
    /// A branch was renamed
    Renamed {
        /// Old branch name
        old_name: String,
        /// New branch name
        new_name: String,
    },
    /// The default branch changed
    DefaultChanged {
        /// Previous default branch
        old_default: String,
        /// New default branch
        new_default: String,
    },
    /// Branch was switched to (HEAD changed)
    Switched {
        /// Previous branch
        from_branch: Option<String>,
        /// New branch
        to_branch: String,
    },
}

impl BranchEvent {
    /// Get the primary branch name involved in this event
    pub fn branch_name(&self) -> &str {
        match self {
            BranchEvent::Created { branch, .. } => branch,
            BranchEvent::Deleted { branch } => branch,
            BranchEvent::Renamed { new_name, .. } => new_name,
            BranchEvent::DefaultChanged { new_default, .. } => new_default,
            BranchEvent::Switched { to_branch, .. } => to_branch,
        }
    }

    /// Check if this event affects a specific branch
    pub fn affects_branch(&self, branch: &str) -> bool {
        match self {
            BranchEvent::Created { branch: b, .. } => b == branch,
            BranchEvent::Deleted { branch: b } => b == branch,
            BranchEvent::Renamed { old_name, new_name } => old_name == branch || new_name == branch,
            BranchEvent::DefaultChanged { old_default, new_default } => {
                old_default == branch || new_default == branch
            }
            BranchEvent::Switched { from_branch, to_branch } => {
                from_branch.as_deref() == Some(branch) || to_branch == branch
            }
        }
    }
}

/// State of a tracked branch in the lifecycle detector
#[derive(Debug, Clone)]
struct TrackedBranch {
    /// Branch name
    name: String,
    /// Commit hash the branch points to
    commit_hash: String,
    /// Last modification time of the branch ref file
    last_modified: SystemTime,
}

/// Configuration for branch lifecycle detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchLifecycleConfig {
    /// Enable branch lifecycle tracking
    pub enabled: bool,
    /// Auto-delete branch documents when branch is deleted
    pub auto_delete_on_branch_delete: bool,
    /// Scan interval for detecting branch changes (seconds)
    pub scan_interval_seconds: u64,
    /// Rename correlation timeout (milliseconds)
    /// If a delete is followed by a create within this window, treat as rename
    pub rename_correlation_timeout_ms: u64,
}

impl Default for BranchLifecycleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_delete_on_branch_delete: true,
            scan_interval_seconds: 5,
            rename_correlation_timeout_ms: 500,
        }
    }
}

/// Pending delete event for rename correlation
#[derive(Debug, Clone)]
struct PendingDelete {
    branch: String,
    commit_hash: String,
    deleted_at: Instant,
}

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

//
// ========== BRANCH LIFECYCLE DETECTOR ==========
//

/// Branch lifecycle detector for monitoring branch changes in a repository
///
/// This struct monitors a Git repository's branches and emits events when:
/// - New branches are created
/// - Branches are deleted
/// - Branches are renamed (detected via delete+create correlation)
/// - The default branch changes
pub struct BranchLifecycleDetector {
    /// Repository path being monitored
    repo_path: PathBuf,
    /// Configuration
    config: BranchLifecycleConfig,
    /// Currently tracked branches (name -> TrackedBranch)
    tracked_branches: Arc<RwLock<HashMap<String, TrackedBranch>>>,
    /// Current default branch
    current_default: Arc<RwLock<Option<String>>>,
    /// Pending deletes for rename correlation
    pending_deletes: Arc<RwLock<Vec<PendingDelete>>>,
    /// Event sender channel
    event_sender: Option<mpsc::Sender<BranchEvent>>,
}

impl BranchLifecycleDetector {
    /// Create a new branch lifecycle detector
    ///
    /// # Arguments
    /// * `repo_path` - Path to the Git repository root
    /// * `config` - Configuration for the detector
    pub fn new(repo_path: PathBuf, config: BranchLifecycleConfig) -> Self {
        Self {
            repo_path,
            config,
            tracked_branches: Arc::new(RwLock::new(HashMap::new())),
            current_default: Arc::new(RwLock::new(None)),
            pending_deletes: Arc::new(RwLock::new(Vec::new())),
            event_sender: None,
        }
    }

    /// Create a detector with default configuration
    pub fn with_defaults(repo_path: PathBuf) -> Self {
        Self::new(repo_path, BranchLifecycleConfig::default())
    }

    /// Set the event sender channel for emitting branch events
    pub fn set_event_sender(&mut self, sender: mpsc::Sender<BranchEvent>) {
        self.event_sender = Some(sender);
    }

    /// Initialize the detector by scanning current branches
    ///
    /// This should be called once when starting to monitor a repository.
    pub async fn initialize(&self) -> GitResult<()> {
        let branches = self.list_all_branches()?;

        let mut tracked = self.tracked_branches.write().await;
        tracked.clear();

        for (name, commit_hash, modified) in branches {
            tracked.insert(
                name.clone(),
                TrackedBranch {
                    name,
                    commit_hash,
                    last_modified: modified,
                },
            );
        }

        // Initialize default branch
        let default = self.detect_default_branch()?;
        let mut current_default = self.current_default.write().await;
        *current_default = Some(default);

        info!(
            "BranchLifecycleDetector initialized for {} with {} branches",
            self.repo_path.display(),
            tracked.len()
        );

        Ok(())
    }

    /// List all branches in the repository
    ///
    /// Returns a list of (branch_name, commit_hash, last_modified)
    pub fn list_all_branches(&self) -> GitResult<Vec<(String, String, SystemTime)>> {
        let repo = Repository::open(&self.repo_path).map_err(|e| {
            if e.code() == git2::ErrorCode::NotFound {
                GitError::NotARepository {
                    path: self.repo_path.display().to_string(),
                }
            } else {
                GitError::RepositoryError {
                    message: "Failed to open repository".to_string(),
                    source: e,
                }
            }
        })?;

        let mut branches = Vec::new();

        for branch in repo.branches(Some(git2::BranchType::Local)).map_err(|e| {
            GitError::RepositoryError {
                message: "Failed to list branches".to_string(),
                source: e,
            }
        })? {
            let (branch, _) = branch.map_err(|e| GitError::RepositoryError {
                message: "Failed to read branch".to_string(),
                source: e,
            })?;

            if let Some(name) = branch.name().ok().flatten() {
                let commit = branch.get().peel_to_commit().ok();
                let commit_hash = commit
                    .map(|c| c.id().to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                // Get modification time from refs file
                let refs_path = self.repo_path.join(".git/refs/heads").join(name);
                let modified = refs_path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                branches.push((name.to_string(), commit_hash, modified));
            }
        }

        Ok(branches)
    }

    /// Detect the default branch of the repository
    ///
    /// Reads .git/HEAD to determine the current default branch.
    pub fn detect_default_branch(&self) -> GitResult<String> {
        let head_path = self.repo_path.join(".git/HEAD");

        let head_content = std::fs::read_to_string(&head_path).map_err(|e| {
            GitError::RepositoryError {
                message: format!("Failed to read .git/HEAD: {}", e),
                source: git2::Error::from_str(&e.to_string()),
            }
        })?;

        // HEAD typically contains "ref: refs/heads/main\n"
        if let Some(stripped) = head_content.strip_prefix("ref: refs/heads/") {
            Ok(stripped.trim().to_string())
        } else {
            // Detached HEAD - try to get default from config or remote
            self.get_remote_default_branch().or_else(|_| {
                // Fall back to "main" as reasonable default
                Ok("main".to_string())
            })
        }
    }

    /// Get the default branch from remote
    fn get_remote_default_branch(&self) -> GitResult<String> {
        let repo = Repository::open(&self.repo_path).map_err(|e| GitError::RepositoryError {
            message: "Failed to open repository".to_string(),
            source: e,
        })?;

        // Try to read from config
        let config = repo.config().map_err(|e| GitError::RepositoryError {
            message: "Failed to read git config".to_string(),
            source: e,
        })?;

        // Check init.defaultBranch
        if let Ok(default_branch) = config.get_string("init.defaultBranch") {
            return Ok(default_branch);
        }

        // Check remote.origin.defaultBranch
        if let Ok(default_branch) = config.get_string("remote.origin.defaultBranch") {
            return Ok(default_branch);
        }

        // Fall back to checking if main or master exists
        if repo.find_branch("main", git2::BranchType::Local).is_ok() {
            return Ok("main".to_string());
        }

        if repo.find_branch("master", git2::BranchType::Local).is_ok() {
            return Ok("master".to_string());
        }

        Err(GitError::RepositoryError {
            message: "Could not determine default branch".to_string(),
            source: git2::Error::from_str("No default branch found"),
        })
    }

    /// Scan for branch changes
    ///
    /// Compares current branches against tracked state and emits events for changes.
    /// Returns a list of detected events.
    pub async fn scan_for_changes(&self) -> GitResult<Vec<BranchEvent>> {
        let current_branches = self.list_all_branches()?;
        let current_names: HashSet<String> = current_branches.iter().map(|(n, _, _)| n.clone()).collect();

        let mut events = Vec::new();
        let mut tracked = self.tracked_branches.write().await;
        let tracked_names: HashSet<String> = tracked.keys().cloned().collect();

        // Process pending deletes for rename correlation
        let rename_timeout = Duration::from_millis(self.config.rename_correlation_timeout_ms);
        let mut pending = self.pending_deletes.write().await;

        // Check for new branches
        for (name, commit_hash, modified) in &current_branches {
            if !tracked_names.contains(name) {
                // New branch - check if it's a rename
                let rename_source = pending.iter().position(|pd| {
                    pd.deleted_at.elapsed() < rename_timeout && pd.commit_hash == *commit_hash
                });

                if let Some(idx) = rename_source {
                    let old_delete = pending.remove(idx);
                    let event = BranchEvent::Renamed {
                        old_name: old_delete.branch,
                        new_name: name.clone(),
                    };
                    info!(
                        "Detected branch rename: {} -> {}",
                        event.branch_name(),
                        name
                    );
                    events.push(event);
                } else {
                    let event = BranchEvent::Created {
                        branch: name.clone(),
                        commit_hash: Some(commit_hash.clone()),
                    };
                    info!("Detected new branch: {}", name);
                    events.push(event);
                }

                tracked.insert(
                    name.clone(),
                    TrackedBranch {
                        name: name.clone(),
                        commit_hash: commit_hash.clone(),
                        last_modified: *modified,
                    },
                );
            }
        }

        // Check for deleted branches
        for name in tracked_names.difference(&current_names) {
            if let Some(old_branch) = tracked.remove(name) {
                // Add to pending for rename correlation
                pending.push(PendingDelete {
                    branch: name.clone(),
                    commit_hash: old_branch.commit_hash,
                    deleted_at: Instant::now(),
                });
            }
        }

        // Clean up expired pending deletes and emit delete events
        let expired_deletes: Vec<_> = pending
            .iter()
            .filter(|pd| pd.deleted_at.elapsed() >= rename_timeout)
            .map(|pd| pd.branch.clone())
            .collect();

        for branch in expired_deletes {
            pending.retain(|pd| pd.branch != branch);
            let event = BranchEvent::Deleted {
                branch: branch.clone(),
            };
            info!("Detected branch deletion: {}", branch);
            events.push(event);
        }

        // Check for default branch change
        let current_default = self.detect_default_branch()?;
        let mut stored_default = self.current_default.write().await;

        if let Some(old_default) = stored_default.as_ref() {
            if old_default != &current_default {
                let event = BranchEvent::DefaultChanged {
                    old_default: old_default.clone(),
                    new_default: current_default.clone(),
                };
                info!(
                    "Detected default branch change: {} -> {}",
                    old_default, current_default
                );
                events.push(event);
            }
        }
        *stored_default = Some(current_default);

        // Send events through channel if configured
        if let Some(ref sender) = self.event_sender {
            for event in &events {
                if sender.send(event.clone()).await.is_err() {
                    warn!("Failed to send branch event - receiver dropped");
                    break;
                }
            }
        }

        Ok(events)
    }

    /// Get the current list of tracked branches
    pub async fn get_tracked_branches(&self) -> Vec<String> {
        self.tracked_branches
            .read()
            .await
            .keys()
            .cloned()
            .collect()
    }

    /// Get the current default branch
    pub async fn get_default_branch(&self) -> Option<String> {
        self.current_default.read().await.clone()
    }

    /// Get commit hash for a specific branch
    pub async fn get_branch_commit(&self, branch: &str) -> Option<String> {
        self.tracked_branches
            .read()
            .await
            .get(branch)
            .map(|tb| tb.commit_hash.clone())
    }

    /// Statistics about tracked branches
    pub async fn stats(&self) -> BranchLifecycleStats {
        let tracked = self.tracked_branches.read().await;
        let pending = self.pending_deletes.read().await;
        let default = self.current_default.read().await;

        BranchLifecycleStats {
            tracked_branches: tracked.len(),
            pending_deletes: pending.len(),
            default_branch: default.clone(),
        }
    }
}

/// Statistics about branch lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchLifecycleStats {
    /// Number of tracked branches
    pub tracked_branches: usize,
    /// Number of pending delete events (waiting for rename correlation)
    pub pending_deletes: usize,
    /// Current default branch
    pub default_branch: Option<String>,
}

//
// ========== BRANCH LIFECYCLE HANDLER ==========
//

/// Handler for branch lifecycle events that integrates with Qdrant
///
/// This trait defines the interface for handling branch events.
/// Implementations can integrate with the storage layer to manage
/// documents when branches change.
#[async_trait::async_trait]
pub trait BranchEventHandler: Send + Sync {
    /// Handle a branch creation event
    async fn handle_branch_created(
        &self,
        project_id: &str,
        branch: &str,
        commit_hash: Option<&str>,
    ) -> GitResult<()>;

    /// Handle a branch deletion event
    async fn handle_branch_deleted(&self, project_id: &str, branch: &str) -> GitResult<()>;

    /// Handle a branch rename event
    async fn handle_branch_renamed(
        &self,
        project_id: &str,
        old_branch: &str,
        new_branch: &str,
    ) -> GitResult<()>;

    /// Handle a default branch change event
    async fn handle_default_changed(
        &self,
        project_id: &str,
        old_default: &str,
        new_default: &str,
    ) -> GitResult<()>;
}

/// SQL schemas for branch lifecycle tracking
///
/// NOTE: Per 3-table SQLite compliance, only schema_version, unified_queue,
/// and watch_folders tables are allowed. Branch events should be logged via
/// structured logging (tracing) rather than stored in SQLite.
pub mod branch_schema {
    /// Add default_branch column to watch_folders
    ///
    /// NOTE: This schema already targets `watch_folders` table per spec.
    /// Run as an optional migration if default_branch tracking is needed.
    pub const ALTER_ADD_DEFAULT_BRANCH: &str = r#"
        ALTER TABLE watch_folders ADD COLUMN default_branch TEXT
    "#;
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

    // ========== Branch Lifecycle Tests ==========

    #[test]
    fn test_branch_event_branch_name() {
        let event = BranchEvent::Created {
            branch: "feature/test".to_string(),
            commit_hash: Some("abc123".to_string()),
        };
        assert_eq!(event.branch_name(), "feature/test");

        let event = BranchEvent::Deleted {
            branch: "old-branch".to_string(),
        };
        assert_eq!(event.branch_name(), "old-branch");

        let event = BranchEvent::Renamed {
            old_name: "old".to_string(),
            new_name: "new".to_string(),
        };
        assert_eq!(event.branch_name(), "new");
    }

    #[test]
    fn test_branch_event_affects_branch() {
        let event = BranchEvent::Renamed {
            old_name: "feature/old".to_string(),
            new_name: "feature/new".to_string(),
        };
        assert!(event.affects_branch("feature/old"));
        assert!(event.affects_branch("feature/new"));
        assert!(!event.affects_branch("main"));
    }

    #[test]
    fn test_branch_lifecycle_config_default() {
        let config = BranchLifecycleConfig::default();
        assert!(config.enabled);
        assert!(config.auto_delete_on_branch_delete);
        assert_eq!(config.scan_interval_seconds, 5);
        assert_eq!(config.rename_correlation_timeout_ms, 500);
    }

    #[tokio::test]
    async fn test_branch_lifecycle_detector_initialize() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        create_test_repo(repo_path).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        let branches = detector.get_tracked_branches().await;
        assert!(!branches.is_empty());

        let default = detector.get_default_branch().await;
        assert!(default.is_some());
    }

    #[tokio::test]
    async fn test_branch_lifecycle_detector_list_branches() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        // Create additional branches
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("feature/test", &commit, false).unwrap();
        repo.branch("develop", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        let branches = detector.list_all_branches().unwrap();

        // Should have at least 3 branches: main/master + feature/test + develop
        assert!(branches.len() >= 3);

        let branch_names: Vec<_> = branches.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(branch_names.contains(&"feature/test"));
        assert!(branch_names.contains(&"develop"));
    }

    #[tokio::test]
    async fn test_branch_lifecycle_detect_creation() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        // Create a new branch
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("new-feature", &commit, false).unwrap();

        // Scan for changes
        let events = detector.scan_for_changes().await.unwrap();

        // Should detect the new branch
        assert!(!events.is_empty());
        assert!(events.iter().any(|e| matches!(
            e,
            BranchEvent::Created { branch, .. } if branch == "new-feature"
        )));
    }

    #[tokio::test]
    async fn test_branch_lifecycle_detect_deletion() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        // Create a branch first
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("to-delete", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        // Delete the branch
        let mut branch = repo.find_branch("to-delete", git2::BranchType::Local).unwrap();
        branch.delete().unwrap();

        // Scan for changes - first scan should detect deletion after timeout
        let events = detector.scan_for_changes().await.unwrap();

        // Wait for rename correlation timeout
        tokio::time::sleep(Duration::from_millis(600)).await;

        // Second scan should emit the delete event
        let events = detector.scan_for_changes().await.unwrap();

        // Should detect the deletion
        assert!(events.iter().any(|e| matches!(
            e,
            BranchEvent::Deleted { branch } if branch == "to-delete"
        )));
    }

    #[tokio::test]
    async fn test_branch_lifecycle_stats() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path();

        let repo = create_test_repo(repo_path).unwrap();

        // Create some branches
        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("feature-a", &commit, false).unwrap();
        repo.branch("feature-b", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        let stats = detector.stats().await;
        assert!(stats.tracked_branches >= 3); // main/master + feature-a + feature-b
        assert_eq!(stats.pending_deletes, 0);
        assert!(stats.default_branch.is_some());
    }

    #[test]
    fn test_branch_schema_sql() {
        // Verify SQL statement is valid syntax
        // NOTE: Per 3-table SQLite compliance, only watch_folders modifications allowed
        assert!(branch_schema::ALTER_ADD_DEFAULT_BRANCH.contains("ALTER TABLE"));
        assert!(branch_schema::ALTER_ADD_DEFAULT_BRANCH.contains("watch_folders"));
    }
}
