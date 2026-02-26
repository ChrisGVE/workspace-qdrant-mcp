use git2::Repository;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use super::types::{GitError, GitResult};

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
    /// Commit hash the branch points to
    commit_hash: String,
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

/// Branch lifecycle detector for monitoring branch changes in a repository
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
    pub async fn initialize(&self) -> GitResult<()> {
        let branches = self.list_all_branches()?;

        let mut tracked = self.tracked_branches.write().await;
        tracked.clear();

        for (name, commit_hash, _modified) in branches {
            tracked.insert(
                name.clone(),
                TrackedBranch { commit_hash },
            );
        }

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
    pub fn detect_default_branch(&self) -> GitResult<String> {
        let head_path = self.repo_path.join(".git/HEAD");

        let head_content = std::fs::read_to_string(&head_path).map_err(|e| {
            GitError::RepositoryError {
                message: format!("Failed to read .git/HEAD: {}", e),
                source: git2::Error::from_str(&e.to_string()),
            }
        })?;

        if let Some(stripped) = head_content.strip_prefix("ref: refs/heads/") {
            Ok(stripped.trim().to_string())
        } else {
            self.get_remote_default_branch().or_else(|_| {
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

        let config = repo.config().map_err(|e| GitError::RepositoryError {
            message: "Failed to read git config".to_string(),
            source: e,
        })?;

        if let Ok(default_branch) = config.get_string("init.defaultBranch") {
            return Ok(default_branch);
        }

        if let Ok(default_branch) = config.get_string("remote.origin.defaultBranch") {
            return Ok(default_branch);
        }

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
    pub async fn scan_for_changes(&self) -> GitResult<Vec<BranchEvent>> {
        let current_branches = self.list_all_branches()?;
        let current_names: HashSet<String> = current_branches.iter().map(|(n, _, _)| n.clone()).collect();

        let mut events = Vec::new();
        let mut tracked = self.tracked_branches.write().await;
        let tracked_names: HashSet<String> = tracked.keys().cloned().collect();

        let rename_timeout = Duration::from_millis(self.config.rename_correlation_timeout_ms);
        let mut pending = self.pending_deletes.write().await;

        // Check for new branches
        for (name, commit_hash, _) in &current_branches {
            if !tracked_names.contains(name) {
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
                        commit_hash: commit_hash.clone(),
                    },
                );
            }
        }

        // Check for deleted branches
        for name in tracked_names.difference(&current_names) {
            if let Some(old_branch) = tracked.remove(name) {
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

/// Handler for branch lifecycle events that integrates with Qdrant
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
pub mod branch_schema {
    /// Add default_branch column to watch_folders
    pub const ALTER_ADD_DEFAULT_BRANCH: &str = r#"
        ALTER TABLE watch_folders ADD COLUMN default_branch TEXT
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::path::Path;

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

        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("feature/test", &commit, false).unwrap();
        repo.branch("develop", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        let branches = detector.list_all_branches().unwrap();

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

        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("new-feature", &commit, false).unwrap();

        let events = detector.scan_for_changes().await.unwrap();

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

        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("to-delete", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        let mut branch = repo.find_branch("to-delete", git2::BranchType::Local).unwrap();
        branch.delete().unwrap();

        let _events = detector.scan_for_changes().await.unwrap();

        tokio::time::sleep(Duration::from_millis(600)).await;

        let events = detector.scan_for_changes().await.unwrap();

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

        let head = repo.head().unwrap();
        let commit = head.peel_to_commit().unwrap();
        repo.branch("feature-a", &commit, false).unwrap();
        repo.branch("feature-b", &commit, false).unwrap();

        let detector = BranchLifecycleDetector::with_defaults(repo_path.to_path_buf());
        detector.initialize().await.unwrap();

        let stats = detector.stats().await;
        assert!(stats.tracked_branches >= 3);
        assert_eq!(stats.pending_deletes, 0);
        assert!(stats.default_branch.is_some());
    }

    #[test]
    fn test_branch_schema_sql() {
        assert!(branch_schema::ALTER_ADD_DEFAULT_BRANCH.contains("ALTER TABLE"));
        assert!(branch_schema::ALTER_ADD_DEFAULT_BRANCH.contains("watch_folders"));
    }
}
