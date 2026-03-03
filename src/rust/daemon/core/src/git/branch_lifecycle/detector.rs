//! BranchLifecycleDetector for monitoring branch changes in a repository.

use git2::Repository;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::git::types::{GitError, GitResult};
use super::{BranchEvent, BranchLifecycleConfig, BranchLifecycleStats};

/// State of a tracked branch in the lifecycle detector
#[derive(Debug, Clone)]
struct TrackedBranch {
    /// Commit hash the branch points to
    commit_hash: String,
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

        self.detect_new_branches(&current_branches, &tracked_names, &mut tracked, &mut pending, rename_timeout, &mut events);
        self.detect_deleted_branches(&current_names, &tracked_names, &mut tracked, &mut pending);
        self.emit_expired_deletes(&mut pending, rename_timeout, &mut events);

        let current_default = self.detect_default_branch()?;
        let mut stored_default = self.current_default.write().await;
        emit_default_branch_change(stored_default.as_deref(), &current_default, &mut events);
        *stored_default = Some(current_default);

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

    fn detect_new_branches(
        &self,
        current_branches: &[(String, String, std::time::SystemTime)],
        tracked_names: &HashSet<String>,
        tracked: &mut std::collections::HashMap<String, TrackedBranch>,
        pending: &mut Vec<PendingDelete>,
        rename_timeout: Duration,
        events: &mut Vec<BranchEvent>,
    ) {
        for (name, commit_hash, _) in current_branches {
            if tracked_names.contains(name) {
                continue;
            }
            let rename_idx = pending.iter().position(|pd| {
                pd.deleted_at.elapsed() < rename_timeout && pd.commit_hash == *commit_hash
            });
            if let Some(idx) = rename_idx {
                let old_delete = pending.remove(idx);
                let event = BranchEvent::Renamed {
                    old_name: old_delete.branch,
                    new_name: name.clone(),
                };
                info!("Detected branch rename: {} -> {}", event.branch_name(), name);
                events.push(event);
            } else {
                let event = BranchEvent::Created {
                    branch: name.clone(),
                    commit_hash: Some(commit_hash.clone()),
                };
                info!("Detected new branch: {}", name);
                events.push(event);
            }
            tracked.insert(name.clone(), TrackedBranch { commit_hash: commit_hash.clone() });
        }
    }

    fn detect_deleted_branches(
        &self,
        current_names: &HashSet<String>,
        tracked_names: &HashSet<String>,
        tracked: &mut std::collections::HashMap<String, TrackedBranch>,
        pending: &mut Vec<PendingDelete>,
    ) {
        for name in tracked_names.difference(current_names) {
            if let Some(old_branch) = tracked.remove(name) {
                pending.push(PendingDelete {
                    branch: name.clone(),
                    commit_hash: old_branch.commit_hash,
                    deleted_at: Instant::now(),
                });
            }
        }
    }

    fn emit_expired_deletes(
        &self,
        pending: &mut Vec<PendingDelete>,
        rename_timeout: Duration,
        events: &mut Vec<BranchEvent>,
    ) {
        let expired: Vec<_> = pending
            .iter()
            .filter(|pd| pd.deleted_at.elapsed() >= rename_timeout)
            .map(|pd| pd.branch.clone())
            .collect();
        for branch in expired {
            pending.retain(|pd| pd.branch != branch);
            info!("Detected branch deletion: {}", branch);
            events.push(BranchEvent::Deleted { branch });
        }
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

/// Emit a `DefaultChanged` event if the default branch has changed.
fn emit_default_branch_change(
    old_default: Option<&str>,
    current_default: &str,
    events: &mut Vec<BranchEvent>,
) {
    if let Some(old) = old_default {
        if old != current_default {
            info!("Detected default branch change: {} -> {}", old, current_default);
            events.push(BranchEvent::DefaultChanged {
                old_default: old.to_string(),
                new_default: current_default.to_string(),
            });
        }
    }
}
