//! Git Watcher Module
//!
//! Provides a second-layer watcher for git-tracked projects that monitors
//! `.git/HEAD` and `.git/refs/heads/` to detect branch switches, commits,
//! merges, pulls, rebases, resets, and stash operations.
//!
//! The git watcher complements the file watcher (layer 1) by detecting
//! structural git operations that change which files are checked out,
//! enabling precise enqueue of changed files via `git diff-tree`.

use std::path::{Path, PathBuf};
use std::time::Duration;
use notify::{RecursiveMode, Watcher as NotifyWatcher};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use thiserror::Error;

/// Errors from git watcher operations
#[derive(Error, Debug)]
pub enum GitWatcherError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Not a git repository: {0}")]
    NotGitRepo(String),

    #[error("Git directory not found: {0}")]
    GitDirNotFound(String),
}

pub type GitWatcherResult<T> = Result<T, GitWatcherError>;

/// Types of git operations detected from reflog parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitEventType {
    /// Branch switch: `checkout: moving from X to Y`
    BranchSwitch,
    /// New commit: `commit:` or `commit (amend):`
    Commit,
    /// Merge commit: `merge`
    Merge,
    /// Pull (fetch+merge): `pull:`
    Pull,
    /// Rebase: `rebase`
    Rebase,
    /// Reset: `reset:`
    Reset,
    /// Stash operation
    Stash,
    /// Unknown operation
    Unknown,
}

impl GitEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GitEventType::BranchSwitch => "branch_switch",
            GitEventType::Commit => "commit",
            GitEventType::Merge => "merge",
            GitEventType::Pull => "pull",
            GitEventType::Rebase => "rebase",
            GitEventType::Reset => "reset",
            GitEventType::Stash => "stash",
            GitEventType::Unknown => "unknown",
        }
    }
}

/// A git event detected by the watcher
#[derive(Debug, Clone)]
pub struct GitEvent {
    /// Watch folder ID this event belongs to
    pub watch_folder_id: String,
    /// Type of git operation
    pub event_type: GitEventType,
    /// Old commit SHA (before the operation)
    pub old_sha: String,
    /// New commit SHA (after the operation)
    pub new_sha: String,
    /// Current branch name (read from HEAD)
    pub branch: Option<String>,
    /// Previous branch name (for branch switches)
    pub old_branch: Option<String>,
}

/// Git watcher for a single project
pub struct GitWatcher {
    /// Watch folder ID for event attribution
    watch_folder_id: String,
    /// Project root directory
    project_root: PathBuf,
    /// Resolved .git directory (handles worktrees)
    git_dir: PathBuf,
    /// Channel to send git events
    event_tx: mpsc::UnboundedSender<GitEvent>,
    /// Notify watcher handle
    watcher: Option<Box<dyn NotifyWatcher + Send + Sync>>,
    /// Background processor handle
    processor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl GitWatcher {
    /// Create a new git watcher for a project.
    ///
    /// Resolves the `.git` directory (handles worktrees where `.git` is a file)
    /// and validates it exists.
    pub fn new(
        watch_folder_id: String,
        project_root: PathBuf,
        event_tx: mpsc::UnboundedSender<GitEvent>,
    ) -> GitWatcherResult<Self> {
        let git_dir = resolve_git_dir(&project_root)
            .ok_or_else(|| GitWatcherError::GitDirNotFound(
                format!("No .git directory found in {}", project_root.display())
            ))?;

        Ok(Self {
            watch_folder_id,
            project_root,
            git_dir,
            event_tx,
            watcher: None,
            processor_handle: None,
        })
    }

    /// Start watching .git/HEAD and .git/refs/heads/
    pub fn start(&mut self) -> GitWatcherResult<()> {
        let (notify_tx, notify_rx) = mpsc::unbounded_channel();

        let mut watcher = notify::RecommendedWatcher::new(
            move |result: Result<notify::Event, notify::Error>| {
                if let Ok(event) = result {
                    let _ = notify_tx.send(event);
                }
            },
            notify::Config::default()
                .with_poll_interval(Duration::from_secs(2)),
        )?;

        // For worktrees, refs/heads/ lives in the common dir, not the worktree dir.
        let common_dir = resolve_common_dir(&self.git_dir);

        // Watch .git/HEAD (for branch switches and commits affecting HEAD)
        let head_path = self.git_dir.join("HEAD");
        if head_path.exists() {
            // Watch the parent directory (logs/) would be too broad;
            // watch HEAD directly — notify supports file-level watching on macOS/Linux
            if let Err(e) = watcher.watch(&head_path, RecursiveMode::NonRecursive) {
                warn!("Failed to watch .git/HEAD at {}: {}", head_path.display(), e);
            } else {
                debug!("Watching .git/HEAD: {}", head_path.display());
            }
        }

        // Watch .git/refs/heads/ (for branch ref updates from commits/pushes)
        let refs_heads = common_dir.join("refs").join("heads");
        if refs_heads.exists() {
            if let Err(e) = watcher.watch(&refs_heads, RecursiveMode::Recursive) {
                warn!("Failed to watch refs/heads/ at {}: {}", refs_heads.display(), e);
            } else {
                debug!("Watching refs/heads/: {}", refs_heads.display());
            }
        }

        // Watch logs/HEAD for reflog updates (more reliable than HEAD file itself)
        let logs_head = self.git_dir.join("logs").join("HEAD");
        if logs_head.exists() {
            if let Err(e) = watcher.watch(&logs_head, RecursiveMode::NonRecursive) {
                warn!("Failed to watch logs/HEAD at {}: {}", logs_head.display(), e);
            } else {
                debug!("Watching logs/HEAD: {}", logs_head.display());
            }
        }

        self.watcher = Some(Box::new(watcher));

        // Start background processor for debouncing and event emission
        let git_dir = self.git_dir.clone();
        let watch_folder_id = self.watch_folder_id.clone();
        let event_tx = self.event_tx.clone();

        let handle = tokio::spawn(async move {
            Self::process_events(notify_rx, git_dir, watch_folder_id, event_tx).await;
        });

        self.processor_handle = Some(handle);

        info!(
            "Git watcher started for {} (git_dir={})",
            self.project_root.display(),
            self.git_dir.display()
        );

        Ok(())
    }

    /// Stop the git watcher
    pub async fn stop(&mut self) {
        self.watcher = None;
        if let Some(handle) = self.processor_handle.take() {
            handle.abort();
        }
        info!("Git watcher stopped for {}", self.project_root.display());
    }

    /// Background event processor with debouncing
    async fn process_events(
        mut notify_rx: mpsc::UnboundedReceiver<notify::Event>,
        git_dir: PathBuf,
        watch_folder_id: String,
        event_tx: mpsc::UnboundedSender<GitEvent>,
    ) {
        // Debounce: collect events for a short window before processing
        let debounce = Duration::from_millis(200);

        loop {
            // Wait for the first event
            let Some(_first_event) = notify_rx.recv().await else {
                debug!("Git watcher channel closed for {}", watch_folder_id);
                break;
            };

            // Debounce: drain any additional events within the window
            tokio::time::sleep(debounce).await;
            while notify_rx.try_recv().is_ok() {
                // drain
            }

            // Parse reflog to determine what happened
            match parse_reflog_last_entry(&git_dir) {
                Some((old_sha, new_sha, event_type, old_branch)) => {
                    let branch = read_current_branch(&git_dir);

                    let git_event = GitEvent {
                        watch_folder_id: watch_folder_id.clone(),
                        event_type,
                        old_sha,
                        new_sha,
                        branch,
                        old_branch,
                    };

                    info!(
                        "Git event detected: {:?} for {} (old={:.8}..new={:.8})",
                        git_event.event_type,
                        watch_folder_id,
                        &git_event.old_sha[..git_event.old_sha.len().min(8)],
                        &git_event.new_sha[..git_event.new_sha.len().min(8)],
                    );

                    if event_tx.send(git_event).is_err() {
                        debug!("Git event receiver dropped for {}", watch_folder_id);
                        break;
                    }
                }
                None => {
                    debug!(
                        "Git change detected but no parseable reflog entry for {}",
                        watch_folder_id
                    );
                }
            }
        }
    }
}

// ========== Helper functions ==========

/// Resolve the actual .git directory for a project.
///
/// Handles:
/// - Standard repos: `.git/` is a directory
/// - Worktrees: `.git` is a file containing `gitdir: /path/to/actual/.git/worktrees/name`
pub fn resolve_git_dir(project_root: &Path) -> Option<PathBuf> {
    let git_path = project_root.join(".git");

    if git_path.is_dir() {
        return Some(git_path);
    }

    if git_path.is_file() {
        // Worktree: .git is a file with `gitdir: <path>`
        if let Ok(content) = std::fs::read_to_string(&git_path) {
            let content = content.trim();
            if let Some(gitdir) = content.strip_prefix("gitdir: ") {
                let resolved = if Path::new(gitdir).is_absolute() {
                    PathBuf::from(gitdir)
                } else {
                    project_root.join(gitdir)
                };
                // For worktrees, the actual reflog and HEAD are in the worktree-specific dir,
                // but refs/heads/ is in the common git dir. We return the worktree git dir
                // and handle the commondir resolution in consumers.
                if resolved.exists() {
                    return Some(resolved);
                }
            }
        }
    }

    None
}

/// Resolve the common git directory (for worktrees, this is the main repo's .git).
///
/// In a worktree, `<gitdir>/commondir` contains the path to the shared git directory.
pub fn resolve_common_dir(git_dir: &Path) -> PathBuf {
    let commondir_file = git_dir.join("commondir");
    if let Ok(content) = std::fs::read_to_string(&commondir_file) {
        let common_path = content.trim();
        if Path::new(common_path).is_absolute() {
            PathBuf::from(common_path)
        } else {
            git_dir.join(common_path)
        }
    } else {
        // Not a worktree — git_dir is the common dir
        git_dir.to_path_buf()
    }
}

/// Read the current branch name from .git/HEAD
pub fn read_current_branch(git_dir: &Path) -> Option<String> {
    let head_path = git_dir.join("HEAD");
    let content = std::fs::read_to_string(&head_path).ok()?;
    let content = content.trim();

    // HEAD contains "ref: refs/heads/<branch>" or a detached commit SHA
    if let Some(ref_path) = content.strip_prefix("ref: refs/heads/") {
        Some(ref_path.to_string())
    } else {
        // Detached HEAD — return the SHA
        None
    }
}

/// Parse the last entry of the reflog to determine what happened.
///
/// Reflog format: `<old-sha> <new-sha> <author> <timestamp> \t<operation>`
///
/// Returns: (old_sha, new_sha, event_type, old_branch_for_switch)
pub fn parse_reflog_last_entry(git_dir: &Path) -> Option<(String, String, GitEventType, Option<String>)> {
    let reflog_path = git_dir.join("logs").join("HEAD");

    // For worktrees, the HEAD log may be in the worktree dir
    let content = std::fs::read_to_string(&reflog_path).ok()?;
    let last_line = content.lines().last()?;

    parse_reflog_line(last_line)
}

/// Parse a single reflog line into its components.
///
/// Format: `<old-sha> <new-sha> Author Name <email> <timestamp> <tz>\t<operation description>`
pub fn parse_reflog_line(line: &str) -> Option<(String, String, GitEventType, Option<String>)> {
    // Split at tab to separate metadata from operation description
    let (metadata, operation) = line.split_once('\t')?;

    // Parse metadata: old_sha new_sha Author Name <email> timestamp tz
    let parts: Vec<&str> = metadata.splitn(3, ' ').collect();
    if parts.len() < 3 {
        return None;
    }

    let old_sha = parts[0].to_string();
    let new_sha = parts[1].to_string();

    // Validate SHAs (should be 40 hex chars)
    if old_sha.len() != 40 || new_sha.len() != 40 {
        return None;
    }

    let operation = operation.trim();

    // Classify operation
    let (event_type, old_branch) = classify_reflog_operation(operation);

    Some((old_sha, new_sha, event_type, old_branch))
}

/// Classify a reflog operation string into a GitEventType.
///
/// Common reflog patterns:
/// - `checkout: moving from <old> to <new>` → BranchSwitch
/// - `commit: <message>` / `commit (amend): <message>` / `commit (initial): <message>` → Commit
/// - `merge <branch>: <details>` → Merge
/// - `pull: <details>` / `pull --rebase: <details>` → Pull
/// - `rebase (start):` / `rebase (continue):` / `rebase (finish):` → Rebase
/// - `reset: moving to <ref>` → Reset
fn classify_reflog_operation(operation: &str) -> (GitEventType, Option<String>) {
    let op_lower = operation.to_lowercase();

    if op_lower.starts_with("checkout: moving from ") {
        // Extract old branch name: "checkout: moving from X to Y"
        let rest = &operation["checkout: moving from ".len()..];
        let old_branch = rest.split(" to ").next().map(|s| s.to_string());
        return (GitEventType::BranchSwitch, old_branch);
    }

    if op_lower.starts_with("commit")
        && (op_lower.starts_with("commit:")
            || op_lower.starts_with("commit (amend):")
            || op_lower.starts_with("commit (initial):")
            || op_lower.starts_with("commit (merge):"))
    {
        return (GitEventType::Commit, None);
    }

    if op_lower.starts_with("merge ") {
        return (GitEventType::Merge, None);
    }

    if op_lower.starts_with("pull") {
        return (GitEventType::Pull, None);
    }

    if op_lower.starts_with("rebase") {
        return (GitEventType::Rebase, None);
    }

    if op_lower.starts_with("reset:") {
        return (GitEventType::Reset, None);
    }

    // Stash operations show up differently — they modify refs/stash
    // rather than HEAD reflog, but some operations do appear in HEAD reflog
    if op_lower.contains("stash") {
        return (GitEventType::Stash, None);
    }

    (GitEventType::Unknown, None)
}

// ========== Diff-Tree Integration (Task 8) ==========

/// File change status from git diff-tree output
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileChangeStatus {
    /// Modified file (M)
    Modified,
    /// Added file (A)
    Added,
    /// Deleted file (D)
    Deleted,
    /// Renamed file (R) with similarity percentage
    Renamed { old_path: String, similarity: u8 },
    /// Copied file (C) with similarity percentage
    Copied { src_path: String, similarity: u8 },
    /// Type changed (T) — e.g., file became symlink
    TypeChanged,
}

/// A single file change from git diff-tree
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Change status (modified, added, deleted, etc.)
    pub status: FileChangeStatus,
    /// Path relative to repo root
    pub path: String,
}

/// Run git diff-tree between two commits to get the list of changed files.
///
/// Uses the git2 crate for safety (no shell escaping issues).
/// Returns file changes relative to the repository root.
pub fn diff_tree(
    repo_root: &Path,
    old_sha: &str,
    new_sha: &str,
) -> Result<Vec<FileChange>, GitWatcherError> {
    let repo = git2::Repository::open(repo_root)
        .map_err(|e| GitWatcherError::NotGitRepo(format!("{}: {}", repo_root.display(), e)))?;

    let old_oid = git2::Oid::from_str(old_sha)
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid old SHA: {}", e))))?;
    let new_oid = git2::Oid::from_str(new_sha)
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid new SHA: {}", e))))?;

    // Handle the null SHA (initial commit or creation)
    let is_initial = old_sha == "0000000000000000000000000000000000000000";

    let old_tree = if is_initial {
        None
    } else {
        let old_commit = repo.find_commit(old_oid)
            .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("Old commit not found: {}", e))))?;
        Some(old_commit.tree()
            .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Old tree error: {}", e))))?)
    };

    let new_commit = repo.find_commit(new_oid)
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("New commit not found: {}", e))))?;
    let new_tree = new_commit.tree()
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("New tree error: {}", e))))?;

    let mut diff_opts = git2::DiffOptions::new();
    diff_opts.include_untracked(false);

    let diff = repo.diff_tree_to_tree(
        old_tree.as_ref(),
        Some(&new_tree),
        Some(&mut diff_opts),
    ).map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Diff error: {}", e))))?;

    // Enable rename/copy detection
    let mut find_opts = git2::DiffFindOptions::new();
    find_opts.renames(true);
    find_opts.copies(true);
    let diff = {
        let mut d = diff;
        d.find_similar(Some(&mut find_opts))
            .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Find similar error: {}", e))))?;
        d
    };

    let mut changes = Vec::new();

    diff.foreach(
        &mut |delta, _| {
            let status = delta.status();
            let new_file = delta.new_file();
            let old_file = delta.old_file();

            let path = new_file.path()
                .or_else(|| old_file.path())
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            let change_status = match status {
                git2::Delta::Added => FileChangeStatus::Added,
                git2::Delta::Deleted => {
                    // For deletes, the path is in old_file
                    let delete_path = old_file.path()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default();
                    changes.push(FileChange {
                        status: FileChangeStatus::Deleted,
                        path: delete_path,
                    });
                    return true;
                }
                git2::Delta::Modified => FileChangeStatus::Modified,
                git2::Delta::Renamed => {
                    let old_path = old_file.path()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default();
                    FileChangeStatus::Renamed {
                        old_path,
                        similarity: 0, // git2 doesn't expose similarity in delta directly
                    }
                }
                git2::Delta::Copied => {
                    let src_path = old_file.path()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default();
                    FileChangeStatus::Copied {
                        src_path,
                        similarity: 0,
                    }
                }
                git2::Delta::Typechange => FileChangeStatus::TypeChanged,
                _ => return true, // skip unmodified, ignored, etc.
            };

            changes.push(FileChange {
                status: change_status,
                path,
            });
            true
        },
        None,
        None,
        None,
    ).map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Diff foreach error: {}", e))))?;

    Ok(changes)
}

/// Parse git diff-tree --name-status output (fallback for when git2 is insufficient).
///
/// Format per line: `<status>\t<path>` or `<status>\t<old_path>\t<new_path>` for renames.
pub fn parse_diff_tree_output(output: &str) -> Vec<FileChange> {
    let mut changes = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }

        let status_str = parts[0];
        let path = parts[1].to_string();

        let status = if status_str == "M" {
            FileChangeStatus::Modified
        } else if status_str == "A" {
            FileChangeStatus::Added
        } else if status_str == "D" {
            FileChangeStatus::Deleted
        } else if status_str.starts_with('R') {
            let similarity = status_str[1..].parse::<u8>().unwrap_or(0);
            let new_path = parts.get(2).map(|s| s.to_string()).unwrap_or_default();
            changes.push(FileChange {
                status: FileChangeStatus::Renamed {
                    old_path: path,
                    similarity,
                },
                path: new_path,
            });
            continue;
        } else if status_str.starts_with('C') {
            let similarity = status_str[1..].parse::<u8>().unwrap_or(0);
            let new_path = parts.get(2).map(|s| s.to_string()).unwrap_or_default();
            changes.push(FileChange {
                status: FileChangeStatus::Copied {
                    src_path: path,
                    similarity,
                },
                path: new_path,
            });
            continue;
        } else if status_str == "T" {
            FileChangeStatus::TypeChanged
        } else {
            continue; // Unknown status, skip
        };

        changes.push(FileChange { status, path });
    }

    changes
}

/// Get the blob hash (git object SHA) for a file at a given revision.
///
/// Uses git2 to look up the tree entry for the path.
pub fn get_blob_hash(repo_root: &Path, relative_path: &str, revision: &str) -> Result<String, GitWatcherError> {
    let repo = git2::Repository::open(repo_root)
        .map_err(|e| GitWatcherError::NotGitRepo(format!("{}: {}", repo_root.display(), e)))?;

    let rev = repo.revparse_single(revision)
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("Revision not found: {}", e))))?;

    let commit = rev.peel_to_commit()
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Not a commit: {}", e))))?;

    let tree = commit.tree()
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Tree error: {}", e))))?;

    let entry = tree.get_path(Path::new(relative_path))
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("Path not in tree: {}", e))))?;

    Ok(entry.id().to_string())
}

/// List submodule entries from a commit tree (160000 mode).
///
/// Returns Vec of (submodule_path, pinned_sha).
pub fn ls_tree_submodules(repo_root: &Path, revision: &str) -> Result<Vec<(String, String)>, GitWatcherError> {
    let repo = git2::Repository::open(repo_root)
        .map_err(|e| GitWatcherError::NotGitRepo(format!("{}: {}", repo_root.display(), e)))?;

    let rev = repo.revparse_single(revision)
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("Revision not found: {}", e))))?;

    let commit = rev.peel_to_commit()
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Not a commit: {}", e))))?;

    let tree = commit.tree()
        .map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Tree error: {}", e))))?;

    let mut submodules = Vec::new();

    tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
        // git submodule entries have filemode 0o160000 (S_IFGITLINK)
        if entry.filemode() == 0o160000 {
            let path = if root.is_empty() {
                entry.name().unwrap_or("").to_string()
            } else {
                format!("{}{}", root, entry.name().unwrap_or(""))
            };
            submodules.push((path, entry.id().to_string()));
        }
        git2::TreeWalkResult::Ok
    }).map_err(|e| GitWatcherError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Tree walk error: {}", e))))?;

    Ok(submodules)
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_branch_switch() {
        let (event_type, old_branch) =
            classify_reflog_operation("checkout: moving from main to feature/new-stuff");
        assert_eq!(event_type, GitEventType::BranchSwitch);
        assert_eq!(old_branch.as_deref(), Some("main"));
    }

    #[test]
    fn test_classify_commit() {
        let (event_type, _) = classify_reflog_operation("commit: fix authentication bug");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (amend): fix typo");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (initial): initial commit");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (merge): Merge branch 'dev'");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_classify_merge() {
        let (event_type, _) = classify_reflog_operation("merge feature/auth: Fast-forward");
        assert_eq!(event_type, GitEventType::Merge);
    }

    #[test]
    fn test_classify_pull() {
        let (event_type, _) = classify_reflog_operation("pull: Fast-forward");
        assert_eq!(event_type, GitEventType::Pull);

        let (event_type, _) = classify_reflog_operation("pull --rebase: checkout abc123");
        assert_eq!(event_type, GitEventType::Pull);
    }

    #[test]
    fn test_classify_rebase() {
        let (event_type, _) = classify_reflog_operation("rebase (start): checkout origin/main");
        assert_eq!(event_type, GitEventType::Rebase);

        let (event_type, _) = classify_reflog_operation("rebase (finish): returning to refs/heads/feature");
        assert_eq!(event_type, GitEventType::Rebase);
    }

    #[test]
    fn test_classify_reset() {
        let (event_type, _) = classify_reflog_operation("reset: moving to HEAD~1");
        assert_eq!(event_type, GitEventType::Reset);
    }

    #[test]
    fn test_classify_unknown() {
        let (event_type, _) = classify_reflog_operation("some-unknown-operation");
        assert_eq!(event_type, GitEventType::Unknown);
    }

    #[test]
    fn test_parse_reflog_line_commit() {
        let line = "abc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John Doe <john@example.com> 1708300000 +0000\tcommit: add new feature";
        let result = parse_reflog_line(line);
        assert!(result.is_some());
        let (old_sha, new_sha, event_type, _) = result.unwrap();
        assert_eq!(old_sha, "abc1234567890123456789012345678901234567");
        assert_eq!(new_sha, "def1234567890123456789012345678901234567");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_parse_reflog_line_branch_switch() {
        let line = "abc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John Doe <john@example.com> 1708300000 +0000\tcheckout: moving from main to feature/auth";
        let result = parse_reflog_line(line);
        assert!(result.is_some());
        let (_, _, event_type, old_branch) = result.unwrap();
        assert_eq!(event_type, GitEventType::BranchSwitch);
        assert_eq!(old_branch.as_deref(), Some("main"));
    }

    #[test]
    fn test_parse_reflog_line_invalid() {
        // Missing tab separator
        let result = parse_reflog_line("no tab in this line");
        assert!(result.is_none());

        // Invalid SHA (too short)
        let result = parse_reflog_line("abc def Author <email> 12345 +0000\tcommit: test");
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_git_dir_standard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();

        let result = resolve_git_dir(temp_dir.path());
        assert_eq!(result, Some(git_dir));
    }

    #[test]
    fn test_resolve_git_dir_worktree() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create a fake worktree .git file
        let actual_git_dir = temp_dir.path().join("actual_git");
        std::fs::create_dir(&actual_git_dir).unwrap();

        let worktree_dir = temp_dir.path().join("worktree");
        std::fs::create_dir(&worktree_dir).unwrap();

        let git_file = worktree_dir.join(".git");
        std::fs::write(&git_file, format!("gitdir: {}", actual_git_dir.display())).unwrap();

        let result = resolve_git_dir(&worktree_dir);
        assert_eq!(result, Some(actual_git_dir));
    }

    #[test]
    fn test_resolve_git_dir_not_git() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = resolve_git_dir(temp_dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_read_current_branch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path();

        // Create HEAD file pointing to a branch
        std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        assert_eq!(read_current_branch(git_dir), Some("main".to_string()));

        // Detached HEAD
        std::fs::write(git_dir.join("HEAD"), "abc1234567890123456789012345678901234567\n").unwrap();
        assert_eq!(read_current_branch(git_dir), None);
    }

    #[test]
    fn test_parse_reflog_last_entry() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path();

        // Create logs/HEAD with a reflog entry
        let logs_dir = git_dir.join("logs");
        std::fs::create_dir_all(&logs_dir).unwrap();

        let reflog_content = "0000000000000000000000000000000000000000 abc1234567890123456789012345678901234567 John <john@x.com> 1708300000 +0000\tcommit (initial): init\nabc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John <john@x.com> 1708301000 +0000\tcommit: second commit\n";
        std::fs::write(logs_dir.join("HEAD"), reflog_content).unwrap();

        let result = parse_reflog_last_entry(git_dir);
        assert!(result.is_some());
        let (old_sha, new_sha, event_type, _) = result.unwrap();
        assert_eq!(old_sha, "abc1234567890123456789012345678901234567");
        assert_eq!(new_sha, "def1234567890123456789012345678901234567");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_parse_reflog_last_entry_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = parse_reflog_last_entry(temp_dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_common_dir_standard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();

        // No commondir file → returns git_dir itself
        let result = resolve_common_dir(&git_dir);
        assert_eq!(result, git_dir);
    }

    #[test]
    fn test_resolve_common_dir_worktree() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create fake worktree structure
        let main_git_dir = temp_dir.path().join("main_repo").join(".git");
        std::fs::create_dir_all(&main_git_dir).unwrap();

        let worktree_git_dir = main_git_dir.join("worktrees").join("wt1");
        std::fs::create_dir_all(&worktree_git_dir).unwrap();

        // Write commondir pointing to main git dir (relative)
        std::fs::write(worktree_git_dir.join("commondir"), "../..").unwrap();

        let result = resolve_common_dir(&worktree_git_dir);
        // Should resolve to main_git_dir
        assert_eq!(
            result.canonicalize().unwrap(),
            main_git_dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn test_git_event_type_as_str() {
        assert_eq!(GitEventType::BranchSwitch.as_str(), "branch_switch");
        assert_eq!(GitEventType::Commit.as_str(), "commit");
        assert_eq!(GitEventType::Merge.as_str(), "merge");
        assert_eq!(GitEventType::Pull.as_str(), "pull");
        assert_eq!(GitEventType::Rebase.as_str(), "rebase");
        assert_eq!(GitEventType::Reset.as_str(), "reset");
        assert_eq!(GitEventType::Stash.as_str(), "stash");
        assert_eq!(GitEventType::Unknown.as_str(), "unknown");
    }

    // === Diff-tree tests (Task 8) ===

    #[test]
    fn test_parse_diff_tree_output_basic() {
        let output = "M\tsrc/main.rs\nA\tsrc/new_file.rs\nD\tsrc/deleted.rs\n";
        let changes = parse_diff_tree_output(output);
        assert_eq!(changes.len(), 3);
        assert_eq!(changes[0].status, FileChangeStatus::Modified);
        assert_eq!(changes[0].path, "src/main.rs");
        assert_eq!(changes[1].status, FileChangeStatus::Added);
        assert_eq!(changes[1].path, "src/new_file.rs");
        assert_eq!(changes[2].status, FileChangeStatus::Deleted);
        assert_eq!(changes[2].path, "src/deleted.rs");
    }

    #[test]
    fn test_parse_diff_tree_output_rename() {
        let output = "R100\told_name.rs\tnew_name.rs\n";
        let changes = parse_diff_tree_output(output);
        assert_eq!(changes.len(), 1);
        match &changes[0].status {
            FileChangeStatus::Renamed { old_path, similarity } => {
                assert_eq!(old_path, "old_name.rs");
                assert_eq!(*similarity, 100);
                assert_eq!(changes[0].path, "new_name.rs");
            }
            other => panic!("Expected Renamed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_diff_tree_output_copy() {
        let output = "C095\toriginal.rs\tcopy.rs\n";
        let changes = parse_diff_tree_output(output);
        assert_eq!(changes.len(), 1);
        match &changes[0].status {
            FileChangeStatus::Copied { src_path, similarity } => {
                assert_eq!(src_path, "original.rs");
                assert_eq!(*similarity, 95);
                assert_eq!(changes[0].path, "copy.rs");
            }
            other => panic!("Expected Copied, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_diff_tree_output_type_change() {
        let output = "T\tsome_link\n";
        let changes = parse_diff_tree_output(output);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].status, FileChangeStatus::TypeChanged);
    }

    #[test]
    fn test_parse_diff_tree_output_empty() {
        let changes = parse_diff_tree_output("");
        assert!(changes.is_empty());

        let changes = parse_diff_tree_output("\n\n");
        assert!(changes.is_empty());
    }

    #[test]
    fn test_parse_diff_tree_output_unknown_status() {
        let output = "X\tunknown.rs\n";
        let changes = parse_diff_tree_output(output);
        assert!(changes.is_empty()); // Unknown statuses are skipped
    }

    #[test]
    fn test_diff_tree_with_real_repo() {
        // Create a temporary git repository with two commits
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        // Configure git identity for commits
        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        // Create first commit with a file
        let file1 = temp_dir.path().join("hello.txt");
        std::fs::write(&file1, "hello world").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("hello.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        let first_commit = repo.commit(
            Some("HEAD"), &sig, &sig, "first commit", &tree, &[],
        ).unwrap();

        // Create second commit: modify hello.txt and add new.txt
        std::fs::write(&file1, "hello world modified").unwrap();
        let file2 = temp_dir.path().join("new.txt");
        std::fs::write(&file2, "new file content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("hello.txt")).unwrap();
        index.add_path(Path::new("new.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let first = repo.find_commit(first_commit).unwrap();
        let second_commit = repo.commit(
            Some("HEAD"), &sig, &sig, "second commit", &tree, &[&first],
        ).unwrap();

        // Run diff_tree between the two commits
        let changes = diff_tree(
            temp_dir.path(),
            &first_commit.to_string(),
            &second_commit.to_string(),
        ).unwrap();

        // Should have 2 changes: Modified hello.txt, Added new.txt
        assert_eq!(changes.len(), 2);

        let modified = changes.iter().find(|c| c.path == "hello.txt").unwrap();
        assert_eq!(modified.status, FileChangeStatus::Modified);

        let added = changes.iter().find(|c| c.path == "new.txt").unwrap();
        assert_eq!(added.status, FileChangeStatus::Added);
    }

    #[test]
    fn test_diff_tree_with_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        // First commit with two files
        std::fs::write(temp_dir.path().join("keep.txt"), "keep").unwrap();
        std::fs::write(temp_dir.path().join("delete_me.txt"), "will be deleted").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("keep.txt")).unwrap();
        index.add_path(Path::new("delete_me.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        let first = repo.commit(Some("HEAD"), &sig, &sig, "first", &tree, &[]).unwrap();

        // Second commit: delete delete_me.txt
        let mut index = repo.index().unwrap();
        index.remove(Path::new("delete_me.txt"), 0).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let first_c = repo.find_commit(first).unwrap();
        let second = repo.commit(Some("HEAD"), &sig, &sig, "second", &tree, &[&first_c]).unwrap();

        let changes = diff_tree(temp_dir.path(), &first.to_string(), &second.to_string()).unwrap();

        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].status, FileChangeStatus::Deleted);
        assert_eq!(changes[0].path, "delete_me.txt");
    }

    #[test]
    fn test_get_blob_hash() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        std::fs::write(temp_dir.path().join("test.txt"), "test content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "commit", &tree, &[]).unwrap();

        // Get blob hash
        let hash = get_blob_hash(temp_dir.path(), "test.txt", "HEAD").unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 40); // SHA1 hex string
    }

    #[test]
    fn test_get_blob_hash_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        std::fs::write(temp_dir.path().join("exists.txt"), "content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("exists.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "commit", &tree, &[]).unwrap();

        let result = get_blob_hash(temp_dir.path(), "nonexistent.txt", "HEAD");
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_tree_not_git_repo() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = diff_tree(temp_dir.path(), "abc", "def");
        assert!(result.is_err());
    }
}
