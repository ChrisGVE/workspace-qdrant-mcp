use notify::{RecursiveMode, Watcher as NotifyWatcher};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::reflog::{
    parse_reflog_last_entry, read_current_branch, resolve_common_dir, resolve_git_dir,
};
use super::watcher_types::{GitEvent, GitWatcherError, GitWatcherResult};

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
        let git_dir = resolve_git_dir(&project_root).ok_or_else(|| {
            GitWatcherError::GitDirNotFound(format!(
                "No .git directory found in {}",
                project_root.display()
            ))
        })?;

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
            notify::Config::default().with_poll_interval(Duration::from_secs(2)),
        )?;

        // For worktrees, refs/heads/ lives in the common dir, not the worktree dir.
        let common_dir = resolve_common_dir(&self.git_dir);

        // Watch .git/HEAD (for branch switches and commits affecting HEAD)
        let head_path = self.git_dir.join("HEAD");
        if head_path.exists() {
            if let Err(e) = watcher.watch(&head_path, RecursiveMode::NonRecursive) {
                warn!(
                    "Failed to watch .git/HEAD at {}: {}",
                    head_path.display(),
                    e
                );
            } else {
                debug!("Watching .git/HEAD: {}", head_path.display());
            }
        }

        // Watch .git/refs/heads/ (for branch ref updates from commits/pushes)
        let refs_heads = common_dir.join("refs").join("heads");
        if refs_heads.exists() {
            if let Err(e) = watcher.watch(&refs_heads, RecursiveMode::Recursive) {
                warn!(
                    "Failed to watch refs/heads/ at {}: {}",
                    refs_heads.display(),
                    e
                );
            } else {
                debug!("Watching refs/heads/: {}", refs_heads.display());
            }
        }

        // Watch logs/HEAD for reflog updates (more reliable than HEAD file itself)
        let logs_head = self.git_dir.join("logs").join("HEAD");
        if logs_head.exists() {
            if let Err(e) = watcher.watch(&logs_head, RecursiveMode::NonRecursive) {
                warn!(
                    "Failed to watch logs/HEAD at {}: {}",
                    logs_head.display(),
                    e
                );
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
        let debounce = Duration::from_millis(200);

        loop {
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
