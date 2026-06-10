//! WatchManager - master coordinator for multiple file watchers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tracing::{debug, error, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::queue_operations::QueueManager;

use super::error_state::WatchErrorState;
use super::error_types::{WatchErrorSummary, WatchHealthStatus};
use super::file_watcher::FileWatcherQueue;
use super::types::{WatchConfig, WatchType, WatchingQueueResult, WatchingQueueStats};

/// Consecutive start attempts against a missing path before the watch folder
/// is auto-disabled in SQLite. The refresh loop polls every 5 minutes, so the
/// threshold gives ~10-15 minutes of grace for transient unmounts before an
/// orphaned watch folder (e.g. a deleted worktree) stops being retried.
pub(super) const MISSING_PATH_DISABLE_THRESHOLD: u32 = 3;

/// Watch manager for multiple watchers
pub struct WatchManager {
    pub(super) pool: SqlitePool,
    pub(super) watchers: Arc<RwLock<HashMap<String, Arc<FileWatcherQueue>>>>,
    pub(super) git_watchers: Arc<Mutex<HashMap<String, crate::git::GitWatcher>>>,
    pub(super) git_event_tx: mpsc::UnboundedSender<crate::git::GitEvent>,
    pub(super) git_event_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<crate::git::GitEvent>>>>,
    pub(super) allowed_extensions: Arc<AllowedExtensions>,
    pub(super) refresh_signal: Option<Arc<Notify>>,
    pub(super) missing_path_strikes: Arc<Mutex<HashMap<String, u32>>>,
}

impl WatchManager {
    /// Create a new watch manager
    pub fn new(pool: SqlitePool, allowed_extensions: Arc<AllowedExtensions>) -> Self {
        let (git_event_tx, git_event_rx) = mpsc::unbounded_channel();
        Self {
            pool,
            watchers: Arc::new(RwLock::new(HashMap::new())),
            git_watchers: Arc::new(Mutex::new(HashMap::new())),
            git_event_tx,
            git_event_rx: Arc::new(Mutex::new(Some(git_event_rx))),
            allowed_extensions,
            refresh_signal: None,
            missing_path_strikes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Resolve a watch path to the local filesystem view used by this process.
    ///
    /// The daemon stores canonical paths in SQLite, but Docker Desktop on
    /// Windows can expose the same bind mount under `/run/desktop/mnt/host/...`
    /// instead of `/mnt/...`. Keep the canonical path when it already exists,
    /// otherwise try the common Docker Desktop aliases before falling back to
    /// the raw value so logs still show the stored DB path.
    pub(crate) fn resolve_local_watch_path(raw_path: &str) -> PathBuf {
        let primary = PathBuf::from(raw_path);
        if primary.exists() {
            return primary;
        }

        for alias in Self::watch_path_aliases(raw_path) {
            if alias.exists() {
                debug!(
                    "Resolved watch path {} to existing container path {}",
                    raw_path,
                    alias.display()
                );
                return alias;
            }
        }

        primary
    }

    /// Generate Docker Desktop alias candidates for a raw watch path.
    pub(crate) fn watch_path_aliases(raw_path: &str) -> Vec<PathBuf> {
        let mut aliases = Vec::new();

        if let Some(rest) = raw_path.strip_prefix("/mnt/") {
            aliases.push(PathBuf::from(format!("/run/desktop/mnt/host/{rest}")));
            aliases.push(PathBuf::from(format!("/host_mnt/{rest}")));
        } else if let Some(rest) = raw_path.strip_prefix("/host_mnt/") {
            aliases.push(PathBuf::from(format!("/mnt/{rest}")));
            aliases.push(PathBuf::from(format!("/run/desktop/mnt/host/{rest}")));
        } else if let Some(rest) = raw_path.strip_prefix("/run/desktop/mnt/host/") {
            aliases.push(PathBuf::from(format!("/mnt/{rest}")));
            aliases.push(PathBuf::from(format!("/host_mnt/{rest}")));
        }

        aliases
    }

    /// Take the git event receiver (can only be called once).
    pub async fn take_git_event_rx(&self) -> Option<mpsc::UnboundedReceiver<crate::git::GitEvent>> {
        let mut rx_lock = self.git_event_rx.lock().await;
        rx_lock.take()
    }

    /// Set the refresh signal for event-driven watch folder refresh
    pub fn with_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.refresh_signal = Some(signal);
        self
    }

    /// Load watch configurations from database and start watchers
    pub async fn start_all_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Load and start project watches from watch_folders table
        self.load_watch_folders(&queue_manager).await?;

        // Load and start library watches from watch_folders table
        self.load_library_watches(&queue_manager).await?;

        Ok(())
    }

    /// Load watch configurations from watch_folders table (project watches)
    async fn load_watch_folders(
        &self,
        queue_manager: &Arc<QueueManager>,
    ) -> WatchingQueueResult<()> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   COALESCE(is_git_tracked, 0) AS is_git_tracked
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        info!(
            "Loading {} project watches from watch_folders table",
            rows.len()
        );

        for row in rows {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");
            let is_git_tracked: bool = row.get::<i32, _>("is_git_tracked") != 0;

            let config = WatchConfig {
                id: id.clone(),
                path: Self::resolve_local_watch_path(&path),
                tenant_id,
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Project,
                library_name: None,
            };

            self.start_watcher(id.clone(), &id, config, queue_manager.clone())
                .await;

            if is_git_tracked {
                self.start_git_watcher(&id, &path).await;
            }
        }

        Ok(())
    }

    /// Start a git watcher for a git-tracked project
    pub(super) async fn start_git_watcher(&self, watch_id: &str, project_path: &str) {
        use crate::git::GitWatcher;

        let project_root = Self::resolve_local_watch_path(project_path);
        match GitWatcher::new(
            watch_id.to_string(),
            project_root,
            self.git_event_tx.clone(),
        ) {
            Ok(mut git_watcher) => match git_watcher.start() {
                Ok(()) => {
                    info!("Started git watcher for project: {}", watch_id);
                    let mut git_watchers = self.git_watchers.lock().await;
                    git_watchers.insert(watch_id.to_string(), git_watcher);
                }
                Err(e) => {
                    warn!("Failed to start git watcher for {}: {}", watch_id, e);
                }
            },
            Err(e) => {
                debug!(
                    "No git watcher for {} (not a git repo or .git not found): {}",
                    watch_id, e
                );
            }
        }
    }

    /// Load library watch configurations from watch_folders table
    async fn load_library_watches(
        &self,
        queue_manager: &Arc<QueueManager>,
    ) -> WatchingQueueResult<()> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'libraries'
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        info!(
            "Loading {} library watches from watch_folders table",
            rows.len()
        );

        for row in rows {
            let db_watch_id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            let id = format!("lib_{}", tenant_id);

            let config = WatchConfig {
                id: id.clone(),
                path: Self::resolve_local_watch_path(&path),
                tenant_id: tenant_id.clone(),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Library,
                library_name: Some(tenant_id.clone()),
            };

            self.start_watcher(id, &db_watch_id, config, queue_manager.clone())
                .await;
        }

        Ok(())
    }

    /// Start a single watcher with the given configuration.
    ///
    /// `db_watch_id` is the `watch_folders.watch_id` backing this watcher; it
    /// can differ from the runtime `id` (library watchers run as
    /// `lib_<tenant>`). It is used to auto-disable the row when the watched
    /// path no longer exists, instead of retrying the start forever.
    pub(super) async fn start_watcher(
        &self,
        id: String,
        db_watch_id: &str,
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
    ) {
        if !config.path.exists() {
            self.handle_missing_watch_path(&id, db_watch_id, &config.path)
                .await;
            return;
        }

        match FileWatcherQueue::new(config, queue_manager, self.allowed_extensions.clone()) {
            Ok(watcher) => {
                let watcher = Arc::new(watcher);
                match watcher.start().await {
                    Ok(_) => {
                        info!(
                            "Started watcher: {} (type: {:?})",
                            id,
                            if id.starts_with("lib_") {
                                "library"
                            } else {
                                "project"
                            }
                        );
                        self.missing_path_strikes.lock().await.remove(&id);
                        let mut watchers = self.watchers.write().await;
                        watchers.insert(id, watcher);
                    }
                    Err(e) => {
                        error!("Failed to start watcher {}: {}", id, e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to create watcher {}: {}", id, e);
            }
        }
    }

    /// Record a start attempt against a missing watch path; once the strike
    /// threshold is reached, disable the watch folder in SQLite so startup and
    /// refresh stop retrying it forever (orphaned watcher, e.g. a worktree
    /// deleted after registration).
    async fn handle_missing_watch_path(&self, id: &str, db_watch_id: &str, path: &Path) {
        let strikes = {
            let mut strikes = self.missing_path_strikes.lock().await;
            let count = strikes.entry(id.to_string()).or_insert(0);
            *count += 1;
            *count
        };

        if strikes < MISSING_PATH_DISABLE_THRESHOLD {
            warn!(
                "Watch path does not exist for {}: {} (attempt {}/{}, auto-disable at threshold)",
                id,
                path.display(),
                strikes,
                MISSING_PATH_DISABLE_THRESHOLD
            );
            return;
        }

        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET enabled = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?1 AND enabled = 1
            "#,
        )
        .bind(db_watch_id)
        .execute(&self.pool)
        .await;

        match result {
            Ok(r) if r.rows_affected() > 0 => {
                error!(
                    "Auto-disabled watch folder {} ({}): path missing for {} consecutive start attempts: {}",
                    db_watch_id,
                    id,
                    strikes,
                    path.display()
                );
                self.missing_path_strikes.lock().await.remove(id);
            }
            Ok(_) => {
                // Row already disabled or removed concurrently; drop the counter.
                self.missing_path_strikes.lock().await.remove(id);
            }
            Err(e) => {
                // Keep the strikes so the next refresh cycle retries the disable.
                warn!(
                    "Failed to auto-disable watch folder {} with missing path: {}",
                    db_watch_id, e
                );
            }
        }
    }

    /// Get statistics for all watchers
    pub async fn get_all_stats(&self) -> HashMap<String, WatchingQueueStats> {
        let watchers = self.watchers.read().await;
        let mut stats = HashMap::new();

        for (id, watcher) in watchers.iter() {
            stats.insert(id.clone(), watcher.get_stats().await);
        }

        stats
    }

    /// Get count of active watchers
    pub async fn active_watcher_count(&self) -> usize {
        let watchers = self.watchers.read().await;
        watchers.len()
    }

    /// Check if a specific watch is active
    pub async fn is_watch_active(&self, id: &str) -> bool {
        let watchers = self.watchers.read().await;
        watchers.contains_key(id)
    }

    /// Get error state for a specific watcher (Task 461.5)
    pub async fn get_error_state(&self, watch_id: &str) -> Option<WatchErrorState> {
        let watchers = self.watchers.read().await;
        watchers
            .get(watch_id)
            .and_then(|w| w.error_tracker().get_state(watch_id))
    }

    /// Get error summaries for all watchers (Task 461.5)
    pub async fn get_all_error_summaries(&self) -> HashMap<String, WatchErrorSummary> {
        let watchers = self.watchers.read().await;
        let mut summaries = HashMap::new();

        for (id, watcher) in watchers.iter() {
            if let Some(summary) = watcher.error_tracker().get_summary(id) {
                summaries.insert(id.clone(), summary);
            }
        }

        summaries
    }

    /// Get watches with health status worse than healthy (Task 461.5)
    pub async fn get_unhealthy_watches(&self) -> Vec<(String, WatchErrorSummary)> {
        let summaries = self.get_all_error_summaries().await;
        summaries
            .into_iter()
            .filter(|(_, summary)| summary.health_status != WatchHealthStatus::Healthy)
            .collect()
    }

    /// Stop all watchers
    pub async fn stop_all_watches(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        for (id, watcher) in watchers.iter() {
            if let Err(e) = watcher.stop().await {
                error!("Failed to stop watcher {}: {}", id, e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docker_desktop_aliases_include_common_windows_paths() {
        let aliases = WatchManager::watch_path_aliases("/mnt/c/Users/alber/dev");
        assert_eq!(
            aliases[0],
            PathBuf::from("/run/desktop/mnt/host/c/Users/alber/dev")
        );
        assert_eq!(aliases[1], PathBuf::from("/host_mnt/c/Users/alber/dev"));
    }
}
