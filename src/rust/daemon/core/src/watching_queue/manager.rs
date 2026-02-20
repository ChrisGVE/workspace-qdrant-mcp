//! WatchManager - master coordinator for multiple file watchers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use tokio::sync::{mpsc, Notify, RwLock, Mutex};
use tracing::{debug, error, info, warn};

use crate::queue_operations::QueueManager;
use crate::allowed_extensions::AllowedExtensions;

use super::types::{WatchConfig, WatchType, WatchingQueueResult, WatchingQueueStats};
use super::file_watcher::FileWatcherQueue;
use super::error_state::WatchErrorState;
use super::error_types::{WatchErrorSummary, WatchHealthStatus};

/// Watch manager for multiple watchers
pub struct WatchManager {
    pub(super) pool: SqlitePool,
    pub(super) watchers: Arc<RwLock<HashMap<String, Arc<FileWatcherQueue>>>>,
    pub(super) git_watchers: Arc<Mutex<HashMap<String, crate::git_watcher::GitWatcher>>>,
    pub(super) git_event_tx: mpsc::UnboundedSender<crate::git_watcher::GitEvent>,
    pub(super) git_event_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<crate::git_watcher::GitEvent>>>>,
    pub(super) allowed_extensions: Arc<AllowedExtensions>,
    pub(super) refresh_signal: Option<Arc<Notify>>,
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
        }
    }

    /// Take the git event receiver (can only be called once).
    pub async fn take_git_event_rx(&self) -> Option<mpsc::UnboundedReceiver<crate::git_watcher::GitEvent>> {
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
    async fn load_watch_folders(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   COALESCE(is_git_tracked, 0) AS is_git_tracked
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} project watches from watch_folders table", rows.len());

        for row in rows {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let _tenant_id: String = row.get("tenant_id");
            let is_git_tracked: bool = row.get::<i32, _>("is_git_tracked") != 0;

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(&path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Project,
                library_name: None,
            };

            self.start_watcher(id.clone(), config, queue_manager.clone()).await;

            if is_git_tracked {
                self.start_git_watcher(&id, &path).await;
            }
        }

        Ok(())
    }

    /// Start a git watcher for a git-tracked project
    pub(super) async fn start_git_watcher(&self, watch_id: &str, project_path: &str) {
        use crate::git_watcher::GitWatcher;

        let project_root = PathBuf::from(project_path);
        match GitWatcher::new(
            watch_id.to_string(),
            project_root,
            self.git_event_tx.clone(),
        ) {
            Ok(mut git_watcher) => {
                match git_watcher.start() {
                    Ok(()) => {
                        info!("Started git watcher for project: {}", watch_id);
                        let mut git_watchers = self.git_watchers.lock().await;
                        git_watchers.insert(watch_id.to_string(), git_watcher);
                    }
                    Err(e) => {
                        warn!("Failed to start git watcher for {}: {}", watch_id, e);
                    }
                }
            }
            Err(e) => {
                debug!("No git watcher for {} (not a git repo or .git not found): {}", watch_id, e);
            }
        }
    }

    /// Load library watch configurations from watch_folders table
    async fn load_library_watches(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'libraries'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} library watches from watch_folders table", rows.len());

        for row in rows {
            let _watch_id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            let id = format!("lib_{}", tenant_id);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Library,
                library_name: Some(tenant_id.clone()),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start a single watcher with the given configuration
    pub(super) async fn start_watcher(&self, id: String, config: WatchConfig, queue_manager: Arc<QueueManager>) {
        match FileWatcherQueue::new(config, queue_manager, self.allowed_extensions.clone()) {
            Ok(watcher) => {
                let watcher = Arc::new(watcher);
                match watcher.start().await {
                    Ok(_) => {
                        info!("Started watcher: {} (type: {:?})", id,
                            if id.starts_with("lib_") { "library" } else { "project" });
                        let mut watchers = self.watchers.write().await;
                        watchers.insert(id, watcher);
                    },
                    Err(e) => {
                        error!("Failed to start watcher {}: {}", id, e);
                    }
                }
            },
            Err(e) => {
                error!("Failed to create watcher {}: {}", id, e);
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
        watchers.get(watch_id)
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
        summaries.into_iter()
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
