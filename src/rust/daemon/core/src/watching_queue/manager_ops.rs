//! WatchManager operations - refresh, reconciliation, polling.

use std::sync::Arc;
use std::time::Duration;

use sqlx::Row;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use wqm_common::constants::COLLECTION_LIBRARIES;

use crate::queue_operations::QueueManager;

use super::manager::WatchManager;
use super::types::{WatchConfig, WatchType, WatchingQueueResult};

impl WatchManager {
    /// Refresh watches by checking for config changes (hot-reload support)
    pub async fn refresh_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        let db_watch_ids = self.get_enabled_watch_ids().await?;

        let running_ids: Vec<String> = {
            let watchers = self.watchers.read().await;
            watchers.keys().cloned().collect()
        };

        // Stop watchers that are no longer enabled
        for id in &running_ids {
            if !db_watch_ids.contains(id) {
                info!("Stopping disabled/removed watcher: {}", id);
                let watcher = {
                    let mut watchers = self.watchers.write().await;
                    watchers.remove(id)
                };
                if let Some(w) = watcher {
                    if let Err(e) = w.stop().await {
                        error!("Failed to stop watcher {}: {}", id, e);
                    }
                }
                let mut git_watchers = self.git_watchers.lock().await;
                if let Some(mut gw) = git_watchers.remove(id) {
                    gw.stop().await;
                    info!("Stopped git watcher for removed project: {}", id);
                }
            }
        }

        // Start watchers for newly enabled watches
        for id in &db_watch_ids {
            let already_running = {
                let watchers = self.watchers.read().await;
                watchers.contains_key(id)
            };

            if !already_running {
                info!("Starting newly enabled watcher: {}", id);
                if id.starts_with("lib_") {
                    self.start_single_library_watch(id, &queue_manager).await?;
                } else {
                    self.start_single_watch_folder(id, &queue_manager).await?;
                }
            }
        }

        self.reconcile_git_watchers().await;

        Ok(())
    }

    /// Reconcile git watcher lifecycle based on is_active and is_git_tracked (Task 12).
    async fn reconcile_git_watchers(&self) {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path,
                   COALESCE(is_git_tracked, 0) AS is_git_tracked,
                   COALESCE(is_active, 0) AS is_active
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'
            "#,
        )
        .fetch_all(&self.pool)
        .await;

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Failed to query watch_folders for git watcher reconciliation: {}",
                    e
                );
                return;
            }
        };

        for row in rows {
            let watch_id: String = row.get("watch_id");
            let path: String = row.get("path");
            let is_git_tracked: bool = row.get::<i32, _>("is_git_tracked") != 0;
            let is_active: bool = row.get::<i32, _>("is_active") != 0;

            let has_git_watcher = {
                let git_watchers = self.git_watchers.lock().await;
                git_watchers.contains_key(&watch_id)
            };

            if is_active && is_git_tracked && !has_git_watcher {
                debug!("Starting git watcher for active project: {}", watch_id);
                self.start_git_watcher(&watch_id, &path).await;
            } else if (!is_active || !is_git_tracked) && has_git_watcher {
                let mut git_watchers = self.git_watchers.lock().await;
                if let Some(mut gw) = git_watchers.remove(&watch_id) {
                    gw.stop().await;
                    info!(
                        "Stopped git watcher for {} project: {} (file watcher continues)",
                        if !is_active { "deactivated" } else { "non-git" },
                        watch_id
                    );
                }
            }
        }
    }

    /// Get all enabled, non-archived watch IDs from the watch_folders table.
    ///
    /// Must apply the same `enabled`/`is_archived` filters as the startup
    /// loaders: any row visible here but not startable would be retried by
    /// every refresh cycle forever.
    async fn get_enabled_watch_ids(&self) -> WatchingQueueResult<Vec<String>> {
        let mut ids = Vec::new();

        let rows = sqlx::query(
            "SELECT watch_id, collection, tenant_id FROM watch_folders \
             WHERE enabled = 1 AND is_archived = 0",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in rows {
            let watch_id: String = row.get("watch_id");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            if collection == COLLECTION_LIBRARIES {
                ids.push(format!("lib_{}", tenant_id));
            } else {
                ids.push(watch_id);
            }
        }

        Ok(ids)
    }

    /// Start a single watch folder by ID
    async fn start_single_watch_folder(
        &self,
        id: &str,
        queue_manager: &Arc<QueueManager>,
    ) -> WatchingQueueResult<()> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   COALESCE(is_git_tracked, 0) AS is_git_tracked
            FROM watch_folders
            WHERE watch_id = ? AND enabled = 1 AND is_archived = 0
              AND collection = 'projects'
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
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

    /// Start a single library watch by ID
    async fn start_single_library_watch(
        &self,
        id: &str,
        queue_manager: &Arc<QueueManager>,
    ) -> WatchingQueueResult<()> {
        let library_name = id.strip_prefix("lib_").unwrap_or(id);

        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE tenant_id = ? AND enabled = 1 AND is_archived = 0
              AND collection = 'libraries'
            "#,
        )
        .bind(library_name)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let db_watch_id: String = row.get("watch_id");
            let tenant_id: String = row.get("tenant_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");

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
                library_name: Some(tenant_id),
            };

            self.start_watcher(id, &db_watch_id, config, queue_manager.clone())
                .await;
        }

        Ok(())
    }

    /// Start periodic polling for watch configuration changes
    pub fn start_polling(self: Arc<Self>, poll_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        let refresh_signal = self.refresh_signal.clone();
        info!(
            "Starting watch configuration polling (interval: {}s, signal-driven: {})",
            poll_interval_secs,
            refresh_signal.is_some()
        );

        tokio::spawn(async move {
            let mut poll_interval = interval(Duration::from_secs(poll_interval_secs));

            loop {
                if let Some(ref signal) = refresh_signal {
                    tokio::select! {
                        _ = poll_interval.tick() => {
                            debug!("Periodic watch configuration refresh...");
                        }
                        _ = signal.notified() => {
                            info!("Watch configuration refresh triggered by signal");
                        }
                    }
                } else {
                    poll_interval.tick().await;
                    debug!("Polling for watch configuration changes...");
                }

                if let Err(e) = self.refresh_watches().await {
                    error!("Failed to refresh watches: {}", e);
                }
            }
        })
    }
}
