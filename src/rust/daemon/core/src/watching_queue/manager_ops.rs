//! WatchManager operations - refresh, reconciliation, persistence, polling.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sqlx::Row;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::queue_operations::QueueManager;

use super::types::{WatchConfig, WatchType, WatchingQueueResult};
use super::error_state::WatchErrorState;
use super::error_types::WatchHealthStatus;
use super::manager::WatchManager;

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
            "#
        )
        .fetch_all(&self.pool)
        .await;

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to query watch_folders for git watcher reconciliation: {}", e);
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

    /// Get all enabled watch IDs from watch_folders table
    async fn get_enabled_watch_ids(&self) -> WatchingQueueResult<Vec<String>> {
        let mut ids = Vec::new();

        let rows = sqlx::query("SELECT watch_id, collection, tenant_id FROM watch_folders WHERE enabled = 1")
            .fetch_all(&self.pool)
            .await?;
        for row in rows {
            let watch_id: String = row.get("watch_id");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            if collection == "libraries" {
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
            WHERE watch_id = ? AND enabled = 1 AND collection = 'projects'
            "#
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
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
            WHERE tenant_id = ? AND enabled = 1 AND collection = 'libraries'
            "#
        )
        .bind(library_name)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let tenant_id: String = row.get("tenant_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");

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
                library_name: Some(tenant_id),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start periodic polling for watch configuration changes
    pub fn start_polling(self: Arc<Self>, poll_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        let refresh_signal = self.refresh_signal.clone();
        info!("Starting watch configuration polling (interval: {}s, signal-driven: {})",
            poll_interval_secs, refresh_signal.is_some());

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

    /// Persist error states to SQLite watch_folders table (Task 461.5)
    pub async fn persist_error_states(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        for (id, watcher) in watchers.iter() {
            if id.starts_with("lib_") {
                continue;
            }

            if let Some(state) = watcher.error_tracker().get_state(id) {
                let last_error_at = state.last_error_time
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                let last_success_at = state.last_successful_processing
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                let backoff_until = state.backoff_until
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                let result = sqlx::query(
                    r#"
                    UPDATE watch_folders
                    SET consecutive_errors = ?,
                        total_errors = ?,
                        last_error_at = ?,
                        last_error_message = ?,
                        backoff_until = ?,
                        last_success_at = ?,
                        health_status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE watch_id = ?
                    "#
                )
                .bind(state.consecutive_errors as i64)
                .bind(state.total_errors as i64)
                .bind(last_error_at)
                .bind(&state.last_error_message)
                .bind(backoff_until)
                .bind(last_success_at)
                .bind(state.health_status.as_str())
                .bind(id)
                .execute(&self.pool)
                .await;

                if let Err(e) = result {
                    warn!("Failed to persist error state for watch {}: {}", id, e);
                } else {
                    debug!("Persisted error state for watch {}: health={}", id, state.health_status.as_str());
                }
            }
        }

        Ok(())
    }

    /// Load error states from SQLite into watchers (Task 461.5)
    pub async fn load_error_states(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        let rows = sqlx::query(
            r#"
            SELECT watch_id, consecutive_errors, total_errors, last_error_at,
                   last_error_message, backoff_until, last_success_at, health_status
            FROM watch_folders
            WHERE consecutive_errors > 0 OR health_status != 'healthy'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        for row in rows {
            let watch_id: String = row.get("watch_id");

            if let Some(watcher) = watchers.get(&watch_id) {
                let consecutive_errors: i64 = row.get("consecutive_errors");
                let total_errors: i64 = row.get("total_errors");
                let last_error_message: Option<String> = row.get("last_error_message");
                let health_status_str: String = row.get("health_status");

                let health_status = match health_status_str.as_str() {
                    "healthy" => WatchHealthStatus::Healthy,
                    "degraded" => WatchHealthStatus::Degraded,
                    "backoff" => WatchHealthStatus::Backoff,
                    "disabled" => WatchHealthStatus::Disabled,
                    "half_open" => WatchHealthStatus::HalfOpen,
                    _ => WatchHealthStatus::Healthy,
                };

                let restored_state = WatchErrorState {
                    consecutive_errors: consecutive_errors as u32,
                    total_errors: total_errors as u64,
                    last_error_time: None,
                    last_error_message,
                    backoff_level: 0,
                    last_successful_processing: None,
                    health_status,
                    consecutive_successes: 0,
                    backoff_until: None,
                    errors_in_window: Vec::new(),
                    circuit_opened_at: None,
                    half_open_attempts: 0,
                    half_open_successes: 0,
                };

                watcher.error_tracker().set_state(&watch_id, restored_state);

                debug!("Restored error state for watch {}: errors={}, health={}",
                    watch_id, consecutive_errors, health_status_str);
            }
        }

        Ok(())
    }

    /// Start periodic error state persistence (Task 461.5)
    pub fn start_error_state_persistence(self: Arc<Self>, persist_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        info!("Starting error state persistence (interval: {}s)", persist_interval_secs);

        tokio::spawn(async move {
            let mut persist_interval = interval(Duration::from_secs(persist_interval_secs));

            loop {
                persist_interval.tick().await;

                debug!("Persisting error states to SQLite...");
                if let Err(e) = self.persist_error_states().await {
                    error!("Failed to persist error states: {}", e);
                }
            }
        })
    }
}
