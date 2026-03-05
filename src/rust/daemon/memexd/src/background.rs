//! Phase 3: Background periodic task spawns.
//!
//! Spawns long-running tokio tasks for periodic maintenance: pause-state polling,
//! metrics collection, processing-timings cleanup, log pruning, inactivity timeout,
//! remote URL monitoring, git state change detection, uptime tracking, and the
//! Prometheus metrics endpoint.

use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use workspace_qdrant_core::{
    check_git_state_changes, check_remote_url_changes, metrics_history, poll_pause_state,
    processing_timings, MetricsServer, METRICS,
};

/// Handles for all background tasks so the orchestrator can abort them on shutdown.
pub struct BackgroundHandles {
    pub uptime_handle: JoinHandle<()>,
    pub pause_poll_handle: JoinHandle<()>,
    pub metrics_collect_handle: JoinHandle<()>,
    pub metrics_maint_handle: JoinHandle<()>,
    pub grpc_handle: Option<JoinHandle<()>>,
    pub metrics_handle: Option<JoinHandle<()>>,
}

/// Start the Prometheus metrics endpoint (if port is specified).
pub fn start_metrics_server(metrics_port: Option<u16>) -> Option<JoinHandle<()>> {
    if let Some(port) = metrics_port {
        info!("Starting Prometheus metrics endpoint on port {}", port);
        let mut metrics_server = MetricsServer::new(port);
        let handle = tokio::spawn(async move {
            if let Err(e) = metrics_server.start().await {
                error!("Metrics server error: {}", e);
            }
        });
        Some(handle)
    } else {
        info!("Metrics endpoint disabled (use --metrics-port to enable)");
        None
    }
}

/// Start uptime tracking (updates the global METRICS gauge every second).
pub fn start_uptime_tracker() -> JoinHandle<()> {
    let start_time = std::time::Instant::now();
    tokio::spawn(async move {
        loop {
            METRICS.set_uptime(start_time.elapsed().as_secs_f64());
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    })
}

/// Start periodic DB polling for CLI-driven pause state changes (Task 543.10).
pub fn start_pause_polling(
    pool: SqlitePool,
    pause_flag: Arc<std::sync::atomic::AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            match poll_pause_state(&pool, &pause_flag).await {
                Ok(true) => {
                    let is_paused = pause_flag.load(std::sync::atomic::Ordering::SeqCst);
                    info!(
                        "Pause state changed via DB: watchers are now {}",
                        if is_paused { "PAUSED" } else { "ACTIVE" }
                    );
                }
                Ok(false) => {} // No change
                Err(e) => {
                    warn!("Failed to poll pause state: {}", e);
                }
            }
        }
    })
}

/// Start periodic metrics history collection (Task 544.6).
pub fn start_metrics_collection(pool: SqlitePool) -> JoinHandle<()> {
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            match metrics_history::write_snapshot(&pool).await {
                Ok(count) => {
                    if count > 0 {
                        debug!("Collected {} metrics to history", count);
                    }
                }
                Err(e) => {
                    warn!("Failed to collect metrics history: {}", e);
                }
            }
        }
    });
    info!("Metrics history collection started (60s interval)");
    handle
}

/// Start hourly metrics maintenance: aggregation + retention (Task 544.11-14).
pub fn start_metrics_maintenance(pool: SqlitePool) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            if let Err(e) = metrics_history::run_maintenance_now(&pool).await {
                warn!("Metrics maintenance failed: {}", e);
            }
        }
    })
}

/// Start hourly processing timings cleanup (Task 42).
pub fn start_timings_cleanup(pool: SqlitePool) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            processing_timings::cleanup_old_timings(&pool, 30).await;
        }
    })
}

/// Start periodic log pruning (cli-qol task 12).
pub fn start_log_pruning(pool: SqlitePool) -> JoinHandle<()> {
    let log_prune_dir = workspace_qdrant_core::logging::get_canonical_log_dir();
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            match workspace_qdrant_core::log_pruner::run_if_due(
                &pool,
                &log_prune_dir,
                36, // retention: 36 hours
                12, // check interval: 12 hours
            )
            .await
            {
                Ok(Some(result)) => {
                    if result.files_deleted > 0 {
                        info!(
                            files = result.files_deleted,
                            bytes = result.bytes_freed,
                            "Log pruning completed"
                        );
                    }
                }
                Ok(None) => {} // not due yet
                Err(e) => warn!("Log pruning failed: {}", e),
            }
        }
    });
    info!("Log pruning started (36h retention, 12h interval)");
    handle
}

/// Start periodic inactivity timeout check (Task 569).
pub fn start_inactivity_timeout(pool: SqlitePool) -> JoinHandle<()> {
    let inactivity_timeout_secs: i64 = std::env::var("WQM_INACTIVITY_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(43200);

    let handle = tokio::spawn(async move {
        let state_mgr = workspace_qdrant_core::daemon_state::DaemonStateManager::with_pool(pool);
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // every 5 min
        loop {
            interval.tick().await;
            match state_mgr
                .deactivate_inactive_projects(inactivity_timeout_secs)
                .await
            {
                Ok(0) => {} // no stale projects
                Ok(n) => info!("Inactivity timeout: deactivated {} project group(s)", n),
                Err(e) => warn!("Inactivity timeout check failed: {}", e),
            }
        }
    });
    info!(
        "Inactivity timeout polling started (5min interval, {}s timeout)",
        std::env::var("WQM_INACTIVITY_TIMEOUT_SECS").unwrap_or_else(|_| "43200".to_string())
    );
    handle
}

/// Start periodic remote URL change detection (Task 584).
pub fn start_remote_url_monitor(pool: SqlitePool) -> JoinHandle<()> {
    let handle = tokio::spawn(async move {
        let queue_manager =
            workspace_qdrant_core::queue_operations::QueueManager::new(pool.clone());
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            match check_remote_url_changes(&pool, &queue_manager).await {
                Ok(result) => {
                    if result.changes_detected > 0 {
                        info!(
                            "Remote URL monitoring: {} change(s) detected, {} checked, {} error(s)",
                            result.changes_detected, result.projects_checked, result.errors
                        );
                    }
                }
                Err(e) => {
                    warn!("Remote URL monitoring failed: {}", e);
                }
            }
        }
    });
    info!("Remote URL monitoring started (30s interval)");
    handle
}

/// Start periodic git state change detection (transitions 1-5).
pub fn start_git_state_monitor(pool: SqlitePool) -> JoinHandle<()> {
    let handle = tokio::spawn(async move {
        let queue_manager =
            workspace_qdrant_core::queue_operations::QueueManager::new(pool.clone());
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            match check_git_state_changes(&pool, &queue_manager).await {
                Ok(result) => {
                    if result.transitions_detected > 0 {
                        info!(
                            "Git state monitoring: {} transition(s) detected, {} checked, {} error(s)",
                            result.transitions_detected, result.projects_checked, result.errors
                        );
                    }
                }
                Err(e) => {
                    warn!("Git state monitoring failed: {}", e);
                }
            }
        }
    });
    info!("Git state monitoring started (60s interval)");
    handle
}

/// Spawn all periodic background tasks and return their handles.
pub fn spawn_all(
    pool: &SqlitePool,
    pause_flag: &Arc<std::sync::atomic::AtomicBool>,
    metrics_port: Option<u16>,
) -> BackgroundHandles {
    let metrics_handle = start_metrics_server(metrics_port);
    let uptime_handle = start_uptime_tracker();
    let pause_poll_handle = start_pause_polling(pool.clone(), Arc::clone(pause_flag));
    let metrics_collect_handle = start_metrics_collection(pool.clone());
    let metrics_maint_handle = start_metrics_maintenance(pool.clone());

    // Fire-and-forget background tasks (handles not needed for shutdown abort)
    let _timings = start_timings_cleanup(pool.clone());
    let _log_prune = start_log_pruning(pool.clone());
    let _inactivity = start_inactivity_timeout(pool.clone());
    let _remote = start_remote_url_monitor(pool.clone());
    let _git_state = start_git_state_monitor(pool.clone());

    BackgroundHandles {
        uptime_handle,
        pause_poll_handle,
        metrics_collect_handle,
        metrics_maint_handle,
        grpc_handle: None, // Filled in later by grpc_setup
        metrics_handle,
    }
}
