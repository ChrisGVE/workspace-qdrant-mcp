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

use workspace_qdrant_core::config::PrometheusExportConfig;
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

/// Start the Prometheus metrics endpoint when `config.enabled` is true.
pub fn start_metrics_server(config: &PrometheusExportConfig) -> Option<JoinHandle<()>> {
    if !config.enabled {
        info!(
            "Prometheus metrics endpoint disabled (set telemetry.prometheus.enabled=true \
             or pass --metrics-port to enable)"
        );
        return None;
    }
    let mut metrics_server = match MetricsServer::from_config(config) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to build metrics server from config: {}", e);
            return None;
        }
    };
    info!(
        "Starting Prometheus metrics endpoint on {}:{}",
        config.bind, config.port
    );
    let handle = tokio::spawn(async move {
        if let Err(e) = metrics_server.start().await {
            error!("Metrics server error: {}", e);
        }
    });
    Some(handle)
}

/// Build an effective PrometheusExportConfig by merging the CLI `--metrics-port`
/// override (if provided) on top of the config-file values. The CLI flag flips
/// `enabled=true` when set, preserving the documented behavior of the flag.
pub fn resolve_prometheus_config(
    base: PrometheusExportConfig,
    cli_override_port: Option<u16>,
) -> PrometheusExportConfig {
    match cli_override_port {
        Some(port) => PrometheusExportConfig {
            enabled: true,
            port,
            bind: base.bind,
        },
        None => base,
    }
}

/// Start periodic graph-metrics snapshotting (PRD D5). Every `interval_secs`
/// (the daemon `collection_interval`, default 60s) it runs a single bounded
/// SQLite read transaction over the graph DB and updates the Phase-1 graph
/// gauges. Failures are logged, never fatal.
pub fn start_graph_metrics_collection(
    graph_pool: SqlitePool,
    interval_secs: u64,
) -> JoinHandle<()> {
    let period = std::time::Duration::from_secs(interval_secs.max(1));
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        loop {
            interval.tick().await;
            if let Err(e) =
                workspace_qdrant_core::graph::metrics::snapshot_graph_metrics(&graph_pool).await
            {
                warn!("Failed to snapshot graph metrics: {}", e);
            }
        }
    });
    info!(
        "Graph metrics collection started ({}s interval)",
        interval_secs
    );
    handle
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
            // State-DB maintenance gauges (D5 DATA-N1): last vacuum/integrity
            // timestamps from the db_maintenance table on the same state pool.
            if let Err(e) =
                workspace_qdrant_core::monitoring::state_db_metrics::snapshot_state_db_metrics(
                    &pool,
                )
                .await
            {
                warn!("Failed to snapshot state-DB metrics: {}", e);
            }
            // Phase-1 SQLite state gauges (D5): schema version, sizes, WAL,
            // free pages, per-table rows, weekly integrity check.
            if let Err(e) =
                workspace_qdrant_core::monitoring::sqlite_metrics::snapshot_sqlite_state_metrics(
                    &pool,
                )
                .await
            {
                warn!("Failed to snapshot SQLite state metrics: {}", e);
            }
        }
    });
    info!("Metrics history collection started (60s interval)");
    handle
}

/// Periodically refresh unified queue depth gauges (issue-64 Task 4).
///
/// Queries the queue for pending/in_progress/failed counts grouped by
/// `(item_type, status)` and pushes them into the Prometheus gauge so
/// `/metrics` reflects real queue size without instrumenting every DB mutation.
pub fn start_queue_depth_exporter(pool: SqlitePool) -> JoinHandle<()> {
    use workspace_qdrant_core::queue_operations::QueueManager;

    let handle = tokio::spawn(async move {
        let manager = QueueManager::new(pool.clone());
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        let known_pairs: std::sync::Arc<
            tokio::sync::Mutex<std::collections::HashSet<(String, String)>>,
        > = std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashSet::new()));
        loop {
            interval.tick().await;
            match manager.get_unified_queue_depth_by_type_status().await {
                Ok(rows) => {
                    let mut seen = std::collections::HashSet::new();
                    for (item_type, status, count) in rows {
                        METRICS.set_unified_queue_depth(&item_type, &status, count);
                        seen.insert((item_type, status));
                    }
                    // Zero-out any (item_type, status) pairs we've seen before
                    // but aren't present now, so gauges don't get stuck.
                    let mut guard = known_pairs.lock().await;
                    for pair in guard.iter() {
                        if !seen.contains(pair) {
                            METRICS.set_unified_queue_depth(&pair.0, &pair.1, 0);
                        }
                    }
                    guard.extend(seen);
                }
                Err(e) => debug!("queue depth refresh failed: {}", e),
            }
            match manager.get_unified_queue_stats().await {
                Ok(stats) => METRICS.set_unified_queue_stale_items(stats.stale_leases),
                Err(e) => debug!("queue stats refresh failed: {}", e),
            }
        }
    });
    info!("Queue depth exporter started (10s interval)");
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
    prometheus_config: &PrometheusExportConfig,
) -> BackgroundHandles {
    let metrics_handle = start_metrics_server(prometheus_config);
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
    let _queue_depth = start_queue_depth_exporter(pool.clone());

    BackgroundHandles {
        uptime_handle,
        pause_poll_handle,
        metrics_collect_handle,
        metrics_maint_handle,
        grpc_handle: None, // Filled in later by grpc_setup
        metrics_handle,
    }
}
