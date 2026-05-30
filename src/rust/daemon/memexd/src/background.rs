//! Phase 3: Background periodic task spawns.
//!
//! Spawns long-running tokio tasks for periodic maintenance: pause-state polling,
//! metrics collection, processing-timings cleanup, log pruning, inactivity timeout,
//! remote URL monitoring, git state change detection, uptime tracking, and the
//! Prometheus metrics endpoint.

use std::sync::Arc;

use chrono::DateTime;
use sqlx::{Row, SqlitePool};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use workspace_qdrant_core::config::PrometheusExportConfig;
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::{
    check_git_state_changes, check_remote_url_changes, metrics_history, poll_pause_state,
    processing_timings, LanguageServerManager, MetricsServer, METRICS,
};

/// Handles for all background tasks so the orchestrator can abort them on shutdown.
pub struct BackgroundHandles {
    pub uptime_handle: JoinHandle<()>,
    pub pause_poll_handle: JoinHandle<()>,
    pub metrics_collect_handle: JoinHandle<()>,
    pub metrics_maint_handle: JoinHandle<()>,
    pub grpc_handle: Option<JoinHandle<()>>,
    pub metrics_handle: Option<JoinHandle<()>>,
    /// Handle for the LSP Prometheus metrics poller.  `None` when LSP is
    /// disabled or when the manager could not be initialized.
    pub lsp_metrics_handle: Option<JoinHandle<()>>,
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
        let known_tenant_pairs: std::sync::Arc<
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
            // Per-tenant indexing-progress gauge (Grafana / MCP indexing block).
            match manager.get_unified_queue_depth_by_tenant_status().await {
                Ok(rows) => {
                    let mut seen = std::collections::HashSet::new();
                    // Accumulate (pending + in_progress) per tenant for ETA.
                    let mut in_flight_by_tenant: std::collections::HashMap<String, i64> =
                        std::collections::HashMap::new();
                    for (tenant_id, status, count) in rows {
                        METRICS.set_unified_queue_depth_by_tenant(&tenant_id, &status, count);
                        if status == "pending" || status == "in_progress" {
                            *in_flight_by_tenant.entry(tenant_id.clone()).or_insert(0) += count;
                        }
                        seen.insert((tenant_id, status));
                    }
                    let mut guard = known_tenant_pairs.lock().await;
                    for pair in guard.iter() {
                        if !seen.contains(pair) {
                            METRICS.set_unified_queue_depth_by_tenant(&pair.0, &pair.1, 0);
                        }
                    }
                    guard.extend(seen);
                    drop(guard);

                    // Per-tenant ETA gauge: query the rate from
                    // tracked_files for every tenant we just observed.
                    // Tenants that drained (in-flight == 0) get ETA = 0
                    // so the Grafana panel shows "done" instead of stale.
                    use workspace_qdrant_core::indexing_progress::{
                        estimate_eta_seconds, eta_for_gauge, rate_files_per_sec,
                    };
                    for (tenant_id, in_flight) in in_flight_by_tenant {
                        let rate = rate_files_per_sec(&pool, &tenant_id).await;
                        // Split in_flight back into a (pending, in_progress)
                        // pair only for the API shape — the sum is what
                        // matters for ETA, so pass it all as `pending`.
                        let eta = estimate_eta_seconds(in_flight, 0, rate);
                        METRICS.set_indexing_eta_seconds(&tenant_id, eta_for_gauge(eta));
                    }
                }
                Err(e) => debug!("per-tenant queue depth refresh failed: {}", e),
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

/// Periodically refresh the indexed-project inventory metric from SQLite.
///
/// Exports one Prometheus series per indexed project row so Grafana can render
/// a live table of project metadata without querying SQLite directly.
pub fn start_indexed_project_inventory_exporter(pool: SqlitePool) -> JoinHandle<()> {
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let query = r#"
                SELECT
                    wf.watch_id,
                    wf.tenant_id,
                    wf.path,
                    wf.enabled,
                    wf.is_active,
                    wf.is_paused,
                    wf.is_archived,
                    wf.is_worktree,
                    wf.is_git_tracked,
                    wf.git_remote_url,
                    COUNT(tf.file_id) AS document_count,
                    COALESCE(SUM(tf.chunk_count), 0) AS point_count,
                    wf.last_scan,
                    wf.last_activity_at
                FROM watch_folders wf
                LEFT JOIN tracked_files tf
                    ON tf.watch_folder_id = wf.watch_id
                WHERE wf.collection = 'projects'
                GROUP BY
                    wf.watch_id,
                    wf.tenant_id,
                    wf.path,
                    wf.enabled,
                    wf.is_active,
                    wf.is_paused,
                    wf.is_archived,
                    wf.is_worktree,
                    wf.is_git_tracked,
                    wf.git_remote_url
                ORDER BY document_count DESC, point_count DESC, wf.tenant_id ASC, wf.path ASC
            "#;

            match sqlx::query(query).fetch_all(&pool).await {
                Ok(rows) => {
                    METRICS.indexed_project_tracked_files.reset();
                    METRICS.indexed_project_points.reset();
                    METRICS.indexed_project_last_scan_seconds.reset();
                    METRICS.indexed_project_last_activity_seconds.reset();

                    for row in rows {
                        let watch_id: String = row.get("watch_id");
                        let tenant_id: String = row.get("tenant_id");
                        let path: String = row.get("path");
                        let enabled: i32 = row.get("enabled");
                        let is_active: i32 = row.get("is_active");
                        let is_paused: i32 = row.get("is_paused");
                        let is_archived: i32 = row.get("is_archived");
                        let is_worktree: i32 = row.get("is_worktree");
                        let is_git_tracked: i32 = row.get("is_git_tracked");
                        let git_remote_url: Option<String> = row.get("git_remote_url");
                        let document_count: i64 = row.get("document_count");
                        let point_count: i64 = row.get("point_count");
                        let last_scan_epoch: Option<i64> =
                            row.get::<Option<String>, _>("last_scan").and_then(|value| {
                                DateTime::parse_from_rfc3339(&value)
                                    .ok()
                                    .map(|dt| dt.timestamp())
                            });
                        let last_activity_epoch: Option<i64> = row
                            .get::<Option<String>, _>("last_activity_at")
                            .and_then(|value| {
                                DateTime::parse_from_rfc3339(&value)
                                    .ok()
                                    .map(|dt| dt.timestamp())
                            });

                        METRICS.set_indexed_project_tracked_files(
                            &watch_id,
                            &tenant_id,
                            &path,
                            enabled != 0,
                            is_active != 0,
                            is_paused != 0,
                            is_archived != 0,
                            is_worktree != 0,
                            is_git_tracked != 0,
                            git_remote_url.as_deref().unwrap_or(""),
                            document_count,
                        );
                        METRICS.set_indexed_project_points(&watch_id, point_count);
                        METRICS.set_indexed_project_last_scan(&watch_id, last_scan_epoch);
                        METRICS.set_indexed_project_last_activity(&watch_id, last_activity_epoch);
                    }
                }
                Err(e) => debug!("indexed project inventory refresh failed: {}", e),
            }
        }
    });
    info!("Indexed project inventory exporter started (30s interval)");
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

/// Periodically refresh `file_metadata`-derived gauges from search.db
/// (Task #3 of the FTS5 size-guard series).
///
/// Queries `file_metadata` grouped by `(tenant_id, branch)` every 30s and
/// pushes file_count, total_bytes, and fts5_skipped_count into the matching
/// Prometheus gauges. Skipped pairs that disappear (e.g., after a project
/// is removed) are zeroed-out so panels don't show stale series — mirrors
/// the convention in `start_queue_depth_exporter`.
///
/// Cardinality: one series per (tenant_id, branch) pair across each gauge.
/// On this stack that's ~5 tenants × ~5 branches = ~25 series total, well
/// within Prometheus comfort zone. Adding a path-level label would explode
/// to ~10k series and is intentionally NOT included; for per-file inspection
/// use the admin UI / sidecar SQL queries.
pub fn start_file_metadata_exporter(search_db: Arc<SearchDbManager>) -> JoinHandle<()> {
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        // Tracks every (tenant_id, branch) pair we've ever emitted so we can
        // zero-out gauges for pairs that vanish (deleted projects / branches).
        let known: std::sync::Arc<tokio::sync::Mutex<std::collections::HashSet<(String, String)>>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashSet::new()));

        loop {
            interval.tick().await;
            match search_db.file_metadata_stats_by_tenant_branch().await {
                Ok(rows) => {
                    let mut seen = std::collections::HashSet::new();
                    for row in rows {
                        METRICS.set_file_metadata_stats(
                            &row.tenant_id,
                            &row.branch,
                            row.file_count,
                            row.total_bytes,
                            row.skipped_count,
                        );
                        seen.insert((row.tenant_id, row.branch));
                    }
                    let mut guard = known.lock().await;
                    for pair in guard.iter() {
                        if !seen.contains(pair) {
                            METRICS.set_file_metadata_stats(&pair.0, &pair.1, 0, 0, 0);
                        }
                    }
                    guard.extend(seen);
                }
                Err(e) => debug!("file_metadata stats refresh failed: {}", e),
            }
        }
    });
    info!("file_metadata exporter started (30s interval)");
    handle
}

/// Spawn the LSP Prometheus metrics collector (30-second polling loop).
///
/// Every tick it reads `LanguageServerManager::stats()` /
/// `available_languages()` / `active_languages()` and pushes the snapshot
/// into the global `METRICS` gauges.  The task is fire-and-forget during
/// normal operation; `abort_background_tasks` stops it on shutdown.
pub fn start_lsp_metrics_collector(
    lsp_manager: Arc<tokio::sync::RwLock<LanguageServerManager>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        info!("LSP metrics collector started (30s interval)");
        loop {
            interval.tick().await;
            let mgr = lsp_manager.read().await;

            let stats = mgr.stats().await;
            let available = mgr.available_languages().await;
            let active = mgr.active_languages().await;

            METRICS.set_lsp_snapshot(
                available.len() as i64,
                stats.active_servers as i64,
            );

            // Mark all detected-available languages as their running state.
            let active_set: std::collections::HashSet<&str> =
                active.iter().map(|l| l.identifier()).collect();

            for lang in &available {
                METRICS.set_lsp_server_state(
                    lang.identifier(),
                    active_set.contains(lang.identifier()),
                );
            }

            debug!(
                available = available.len(),
                active_servers = stats.active_servers,
                "LSP metrics snapshot updated"
            );
        }
    })
}

/// Spawn the graph stub-edge resolver (120-second loop).
///
/// Tree-sitter emits name-only "stub" callee/import targets (empty file_path)
/// that never match the callee's real node — so the raw call graph is 100%
/// dangling. This task periodically repoints each resolvable stub edge to the
/// real project symbol of the same name (see
/// `GraphStore::resolve_stub_edges`), turning the dangling baseline into a
/// usable intra-project relationship graph for PageRank/communities/impact.
/// Stdlib/external names (no project node) stay dangling and are excluded.
///
/// Runs periodically (not per-file) so it stays O(dangling) per tenant and
/// converges as indexing settles. It also heals the *existing* graph in place,
/// so no reindex is required to benefit.
pub fn start_graph_stub_resolver(graph_store: crate::database::ConcreteGraphStore) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(120));
        info!("Graph stub-edge resolver started (120s interval)");
        loop {
            interval.tick().await;
            // Enumerate tenants present in the graph. The read guard is dropped
            // before resolve_stub_edges (which takes the write lock) to avoid
            // self-deadlock on the shared store's RwLock.
            let tenants: Vec<String> = {
                let guard = graph_store.read().await;
                sqlx::query_scalar("SELECT DISTINCT tenant_id FROM graph_edges")
                    .fetch_all(guard.pool())
                    .await
                    .unwrap_or_default()
            };
            for tenant in tenants {
                match graph_store.resolve_stub_edges(&tenant).await {
                    Ok(n) if n > 0 => {
                        info!(tenant = %tenant, repointed = n, "Graph stub resolver repointed edges")
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!(tenant = %tenant, error = %e, "Graph stub resolution failed")
                    }
                }
            }
        }
    })
}

/// Spawn all periodic background tasks and return their handles.
pub fn spawn_all(
    pool: &SqlitePool,
    search_db: &Arc<SearchDbManager>,
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
    let _project_inventory = start_indexed_project_inventory_exporter(pool.clone());
    let _file_metadata = start_file_metadata_exporter(Arc::clone(search_db));

    BackgroundHandles {
        uptime_handle,
        pause_poll_handle,
        metrics_collect_handle,
        metrics_maint_handle,
        grpc_handle: None,      // Filled in later by grpc_setup
        metrics_handle,
        lsp_metrics_handle: None, // Filled in after Phase 4 (LSP manager init)
    }
}
