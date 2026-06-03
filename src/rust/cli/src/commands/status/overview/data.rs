//! Data collection for the status overview.
//!
//! Gathers metrics from gRPC (daemon) and SQLite (local DB) and returns
//! a single `StatusData` struct consumed by the rendering layer.

use crate::data::db::connect_readonly;
use crate::data::health;
use crate::data::queries::{self, HealthLevel, QueueStats};
use crate::grpc::client::workspace_daemon::SystemStatusResponse;

/// All data needed to render the status overview.
pub(super) struct StatusData {
    pub daemon_up: bool,
    pub daemon_status: Option<SystemStatusResponse>,
    pub grpc_ms: u64,
    pub qdrant_reachable: bool,
    pub qdrant_error: Option<String>,
    pub qdrant_ms: u64,
    pub collection_count: usize,
    pub document_count: usize,
    pub active_project_count: usize,
    pub project_names: Vec<String>,
    pub queue_stats: QueueStats,
    pub avg_processing_ms: Option<f64>,
    pub overall: HealthLevel,
    pub worker_health: HealthLevel,
    pub qdrant_level: HealthLevel,
    pub queue_health: HealthLevel,
}

/// Collect all status data from gRPC and SQLite.
pub(super) async fn collect() -> StatusData {
    let (daemon_up, daemon_status, grpc_ms) = probe_daemon().await;
    let (qdrant_reachable, qdrant_error, qdrant_ms, qdrant_collection_count) = probe_qdrant().await;
    let (collection_count, document_count, active_project_count, project_names, queue_stats) =
        query_sqlite();
    let avg_processing_ms = load_avg_processing_ms();

    let total_collections = if qdrant_reachable {
        qdrant_collection_count.max(collection_count)
    } else {
        collection_count
    };

    let worker_health = if daemon_up {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let qdrant_level = if qdrant_reachable {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let queue_health = queue_stats.health();
    let overall = worker_health.worst(qdrant_level).worst(queue_health);

    StatusData {
        daemon_up,
        daemon_status,
        grpc_ms,
        qdrant_reachable,
        qdrant_error,
        qdrant_ms,
        collection_count: total_collections,
        document_count,
        active_project_count,
        project_names,
        queue_stats,
        avg_processing_ms,
        overall,
        worker_health,
        qdrant_level,
        queue_health,
    }
}

/// Probe the daemon via gRPC. Returns (up, status, elapsed_ms).
async fn probe_daemon() -> (bool, Option<SystemStatusResponse>, u64) {
    let start = std::time::Instant::now();
    let (up, status) = match crate::grpc::connect_default().await {
        Ok(mut client) => {
            let status = client
                .system()
                .get_status(())
                .await
                .ok()
                .map(|r| r.into_inner());
            (true, status)
        }
        Err(_) => (false, None),
    };
    let ms = start.elapsed().as_millis() as u64;
    (up, status, ms)
}

/// Probe Qdrant health. Returns (reachable, error, elapsed_ms, collection_count).
async fn probe_qdrant() -> (bool, Option<String>, u64, usize) {
    let start = std::time::Instant::now();
    let result = health::check_qdrant().await;
    let ms = start.elapsed().as_millis() as u64;
    (result.reachable, result.error, ms, result.collection_count)
}

/// Query SQLite for metrics. Returns (collections, documents, active_projects, names, queue).
fn query_sqlite() -> (usize, usize, usize, Vec<String>, QueueStats) {
    match connect_readonly() {
        Ok(conn) => {
            let collections = queries::get_active_collection_count(&conn).unwrap_or(0);
            let documents = queries::get_total_document_count(&conn, "projects").unwrap_or(0);
            let active = queries::get_active_project_count(&conn).unwrap_or(0);
            let projects = queries::get_projects(&conn).unwrap_or_default();
            let names: Vec<String> = projects
                .iter()
                .filter(|p| p.is_active)
                .map(|p| {
                    p.path
                        .rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&p.tenant_id)
                        .to_string()
                })
                .collect();
            let q = queries::get_queue_stats(&conn).unwrap_or_default();
            (collections, documents, active, names, q)
        }
        Err(_) => (0, 0, 0, Vec::new(), QueueStats::default()),
    }
}

/// Load average processing time from SQLite (separate connection for lazy access).
fn load_avg_processing_ms() -> Option<f64> {
    connect_readonly()
        .ok()
        .and_then(|conn| queries::get_avg_processing_ms(&conn))
}
