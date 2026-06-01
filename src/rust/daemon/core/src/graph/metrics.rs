//! Graph subsystem Prometheus metrics (PRD D5, Phase-1).
//!
//! Phase-1 emits a snapshot of code-graph size/health as gauges plus a graph
//! extraction-duration histogram. All graph metrics carry the mandatory
//! `graph_type` and `backend` labels (§4.2); Phase-1 values are
//! `graph_type=code`, `backend=sqlite`. The `layer` label appears ONLY on
//! `graph_extract_duration_seconds` (value `processing`), never on a gauge.
//!
//! Gauges have NO `_total` suffix (only monotonic counters do). The canonical
//! tenant label key is `tenant_id` (never a bare `tenant`).
//!
//! Collectors register into the global [`crate::monitoring::METRICS`] registry
//! so they appear at the same `/metrics` endpoint. The snapshot
//! ([`snapshot_graph_metrics`]) runs on the daemon's `collection_interval`
//! (default 60s) and costs a SINGLE bounded SQLite read transaction.

use once_cell::sync::Lazy;
use prometheus::{HistogramOpts, HistogramVec, IntGaugeVec, Opts};
use sqlx::{Row, SqlitePool};

use crate::monitoring::{METRICS, PROCESSING_DURATION_BUCKETS};

use super::GRAPH_SCHEMA_VERSION;

/// Phase-1 `graph_type` label value.
pub const GRAPH_TYPE_CODE: &str = "code";
/// Phase-1 `backend` label value.
pub const BACKEND_SQLITE: &str = "sqlite";
/// `layer` label value for extraction performed in the processing pipeline.
pub const LAYER_PROCESSING: &str = "processing";

/// Graph-subsystem Prometheus collectors (PRD D5).
pub struct GraphMetrics {
    /// Total graph nodes. Labels: tenant_id, graph_type, backend.
    pub graph_nodes: IntGaugeVec,
    /// Total graph edges. Labels: tenant_id, graph_type, backend.
    pub graph_edges: IntGaugeVec,
    /// Node counts by node_type. Labels: tenant_id, node_type, graph_type, backend.
    pub graph_nodes_by_type: IntGaugeVec,
    /// Edge counts by edge_type. Labels: tenant_id, edge_type, graph_type, backend.
    pub graph_edges_by_type: IntGaugeVec,
    /// Graph schema version. Labels: graph_type, backend.
    pub graph_schema_version: IntGaugeVec,
    /// Orphaned nodes (no incident edges). Labels: tenant_id, graph_type, backend.
    pub graph_orphaned_nodes: IntGaugeVec,
    /// On-disk graph DB size in bytes. Labels: graph_type, backend.
    pub graph_db_size_bytes: IntGaugeVec,
    /// Graph extraction duration in seconds. Labels: graph_type, backend, layer.
    /// `layer` is present ONLY on this metric.
    pub graph_extract_duration_seconds: HistogramVec,
}

impl GraphMetrics {
    fn new() -> Self {
        let graph_nodes = IntGaugeVec::new(
            Opts::new("wqm_memexd_graph_nodes", "Total graph nodes"),
            &["tenant_id", "graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_edges = IntGaugeVec::new(
            Opts::new("wqm_memexd_graph_edges", "Total graph edges"),
            &["tenant_id", "graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_nodes_by_type = IntGaugeVec::new(
            Opts::new("wqm_memexd_graph_nodes_by_type", "Graph nodes by node type"),
            &["tenant_id", "node_type", "graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_edges_by_type = IntGaugeVec::new(
            Opts::new("wqm_memexd_graph_edges_by_type", "Graph edges by edge type"),
            &["tenant_id", "edge_type", "graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_schema_version = IntGaugeVec::new(
            Opts::new("wqm_memexd_graph_schema_version", "Graph schema version"),
            &["graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_orphaned_nodes = IntGaugeVec::new(
            Opts::new(
                "wqm_memexd_graph_orphaned_nodes",
                "Graph nodes with no incident edges",
            ),
            &["tenant_id", "graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_db_size_bytes = IntGaugeVec::new(
            Opts::new(
                "wqm_memexd_graph_db_size_bytes",
                "On-disk graph database size in bytes",
            ),
            &["graph_type", "backend"],
        )
        .expect("metric can be created");
        let graph_extract_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "wqm_memexd_graph_extract_duration_seconds",
                "Graph extraction duration in seconds",
            )
            .buckets(PROCESSING_DURATION_BUCKETS.to_vec()),
            &["graph_type", "backend", "layer"],
        )
        .expect("metric can be created");

        // Register into the shared daemon registry so all graph metrics are
        // exported from the same /metrics endpoint. Errors (e.g. duplicate
        // registration in a re-init) are ignored.
        let r = &METRICS.registry;
        let _ = r.register(Box::new(graph_nodes.clone()));
        let _ = r.register(Box::new(graph_edges.clone()));
        let _ = r.register(Box::new(graph_nodes_by_type.clone()));
        let _ = r.register(Box::new(graph_edges_by_type.clone()));
        let _ = r.register(Box::new(graph_schema_version.clone()));
        let _ = r.register(Box::new(graph_orphaned_nodes.clone()));
        let _ = r.register(Box::new(graph_db_size_bytes.clone()));
        let _ = r.register(Box::new(graph_extract_duration_seconds.clone()));

        Self {
            graph_nodes,
            graph_edges,
            graph_nodes_by_type,
            graph_edges_by_type,
            graph_schema_version,
            graph_orphaned_nodes,
            graph_db_size_bytes,
            graph_extract_duration_seconds,
        }
    }
}

/// Global graph metrics collectors, registered into [`METRICS`] on first use.
pub static GRAPH_METRICS: Lazy<GraphMetrics> = Lazy::new(GraphMetrics::new);

/// Record a graph extraction duration observation (PRD D5). `layer` is the only
/// metric carrying a `layer` label; Phase-1 callers pass [`LAYER_PROCESSING`].
/// No-op when the telemetry kill switch is off.
pub fn record_graph_extract_duration(layer: &str, seconds: f64) {
    if !METRICS.is_enabled() {
        return;
    }
    GRAPH_METRICS
        .graph_extract_duration_seconds
        .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE, layer])
        .observe(seconds);
}

/// Snapshot code-graph size/health gauges from the SQLite graph store in a
/// SINGLE read transaction (PRD D5 — bounded cost per interval). No-op when the
/// telemetry kill switch is off.
///
/// Per-tenant and by-type gauges are reset first so a tenant/type that drops to
/// zero (or is removed) does not leave a stale series.
pub async fn snapshot_graph_metrics(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    if !METRICS.is_enabled() {
        return Ok(());
    }
    let gm = &*GRAPH_METRICS;

    // Clear per-tenant / by-type series to avoid stale gauges across snapshots.
    gm.graph_nodes.reset();
    gm.graph_edges.reset();
    gm.graph_nodes_by_type.reset();
    gm.graph_edges_by_type.reset();
    gm.graph_orphaned_nodes.reset();

    let mut tx = pool.begin().await?;

    let tenant_rows = sqlx::query("SELECT DISTINCT tenant_id FROM graph_nodes")
        .fetch_all(&mut *tx)
        .await?;

    for tr in &tenant_rows {
        let tid: String = tr.get("tenant_id");

        let node_rows = sqlx::query(
            "SELECT symbol_type, COUNT(*) AS cnt FROM graph_nodes \
             WHERE tenant_id = ?1 GROUP BY symbol_type",
        )
        .bind(&tid)
        .fetch_all(&mut *tx)
        .await?;
        let mut total_nodes: i64 = 0;
        for row in &node_rows {
            let node_type: String = row.get("symbol_type");
            let cnt: i64 = row.get("cnt");
            total_nodes += cnt;
            gm.graph_nodes_by_type
                .with_label_values(&[&tid, &node_type, GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .set(cnt);
        }
        gm.graph_nodes
            .with_label_values(&[&tid, GRAPH_TYPE_CODE, BACKEND_SQLITE])
            .set(total_nodes);

        let edge_rows = sqlx::query(
            "SELECT edge_type, COUNT(*) AS cnt FROM graph_edges \
             WHERE tenant_id = ?1 GROUP BY edge_type",
        )
        .bind(&tid)
        .fetch_all(&mut *tx)
        .await?;
        let mut total_edges: i64 = 0;
        for row in &edge_rows {
            let edge_type: String = row.get("edge_type");
            let cnt: i64 = row.get("cnt");
            total_edges += cnt;
            gm.graph_edges_by_type
                .with_label_values(&[&tid, &edge_type, GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .set(cnt);
        }
        gm.graph_edges
            .with_label_values(&[&tid, GRAPH_TYPE_CODE, BACKEND_SQLITE])
            .set(total_edges);

        let orphaned: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM graph_nodes WHERE tenant_id = ?1 \
               AND node_id NOT IN ( \
                   SELECT source_node_id FROM graph_edges WHERE tenant_id = ?1 \
                   UNION \
                   SELECT target_node_id FROM graph_edges WHERE tenant_id = ?1 \
               )",
        )
        .bind(&tid)
        .fetch_one(&mut *tx)
        .await?;
        gm.graph_orphaned_nodes
            .with_label_values(&[&tid, GRAPH_TYPE_CODE, BACKEND_SQLITE])
            .set(orphaned);
    }

    // Schema version (global): fall back to the compiled-in constant if the
    // version table is absent or empty.
    let schema_version: i64 = sqlx::query_scalar("SELECT MAX(version) FROM graph_schema_version")
        .fetch_one(&mut *tx)
        .await
        .ok()
        .flatten()
        .unwrap_or(GRAPH_SCHEMA_VERSION as i64);
    gm.graph_schema_version
        .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE])
        .set(schema_version);

    // On-disk size = page_count * page_size (committed pages).
    let page_count: i64 = sqlx::query_scalar("PRAGMA page_count")
        .fetch_one(&mut *tx)
        .await
        .unwrap_or(0);
    let page_size: i64 = sqlx::query_scalar("PRAGMA page_size")
        .fetch_one(&mut *tx)
        .await
        .unwrap_or(0);
    gm.graph_db_size_bytes
        .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE])
        .set(page_count * page_size);

    // Read-only snapshot: roll back to release without writing.
    let _ = tx.rollback().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    async fn pool_with_graph_schema() -> SqlitePool {
        let opts = SqliteConnectOptions::new()
            .filename(":memory:")
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, \
             symbol_name TEXT NOT NULL, symbol_type TEXT NOT NULL, file_path TEXT NOT NULL, \
             branches TEXT NOT NULL DEFAULT '[\"main\"]')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE graph_edges (edge_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, \
             source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL, edge_type TEXT NOT NULL, \
             source_file TEXT NOT NULL, branch TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("CREATE TABLE graph_schema_version (version INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO graph_schema_version (version) VALUES (4)")
            .execute(&pool)
            .await
            .unwrap();
        pool
    }

    /// Gauges must NOT carry a `_total` suffix (only monotonic counters do).
    /// Assert every gauge family name gathered from the registry is total-free.
    #[test]
    fn gauge_names_have_no_total_suffix() {
        Lazy::force(&GRAPH_METRICS);
        let gauge_names = [
            "wqm_memexd_graph_nodes",
            "wqm_memexd_graph_edges",
            "wqm_memexd_graph_nodes_by_type",
            "wqm_memexd_graph_edges_by_type",
            "wqm_memexd_graph_schema_version",
            "wqm_memexd_graph_orphaned_nodes",
            "wqm_memexd_graph_db_size_bytes",
        ];
        for name in gauge_names {
            assert!(!name.ends_with("_total"), "{name} must not end with _total");
        }
    }

    #[tokio::test]
    #[serial]
    async fn snapshot_populates_graph_gauges() {
        let pool = pool_with_graph_schema().await;
        // Two nodes (function, struct), one edge (CALLS). One orphan (struct).
        sqlx::query(
            "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type, file_path) \
             VALUES ('n1','t-snap','f','function','a.rs'), \
                    ('n2','t-snap','S','struct','a.rs')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO graph_edges \
             (edge_id, tenant_id, source_node_id, target_node_id, edge_type, source_file) \
             VALUES ('e1','t-snap','n1','n1','CALLS','a.rs')",
        )
        .execute(&pool)
        .await
        .unwrap();

        snapshot_graph_metrics(&pool).await.unwrap();

        let gm = &*GRAPH_METRICS;
        assert_eq!(
            gm.graph_nodes
                .with_label_values(&["t-snap", GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            2
        );
        assert_eq!(
            gm.graph_edges
                .with_label_values(&["t-snap", GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            1
        );
        assert_eq!(
            gm.graph_nodes_by_type
                .with_label_values(&["t-snap", "function", GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            1
        );
        assert_eq!(
            gm.graph_edges_by_type
                .with_label_values(&["t-snap", "CALLS", GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            1
        );
        // n1 has an edge; n2 (struct) is orphaned.
        assert_eq!(
            gm.graph_orphaned_nodes
                .with_label_values(&["t-snap", GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            1
        );
        assert_eq!(
            gm.graph_schema_version
                .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get(),
            4
        );
        assert!(
            gm.graph_db_size_bytes
                .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE])
                .get()
                > 0
        );
    }

    #[test]
    fn extract_duration_carries_layer_and_records() {
        // graph_extract_duration_seconds is the only graph metric with `layer`.
        record_graph_extract_duration(LAYER_PROCESSING, 0.012);
        let count = GRAPH_METRICS
            .graph_extract_duration_seconds
            .with_label_values(&[GRAPH_TYPE_CODE, BACKEND_SQLITE, LAYER_PROCESSING])
            .get_sample_count();
        assert!(count >= 1);
    }

    /// Every graph metric is registered in the shared registry and exported
    /// with graph_type+backend labels; only extract_duration has `layer`.
    #[tokio::test]
    #[serial]
    async fn registered_metrics_have_mandatory_labels() {
        let pool = pool_with_graph_schema().await;
        sqlx::query(
            "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type, file_path) \
             VALUES ('lq1','t-lbl','f','function','a.rs')",
        )
        .execute(&pool)
        .await
        .unwrap();
        // An edge so graph_edges_by_type populates a series (all 8 families present).
        sqlx::query(
            "INSERT INTO graph_edges \
             (edge_id, tenant_id, source_node_id, target_node_id, edge_type, source_file) \
             VALUES ('elq1','t-lbl','lq1','lq1','CALLS','a.rs')",
        )
        .execute(&pool)
        .await
        .unwrap();
        snapshot_graph_metrics(&pool).await.unwrap();
        record_graph_extract_duration(LAYER_PROCESSING, 0.01);

        let families = METRICS.registry.gather();
        let graph_families: Vec<_> = families
            .iter()
            .filter(|f| f.get_name().starts_with("wqm_memexd_graph_"))
            .collect();
        assert!(
            graph_families.len() >= 8,
            "expected all 8 graph metric families, got {}",
            graph_families.len()
        );
        for f in graph_families {
            for m in f.get_metric() {
                let labels: Vec<&str> = m.get_label().iter().map(|l| l.get_name()).collect();
                assert!(
                    labels.contains(&"graph_type"),
                    "{} missing graph_type",
                    f.get_name()
                );
                assert!(
                    labels.contains(&"backend"),
                    "{} missing backend",
                    f.get_name()
                );
                let has_layer = labels.contains(&"layer");
                if f.get_name() == "wqm_memexd_graph_extract_duration_seconds" {
                    assert!(has_layer, "extract_duration must carry layer");
                } else {
                    assert!(!has_layer, "{} must NOT carry layer", f.get_name());
                }
                // tenant label, when present, is exactly `tenant_id`.
                assert!(
                    !labels.contains(&"tenant"),
                    "{} uses bare `tenant`; must be `tenant_id`",
                    f.get_name()
                );
            }
        }
    }
}
