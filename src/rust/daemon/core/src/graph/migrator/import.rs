//! Import, migration pipeline, and validation functions for graph data.

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::graph::schema::GraphDbResult;
use crate::graph::GraphStore;
use super::{GraphSnapshot, MigrationReport, DEFAULT_BATCH_SIZE};
use super::export::export_sqlite;

/// Import a graph snapshot into a target store in batches.
pub async fn import_to_store<S: GraphStore>(
    snapshot: &GraphSnapshot,
    target: &S,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    let batch_size = if batch_size == 0 {
        DEFAULT_BATCH_SIZE
    } else {
        batch_size
    };

    let mut warnings = Vec::new();

    // Import nodes in batches
    let mut nodes_imported = 0u64;
    for chunk in snapshot.nodes.chunks(batch_size) {
        match target.upsert_nodes(chunk).await {
            Ok(()) => {
                nodes_imported += chunk.len() as u64;
                debug!(
                    "Imported node batch: {}/{}",
                    nodes_imported,
                    snapshot.nodes.len()
                );
            }
            Err(e) => {
                let msg = format!(
                    "Failed to import node batch at offset {}: {}",
                    nodes_imported, e
                );
                warn!("{}", msg);
                warnings.push(msg);
            }
        }
    }

    // Import edges in batches
    let mut edges_imported = 0u64;
    for chunk in snapshot.edges.chunks(batch_size) {
        match target.insert_edges(chunk).await {
            Ok(()) => {
                edges_imported += chunk.len() as u64;
                debug!(
                    "Imported edge batch: {}/{}",
                    edges_imported,
                    snapshot.edges.len()
                );
            }
            Err(e) => {
                let msg = format!(
                    "Failed to import edge batch at offset {}: {}",
                    edges_imported, e
                );
                warn!("{}", msg);
                warnings.push(msg);
            }
        }
    }

    // Collect tenant IDs
    let mut tenant_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for node in &snapshot.nodes {
        tenant_set.insert(&node.tenant_id);
    }

    let report = MigrationReport {
        nodes_exported: snapshot.nodes.len() as u64,
        edges_exported: snapshot.edges.len() as u64,
        nodes_imported,
        edges_imported,
        nodes_match: nodes_imported == snapshot.nodes.len() as u64,
        edges_match: edges_imported == snapshot.edges.len() as u64,
        tenants: tenant_set.into_iter().map(|s| s.to_string()).collect(),
        warnings,
    };

    info!(
        "Migration complete: {} nodes, {} edges, match={}",
        report.nodes_imported,
        report.edges_imported,
        report.nodes_match && report.edges_match
    );

    Ok(report)
}

// ─── Full migration pipelines ───────────────────────────────────────────

/// Migrate from SQLite to any GraphStore target.
pub async fn migrate_from_sqlite<S: GraphStore>(
    source_pool: &SqlitePool,
    target: &S,
    tenant_id: Option<&str>,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    info!(
        "Starting SQLite export (tenant: {:?})",
        tenant_id
    );
    let snapshot = export_sqlite(source_pool, tenant_id).await?;
    info!(
        "Exported {} nodes, {} edges from SQLite",
        snapshot.nodes.len(),
        snapshot.edges.len()
    );
    import_to_store(&snapshot, target, batch_size).await
}

/// Migrate from LadybugDB to any GraphStore target.
#[cfg(feature = "ladybug")]
pub async fn migrate_from_ladybug<S: GraphStore>(
    source: &crate::graph::LadybugGraphStore,
    target: &S,
    tenant_id: Option<&str>,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    use super::export::export_ladybug;

    info!(
        "Starting LadybugDB export (tenant: {:?})",
        tenant_id
    );
    let snapshot = export_ladybug(source, tenant_id)?;
    info!(
        "Exported {} nodes, {} edges from LadybugDB",
        snapshot.nodes.len(),
        snapshot.edges.len()
    );
    import_to_store(&snapshot, target, batch_size).await
}

/// Validate a migration by comparing stats between source and target.
pub async fn validate_migration<S: GraphStore>(
    source_pool: &SqlitePool,
    target: &S,
    tenant_id: Option<&str>,
) -> GraphDbResult<bool> {
    // Count source
    let (source_nodes, source_edges) = match tenant_id {
        Some(tid) => {
            let n: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM graph_nodes WHERE tenant_id = ?1",
            )
            .bind(tid)
            .fetch_one(source_pool)
            .await?;
            let e: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM graph_edges WHERE tenant_id = ?1",
            )
            .bind(tid)
            .fetch_one(source_pool)
            .await?;
            (n.0 as u64, e.0 as u64)
        }
        None => {
            let n: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM graph_nodes")
                    .fetch_one(source_pool)
                    .await?;
            let e: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM graph_edges")
                    .fetch_one(source_pool)
                    .await?;
            (n.0 as u64, e.0 as u64)
        }
    };

    // Count target
    let target_stats = target.stats(tenant_id).await?;

    let nodes_ok = source_nodes == target_stats.total_nodes;
    let edges_ok = source_edges == target_stats.total_edges;

    if nodes_ok && edges_ok {
        info!(
            "Validation passed: {} nodes, {} edges",
            source_nodes, source_edges
        );
    } else {
        warn!(
            "Validation FAILED: source({} nodes, {} edges) vs target({} nodes, {} edges)",
            source_nodes, source_edges, target_stats.total_nodes, target_stats.total_edges
        );
    }

    Ok(nodes_ok && edges_ok)
}
