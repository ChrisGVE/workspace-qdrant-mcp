//! ELABORATES edge maintenance — links shallow narrative nodes to deeper ones
//! covering the same concept.
//!
//! When two narrative nodes (DocumentSection, LibrarySection) both connect to
//! the same ConceptNode via COVERS_TOPIC edges but at different depths, the
//! shallower one ELABORATES (points to) the deeper one.
//!
//! This task runs during idle periods and queries the graph store for
//! COVERS_TOPIC edge pairs sharing a target concept with differing depth
//! metadata, then creates ELABORATES edges from shallow to deep.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::graph::{DepthLevel, EdgeType, GraphEdge, GraphStore};
use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;

/// A COVERS_TOPIC edge with parsed depth.
struct CoverageRecord {
    source_node_id: String,
    target_node_id: String,
    depth: DepthLevel,
}

/// Idle task that creates ELABORATES edges between narrative nodes covering
/// the same concept at different depth levels.
///
/// Runs as a single-batch task: fetches all COVERS_TOPIC edges, groups by
/// concept, and generates ELABORATES edges in one pass. The graph store's
/// INSERT OR IGNORE semantics ensure idempotency.
pub struct ElaboratesMaintenanceTask;

impl ElaboratesMaintenanceTask {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MaintenanceTask for ElaboratesMaintenanceTask {
    fn name(&self) -> &str {
        "elaborates_maintenance"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // Only needs the graph store (SQLite), not Qdrant
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        180 // 3 minutes of idle before activating
    }

    fn cooldown_secs(&self) -> u64 {
        3600 // 1 hour between full cycles
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let Some(graph_store) = ctx.graph_store else {
            debug!("ELABORATES maintenance: graph store unavailable, skipping");
            return MaintenanceResult::Done;
        };

        match run_elaborates_pass(graph_store, cancel).await {
            Ok(count) => {
                log_completion(count);
                MaintenanceResult::Done
            }
            Err(ElaboratesError::Cancelled) => MaintenanceResult::Yielded,
            Err(ElaboratesError::Query(e)) => {
                warn!("ELABORATES maintenance failed: {e}");
                MaintenanceResult::Yielded
            }
        }
    }
}

enum ElaboratesError {
    Cancelled,
    Query(String),
}

/// Execute the full ELABORATES pass: query COVERS_TOPIC edges, group by concept,
/// generate and insert ELABORATES edges.
async fn run_elaborates_pass(
    graph_store: &Arc<dyn GraphStore>,
    cancel: &CancellationToken,
) -> Result<u64, ElaboratesError> {
    // Fetch all COVERS_TOPIC edges from the graph store.
    let covers_edges = graph_store
        .query_edges_by_type(EdgeType::CoversTopic)
        .await
        .map_err(|e| ElaboratesError::Query(e.to_string()))?;

    if cancel.is_cancelled() {
        return Err(ElaboratesError::Cancelled);
    }

    // Parse depth from metadata and collect records.
    let records: Vec<CoverageRecord> = covers_edges
        .into_iter()
        .filter_map(|edge| {
            let depth = edge.depth_level()?;
            Some(CoverageRecord {
                source_node_id: edge.source_node_id,
                target_node_id: edge.target_node_id,
                depth,
            })
        })
        .collect();

    // Group by concept (target_node_id).
    let mut by_concept: HashMap<&str, Vec<&CoverageRecord>> = HashMap::new();
    for rec in &records {
        by_concept.entry(&rec.target_node_id).or_default().push(rec);
    }

    // Generate ELABORATES edges for concepts with 2+ covering nodes.
    let mut new_edges: Vec<GraphEdge> = Vec::new();
    for concept_records in by_concept.values() {
        if cancel.is_cancelled() {
            return Err(ElaboratesError::Cancelled);
        }
        generate_elaborates_edges(concept_records, &mut new_edges);
    }

    let count = new_edges.len() as u64;

    // Batch-insert (INSERT OR IGNORE avoids duplicates).
    if !new_edges.is_empty() {
        graph_store
            .insert_edges(&new_edges)
            .await
            .map_err(|e| ElaboratesError::Query(e.to_string()))?;
    }

    Ok(count)
}

/// Generate ELABORATES edges: for each concept, every shallower node points to
/// every deeper node. Edges use `__global__` tenant and `elaborates_task` source.
fn generate_elaborates_edges(records: &[&CoverageRecord], out: &mut Vec<GraphEdge>) {
    if records.len() < 2 {
        return;
    }

    // Group by depth ordinal to avoid creating edges between same-depth nodes.
    let mut by_depth: HashMap<u8, Vec<&str>> = HashMap::new();
    for rec in records {
        by_depth
            .entry(rec.depth.as_ordinal())
            .or_default()
            .push(&rec.source_node_id);
    }

    // Collect distinct ordinals, sorted ascending (shallow first).
    let mut ordinals: Vec<u8> = by_depth.keys().copied().collect();
    ordinals.sort_unstable();

    if ordinals.len() < 2 {
        // All nodes at the same depth — nothing to link.
        return;
    }

    // For each pair (shallow_ord, deep_ord) where shallow < deep,
    // create edges from every node at shallow_ord to every node at deep_ord.
    for (i, &shallow_ord) in ordinals.iter().enumerate() {
        for &deep_ord in &ordinals[i + 1..] {
            let shallow_nodes = &by_depth[&shallow_ord];
            let deep_nodes = &by_depth[&deep_ord];
            for &shallow_id in shallow_nodes {
                for &deep_id in deep_nodes {
                    out.push(GraphEdge::new(
                        "__global__",
                        shallow_id,
                        deep_id,
                        EdgeType::Elaborates,
                        "elaborates_task",
                    ));
                }
            }
        }
    }
}

/// Log a summary when the maintenance cycle finishes.
fn log_completion(edges_created: u64) {
    if edges_created > 0 {
        info!(
            "ELABORATES maintenance complete: created {} edges",
            edges_created
        );
    } else {
        debug!("ELABORATES maintenance complete: no new edges needed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `CoverageRecord` for testing.
    fn rec(id: &str, concept: &str, depth: DepthLevel) -> CoverageRecord {
        CoverageRecord {
            source_node_id: id.to_string(),
            target_node_id: concept.to_string(),
            depth,
        }
    }

    #[test]
    fn two_nodes_different_depth_creates_elaborates() {
        let records = vec![
            rec("intro-node", "concept-1", DepthLevel::Introductory),
            rec("rigorous-node", "concept-1", DepthLevel::Rigorous),
        ];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges = Vec::new();
        generate_elaborates_edges(&refs, &mut edges);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_node_id, "intro-node");
        assert_eq!(edges[0].target_node_id, "rigorous-node");
        assert_eq!(edges[0].edge_type, EdgeType::Elaborates);
        assert_eq!(edges[0].tenant_id, "__global__");
        assert_eq!(edges[0].source_file, "elaborates_task");
    }

    #[test]
    fn two_nodes_same_depth_no_edge() {
        let records = vec![
            rec("node-a", "concept-1", DepthLevel::Intermediate),
            rec("node-b", "concept-1", DepthLevel::Intermediate),
        ];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges = Vec::new();
        generate_elaborates_edges(&refs, &mut edges);

        assert!(edges.is_empty(), "same depth should produce no edges");
    }

    #[test]
    fn three_nodes_different_depths_creates_multiple_edges() {
        let records = vec![
            rec("qualitative-node", "concept-1", DepthLevel::Qualitative),
            rec("intro-node", "concept-1", DepthLevel::Introductory),
            rec("rigorous-node", "concept-1", DepthLevel::Rigorous),
        ];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges = Vec::new();
        generate_elaborates_edges(&refs, &mut edges);

        // qual->intro, qual->rigorous, intro->rigorous = 3 edges
        assert_eq!(edges.len(), 3);

        // Verify shallow->deep direction for all edges
        let pairs: Vec<(&str, &str)> = edges
            .iter()
            .map(|e| (e.source_node_id.as_str(), e.target_node_id.as_str()))
            .collect();

        assert!(pairs.contains(&("qualitative-node", "intro-node")));
        assert!(pairs.contains(&("qualitative-node", "rigorous-node")));
        assert!(pairs.contains(&("intro-node", "rigorous-node")));
    }

    #[test]
    fn single_node_no_edges() {
        let records = vec![rec("lonely", "concept-1", DepthLevel::Reference)];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges = Vec::new();
        generate_elaborates_edges(&refs, &mut edges);

        assert!(edges.is_empty());
    }

    #[test]
    fn multiple_nodes_at_two_depths_creates_cross_product() {
        // 2 shallow + 2 deep = 4 edges
        let records = vec![
            rec("qual-a", "concept-1", DepthLevel::Qualitative),
            rec("qual-b", "concept-1", DepthLevel::Qualitative),
            rec("ref-a", "concept-1", DepthLevel::Reference),
            rec("ref-b", "concept-1", DepthLevel::Reference),
        ];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges = Vec::new();
        generate_elaborates_edges(&refs, &mut edges);

        assert_eq!(edges.len(), 4);

        // All edges should go from qualitative -> reference
        for edge in &edges {
            assert!(
                edge.source_node_id.starts_with("qual-"),
                "source should be qualitative"
            );
            assert!(
                edge.target_node_id.starts_with("ref-"),
                "target should be reference"
            );
        }
    }

    #[test]
    fn edge_ids_are_deterministic() {
        let records = vec![
            rec("node-a", "concept-1", DepthLevel::Introductory),
            rec("node-b", "concept-1", DepthLevel::Rigorous),
        ];
        let refs: Vec<&CoverageRecord> = records.iter().collect();
        let mut edges1 = Vec::new();
        let mut edges2 = Vec::new();
        generate_elaborates_edges(&refs, &mut edges1);
        generate_elaborates_edges(&refs, &mut edges2);

        assert_eq!(edges1[0].edge_id, edges2[0].edge_id);
    }

    #[test]
    fn task_metadata() {
        let task = ElaboratesMaintenanceTask::new();
        assert_eq!(task.name(), "elaborates_maintenance");
        assert_eq!(task.idle_delay_secs(), 180);
        assert_eq!(task.cooldown_secs(), 3600);
        assert!(task.can_run_in(IdleState::FullIdle));
        assert!(task.can_run_in(IdleState::QdrantDownIdle));
        assert!(!task.can_run_in(IdleState::Active));
        assert!(!task.can_run_in(IdleState::ResourceConstrained));
    }
}
