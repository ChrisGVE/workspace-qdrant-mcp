/// Graph migration utility for moving data between SQLite and LadybugDB backends.
///
/// Exports nodes and edges from one backend, imports to another in batches,
/// then validates counts match. Designed for `wqm graph migrate` CLI command.

mod export;
mod import;

pub use export::*;
pub use import::*;

use serde::{Deserialize, Serialize};

use super::{GraphEdge, GraphNode};

/// Default batch size for import operations.
pub(super) const DEFAULT_BATCH_SIZE: usize = 500;

/// Report produced after a migration completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    /// Number of nodes exported from the source.
    pub nodes_exported: u64,
    /// Number of edges exported from the source.
    pub edges_exported: u64,
    /// Number of nodes imported to the target.
    pub nodes_imported: u64,
    /// Number of edges imported to the target.
    pub edges_imported: u64,
    /// Whether node counts match.
    pub nodes_match: bool,
    /// Whether edge counts match.
    pub edges_match: bool,
    /// Tenant IDs that were migrated (None = all).
    pub tenants: Vec<String>,
    /// Any warnings or issues encountered.
    pub warnings: Vec<String>,
}

/// Snapshot of graph data for migration.
#[derive(Debug)]
pub struct GraphSnapshot {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[cfg(test)]
mod tests;
