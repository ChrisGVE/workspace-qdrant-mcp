//! GraphServiceImpl struct definition and constructor.

use workspace_qdrant_core::graph::{SharedGraphStore, SqliteGraphStore};

/// GraphService implementation backed by SharedGraphStore.
pub struct GraphServiceImpl {
    pub(crate) graph_store: SharedGraphStore<SqliteGraphStore>,
}

impl GraphServiceImpl {
    /// Create a new GraphService with a shared graph store handle.
    pub fn new(graph_store: SharedGraphStore<SqliteGraphStore>) -> Self {
        Self { graph_store }
    }
}
