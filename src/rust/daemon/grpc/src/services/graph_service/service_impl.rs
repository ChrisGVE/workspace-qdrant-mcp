//! GraphServiceImpl struct definition and constructor.

use std::sync::Arc;

use workspace_qdrant_core::graph::{GraphStore, SharedGraphStore, SqliteGraphStore};

/// GraphService implementation.
///
/// `graph_store` is backend-agnostic (`Arc<dyn GraphStore>`) and serves every
/// RPC that goes through `GraphStore` trait methods (analytics, traversal,
/// impact analysis, stats, …) — these work identically on SQLite and LadybugDB.
///
/// `sqlite_store` is the concrete SQLite handle retained for the two RPCs that
/// still require raw SQLite access — `NarrativeQuery` (raw recursive SQL) and
/// `MigrateGraph` (SQLite export/import). It is `None` on non-SQLite backends,
/// where those two RPCs return `unimplemented` rather than a silent empty
/// result. Both handles share the same underlying lock when the backend is
/// SQLite (a `SharedGraphStore` clone is an `Arc` bump).
pub struct GraphServiceImpl {
    pub(crate) graph_store: Arc<dyn GraphStore>,
    pub(crate) sqlite_store: Option<SharedGraphStore<SqliteGraphStore>>,
}

impl GraphServiceImpl {
    /// Create a new GraphService.
    ///
    /// `graph_store` is the active backend (any `GraphStore`). `sqlite_store` is
    /// `Some` only when the active backend is SQLite, enabling the raw-SQL RPCs.
    pub fn new(
        graph_store: Arc<dyn GraphStore>,
        sqlite_store: Option<SharedGraphStore<SqliteGraphStore>>,
    ) -> Self {
        Self {
            graph_store,
            sqlite_store,
        }
    }
}
