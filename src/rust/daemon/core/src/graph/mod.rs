//! Graph database module for code relationship storage and querying.
//!
//! Provides a `GraphStore` trait abstracting graph operations, with
//! `SqliteGraphStore` (recursive CTEs) and `LadybugGraphStore` (Kuzu
//! fork, behind `ladybug` feature flag) implementations.
//!
//! Use `factory::create_sqlite_graph_store` or the LadybugDB variant
//! to instantiate the appropriate backend based on configuration.
//! The graph is stored in a dedicated `graph.db` file separate from
//! `state.db` to avoid lock contention with queue processing.

pub mod algorithms;
pub mod extractor;
pub mod factory;
mod schema;
mod shared;
mod sqlite_store;

#[cfg(feature = "ladybug")]
pub mod ladybug_store;

#[cfg(test)]
mod tests;

pub use factory::{GraphBackend, GraphConfig, create_sqlite_graph_store};
#[cfg(feature = "ladybug")]
pub use factory::create_ladybug_graph_store;
pub use schema::{GraphDbManager, GraphDbError, GraphDbResult, GRAPH_DB_FILENAME, GRAPH_SCHEMA_VERSION};
#[cfg(feature = "ladybug")]
pub use ladybug_store::{LadybugGraphStore, LadybugConfig};
pub use shared::SharedGraphStore;
pub use sqlite_store::SqliteGraphStore;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Node types in the code graph, mapping to semantic chunk types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    File,
    Function,
    AsyncFunction,
    Class,
    Method,
    Struct,
    Trait,
    Interface,
    Enum,
    Impl,
    Module,
    Constant,
    TypeAlias,
    Macro,
}

impl NodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::File => "file",
            NodeType::Function => "function",
            NodeType::AsyncFunction => "async_function",
            NodeType::Class => "class",
            NodeType::Method => "method",
            NodeType::Struct => "struct",
            NodeType::Trait => "trait",
            NodeType::Interface => "interface",
            NodeType::Enum => "enum",
            NodeType::Impl => "impl",
            NodeType::Module => "module",
            NodeType::Constant => "constant",
            NodeType::TypeAlias => "type_alias",
            NodeType::Macro => "macro",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "file" => Some(NodeType::File),
            "function" => Some(NodeType::Function),
            "async_function" => Some(NodeType::AsyncFunction),
            "class" => Some(NodeType::Class),
            "method" => Some(NodeType::Method),
            "struct" => Some(NodeType::Struct),
            "trait" => Some(NodeType::Trait),
            "interface" => Some(NodeType::Interface),
            "enum" => Some(NodeType::Enum),
            "impl" => Some(NodeType::Impl),
            "module" => Some(NodeType::Module),
            "constant" => Some(NodeType::Constant),
            "type_alias" => Some(NodeType::TypeAlias),
            "macro" => Some(NodeType::Macro),
            _ => None,
        }
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Edge types representing relationships between code entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EdgeType {
    /// Function/method call relationship.
    Calls,
    /// Parent-child containment (class contains method, impl contains fn).
    Contains,
    /// Import/use statement dependency.
    Imports,
    /// Type reference in signature (parameter types, return types).
    UsesType,
    /// Class/trait inheritance.
    Extends,
    /// Trait/interface implementation.
    Implements,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::Calls => "CALLS",
            EdgeType::Contains => "CONTAINS",
            EdgeType::Imports => "IMPORTS",
            EdgeType::UsesType => "USES_TYPE",
            EdgeType::Extends => "EXTENDS",
            EdgeType::Implements => "IMPLEMENTS",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "CALLS" => Some(EdgeType::Calls),
            "CONTAINS" => Some(EdgeType::Contains),
            "IMPORTS" => Some(EdgeType::Imports),
            "USES_TYPE" => Some(EdgeType::UsesType),
            "EXTENDS" => Some(EdgeType::Extends),
            "IMPLEMENTS" => Some(EdgeType::Implements),
            _ => None,
        }
    }
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A node in the code graph representing a code entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub node_id: String,
    pub tenant_id: String,
    pub symbol_name: String,
    pub symbol_type: NodeType,
    pub file_path: String,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    pub signature: Option<String>,
    pub language: Option<String>,
}

impl GraphNode {
    /// Create a new graph node, computing the node_id deterministically.
    pub fn new(
        tenant_id: impl Into<String>,
        file_path: impl Into<String>,
        symbol_name: impl Into<String>,
        symbol_type: NodeType,
    ) -> Self {
        let tenant_id = tenant_id.into();
        let file_path = file_path.into();
        let symbol_name = symbol_name.into();
        let node_id = compute_node_id(&tenant_id, &file_path, &symbol_name, symbol_type);
        Self {
            node_id,
            tenant_id,
            symbol_name,
            symbol_type,
            file_path,
            start_line: None,
            end_line: None,
            signature: None,
            language: None,
        }
    }

    /// Create a stub node (unresolved target — only name and type known).
    pub fn stub(
        tenant_id: impl Into<String>,
        symbol_name: impl Into<String>,
        symbol_type: NodeType,
    ) -> Self {
        let tenant_id = tenant_id.into();
        let symbol_name = symbol_name.into();
        // Stub nodes use empty file_path — updated when the target file is processed
        let node_id = compute_node_id(&tenant_id, "", &symbol_name, symbol_type);
        Self {
            node_id,
            tenant_id,
            symbol_name,
            symbol_type,
            file_path: String::new(),
            start_line: None,
            end_line: None,
            signature: None,
            language: None,
        }
    }
}

/// An edge in the code graph representing a relationship between entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub edge_id: String,
    pub tenant_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub edge_type: EdgeType,
    /// The file that "owns" this edge (for deletion on re-ingestion).
    pub source_file: String,
    pub weight: f64,
    pub metadata_json: Option<String>,
}

impl GraphEdge {
    /// Create a new edge, computing the edge_id deterministically.
    pub fn new(
        tenant_id: impl Into<String>,
        source_node_id: impl Into<String>,
        target_node_id: impl Into<String>,
        edge_type: EdgeType,
        source_file: impl Into<String>,
    ) -> Self {
        let source_node_id = source_node_id.into();
        let target_node_id = target_node_id.into();
        let edge_id = compute_edge_id(&source_node_id, &target_node_id, edge_type);
        Self {
            edge_id,
            tenant_id: tenant_id.into(),
            source_node_id,
            target_node_id,
            edge_type,
            source_file: source_file.into(),
            weight: 1.0,
            metadata_json: None,
        }
    }
}

/// A node encountered during graph traversal, with path context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalNode {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub edge_type: String,
    pub depth: u32,
    pub path: String,
}

/// Result of an impact analysis query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactReport {
    pub symbol_name: String,
    pub impacted_nodes: Vec<ImpactNode>,
    pub total_impacted: u32,
}

/// A node impacted by a symbol change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactNode {
    pub node_id: String,
    pub symbol_name: String,
    pub file_path: String,
    pub impact_type: String,
    pub distance: u32,
}

/// Graph statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_nodes: u64,
    pub total_edges: u64,
    pub nodes_by_type: std::collections::HashMap<String, u64>,
    pub edges_by_type: std::collections::HashMap<String, u64>,
}

/// Trait abstracting graph storage operations.
///
/// Implementations:
/// - `SqliteGraphStore`: SQLite with recursive CTEs (default)
/// - `LadybugGraphStore`: Kuzu fork with Cypher queries (`ladybug` feature)
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Insert or update a node. If the node_id already exists, update metadata.
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()>;

    /// Batch upsert multiple nodes in a single transaction.
    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()>;

    /// Insert an edge. Ignores duplicates (same edge_id).
    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()>;

    /// Batch insert multiple edges in a single transaction.
    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()>;

    /// Delete all edges owned by a specific file.
    async fn delete_edges_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64>;

    /// Delete all nodes and edges for a tenant.
    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64>;

    /// Query nodes related to a given node within N hops.
    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<Vec<TraversalNode>>;

    /// Find all nodes that would be affected by changing a given symbol.
    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
    ) -> GraphDbResult<ImpactReport>;

    /// Get graph statistics, optionally filtered by tenant.
    async fn stats(&self, tenant_id: Option<&str>) -> GraphDbResult<GraphStats>;

    /// Delete orphaned nodes (nodes with no edges).
    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64>;
}

/// Compute deterministic node ID from its identifying fields.
pub fn compute_node_id(
    tenant_id: &str,
    file_path: &str,
    symbol_name: &str,
    symbol_type: NodeType,
) -> String {
    let input = format!(
        "{}|{}|{}|{}",
        tenant_id, file_path, symbol_name, symbol_type.as_str()
    );
    let hash = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in &hash[..16] {
        let _ = write!(out, "{:02x}", b);
    }
    out
}

/// Compute deterministic edge ID from source, target, and type.
pub fn compute_edge_id(
    source_node_id: &str,
    target_node_id: &str,
    edge_type: EdgeType,
) -> String {
    let input = format!(
        "{}|{}|{}",
        source_node_id, target_node_id, edge_type.as_str()
    );
    let hash = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in &hash[..16] {
        let _ = write!(out, "{:02x}", b);
    }
    out
}
