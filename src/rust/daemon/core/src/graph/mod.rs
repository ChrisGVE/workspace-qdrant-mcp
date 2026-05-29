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
pub mod migrator;
mod schema;
mod shared;
mod sqlite_store;

#[cfg(feature = "ladybug")]
pub mod ladybug_store;

#[cfg(test)]
mod tests;

#[cfg(feature = "ladybug")]
pub use factory::create_ladybug_graph_store;
pub use factory::{create_sqlite_graph_store, GraphBackend, GraphConfig};
#[cfg(feature = "ladybug")]
pub use ladybug_store::{LadybugConfig, LadybugGraphStore};
pub use schema::{
    GraphDbError, GraphDbManager, GraphDbResult, GRAPH_DB_FILENAME, GRAPH_SCHEMA_VERSION,
};
pub use shared::SharedGraphStore;
pub use sqlite_store::SqliteGraphStore;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Default branches JSON for new nodes: `["main"]`.
const DEFAULT_BRANCHES_JSON: &str = r#"["main"]"#;

/// Node types in the code graph, mapping to semantic chunk types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    // Structural layer (code entities)
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
    // Narrative layer (human-authored documentation)
    DocumentSection,
    CodeComment,
    Docstring,
    LibrarySection,
    // Concept layer (cross-boundary bridging)
    ConceptNode,
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
            NodeType::DocumentSection => "document_section",
            NodeType::CodeComment => "code_comment",
            NodeType::Docstring => "docstring",
            NodeType::LibrarySection => "library_section",
            NodeType::ConceptNode => "concept_node",
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
            "document_section" => Some(NodeType::DocumentSection),
            "code_comment" => Some(NodeType::CodeComment),
            "docstring" => Some(NodeType::Docstring),
            "library_section" => Some(NodeType::LibrarySection),
            "concept_node" => Some(NodeType::ConceptNode),
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
    // Structural layer
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
    // Narrative layer
    /// Docstring/comment explains a code symbol (via Aho-Corasick matching).
    Explains,
    /// Narrative node describes a concept or entity.
    Describes,
    /// Markdown/doc cross-reference to another document or code symbol.
    ReferencesDoc,
    /// Back-link from concept node to a more detailed narrative node.
    Elaborates,
    // Concept layer
    /// Narrative or code node covers a taxonomy concept (cosine similarity).
    CoversTopic,
    /// Code symbol implements a concept (structural ↔ concept bridge).
    ImplementsConcept,
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
            EdgeType::Explains => "EXPLAINS",
            EdgeType::Describes => "DESCRIBES",
            EdgeType::ReferencesDoc => "REFERENCES_DOC",
            EdgeType::Elaborates => "ELABORATES",
            EdgeType::CoversTopic => "COVERS_TOPIC",
            EdgeType::ImplementsConcept => "IMPLEMENTS_CONCEPT",
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
            "EXPLAINS" => Some(EdgeType::Explains),
            "DESCRIBES" => Some(EdgeType::Describes),
            "REFERENCES_DOC" => Some(EdgeType::ReferencesDoc),
            "ELABORATES" => Some(EdgeType::Elaborates),
            "COVERS_TOPIC" => Some(EdgeType::CoversTopic),
            "IMPLEMENTS_CONCEPT" => Some(EdgeType::ImplementsConcept),
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
    /// JSON array of branch names this node belongs to, e.g. `["main"]`.
    pub branches: String,
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
            branches: DEFAULT_BRANCHES_JSON.to_string(),
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
            branches: DEFAULT_BRANCHES_JSON.to_string(),
        }
    }

    /// Set the branches JSON from a branch name (wraps in a single-element array).
    pub fn with_branch(mut self, branch: &str) -> Self {
        self.branches = format!(r#"["{}"]"#, branch);
        self
    }
}

/// Depth level for concept coverage — how deeply a source covers a topic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthLevel {
    Qualitative,
    Introductory,
    Intermediate,
    Rigorous,
    Reference,
}

impl DepthLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            DepthLevel::Qualitative => "qualitative",
            DepthLevel::Introductory => "introductory",
            DepthLevel::Intermediate => "intermediate",
            DepthLevel::Rigorous => "rigorous",
            DepthLevel::Reference => "reference",
        }
    }

    /// Numeric ordering: Qualitative (0) < Introductory (1) < Intermediate (2) < Rigorous (3) < Reference (4).
    pub fn as_ordinal(self) -> u8 {
        match self {
            DepthLevel::Qualitative => 0,
            DepthLevel::Introductory => 1,
            DepthLevel::Intermediate => 2,
            DepthLevel::Rigorous => 3,
            DepthLevel::Reference => 4,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "qualitative" => Some(DepthLevel::Qualitative),
            "introductory" => Some(DepthLevel::Introductory),
            "intermediate" => Some(DepthLevel::Intermediate),
            "rigorous" => Some(DepthLevel::Rigorous),
            "reference" => Some(DepthLevel::Reference),
            _ => None,
        }
    }

    pub fn to_metadata_json(&self) -> String {
        format!(r#"{{"depth":"{}"}}"#, self.as_str())
    }

    pub fn from_metadata_json(json: &str) -> Option<Self> {
        let trimmed = json.trim();
        if let Some(start) = trimmed.find("\"depth\"") {
            let rest = &trimmed[start + 7..];
            if let Some(colon) = rest.find(':') {
                let after_colon = rest[colon + 1..].trim().trim_start_matches('"');
                if let Some(end) = after_colon.find('"') {
                    return Self::from_str(&after_colon[..end]);
                }
            }
        }
        None
    }
}

impl std::fmt::Display for DepthLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
    /// The branch this edge was extracted on. `None` means globally inferred.
    pub branch: Option<String>,
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
            branch: None,
        }
    }

    /// Set the branch for this edge.
    pub fn with_branch(mut self, branch: &str) -> Self {
        self.branch = Some(branch.to_string());
        self
    }

    /// Set depth level metadata (for CoversTopic edges).
    pub fn with_depth(mut self, depth: DepthLevel) -> Self {
        self.metadata_json = Some(depth.to_metadata_json());
        self
    }

    /// Extract depth level from metadata_json (if present).
    pub fn depth_level(&self) -> Option<DepthLevel> {
        self.metadata_json
            .as_deref()
            .and_then(DepthLevel::from_metadata_json)
    }
}

/// A resolved code-graph symbol: its name plus the real node identity it was
/// stored under. Returned by [`GraphStore::query_code_symbols`] so narrative
/// EXPLAINS resolution can target an existing node id directly rather than
/// recomputing one from guessed fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRow {
    pub symbol_name: String,
    pub node_id: String,
    pub file_path: String,
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
    /// Tenant the reached node belongs to. For Layer-1 (same-tenant) queries
    /// this equals the query tenant; for cross-boundary queries it may be the
    /// source tenant, `__global__`, or a library tenant.
    pub tenant_id: String,
    /// Confidence of the edge that reached this node. Structural Layer-1 paths
    /// use `1.0`; cross-boundary concept/narrative paths scale per edge type.
    pub edge_confidence: f64,
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

/// Check whether a branch filter should be applied.
///
/// Returns `true` when the caller wants cross-branch (unscoped) results:
/// `None`, empty string, or the wildcard `"*"`.
pub fn is_cross_branch(branch: Option<&str>) -> bool {
    match branch {
        None => true,
        Some(b) => b.is_empty() || b == "*",
    }
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
    async fn delete_edges_by_file(&self, tenant_id: &str, file_path: &str) -> GraphDbResult<u64>;

    /// Delete all nodes and edges for a tenant.
    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64>;

    /// Query nodes related to a given node within N hops.
    ///
    /// When `branch` is `Some(name)` (and not `"*"`), only nodes whose
    /// `branches` JSON array contains `name` and edges whose `branch`
    /// column equals `name` (or is `NULL`) are traversed.
    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>>;

    /// Find all nodes that would be affected by changing a given symbol.
    ///
    /// When `branch` is provided (and not `"*"`), only edges and nodes
    /// belonging to that branch are considered.
    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport>;

    /// Get graph statistics, optionally filtered by tenant and branch.
    async fn stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats>;

    /// Delete orphaned nodes (nodes with no edges).
    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64>;

    /// Find shortest path between two nodes using BFS.
    ///
    /// When `branch` is provided (and not `"*"`), only edges belonging
    /// to that branch (or with `NULL` branch) are traversed.
    async fn find_path(
        &self,
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>>;

    /// Traverse graph crossing tenant boundaries via concept/narrative edges.
    ///
    /// Starts from `source_node_id` in `source_tenant`, then follows edges
    /// regardless of target node tenant_id. Callers should clamp `max_hops`
    /// to 1..=10 before calling.
    async fn query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
    ) -> GraphDbResult<Vec<TraversalNode>>;

    /// Query all edges of a given type, returning full `GraphEdge` records.
    ///
    /// Used by maintenance tasks that need to scan edge metadata (e.g. depth
    /// levels on COVERS_TOPIC edges). Backends that cannot support bulk edge
    /// queries return an empty vec by default.
    async fn query_edges_by_type(&self, edge_type: EdgeType) -> GraphDbResult<Vec<GraphEdge>> {
        let _ = edge_type;
        Ok(Vec::new())
    }

    /// Query distinct code-graph symbol names for a tenant, returning each
    /// symbol's real `node_id` and `file_path` so callers can target the
    /// existing node directly.
    ///
    /// Only structural (code) node types are returned — narrative and concept
    /// nodes are excluded. Backends that cannot support this return an empty
    /// vec by default (narrative EXPLAINS resolution then yields no edges).
    async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        let _ = tenant_id;
        Ok(Vec::new())
    }

    /// Delete file-owned narrative nodes (document_section / code_comment /
    /// docstring) for a file, so re-ingestion does not accumulate orphaned
    /// narrative nodes when headings shift or comments move (re-keying ids).
    ///
    /// Must NOT delete `library_section` (scoped by library_name, not file),
    /// `concept_node` (global, shared across files), or code-graph nodes.
    /// Returns the number of nodes deleted. Default no-op for backends that
    /// manage their own cleanup.
    async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let _ = (tenant_id, file_path);
        Ok(0)
    }

    /// Atomically re-ingest a file: delete old edges, upsert nodes, insert new
    /// edges — all within a single transaction. On error the database remains
    /// unchanged (all-or-nothing).
    ///
    /// The default implementation calls the three operations sequentially,
    /// which is correct for backends that provide their own atomicity
    /// guarantees (e.g. LadybugDB). `SqliteGraphStore` overrides this to
    /// use a single SQLite transaction.
    async fn reingest_file(
        &self,
        tenant_id: &str,
        file_path: &str,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> GraphDbResult<()> {
        self.delete_edges_by_file(tenant_id, file_path).await?;
        self.delete_narrative_nodes_by_file(tenant_id, file_path)
            .await?;
        self.upsert_nodes(nodes).await?;
        self.insert_edges(edges).await?;
        Ok(())
    }
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
        tenant_id,
        file_path,
        symbol_name,
        symbol_type.as_str()
    );
    let hash = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in &hash[..16] {
        let _ = write!(out, "{:02x}", b);
    }
    out
}

/// Fields for computing node IDs for narrative/concept node types.
#[derive(Debug, Clone)]
pub struct NodeIdFields<'a> {
    pub tenant_id: &'a str,
    pub file_path: &'a str,
    pub symbol_name: &'a str,
    pub symbol_type: NodeType,
    pub section_index: Option<u32>,
    pub start_line: Option<u32>,
    pub library_name: Option<&'a str>,
}

impl<'a> NodeIdFields<'a> {
    pub fn new(
        tenant_id: &'a str,
        file_path: &'a str,
        symbol_name: &'a str,
        symbol_type: NodeType,
    ) -> Self {
        Self {
            tenant_id,
            file_path,
            symbol_name,
            symbol_type,
            section_index: None,
            start_line: None,
            library_name: None,
        }
    }
}

/// Compute node ID dispatching to type-specific hashing logic.
pub fn compute_node_id_for_type(fields: &NodeIdFields<'_>) -> String {
    let input = match fields.symbol_type {
        NodeType::ConceptNode => {
            format!("concept_node|{}", fields.symbol_name)
        }
        NodeType::DocumentSection => {
            let idx = fields.section_index.unwrap_or(0);
            format!(
                "{}|{}|{}|document_section|{}",
                fields.tenant_id, fields.file_path, fields.symbol_name, idx
            )
        }
        NodeType::CodeComment => {
            let line = fields.start_line.unwrap_or(0);
            format!(
                "{}|{}|code_comment|{}",
                fields.tenant_id, fields.file_path, line
            )
        }
        NodeType::Docstring => {
            format!(
                "{}|{}|{}|docstring",
                fields.tenant_id, fields.file_path, fields.symbol_name
            )
        }
        NodeType::LibrarySection => {
            let lib = fields.library_name.unwrap_or(fields.tenant_id);
            let idx = fields.section_index.unwrap_or(0);
            format!(
                "{}|{}|{}|library_section|{}",
                lib, fields.file_path, fields.symbol_name, idx
            )
        }
        _ => {
            return compute_node_id(
                fields.tenant_id,
                fields.file_path,
                fields.symbol_name,
                fields.symbol_type,
            );
        }
    };

    let hash = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in &hash[..16] {
        let _ = write!(out, "{:02x}", b);
    }
    out
}

/// Compute deterministic edge ID from source, target, and type.
pub fn compute_edge_id(source_node_id: &str, target_node_id: &str, edge_type: EdgeType) -> String {
    let input = format!(
        "{}|{}|{}",
        source_node_id,
        target_node_id,
        edge_type.as_str()
    );
    let hash = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in &hash[..16] {
        let _ = write!(out, "{:02x}", b);
    }
    out
}
