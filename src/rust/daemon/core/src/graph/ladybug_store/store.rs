//! LadybugDB-backed graph store implementation.
//!
//! Uses `lbug::Database` for an in-process property graph with native Cypher
//! queries. All user-supplied values pass through parameterized queries
//! (`$param` + `PreparedStatement`) to prevent injection.

use std::path::Path;

use async_trait::async_trait;
use lbug::{Connection, Database, SystemConfig, Value};
use tokio::sync::Mutex;
use tracing::debug;

use crate::graph::{
    cross_boundary::{apply_fan_out_caps, tenant_relaxation_set, CROSS_BOUNDARY_MAX_HOPS},
    schema::{GraphDbError, GraphDbResult},
    EdgeType, GraphEdge, GraphNode, GraphStats, GraphStore, ImpactNode, ImpactReport, SymbolRow,
    TraversalNode,
};

use super::config::LadybugConfig;

// ---- Constants ---------------------------------------------------------------

/// Every relationship type in the schema. Used for operations that must iterate
/// over all rel tables (delete-by-file, delete-tenant, stats, prune) so that
/// narrative and concept edges are not silently skipped.
const ALL_REL_TYPES: &[&str] = &[
    // Structural
    "CALLS",
    "CONTAINS",
    "IMPORTS",
    "USES_TYPE",
    "EXTENDS",
    "IMPLEMENTS",
    // Narrative
    "EXPLAINS",
    "DESCRIBES",
    "REFERENCES_DOC",
    "ELABORATES",
    // Concept
    "COVERS_TOPIC",
    "IMPLEMENTS_CONCEPT",
];

/// Structural code-graph node types (mirrors `SqliteGraphStore::query_code_symbols`).
const CODE_SYMBOL_TYPES: &[&str] = &[
    "function",
    "async_function",
    "class",
    "method",
    "struct",
    "trait",
    "interface",
    "enum",
    "impl",
    "module",
    "constant",
    "type_alias",
    "macro",
];

/// File-owned narrative node types deleted on re-ingestion (mirrors SQLite).
const NARRATIVE_FILE_NODE_TYPES: &[&str] = &["document_section", "code_comment", "docstring"];

// ---- Store struct ------------------------------------------------------------

/// LadybugDB-backed graph store.
///
/// Uses an in-process LadybugDB instance for graph storage and Cypher queries.
/// The `Database` is stored inline; connections are created on demand since
/// `Connection<'a>` borrows `Database` and cannot be stored alongside it.
pub struct LadybugGraphStore {
    /// The LadybugDB database instance.
    db: Database,
    /// Serialize write access (LadybugDB supports concurrent reads but
    /// single writer).
    write_lock: Mutex<()>,
    config: LadybugConfig,
    /// Cross-boundary fan-out caps and fusion settings (shared with SQLite).
    graph_rag: crate::config::GraphRagConfig,
}

// Safety: Database is Send+Sync per lbug crate documentation.
// Connection is also Send+Sync per the lbug crate.
unsafe impl Send for LadybugGraphStore {}
unsafe impl Sync for LadybugGraphStore {}

// ---- Constructor and helpers -------------------------------------------------

impl LadybugGraphStore {
    /// Create a new LadybugDB graph store.
    ///
    /// Opens (or creates) a database at the configured path and initializes
    /// the graph schema if needed.
    pub fn new(config: LadybugConfig) -> GraphDbResult<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let sys_config = SystemConfig::default()
            .buffer_pool_size(config.buffer_pool_size)
            .max_num_threads(config.max_num_threads);

        let db = Database::new(&config.db_path, sys_config)
            .map_err(|e| GraphDbError::Migration(format!("Failed to open LadybugDB: {e}")))?;

        let store = Self {
            db,
            write_lock: Mutex::new(()),
            config,
            graph_rag: crate::config::GraphRagConfig::default(),
        };

        // Initialize schema (idempotent via IF NOT EXISTS)
        store.init_schema()?;

        Ok(store)
    }

    /// Override the cross-boundary graph-RAG configuration (fan-out caps).
    pub fn with_graph_rag_config(mut self, config: crate::config::GraphRagConfig) -> Self {
        self.graph_rag = config;
        self
    }

    /// Create a connection to the database.
    fn connect(&self) -> GraphDbResult<Connection<'_>> {
        Connection::new(&self.db)
            .map_err(|e| GraphDbError::Migration(format!("Connection failed: {e}")))
    }

    /// Execute a DDL/mutation query (no parameters), mapping lbug errors.
    fn exec(&self, conn: &Connection<'_>, cypher: &str) -> GraphDbResult<()> {
        conn.query(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {e}")))?;
        Ok(())
    }

    /// Initialize the graph schema (node tables, rel tables).
    ///
    /// Uses `IF NOT EXISTS` so this is idempotent across restarts.
    fn init_schema(&self) -> GraphDbResult<()> {
        let conn = self.connect()?;

        let ddl_statements = [
            // Node table with all properties from GraphNode
            r#"CREATE NODE TABLE IF NOT EXISTS GraphNode(
                node_id STRING,
                tenant_id STRING,
                symbol_name STRING,
                symbol_type STRING,
                file_path STRING,
                start_line INT64,
                end_line INT64,
                signature STRING,
                language STRING,
                PRIMARY KEY (node_id)
            )"#,
            // One rel table per EdgeType, each carrying edge metadata
            "CREATE REL TABLE IF NOT EXISTS CALLS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS USES_TYPE(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS EXTENDS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            // Narrative layer
            "CREATE REL TABLE IF NOT EXISTS EXPLAINS(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS DESCRIBES(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS REFERENCES_DOC(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS ELABORATES(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            // Concept layer
            "CREATE REL TABLE IF NOT EXISTS COVERS_TOPIC(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
            "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS_CONCEPT(FROM GraphNode TO GraphNode, \
             weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING, \
             metadata_json STRING)",
        ];

        for ddl in &ddl_statements {
            self.exec(&conn, ddl)?;
        }

        Ok(())
    }

    /// Get the database path.
    pub fn db_path(&self) -> &Path {
        &self.config.db_path
    }

    /// Execute a raw Cypher query and return results as formatted strings.
    ///
    /// This is LadybugDB-specific (not part of GraphStore trait).
    pub fn execute_cypher(&self, cypher: &str) -> GraphDbResult<Vec<Vec<String>>> {
        let conn = self.connect()?;
        let result = conn
            .query(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {e}")))?;

        let mut rows = Vec::new();
        for row in result {
            let formatted: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
            rows.push(formatted);
        }

        Ok(rows)
    }

    /// Upsert a single node using parameterized MERGE+SET.
    ///
    /// MERGE matches on `node_id` (primary key); SET updates all properties.
    /// Per T34 findings, no ON CREATE/ON MATCH needed -- bare SET after MERGE
    /// handles both cases.
    fn upsert_node_with_conn(&self, conn: &Connection<'_>, node: &GraphNode) -> GraphDbResult<()> {
        let mut stmt = conn
            .prepare(
                "MERGE (n:GraphNode {node_id: $node_id}) \
                 SET n.tenant_id = $tenant_id, \
                     n.symbol_name = $symbol_name, \
                     n.symbol_type = $symbol_type, \
                     n.file_path = $file_path, \
                     n.start_line = $start_line, \
                     n.end_line = $end_line, \
                     n.signature = $signature, \
                     n.language = $language",
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare upsert_node failed: {e}")))?;

        conn.execute(
            &mut stmt,
            vec![
                ("node_id", Value::String(node.node_id.clone())),
                ("tenant_id", Value::String(node.tenant_id.clone())),
                ("symbol_name", Value::String(node.symbol_name.clone())),
                (
                    "symbol_type",
                    Value::String(node.symbol_type.as_str().to_string()),
                ),
                ("file_path", Value::String(node.file_path.clone())),
                (
                    "start_line",
                    Value::Int64(node.start_line.unwrap_or(0) as i64),
                ),
                ("end_line", Value::Int64(node.end_line.unwrap_or(0) as i64)),
                (
                    "signature",
                    Value::String(node.signature.clone().unwrap_or_default()),
                ),
                (
                    "language",
                    Value::String(node.language.clone().unwrap_or_default()),
                ),
            ],
        )
        .map_err(|e| GraphDbError::InvalidInput(format!("Execute upsert_node failed: {e}")))?;

        Ok(())
    }

    /// Insert a single edge using parameterized MATCH+CREATE with typed
    /// relationship pattern (`-[r:CALLS]->`).
    ///
    /// Since Cypher does not allow parameterized rel types, we use a separate
    /// prepared statement per `EdgeType`. The rel type name is a compile-time
    /// constant, not user input, so this is safe.
    fn insert_edge_with_conn(&self, conn: &Connection<'_>, edge: &GraphEdge) -> GraphDbResult<()> {
        let rel_type = edge.edge_type.as_str();
        // Build the Cypher string with the rel type literal (safe: comes from
        // EdgeType::as_str(), not user input).
        let cypher = format!(
            "MATCH (a:GraphNode {{node_id: $src_id}}), (b:GraphNode {{node_id: $dst_id}}) \
             CREATE (a)-[:{rel_type} {{weight: $weight, source_file: $source_file, \
             edge_id: $edge_id, tenant_id: $tenant_id, metadata_json: $metadata_json}}]->(b)"
        );

        let mut stmt = conn
            .prepare(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare insert_edge failed: {e}")))?;

        conn.execute(
            &mut stmt,
            vec![
                ("src_id", Value::String(edge.source_node_id.clone())),
                ("dst_id", Value::String(edge.target_node_id.clone())),
                ("weight", Value::Double(edge.weight)),
                ("source_file", Value::String(edge.source_file.clone())),
                ("edge_id", Value::String(edge.edge_id.clone())),
                ("tenant_id", Value::String(edge.tenant_id.clone())),
                (
                    "metadata_json",
                    Value::String(edge.metadata_json.clone().unwrap_or_default()),
                ),
            ],
        )
        .map_err(|e| GraphDbError::InvalidInput(format!("Execute insert_edge failed: {e}")))?;

        Ok(())
    }

    /// Fetch the direct (1-hop) neighbours of `current_id` in BOTH directions
    /// over the allowed `rel_pattern`, keeping only neighbours whose tenant is
    /// in `tenants`. Returns the reached node plus the reaching edge's type and
    /// weight so the caller can score the hop.
    fn cross_boundary_neighbours(
        &self,
        conn: &Connection<'_>,
        current_id: &str,
        rel_pattern: &str,
        tenants: &[String],
    ) -> GraphDbResult<Vec<CrossBoundaryNeighbour>> {
        // Tenant IN-list literal. Tenant ids come from project registration,
        // not free-form query input, but we quote-escape defensively.
        let tenant_list = tenants
            .iter()
            .map(|t| format!("'{}'", t.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(",");

        let mut out = Vec::new();

        // Outgoing: (current)-[r]->(n); Incoming: (current)<-[r]-(n).
        let patterns = [
            format!("(c:GraphNode {{node_id: $cid}})-[r:{rel_pattern}]->(n:GraphNode)"),
            format!("(c:GraphNode {{node_id: $cid}})<-[r:{rel_pattern}]-(n:GraphNode)"),
        ];
        for pattern in patterns {
            let cypher = format!(
                "MATCH {pattern} \
                 WHERE n.tenant_id IN [{tenant_list}] \
                 RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path, \
                        n.tenant_id, label(r), r.weight"
            );
            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare cross_boundary_neighbours: {e}"))
            })?;
            let result = conn
                .execute(
                    &mut stmt,
                    vec![("cid", Value::String(current_id.to_string()))],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!("Execute cross_boundary_neighbours: {e}"))
                })?;
            for row in result {
                if row.len() < 7 {
                    continue;
                }
                let weight = match &row[6] {
                    Value::Double(d) => *d,
                    Value::Int64(i) => *i as f64,
                    _ => 1.0,
                };
                out.push(CrossBoundaryNeighbour {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    tenant_id: value_to_string(&row[4]),
                    edge_type: value_to_string(&row[5]),
                    weight,
                });
            }
        }

        Ok(out)
    }
}

// ---- Value helpers -----------------------------------------------------------

/// A node reached in one cross-boundary hop, with the reaching edge's type and
/// weight (used to compute per-hop confidence).
struct CrossBoundaryNeighbour {
    node_id: String,
    symbol_name: String,
    symbol_type: String,
    file_path: String,
    tenant_id: String,
    edge_type: String,
    weight: f64,
}

/// Extract a String from a lbug Value, falling back to Display format.
pub(super) fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        other => format!("{other}"),
    }
}

/// Extract an i64 from a lbug Value.
pub(super) fn value_to_i64(val: &Value) -> i64 {
    match val {
        Value::Int64(n) => *n,
        Value::UInt64(n) => *n as i64,
        Value::Int32(n) => *n as i64,
        _ => 0,
    }
}

/// Escape single quotes in Cypher string literals.
///
/// Retained for test coverage; production code uses `PreparedStatement`
/// parameters for all user-supplied values.
#[cfg(test)]
pub(super) fn escape_cypher(s: &str) -> String {
    s.replace('\'', "\\'")
}

// ---- GraphStore trait implementation -----------------------------------------

#[async_trait]
impl GraphStore for LadybugGraphStore {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        self.upsert_node_with_conn(&conn, node)
    }

    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for node in nodes {
            self.upsert_node_with_conn(&conn, node)?;
        }
        debug!("Upserted {} graph nodes (LadybugDB)", nodes.len());
        Ok(())
    }

    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        self.insert_edge_with_conn(&conn, edge)
    }

    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for edge in edges {
            self.insert_edge_with_conn(&conn, edge)?;
        }
        debug!("Inserted {} graph edges (LadybugDB)", edges.len());
        Ok(())
    }

    async fn delete_edges_by_file(&self, tenant_id: &str, file_path: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Must delete from each rel table separately (LadybugDB requires
        // typed MATCH patterns per T34 findings).
        for rel_type in ALL_REL_TYPES {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                 WHERE r.tenant_id = $tid AND r.source_file = $fp \
                 DELETE r"
            );
            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare delete_edges failed: {e}"))
            })?;
            conn.execute(
                &mut stmt,
                vec![
                    ("tid", Value::String(tenant_id.to_string())),
                    ("fp", Value::String(file_path.to_string())),
                ],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute delete_edges failed: {e}")))?;
        }

        debug!(
            "Deleted edges for file {} in tenant {} (LadybugDB)",
            file_path, tenant_id
        );
        // LadybugDB does not return affected row counts from DELETE
        Ok(0)
    }

    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Delete all edges first (per rel table)
        for rel_type in ALL_REL_TYPES {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                 WHERE r.tenant_id = $tid DELETE r"
            );
            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare delete_tenant edges: {e}"))
            })?;
            conn.execute(
                &mut stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute delete_tenant edges: {e}")))?;
        }

        // Then delete all nodes for this tenant
        let mut stmt = conn
            .prepare("MATCH (n:GraphNode) WHERE n.tenant_id = $tid DELETE n")
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare delete_tenant nodes: {e}")))?;
        conn.execute(
            &mut stmt,
            vec![("tid", Value::String(tenant_id.to_string()))],
        )
        .map_err(|e| GraphDbError::InvalidInput(format!("Execute delete_tenant nodes: {e}")))?;

        debug!("Deleted tenant {} data (LadybugDB)", tenant_id);
        Ok(0)
    }

    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        _branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        if max_hops == 0 {
            return Ok(Vec::new());
        }
        let conn = self.connect()?;

        // Rel type pattern. Rel types come from the EdgeType enum (static
        // literals), so string interpolation is injection-safe.
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // SQLite reports the TRUE minimum depth per reached node. Kuzu's
        // variable-length `*1..n` pattern cannot expose per-row depth directly,
        // so we query each exact hop length `*k..k` in ascending order and keep
        // the first (minimum) depth at which a node is reached. The last edge's
        // type is captured for parity with the SQLite `edge_type` column.
        let mut by_node: std::collections::HashMap<String, TraversalNode> =
            std::collections::HashMap::new();

        for hop in 1..=max_hops {
            // Kuzu does not allow projecting a property off an indexed element
            // of a recursive-rel list (`rels[i].edge_type`), so we return only
            // node identity. `edge_type` is left empty for the per-hop result;
            // the SQLite backend populates it, but cross-backend conformance is
            // asserted on the (node_id, depth) and identity maps, not on the
            // reaching edge type.
            let cypher = format!(
                "MATCH (start:GraphNode {{node_id: $start_id}})\
                 -[rels:{rel_pattern}*{hop}..{hop}]->(related:GraphNode) \
                 WHERE related.tenant_id = $tid \
                 RETURN related.node_id, related.symbol_name, related.symbol_type, \
                        related.file_path"
            );

            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare query_related (hop {hop}): {e}"))
            })?;

            let result = conn
                .execute(
                    &mut stmt,
                    vec![
                        ("start_id", Value::String(node_id.to_string())),
                        ("tid", Value::String(tenant_id.to_string())),
                    ],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!("Execute query_related (hop {hop}): {e}"))
                })?;

            for row in result {
                if row.len() < 4 {
                    continue;
                }
                let nid = value_to_string(&row[0]);
                // Do not overwrite a shallower entry recorded at a smaller hop.
                by_node.entry(nid.clone()).or_insert_with(|| TraversalNode {
                    node_id: nid,
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    edge_type: String::new(),
                    depth: hop,
                    path: String::new(),
                    tenant_id: tenant_id.to_string(),
                    edge_confidence: 1.0,
                });
            }
        }
        let mut nodes: Vec<TraversalNode> = by_node.into_values().collect();
        nodes.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then(a.symbol_name.cmp(&b.symbol_name))
        });
        Ok(nodes)
    }

    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        _branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        let conn = self.connect()?;

        // Reverse traversal: find all callers up to 3 hops (matching SQLite
        // implementation's depth limit).
        // We query each rel type separately and merge to get proper impact_type.
        let rel_pattern = ALL_REL_TYPES.join("|");

        let (cypher, params) = if let Some(fp) = file_path {
            let c = format!(
                "MATCH (start:GraphNode)<-[r:{rel_pattern}*1..3]-(caller:GraphNode) \
                 WHERE start.symbol_name = $sym AND start.tenant_id = $tid \
                       AND start.file_path = $fp \
                 RETURN DISTINCT caller.node_id, caller.symbol_name, caller.file_path"
            );
            let p: Vec<(&str, Value)> = vec![
                ("sym", Value::String(symbol_name.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
                ("fp", Value::String(fp.to_string())),
            ];
            (c, p)
        } else {
            let c = format!(
                "MATCH (start:GraphNode)<-[r:{rel_pattern}*1..3]-(caller:GraphNode) \
                 WHERE start.symbol_name = $sym AND start.tenant_id = $tid \
                 RETURN DISTINCT caller.node_id, caller.symbol_name, caller.file_path"
            );
            let p: Vec<(&str, Value)> = vec![
                ("sym", Value::String(symbol_name.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
            ];
            (c, p)
        };

        let mut stmt = conn
            .prepare(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare impact_analysis: {e}")))?;

        let result = conn
            .execute(&mut stmt, params)
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute impact_analysis: {e}")))?;

        let mut impacted = Vec::new();
        for row in result {
            if row.len() >= 3 {
                impacted.push(ImpactNode {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    file_path: value_to_string(&row[2]),
                    impact_type: "caller".to_string(),
                    distance: 1,
                });
            }
        }

        let total = impacted.len() as u32;
        Ok(ImpactReport {
            symbol_name: symbol_name.to_string(),
            impacted_nodes: impacted,
            total_impacted: total,
        })
    }

    async fn find_path(
        &self,
        _tenant_id: &str,
        _source_id: &str,
        _target_id: &str,
        _max_depth: u32,
        _edge_types: Option<&[EdgeType]>,
        _branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        // LadybugDB path-finding will use Cypher SHORTEST PATH in a future iteration.
        // For now, return None (no path found) — callers handle this gracefully.
        Ok(None)
    }

    async fn stats(
        &self,
        tenant_id: Option<&str>,
        _branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        let conn = self.connect()?;

        // --- Node counts by type ---
        let mut nodes_by_type = std::collections::HashMap::new();
        let mut total_nodes = 0u64;

        let (cypher, params): (String, Vec<(&str, Value)>) = match tenant_id {
            Some(tid) => (
                "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
                 RETURN n.symbol_type, count(n)"
                    .to_string(),
                vec![("tid", Value::String(tid.to_string()))],
            ),
            None => (
                "MATCH (n:GraphNode) RETURN n.symbol_type, count(n)".to_string(),
                vec![],
            ),
        };

        let mut stmt = conn
            .prepare(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare stats nodes: {e}")))?;

        let result = conn
            .execute(&mut stmt, params)
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute stats nodes: {e}")))?;

        for row in result {
            if row.len() >= 2 {
                let stype = value_to_string(&row[0]);
                let cnt = value_to_i64(&row[1]) as u64;
                total_nodes += cnt;
                nodes_by_type.insert(stype, cnt);
            }
        }

        // --- Edge counts by type ---
        let mut edges_by_type = std::collections::HashMap::new();
        let mut total_edges = 0u64;

        for rel_type in ALL_REL_TYPES {
            let (cypher, params): (String, Vec<(&str, Value)>) = match tenant_id {
                Some(tid) => (
                    format!("MATCH ()-[r:{rel_type}]->() WHERE r.tenant_id = $tid RETURN count(r)"),
                    vec![("tid", Value::String(tid.to_string()))],
                ),
                None => (
                    format!("MATCH ()-[r:{rel_type}]->() RETURN count(r)"),
                    vec![],
                ),
            };

            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare stats edges ({rel_type}): {e}"))
            })?;

            let result = conn.execute(&mut stmt, params).map_err(|e| {
                GraphDbError::InvalidInput(format!("Execute stats edges ({rel_type}): {e}"))
            })?;

            for row in result {
                if !row.is_empty() {
                    let cnt = value_to_i64(&row[0]) as u64;
                    if cnt > 0 {
                        total_edges += cnt;
                        edges_by_type.insert(rel_type.to_string(), cnt);
                    }
                }
            }
        }

        Ok(GraphStats {
            total_nodes,
            total_edges,
            nodes_by_type,
            edges_by_type,
        })
    }

    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Find nodes with no incident edges across any rel table, then delete.
        // LadybugDB's EXISTS subquery checks all rel tables via the union pattern.
        let all_rels = ALL_REL_TYPES.join("|");
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid \
             AND NOT EXISTS {{ MATCH (n)-[:{all_rels}]-() }} \
             DELETE n"
        );

        // This may not be supported by all LadybugDB versions; if it fails,
        // fall back gracefully.
        let result = (|| -> GraphDbResult<()> {
            let mut stmt = conn
                .prepare(&cypher)
                .map_err(|e| GraphDbError::InvalidInput(format!("Prepare prune_orphans: {e}")))?;
            conn.execute(
                &mut stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute prune_orphans: {e}")))?;
            Ok(())
        })();

        if let Err(e) = result {
            debug!("prune_orphans subquery not supported, skipping: {}", e);
        }

        Ok(0)
    }

    async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        let conn = self.connect()?;
        let type_list = CODE_SYMBOL_TYPES
            .iter()
            .map(|t| format!("'{t}'"))
            .collect::<Vec<_>>()
            .join(",");
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND n.symbol_type IN [{type_list}] \
                   AND n.symbol_name <> '' \
             RETURN n.symbol_name, n.node_id, n.file_path"
        );
        let mut stmt = conn
            .prepare(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare query_code_symbols: {e}")))?;
        let result = conn
            .execute(
                &mut stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute query_code_symbols: {e}")))?;
        let mut rows = Vec::new();
        for row in result {
            if row.len() >= 3 {
                rows.push(SymbolRow {
                    symbol_name: value_to_string(&row[0]),
                    node_id: value_to_string(&row[1]),
                    file_path: value_to_string(&row[2]),
                });
            }
        }
        Ok(rows)
    }

    async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        let type_list = NARRATIVE_FILE_NODE_TYPES
            .iter()
            .map(|t| format!("'{t}'"))
            .collect::<Vec<_>>()
            .join(",");
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND n.file_path = $fp \
                   AND n.symbol_type IN [{type_list}] \
             DELETE n"
        );
        let mut stmt = conn.prepare(&cypher).map_err(|e| {
            GraphDbError::InvalidInput(format!("Prepare delete_narrative_nodes_by_file: {e}"))
        })?;
        conn.execute(
            &mut stmt,
            vec![
                ("tid", Value::String(tenant_id.to_string())),
                ("fp", Value::String(file_path.to_string())),
            ],
        )
        .map_err(|e| {
            GraphDbError::InvalidInput(format!("Execute delete_narrative_nodes_by_file: {e}"))
        })?;
        // LadybugDB does not return affected row counts from DELETE.
        Ok(0)
    }

    async fn query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
        library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        if edge_types.is_empty() || max_hops == 0 {
            return Ok(Vec::new());
        }
        let hops = max_hops.clamp(1, CROSS_BOUNDARY_MAX_HOPS);
        let conn = self.connect()?;

        // Tenant relaxation set: source ∪ {"__global__"} ∪ library_tenants.
        let tenants = tenant_relaxation_set(source_tenant, library_tenants);

        // Rel-type pattern (static literals from EdgeType — injection-safe).
        let rel_pattern = edge_types
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join("|");

        // Bidirectional BFS expanded one hop length at a time so we can record
        // the TRUE minimum depth, the reaching edge type, and the node path for
        // each reached node (the shared fan-out caps depend on `path`). Each hop
        // expands the current frontier in both directions, applying the tenant
        // guard so we never traverse through a foreign tenant.
        let mut reached: std::collections::HashMap<String, TraversalNode> =
            std::collections::HashMap::new();
        // Frontier maps node_id -> path string up to that node (source first).
        let mut frontier: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        frontier.insert(source_node_id.to_string(), source_node_id.to_string());

        for depth in 1..=hops {
            let mut next: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for (current_id, current_path) in &frontier {
                let neighbours =
                    self.cross_boundary_neighbours(&conn, current_id, &rel_pattern, &tenants)?;
                for nb in neighbours {
                    // Acyclic guard: skip nodes already on this path.
                    if current_path.split(" -> ").any(|seg| seg == nb.node_id) {
                        continue;
                    }
                    let new_path = format!("{current_path} -> {}", nb.node_id);
                    let confidence = nb.weight
                        * crate::graph::cross_boundary::edge_type_base_confidence(&nb.edge_type);
                    // Record the shallowest reach; on equal depth keep the
                    // higher-confidence reach (matches SQLite MIN depth / MAX conf).
                    match reached.entry(nb.node_id.clone()) {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(TraversalNode {
                                node_id: nb.node_id.clone(),
                                symbol_name: nb.symbol_name,
                                symbol_type: nb.symbol_type,
                                file_path: nb.file_path,
                                edge_type: nb.edge_type,
                                depth,
                                path: new_path.clone(),
                                tenant_id: nb.tenant_id,
                                edge_confidence: confidence,
                            });
                        }
                        std::collections::hash_map::Entry::Occupied(mut o) => {
                            let existing = o.get_mut();
                            if depth < existing.depth
                                || (depth == existing.depth
                                    && confidence > existing.edge_confidence)
                            {
                                existing.depth = depth;
                                existing.edge_type = nb.edge_type;
                                existing.edge_confidence = confidence;
                                existing.path = new_path.clone();
                                existing.tenant_id = nb.tenant_id;
                            }
                        }
                    }
                    // Continue BFS from the first (shortest) path to this node.
                    next.entry(nb.node_id.clone()).or_insert(new_path);
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }

        let results: Vec<TraversalNode> = reached.into_values().collect();
        Ok(apply_fan_out_caps(results, &self.graph_rag))
    }
}
