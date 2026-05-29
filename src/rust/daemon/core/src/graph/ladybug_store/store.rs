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
    schema::{GraphDbError, GraphDbResult},
    EdgeType, GraphEdge, GraphNode, GraphStats, GraphStore, ImpactNode, ImpactReport,
    TraversalNode,
};

use super::config::LadybugConfig;

// ---- Constants ---------------------------------------------------------------

/// All relationship types in the schema. Used for operations that must iterate
/// over every rel table (e.g. delete-by-file, stats).
const ALL_REL_TYPES: &[&str] = &[
    "CALLS",
    "CONTAINS",
    "IMPORTS",
    "USES_TYPE",
    "EXTENDS",
    "IMPLEMENTS",
];

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
        };

        // Initialize schema (idempotent via IF NOT EXISTS)
        store.init_schema()?;

        Ok(store)
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
}

// ---- Value helpers -----------------------------------------------------------

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
        let conn = self.connect()?;

        // Build the rel type pattern. Rel types come from EdgeType enum, not
        // user input, so string formatting is safe here.
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // Use recursive rel pattern for variable-length traversal.
        // Return the full recursive rel so we can extract depth info.
        let cypher = format!(
            "MATCH (start:GraphNode {{node_id: $start_id}})-[rels:{rel_pattern}*1..{max_hops}]->\
             (related:GraphNode) \
             WHERE related.tenant_id = $tid \
             RETURN DISTINCT related.node_id, related.symbol_name, \
                    related.symbol_type, related.file_path"
        );

        let mut stmt = conn
            .prepare(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare query_related: {e}")))?;

        let result = conn
            .execute(
                &mut stmt,
                vec![
                    ("start_id", Value::String(node_id.to_string())),
                    ("tid", Value::String(tenant_id.to_string())),
                ],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute query_related: {e}")))?;

        let mut nodes = Vec::new();
        for row in result {
            if row.len() >= 4 {
                nodes.push(TraversalNode {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    edge_type: String::new(), // aggregate over variable-length path
                    depth: 1,
                    path: String::new(),
                    tenant_id: tenant_id.to_string(),
                    edge_confidence: 1.0,
                });
            }
        }

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

    async fn query_cross_boundary(
        &self,
        _source_tenant: &str,
        _source_node_id: &str,
        _edge_types: &[EdgeType],
        _max_hops: u32,
        _library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        Ok(Vec::new())
    }
}
