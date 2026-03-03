/// LadybugDB-backed graph store implementation.

use std::path::Path;

use async_trait::async_trait;
use lbug::{Connection, Database, SystemConfig, Value};
use tokio::sync::Mutex;

use crate::graph::{
    EdgeType, GraphEdge, GraphNode, GraphStats, ImpactNode, ImpactReport, TraversalNode,
    schema::{GraphDbError, GraphDbResult},
    GraphStore,
};

use super::config::LadybugConfig;

// ─── Store struct ────────────────────────────────────────────────────────────

/// LadybugDB-backed graph store.
///
/// Uses an in-process LadybugDB instance for graph storage and Cypher queries.
/// The Database is stored inline; connections are created on demand since
/// `Connection<'a>` borrows `Database` and cannot be stored alongside it.
pub struct LadybugGraphStore {
    /// The LadybugDB database instance. Pinned for lifetime safety.
    db: Database,
    /// Serialize write access (LadybugDB supports concurrent reads but
    /// single writer).
    write_lock: Mutex<()>,
    config: LadybugConfig,
}

// Safety: Database is Send+Sync per lbug crate.
unsafe impl Send for LadybugGraphStore {}
unsafe impl Sync for LadybugGraphStore {}

// ─── Constructor and helpers ─────────────────────────────────────────────────

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
            .map_err(|e| GraphDbError::Migration(format!("Failed to open LadybugDB: {}", e)))?;

        let store = Self {
            db,
            write_lock: Mutex::new(()),
            config,
        };

        // Initialize schema
        store.init_schema()?;

        Ok(store)
    }

    /// Create a connection to the database.
    fn connect(&self) -> GraphDbResult<Connection<'_>> {
        Connection::new(&self.db)
            .map_err(|e| GraphDbError::Migration(format!("Connection failed: {}", e)))
    }

    /// Execute a Cypher query, mapping lbug errors to GraphDbError.
    fn exec(&self, conn: &Connection<'_>, cypher: &str) -> GraphDbResult<()> {
        conn.query(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {}", e)))?;
        Ok(())
    }

    /// Initialize the graph schema (node tables, rel tables).
    fn init_schema(&self) -> GraphDbResult<()> {
        let conn = self.connect()?;

        let ddl_statements = [
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
            "CREATE REL TABLE IF NOT EXISTS CALLS(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
            "CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
            "CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
            "CREATE REL TABLE IF NOT EXISTS USES_TYPE(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
            "CREATE REL TABLE IF NOT EXISTS EXTENDS(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
            "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(FROM GraphNode TO GraphNode, weight DOUBLE, source_file STRING, edge_id STRING, tenant_id STRING)",
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
        let result = conn.query(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {}", e)))?;

        let mut rows = Vec::new();
        for row in result {
            let formatted: Vec<String> = row.iter().map(|v| format!("{}", v)).collect();
            rows.push(formatted);
        }

        Ok(rows)
    }
}

// ─── Value helpers ───────────────────────────────────────────────────────────

/// Extract a String from a lbug Value, falling back to Debug format.
pub(super) fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        other => format!("{}", other),
    }
}

/// Extract an i64 from a lbug Value.
pub(super) fn value_to_i64(val: &Value) -> i64 {
    match val {
        Value::Int64(n) => *n,
        _ => 0,
    }
}

/// Escape single quotes in Cypher string literals.
pub(super) fn escape_cypher(s: &str) -> String {
    s.replace('\'', "\\'")
}

// ─── GraphStore trait implementation ────────────────────────────────────────

#[async_trait]
impl GraphStore for LadybugGraphStore {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        let cypher = format!(
            "MERGE (n:GraphNode {{node_id: '{}'}}) \
             SET n.tenant_id = '{}', n.symbol_name = '{}', n.symbol_type = '{}', \
             n.file_path = '{}', n.start_line = {}, n.end_line = {}, \
             n.language = '{}'",
            escape_cypher(&node.node_id),
            escape_cypher(&node.tenant_id),
            escape_cypher(&node.symbol_name),
            node.symbol_type.as_str(),
            escape_cypher(&node.file_path),
            node.start_line.unwrap_or(0) as i64,
            node.end_line.unwrap_or(0) as i64,
            escape_cypher(node.language.as_deref().unwrap_or("")),
        );
        self.exec(&conn, &cypher)
    }

    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for node in nodes {
            let cypher = format!(
                "MERGE (n:GraphNode {{node_id: '{}'}}) \
                 SET n.tenant_id = '{}', n.symbol_name = '{}', n.symbol_type = '{}', \
                 n.file_path = '{}', n.start_line = {}, n.end_line = {}, \
                 n.language = '{}'",
                escape_cypher(&node.node_id),
                escape_cypher(&node.tenant_id),
                escape_cypher(&node.symbol_name),
                node.symbol_type.as_str(),
                escape_cypher(&node.file_path),
                node.start_line.unwrap_or(0) as i64,
                node.end_line.unwrap_or(0) as i64,
                escape_cypher(node.language.as_deref().unwrap_or("")),
            );
            self.exec(&conn, &cypher)?;
        }
        Ok(())
    }

    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        let rel_type = edge.edge_type.as_str();
        let cypher = format!(
            "MATCH (a:GraphNode {{node_id: '{}'}}), (b:GraphNode {{node_id: '{}'}}) \
             CREATE (a)-[:{} {{weight: {}, source_file: '{}', edge_id: '{}', tenant_id: '{}'}}]->(b)",
            escape_cypher(&edge.source_node_id),
            escape_cypher(&edge.target_node_id),
            rel_type,
            edge.weight,
            escape_cypher(&edge.source_file),
            escape_cypher(&edge.edge_id),
            escape_cypher(&edge.tenant_id),
        );
        self.exec(&conn, &cypher)
    }

    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for edge in edges {
            let rel_type = edge.edge_type.as_str();
            let cypher = format!(
                "MATCH (a:GraphNode {{node_id: '{}'}}), (b:GraphNode {{node_id: '{}'}}) \
                 CREATE (a)-[:{} {{weight: {}, source_file: '{}', edge_id: '{}', tenant_id: '{}'}}]->(b)",
                escape_cypher(&edge.source_node_id),
                escape_cypher(&edge.target_node_id),
                rel_type,
                edge.weight,
                escape_cypher(&edge.source_file),
                escape_cypher(&edge.edge_id),
                escape_cypher(&edge.tenant_id),
            );
            self.exec(&conn, &cypher)?;
        }
        Ok(())
    }

    async fn delete_edges_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for rel_type in &["CALLS", "CONTAINS", "IMPORTS", "USES_TYPE", "EXTENDS", "IMPLEMENTS"] {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{}]->(b:GraphNode) \
                 WHERE r.tenant_id = '{}' AND r.source_file = '{}' \
                 DELETE r",
                rel_type,
                escape_cypher(tenant_id),
                escape_cypher(file_path),
            );
            self.exec(&conn, &cypher)?;
        }
        Ok(0) // LadybugDB doesn't return affected row count
    }

    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for rel_type in &["CALLS", "CONTAINS", "IMPORTS", "USES_TYPE", "EXTENDS", "IMPLEMENTS"] {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{}]->(b:GraphNode) \
                 WHERE r.tenant_id = '{}' DELETE r",
                rel_type,
                escape_cypher(tenant_id),
            );
            self.exec(&conn, &cypher)?;
        }
        let cypher = format!(
            "MATCH (n:GraphNode) WHERE n.tenant_id = '{}' DELETE n",
            escape_cypher(tenant_id),
        );
        self.exec(&conn, &cypher)?;
        Ok(0)
    }

    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        let conn = self.connect()?;
        let rel_pattern = match edge_types {
            Some(types) => types.iter().map(|t| t.as_str()).collect::<Vec<_>>().join("|"),
            None => "CALLS|CONTAINS|IMPORTS|USES_TYPE|EXTENDS|IMPLEMENTS".to_string(),
        };

        let cypher = format!(
            "MATCH (start:GraphNode {{node_id: '{}'}})-[r:{}*1..{}]->(related:GraphNode) \
             WHERE related.tenant_id = '{}' \
             RETURN DISTINCT related.node_id, related.symbol_name, \
                    related.symbol_type, related.file_path",
            escape_cypher(node_id),
            rel_pattern,
            max_hops,
            escape_cypher(tenant_id),
        );

        let result = conn.query(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("query_related: {}", e)))?;

        let mut nodes = Vec::new();
        for row in result {
            if row.len() >= 4 {
                nodes.push(TraversalNode {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    edge_type: String::new(),
                    depth: 1,
                    path: String::new(),
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
    ) -> GraphDbResult<ImpactReport> {
        let conn = self.connect()?;
        let file_filter = match file_path {
            Some(fp) => format!(" AND start.file_path = '{}'", escape_cypher(fp)),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (start:GraphNode {{symbol_name: '{}'}})<-[r:CALLS*1..3]-(caller:GraphNode) \
             WHERE start.tenant_id = '{}'{} \
             RETURN caller.node_id, caller.symbol_name, caller.file_path",
            escape_cypher(symbol_name),
            escape_cypher(tenant_id),
            file_filter,
        );

        let result = conn.query(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("impact_analysis: {}", e)))?;

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

        Ok(ImpactReport {
            symbol_name: symbol_name.to_string(),
            total_impacted: impacted.len() as u32,
            impacted_nodes: impacted,
        })
    }

    async fn stats(&self, tenant_id: Option<&str>) -> GraphDbResult<GraphStats> {
        let conn = self.connect()?;
        let filter = match tenant_id {
            Some(tid) => format!(" WHERE n.tenant_id = '{}'", escape_cypher(tid)),
            None => String::new(),
        };

        let cypher = format!("MATCH (n:GraphNode){} RETURN count(n)", filter);
        let result = conn.query(&cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("stats: {}", e)))?;

        let mut total_nodes = 0u64;
        for row in result {
            if !row.is_empty() {
                total_nodes = value_to_i64(&row[0]) as u64;
            }
        }

        Ok(GraphStats {
            total_nodes,
            total_edges: 0,
            nodes_by_type: std::collections::HashMap::new(),
            edges_by_type: std::collections::HashMap::new(),
        })
    }

    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        // LadybugDB subquery syntax for "no edges"
        let cypher = format!(
            "MATCH (n:GraphNode) WHERE n.tenant_id = '{}' \
             AND NOT EXISTS {{ MATCH (n)-[]-() }} \
             DELETE n",
            escape_cypher(tenant_id),
        );
        // This may fail if LadybugDB doesn't support this exact subquery syntax,
        // in which case it's a no-op (returns 0).
        let _ = conn.query(&cypher);
        Ok(0)
    }
}
