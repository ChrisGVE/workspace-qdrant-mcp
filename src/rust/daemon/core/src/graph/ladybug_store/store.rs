//! LadybugDB-backed graph store implementation.
//!
//! Uses `lbug::Database` for an in-process property graph with native Cypher
//! queries. All user-supplied values pass through parameterized queries
//! (`$param` + `PreparedStatement`) to prevent injection.

use std::panic::AssertUnwindSafe;
use std::path::Path;

use async_trait::async_trait;
use lbug::{Connection, Database, SystemConfig, Value};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::graph::{
    compute_edge_id,
    cross_boundary::{apply_fan_out_caps, tenant_relaxation_set, CROSS_BOUNDARY_MAX_HOPS},
    schema::{GraphDbError, GraphDbResult},
    AdjacencyExport, EdgeType, GraphEdge, GraphNode, GraphStats, GraphStore, ImpactNode,
    ImpactReport, NodeMetadata, SymbolRow, TraversalNode,
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
/// `library_section` is included so library-document re-ingest is cleaned
/// per-file on this backend too (tenant_id == library_name for libraries),
/// matching `SqliteGraphStore`'s delete filter.
const NARRATIVE_FILE_NODE_TYPES: &[&str] = &[
    "document_section",
    "library_section",
    "code_comment",
    "docstring",
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
    ///
    /// The `Connection::new` call crosses into the lbug C++ layer; it is
    /// wrapped with [`lbug_call`] so a Rust panic in the binding is caught and
    /// converted to [`GraphDbError::InternalError`] rather than unwinding the
    /// tokio runtime (SEC-03).
    fn connect(&self) -> GraphDbResult<Connection<'_>> {
        // lbug FFI boundary: Connection::new calls into C++ kuzu internals.
        lbug_call(|| Connection::new(&self.db)).and_then(|result| {
            result.map_err(|e| GraphDbError::Migration(format!("Connection failed: {e}")))
        })
    }

    /// Execute a DDL/mutation query (no parameters), mapping lbug errors.
    ///
    /// The `conn.query` call crosses into the lbug C++ layer; wrapped with
    /// [`lbug_call`] for panic containment (SEC-03).
    fn exec(&self, conn: &Connection<'_>, cypher: &str) -> GraphDbResult<()> {
        // lbug FFI boundary: conn.query calls into C++ kuzu query execution.
        lbug_call(|| conn.query(cypher)).and_then(|result| {
            result.map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {e}")))?;
            Ok(())
        })
    }

    /// Initialize the graph schema (node tables, rel tables).
    ///
    /// Uses `IF NOT EXISTS` so this is idempotent across restarts.
    fn init_schema(&self) -> GraphDbResult<()> {
        let conn = self.connect()?;

        let ddl_statements = [
            // Node table with all properties from GraphNode.
            // `qdrant_point_id` links chunk-derived nodes to their Qdrant embedding
            // vector; empty string is the sentinel for "no link" (Kuzu does not
            // support NULL for STRING properties cleanly). `point_id_state` tracks
            // whether the link is established ("linked") or absent ("none").
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
                qdrant_point_id STRING,
                point_id_state STRING,
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
    ///
    /// Both `conn.prepare` and `conn.execute` cross into the lbug C++ layer
    /// and are wrapped with [`lbug_call`] for Rust-panic containment (SEC-03).
    fn upsert_node_with_conn(&self, conn: &Connection<'_>, node: &GraphNode) -> GraphDbResult<()> {
        // Kuzu STRING properties do not support NULL; use an empty string as the
        // sentinel for "no Qdrant link" and restore `None` on read (round-trip
        // lossless: empty → None, non-empty → Some(value)). The MERGE+SET query
        // applies COALESCE-equivalent semantics via a CASE expression so a
        // re-upsert with an empty point_id never clobbers an existing link;
        // `point_id_state` always takes the caller-authoritative new value.
        let point_id_str = node.qdrant_point_id.clone().unwrap_or_default();

        // lbug FFI boundary: conn.prepare calls into C++ kuzu statement compilation.
        let mut stmt = lbug_call(|| {
            conn.prepare(
                "MERGE (n:GraphNode {node_id: $node_id}) \
                 SET n.tenant_id = $tenant_id, \
                     n.symbol_name = $symbol_name, \
                     n.symbol_type = $symbol_type, \
                     n.file_path = $file_path, \
                     n.start_line = $start_line, \
                     n.end_line = $end_line, \
                     n.signature = $signature, \
                     n.language = $language, \
                     n.qdrant_point_id = \
                         CASE WHEN $qdrant_point_id <> '' THEN $qdrant_point_id \
                              WHEN n.qdrant_point_id IS NOT NULL AND n.qdrant_point_id <> '' \
                                  THEN n.qdrant_point_id \
                              ELSE $qdrant_point_id END, \
                     n.point_id_state = $point_id_state",
            )
        })
        .and_then(|r| {
            r.map_err(|e| GraphDbError::InvalidInput(format!("Prepare upsert_node failed: {e}")))
        })?;

        // lbug FFI boundary: conn.execute calls into C++ kuzu query execution.
        lbug_call(|| {
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
                    ("qdrant_point_id", Value::String(point_id_str)),
                    ("point_id_state", Value::String(node.point_id_state.clone())),
                ],
            )
        })
        .and_then(|r| {
            r.map_err(|e| GraphDbError::InvalidInput(format!("Execute upsert_node failed: {e}")))?;
            Ok(())
        })
    }

    /// Insert a single edge using parameterized MATCH+CREATE with typed
    /// relationship pattern (`-[r:CALLS]->`).
    ///
    /// Since Cypher does not allow parameterized rel types, we use a separate
    /// prepared statement per `EdgeType`. The rel type name is a compile-time
    /// constant, not user input, so this is safe.
    ///
    /// Both `conn.prepare` and `conn.execute` cross into the lbug C++ layer
    /// and are wrapped with [`lbug_call`] for Rust-panic containment (SEC-03).
    fn insert_edge_with_conn(&self, conn: &Connection<'_>, edge: &GraphEdge) -> GraphDbResult<()> {
        let rel_type = edge.edge_type.as_str();
        // Build the Cypher string with the rel type literal (safe: comes from
        // EdgeType::as_str(), not user input).
        let cypher = format!(
            "MATCH (a:GraphNode {{node_id: $src_id}}), (b:GraphNode {{node_id: $dst_id}}) \
             CREATE (a)-[:{rel_type} {{weight: $weight, source_file: $source_file, \
             edge_id: $edge_id, tenant_id: $tenant_id, metadata_json: $metadata_json}}]->(b)"
        );

        // lbug FFI boundary: conn.prepare calls into C++ kuzu statement compilation.
        let mut stmt = lbug_call(|| conn.prepare(&cypher)).and_then(|r| {
            r.map_err(|e| GraphDbError::InvalidInput(format!("Prepare insert_edge failed: {e}")))
        })?;

        // lbug FFI boundary: conn.execute calls into C++ kuzu query execution.
        lbug_call(|| {
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
        })
        .and_then(|r| {
            r.map_err(|e| GraphDbError::InvalidInput(format!("Execute insert_edge failed: {e}")))?;
            Ok(())
        })
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
        // Parameterized tenant IN-list: `[$t0,$t1,...]` with one bound param per
        // tenant, so tenant ids are never interpolated into the query text
        // (no Cypher string-literal escaping pitfalls).
        let (tenant_list, tenant_names) = tenant_param_list(tenants.len());

        let mut out = Vec::new();

        // Outgoing: (current)-[r]->(n); Incoming: (current)<-[r]-(n).
        let patterns = [
            format!("(c:GraphNode {{node_id: $cid}})-[r:{rel_pattern}]->(n:GraphNode)"),
            format!("(c:GraphNode {{node_id: $cid}})<-[r:{rel_pattern}]-(n:GraphNode)"),
        ];
        for pattern in patterns {
            let cypher = format!(
                "MATCH {pattern} \
                 WHERE n.tenant_id IN {tenant_list} \
                 RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path, \
                        n.tenant_id, label(r), r.weight"
            );
            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare cross_boundary_neighbours: {e}"))
            })?;
            let mut params: Vec<(&str, Value)> = Vec::with_capacity(tenants.len() + 1);
            params.push(("cid", Value::String(current_id.to_string())));
            for (name, tenant) in tenant_names.iter().zip(tenants.iter()) {
                params.push((name.as_str(), Value::String(tenant.clone())));
            }
            let result = conn.execute(&mut stmt, params).map_err(|e| {
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

    // ---- find_path helpers ---------------------------------------------------

    /// Return `Ok(Some([source_node]))` for the self-path case, or `Ok(None)`
    /// when the source node does not exist in the tenant (matching SQLite: the
    /// recursive CTE base case joins `graph_nodes`, so a missing node yields
    /// no rows → None).
    fn find_path_self(
        &self,
        tenant_id: &str,
        node_id: &str,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        let conn = self.connect()?;
        let cypher = "MATCH (n:GraphNode {node_id: $id}) \
                      WHERE n.tenant_id = $tid \
                      RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
        let mut stmt = conn
            .prepare(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare find_path_self: {e}")))?;
        let result = conn
            .execute(
                &mut stmt,
                vec![
                    ("id", Value::String(node_id.to_string())),
                    ("tid", Value::String(tenant_id.to_string())),
                ],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute find_path_self: {e}")))?;

        for row in result {
            if row.len() < 4 {
                continue;
            }
            return Ok(Some(vec![TraversalNode {
                node_id: value_to_string(&row[0]),
                symbol_name: value_to_string(&row[1]),
                symbol_type: value_to_string(&row[2]),
                file_path: value_to_string(&row[3]),
                edge_type: String::new(),
                depth: 0,
                path: String::new(),
                tenant_id: tenant_id.to_string(),
                edge_confidence: 1.0,
            }]));
        }
        Ok(None)
    }

    /// Fetch each node in `path` (ordered vec of node_ids) from the graph and
    /// return them as `TraversalNode`s with `depth` = index in the path.
    ///
    /// Called once the BFS has confirmed the full path exists. Re-queries each
    /// node to populate all `TraversalNode` fields consistently with the rest of
    /// the LadybugDB backend.
    fn reconstruct_path(
        &self,
        tenant_id: &str,
        path: &[String],
        conn: &Connection<'_>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        let mut nodes = Vec::with_capacity(path.len());
        for (depth, nid) in path.iter().enumerate() {
            let cypher = "MATCH (n:GraphNode {node_id: $id}) \
                          WHERE n.tenant_id = $tid \
                          RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
            let mut stmt = conn.prepare(cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare reconstruct_path: {e}"))
            })?;
            let result = conn
                .execute(
                    &mut stmt,
                    vec![
                        ("id", Value::String(nid.clone())),
                        ("tid", Value::String(tenant_id.to_string())),
                    ],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!("Execute reconstruct_path: {e}"))
                })?;

            let mut found = false;
            for row in result {
                if row.len() < 4 {
                    continue;
                }
                nodes.push(TraversalNode {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    edge_type: String::new(),
                    depth: depth as u32,
                    path: String::new(),
                    tenant_id: tenant_id.to_string(),
                    edge_confidence: 1.0,
                });
                found = true;
                break;
            }
            if !found {
                // A node in a valid path disappeared from the graph (concurrent
                // delete race). Treat as no path rather than returning a partial
                // result — matches SQLite's JOIN behaviour where a missing node
                // simply drops the row.
                debug!(
                    "find_path reconstruct: node {} vanished mid-path for tenant {}",
                    nid, tenant_id
                );
                return Ok(None);
            }
        }
        Ok(Some(nodes))
    }
}

/// Build a parameterized Cypher tenant IN-list. Returns the bracketed fragment
/// `[$t0,$t1,...]` and the matching parameter names (`t0`, `t1`, ...), so tenant
/// ids are bound as parameters rather than interpolated into the query text.
/// This avoids Cypher string-literal escaping pitfalls (e.g. a tenant id
/// containing a quote or backslash). Callers bind `Value::String(tenant)` under
/// each returned name in order.
fn tenant_param_list(count: usize) -> (String, Vec<String>) {
    let names: Vec<String> = (0..count).map(|i| format!("t{i}")).collect();
    let fragment = format!(
        "[{}]",
        names
            .iter()
            .map(|n| format!("${n}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    (fragment, names)
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

/// Extract an f64 from a lbug Value (edge weight); defaults to 1.0.
pub(super) fn value_to_f64(val: &Value) -> f64 {
    match val {
        Value::Double(n) => *n,
        Value::Float(n) => *n as f64,
        Value::Int64(n) => *n as f64,
        _ => 1.0,
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

// ---- Panic guard for the lbug FFI boundary -----------------------------------

/// Invoke a synchronous lbug binding call `f`, catching any Rust panics that
/// originate in the C++/lbug layer and converting them to `GraphDbError`.
///
/// # SEC-03 — scope and limitations
///
/// This guard catches **Rust panics** that unwind through the FFI binding code.
/// It CANNOT catch:
/// - C++ exceptions thrown inside `lbug` (UB if they cross the FFI boundary)
/// - `abort()`, `std::terminate()`, or C++ `std::unexpected()`
/// - Out-of-memory (OOM) situations that call `abort()`
/// - OS signals: SIGSEGV, SIGABRT, SIGBUS, SIGILL, etc.
///
/// True fault isolation against C++-level failures requires process isolation
/// (see DEF-7 in the architecture document). This function is **best-effort
/// containment** only — it prevents a Rust `panic!` in the binding glue layer
/// from unwinding through the async tokio runtime and crashing the daemon.
///
/// # Usage
///
/// Wrap the direct synchronous call into an lbug type (e.g., `Connection::new`,
/// `conn.prepare`, `conn.execute`, `conn.query`) — NOT the whole async fn:
///
/// ```ignore
/// let conn = lbug_call(|| Connection::new(&self.db))
///     .map_err(|e| GraphDbError::Migration(format!("Connection failed: {e}")))?;
/// ```
fn lbug_call<F, R>(f: F) -> Result<R, GraphDbError>
where
    F: FnOnce() -> R,
{
    // AssertUnwindSafe is required because lbug types do not implement
    // UnwindSafe (the C++ internals are opaque). We accept the theoretical
    // risk of leaving lbug state in an inconsistent condition after a panic —
    // the caller discards the connection/statement after any error, so the
    // inconsistent object is dropped rather than reused.
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            // Extract a human-readable panic message from the Any payload.
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            warn!("ladybug_panic_trapped: {}", msg);
            Err(GraphDbError::InternalError(
                "Rust panic in lbug binding".to_string(),
            ))
        }
    }
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
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        _branch: Option<&str>,
        // NOTE: branch scoping is intentionally ignored at the LadybugDB layer,
        // matching the convention established in `query_related` (_branch param).
        // SQLite uses branch-aware edge filters; the cross-backend branch-parity
        // gap is formally asserted in the conformance suite (task 8).
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        // Self-path: source == target — return the source node at depth 0,
        // matching SQLite's BFS base-case behaviour.
        if source_id == target_id {
            return self.find_path_self(tenant_id, source_id);
        }

        // Rel-type pattern for the single-hop neighbour query.
        // Rel type names come from EdgeType::as_str() (compile-time literals),
        // so string interpolation is injection-safe.
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // Rust-side BFS replicating SQLite's `WITH RECURSIVE bfs` logic:
        //   - Each candidate is a full path (Vec<String> of node_ids) rather
        //     than just a frontier node, so the no-revisit check is O(path len)
        //     and the reconstruction step is trivial.
        //   - We process complete hop levels in ascending order, so the first
        //     time target_id is reached it is guaranteed to be a minimum-hop path.
        //   - Tie-breaking among equal-depth paths is implementation-defined
        //     (matches SQLite, which also gives no tie-break guarantee via LIMIT 1
        //     on the CTE result); tests use unique-shortest-path fixtures only.

        // frontier: list of in-progress paths, each ending at the current node.
        let mut frontier: Vec<Vec<String>> = vec![vec![source_id.to_string()]];

        let conn = self.connect()?;

        for _hop in 0..max_depth {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier: Vec<Vec<String>> = Vec::new();

            for path in &frontier {
                // A frontier path is never empty by construction (every entry
                // starts with `source_id` and only grows). Handle the empty
                // case gracefully rather than panicking: a panic here would
                // unwind the tokio runtime on a fallible async path (CR-020).
                let Some(current_id) = path.last() else {
                    continue;
                };

                // Single-hop outgoing neighbour query for this node.
                let cypher = format!(
                    "MATCH (a:GraphNode {{node_id: $id}})\
                     -[:{rel_pattern}*1..1]->(b:GraphNode) \
                     WHERE b.tenant_id = $tid \
                     RETURN b.node_id, b.symbol_name, b.symbol_type, b.file_path"
                );
                let mut stmt = conn.prepare(&cypher).map_err(|e| {
                    GraphDbError::InvalidInput(format!("Prepare find_path neighbour: {e}"))
                })?;
                let result = conn
                    .execute(
                        &mut stmt,
                        vec![
                            ("id", Value::String(current_id.clone())),
                            ("tid", Value::String(tenant_id.to_string())),
                        ],
                    )
                    .map_err(|e| {
                        GraphDbError::InvalidInput(format!("Execute find_path neighbour: {e}"))
                    })?;

                for row in result {
                    if row.len() < 4 {
                        continue;
                    }
                    let neighbour_id = value_to_string(&row[0]);

                    // No-revisit within this path (mirrors SQLite's INSTR check).
                    if path.contains(&neighbour_id) {
                        continue;
                    }

                    let mut new_path = path.clone();
                    new_path.push(neighbour_id.clone());

                    if neighbour_id == target_id {
                        // Found the target — reconstruct TraversalNode vec and return.
                        // The final node's fields are already in `row`; all prior
                        // nodes in the path must be fetched from the graph.
                        return self.reconstruct_path(tenant_id, &new_path, &conn);
                    }

                    next_frontier.push(new_path);
                }
            }

            frontier = next_frontier;
        }

        // Exhausted all paths within max_depth without reaching target.
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

    async fn graph_tenants(&self) -> GraphDbResult<Vec<String>> {
        let conn = self.connect()?;
        let mut stmt = conn
            .prepare("MATCH (n:GraphNode) RETURN DISTINCT n.tenant_id")
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare graph_tenants: {e}")))?;
        let result = conn
            .execute(&mut stmt, vec![])
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute graph_tenants: {e}")))?;
        let mut out = Vec::new();
        for row in result {
            if let Some(v) = row.first() {
                let t = value_to_string(v);
                if !t.is_empty() {
                    out.push(t);
                }
            }
        }
        Ok(out)
    }

    async fn resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        use std::collections::HashMap;

        // Only structural code rel types carry tree-sitter name-only stubs.
        const CODE_REL_TYPES: &[&str] = &[
            "CALLS",
            "CONTAINS",
            "IMPORTS",
            "USES_TYPE",
            "EXTENDS",
            "IMPLEMENTS",
        ];

        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Real candidate nodes (resolved file_path, not file-typed), indexed by
        // symbol_name -> [(node_id, file_path)].
        let mut by_name: HashMap<String, Vec<(String, String)>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare(
                    "MATCH (n:GraphNode) \
                     WHERE n.tenant_id = $tid AND n.file_path <> '' AND n.symbol_type <> 'file' \
                     RETURN n.node_id, n.symbol_name, n.file_path",
                )
                .map_err(|e| GraphDbError::InvalidInput(format!("Prepare stub candidates: {e}")))?;
            let result = conn
                .execute(
                    &mut stmt,
                    vec![("tid", Value::String(tenant_id.to_string()))],
                )
                .map_err(|e| GraphDbError::InvalidInput(format!("Execute stub candidates: {e}")))?;
            for row in result {
                if row.len() < 3 {
                    continue;
                }
                let nid = value_to_string(&row[0]);
                let name = value_to_string(&row[1]);
                let fp = value_to_string(&row[2]);
                by_name.entry(name).or_default().push((nid, fp));
            }
        }

        let mut repointed: u64 = 0;
        for rel_type in CODE_REL_TYPES {
            let Some(edge_type) = EdgeType::from_str(rel_type) else {
                continue;
            };

            // Dangling edges of this type: target is a stub (empty file_path).
            // Collect rows first so the connection is free for the follow-up
            // delete/create mutations.
            let find = format!(
                "MATCH (s:GraphNode)-[e:{rel_type}]->(t:GraphNode) \
                 WHERE e.tenant_id = $tid AND t.tenant_id = $tid \
                       AND (t.file_path = '' OR t.file_path IS NULL) \
                 RETURN s.node_id, e.edge_id, e.source_file, e.weight, e.metadata_json, \
                        t.symbol_name"
            );
            let rows: Vec<(String, String, String, f64, String, String)> = {
                let mut stmt = conn.prepare(&find).map_err(|e| {
                    GraphDbError::InvalidInput(format!("Prepare stub dangling ({rel_type}): {e}"))
                })?;
                let result = conn
                    .execute(
                        &mut stmt,
                        vec![("tid", Value::String(tenant_id.to_string()))],
                    )
                    .map_err(|e| {
                        GraphDbError::InvalidInput(format!(
                            "Execute stub dangling ({rel_type}): {e}"
                        ))
                    })?;
                result
                    .into_iter()
                    .filter_map(|row| {
                        if row.len() < 6 {
                            return None;
                        }
                        Some((
                            value_to_string(&row[0]),
                            value_to_string(&row[1]),
                            value_to_string(&row[2]),
                            value_to_f64(&row[3]),
                            value_to_string(&row[4]),
                            value_to_string(&row[5]),
                        ))
                    })
                    .collect()
            };

            for (source_node_id, old_edge_id, source_file, weight, metadata_json, target_name) in
                rows
            {
                let Some(candidates) = by_name.get(&target_name) else {
                    continue; // external/stdlib — no project node with this name.
                };
                // Prefer a definition in the caller's own file; else require a
                // unique tenant-wide match. Ambiguous names are skipped.
                let chosen: Option<&String> = candidates
                    .iter()
                    .find(|(_, fp)| *fp == source_file)
                    .map(|(nid, _)| nid)
                    .or_else(|| {
                        if candidates.len() == 1 {
                            Some(&candidates[0].0)
                        } else {
                            None
                        }
                    });
                let Some(new_target) = chosen else {
                    continue;
                };
                if &source_node_id == new_target {
                    continue; // skip self-loops
                }
                let new_edge_id = compute_edge_id(&source_node_id, new_target, edge_type);

                // Repoint. Kuzu has no multi-statement transaction here, so CREATE
                // the repointed rel FIRST, then DELETE the old dangling one: if the
                // CREATE fails we propagate the error with the old edge still
                // intact (the next sweep retries) rather than losing connectivity.
                // The `branch` scope is not carried — the LadybugDB rel tables have
                // no branch column (this backend does not track edge branches).
                let ins = format!(
                    "MATCH (a:GraphNode {{node_id: $src}}), (b:GraphNode {{node_id: $dst}}) \
                     CREATE (a)-[:{rel_type} {{weight: $weight, source_file: $sf, \
                     edge_id: $neid, tenant_id: $tid, metadata_json: $md}}]->(b)"
                );
                let mut istmt = conn.prepare(&ins).map_err(|e| {
                    GraphDbError::InvalidInput(format!("Prepare stub create ({rel_type}): {e}"))
                })?;
                conn.execute(
                    &mut istmt,
                    vec![
                        ("src", Value::String(source_node_id.clone())),
                        ("dst", Value::String(new_target.clone())),
                        ("weight", Value::Double(weight)),
                        ("sf", Value::String(source_file.clone())),
                        ("neid", Value::String(new_edge_id)),
                        ("tid", Value::String(tenant_id.to_string())),
                        ("md", Value::String(metadata_json.clone())),
                    ],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!("Execute stub create ({rel_type}): {e}"))
                })?;

                let del = format!(
                    "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                     WHERE r.edge_id = $eid AND r.tenant_id = $tid DELETE r"
                );
                let mut dstmt = conn.prepare(&del).map_err(|e| {
                    GraphDbError::InvalidInput(format!("Prepare stub delete ({rel_type}): {e}"))
                })?;
                conn.execute(
                    &mut dstmt,
                    vec![
                        ("eid", Value::String(old_edge_id.clone())),
                        ("tid", Value::String(tenant_id.to_string())),
                    ],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!("Execute stub delete ({rel_type}): {e}"))
                })?;

                repointed += 1;
            }
        }

        // Prune stub nodes (empty file_path) that are now edgeless. Mirrors
        // `prune_orphans` but scoped to stubs; tolerated if the EXISTS subquery
        // is unsupported by the LadybugDB version.
        let all_rels = ALL_REL_TYPES.join("|");
        let prune = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND (n.file_path = '' OR n.file_path IS NULL) \
             AND NOT EXISTS {{ MATCH (n)-[r:{all_rels}]-() WHERE r.tenant_id = $tid }} \
             DELETE n"
        );
        let prune_res = (|| -> GraphDbResult<()> {
            let mut stmt = conn
                .prepare(&prune)
                .map_err(|e| GraphDbError::InvalidInput(format!("Prepare stub prune: {e}")))?;
            conn.execute(
                &mut stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute stub prune: {e}")))?;
            Ok(())
        })();
        if let Err(e) = prune_res {
            debug!("stub-node prune subquery not supported, skipping: {}", e);
        }

        debug!(
            "Resolved {} stub edges for tenant {} (LadybugDB)",
            repointed, tenant_id
        );
        Ok(repointed)
    }

    async fn resolve_all_stub_edges(&self) -> GraphDbResult<u64> {
        let mut total: u64 = 0;
        // Each tenant takes the write lock independently via resolve_stub_edges,
        // so the lock is released between tenants.
        for tenant in self.graph_tenants().await? {
            total += self.resolve_stub_edges(&tenant).await?;
        }
        Ok(total)
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

    async fn fetch_node_metadata(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, NodeMetadata>> {
        let conn = self.connect()?;
        let cypher = "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
                      RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
        let mut stmt = conn
            .prepare(cypher)
            .map_err(|e| GraphDbError::InvalidInput(format!("Prepare fetch_node_metadata: {e}")))?;
        let result = conn
            .execute(
                &mut stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| GraphDbError::InvalidInput(format!("Execute fetch_node_metadata: {e}")))?;
        let mut map = std::collections::HashMap::new();
        for row in result {
            if row.len() >= 4 {
                map.insert(
                    value_to_string(&row[0]),
                    NodeMetadata {
                        symbol_name: value_to_string(&row[1]),
                        symbol_type: value_to_string(&row[2]),
                        file_path: value_to_string(&row[3]),
                    },
                );
            }
        }
        Ok(map)
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

        // Edge-first deletion (mirrors `delete_tenant`). LadybugDB enforces
        // node<->edge referential integrity, so a narrative node that is the
        // endpoint of any edge (e.g. an EXPLAINS edge to the code it documents)
        // cannot be deleted while that edge exists. Delete every incident edge
        // across all rel types -- in both directions, since a narrative node
        // may be the source (EXPLAINS) or the target (REFERENCES_DOC) -- before
        // removing the nodes themselves.
        let node_predicate = format!(
            "m.tenant_id = $tid AND m.file_path = $fp \
             AND m.symbol_type IN [{type_list}]"
        );
        for rel_type in ALL_REL_TYPES {
            // Outgoing (narrative -> other) and incoming (other -> narrative).
            let out_cypher = format!(
                "MATCH (m:GraphNode)-[r:{rel_type}]->(:GraphNode) \
                 WHERE {node_predicate} DELETE r"
            );
            let in_cypher = format!(
                "MATCH (:GraphNode)-[r:{rel_type}]->(m:GraphNode) \
                 WHERE {node_predicate} DELETE r"
            );
            for cypher in [out_cypher, in_cypher] {
                let mut stmt = conn.prepare(&cypher).map_err(|e| {
                    GraphDbError::InvalidInput(format!(
                        "Prepare delete_narrative_nodes_by_file edges: {e}"
                    ))
                })?;
                conn.execute(
                    &mut stmt,
                    vec![
                        ("tid", Value::String(tenant_id.to_string())),
                        ("fp", Value::String(file_path.to_string())),
                    ],
                )
                .map_err(|e| {
                    GraphDbError::InvalidInput(format!(
                        "Execute delete_narrative_nodes_by_file edges: {e}"
                    ))
                })?;
            }
        }

        // Now the (edge-free) narrative nodes can be removed.
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

        // Seed-ownership guard (parity with SQLite): only traverse when the
        // source node belongs to the relaxation set (source_tenant ∪
        // __global__ ∪ library_tenants); otherwise a foreign seed could reach
        // __global__ / library nodes and bypass tenant scoping. Concept and
        // library seeds remain valid. The per-hop query guards reached nodes.
        {
            let (tenant_list, tenant_names) = tenant_param_list(tenants.len());
            let cypher = format!(
                "MATCH (n:GraphNode {{node_id: $id}}) \
                 WHERE n.tenant_id IN {tenant_list} RETURN n.node_id"
            );
            let mut stmt = conn.prepare(&cypher).map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare cross_boundary seed guard: {e}"))
            })?;
            let mut params: Vec<(&str, Value)> = Vec::with_capacity(tenants.len() + 1);
            params.push(("id", Value::String(source_node_id.to_string())));
            for (name, tenant) in tenant_names.iter().zip(tenants.iter()) {
                params.push((name.as_str(), Value::String(tenant.clone())));
            }
            let result = conn.execute(&mut stmt, params).map_err(|e| {
                GraphDbError::InvalidInput(format!("Execute cross_boundary seed guard: {e}"))
            })?;
            if result.into_iter().next().is_none() {
                return Ok(Vec::new());
            }
        }

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

    async fn export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        use std::collections::HashMap;

        let conn = self.connect()?;

        // 1. Load all nodes for the tenant, sorted deterministically (DOM-01).
        let mut node_stmt = conn
            .prepare(
                "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
                 RETURN n.node_id ORDER BY n.node_id",
            )
            .map_err(|e| {
                GraphDbError::InvalidInput(format!("Prepare export_adjacency nodes: {e}"))
            })?;

        let node_result = conn
            .execute(
                &mut node_stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| {
                GraphDbError::InvalidInput(format!("Execute export_adjacency nodes: {e}"))
            })?;

        let node_ids: Vec<String> = node_result
            .into_iter()
            .filter_map(|row| row.into_iter().next().map(|v| value_to_string(&v)))
            .collect();

        // Build a node_id→index map for O(1) edge lookups.
        let index_map: HashMap<String, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        // 2. Build the rel-type pattern.
        //
        //    Rel type names come from EdgeType::as_str() (compile-time constants),
        //    so string interpolation into the rel-pattern position is injection-safe
        //    (same convention as query_related, find_path, etc. in this backend).
        //
        //    NOTE: LadybugDB rel tables carry a `weight DOUBLE` property (confirmed
        //    in init_schema DDL). Weight is returned and stored directly; no default
        //    substitution is needed. This ensures cross-backend weight equivalence
        //    with the SQLite backend (relevant for the conformance suite, task 8).
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // 3. Query edges: source node_id, target node_id, weight.
        let edge_cypher = format!(
            "MATCH (a:GraphNode)-[r:{rel_pattern}]->(b:GraphNode) \
             WHERE a.tenant_id = $tid \
             RETURN a.node_id, b.node_id, r.weight \
             ORDER BY a.node_id, b.node_id"
        );

        let mut edge_stmt = conn.prepare(&edge_cypher).map_err(|e| {
            GraphDbError::InvalidInput(format!("Prepare export_adjacency edges: {e}"))
        })?;

        let edge_result = conn
            .execute(
                &mut edge_stmt,
                vec![("tid", Value::String(tenant_id.to_string()))],
            )
            .map_err(|e| {
                GraphDbError::InvalidInput(format!("Execute export_adjacency edges: {e}"))
            })?;

        // 4. Convert to indexed edges; skip orphan endpoints.
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for row in edge_result {
            if row.len() < 3 {
                continue;
            }
            let src = value_to_string(&row[0]);
            let tgt = value_to_string(&row[1]);
            let weight = value_to_f64(&row[2]);

            let (Some(&si), Some(&ti)) = (index_map.get(&src), index_map.get(&tgt)) else {
                // Orphan edge: at least one endpoint absent from the node list.
                continue;
            };
            edges.push((si, ti, weight));
        }

        Ok(AdjacencyExport { node_ids, edges })
    }

    /// Export all nodes for a tenant, ordered by node_id (DATA-05 content diff).
    ///
    /// Delegates to the migrator's Cypher-based exporter so that
    /// [`crate::graph::migrator::diff_graph_contents`] can compare this backend
    /// against SQLite through the trait. The export restores the empty-string
    /// point-id sentinel back to `None` for lossless round-tripping.
    async fn export_nodes_for_tenant(&self, tenant_id: &str) -> GraphDbResult<Vec<GraphNode>> {
        crate::graph::migrator::export_nodes_ladybug(self, Some(tenant_id))
    }

    /// Export all edges for a tenant, ordered by edge_id (DATA-05 content diff).
    async fn export_edges_for_tenant(&self, tenant_id: &str) -> GraphDbResult<Vec<GraphEdge>> {
        crate::graph::migrator::export_edges_ladybug(self, Some(tenant_id))
    }
}

// ---- Unit tests for the lbug panic guard (A0.3) ------------------------------
//
// These tests verify `lbug_call` behaviour in isolation, without a real lbug
// database. They use a fault-injecting closure to trigger a Rust panic and
// assert that:
//   1. `lbug_call` returns `Err(GraphDbError::InternalError(...))`.
//   2. The error message is "Rust panic in lbug binding".
//   3. A warning log line containing "ladybug_panic_trapped" is emitted.
//   4. The current thread (standing in for the tokio runtime) does NOT crash.
//
// The tests do NOT require a real lbug::Database — the closure is a plain Rust
// closure that panics, exercising catch_unwind directly.

#[cfg(test)]
mod panic_guard_tests {
    use tracing_test::traced_test;

    use super::{lbug_call, GraphDbError};

    /// A successful closure must pass its return value through unchanged.
    #[test]
    fn lbug_call_ok_passes_value_through() {
        let result = lbug_call(|| 42u32);
        assert_eq!(result.unwrap(), 42u32);
    }

    /// A closure that panics with a string message must:
    ///   - be caught (thread does not crash),
    ///   - return `Err(GraphDbError::InternalError("Rust panic in lbug binding"))`,
    ///   - emit a `warn!` log containing "ladybug_panic_trapped" and the message.
    #[test]
    #[traced_test]
    fn lbug_call_traps_str_panic_and_logs() {
        // Fault-injecting fake: simulates a Rust panic in the lbug binding layer.
        let result = lbug_call(|| -> u32 { panic!("simulated lbug binding panic") });

        // The thread (runtime) must not have crashed — we reach this assertion.
        match result {
            Err(GraphDbError::InternalError(msg)) => {
                assert_eq!(msg, "Rust panic in lbug binding");
            }
            other => panic!("expected InternalError, got {other:?}"),
        }

        // Verify the warning log was emitted with the expected prefix and message.
        assert!(
            logs_contain("ladybug_panic_trapped"),
            "expected 'ladybug_panic_trapped' in logs"
        );
        assert!(
            logs_contain("simulated lbug binding panic"),
            "expected panic message in logs"
        );
    }

    /// A closure that panics with an owned String payload must also be caught.
    #[test]
    #[traced_test]
    fn lbug_call_traps_string_panic_and_logs() {
        let result =
            lbug_call(|| -> u32 { panic!("{}", "owned string panic from lbug".to_string()) });

        match result {
            Err(GraphDbError::InternalError(msg)) => {
                assert_eq!(msg, "Rust panic in lbug binding");
            }
            other => panic!("expected InternalError, got {other:?}"),
        }

        assert!(
            logs_contain("ladybug_panic_trapped"),
            "expected 'ladybug_panic_trapped' in logs"
        );
        assert!(
            logs_contain("owned string panic from lbug"),
            "expected panic message in logs"
        );
    }

    /// A closure that panics with a non-string payload (box of u32) must still
    /// be caught and emit the generic "<non-string panic payload>" message.
    #[test]
    #[traced_test]
    fn lbug_call_traps_non_string_panic_and_logs() {
        use std::panic;

        // Suppress the default "panicked at …" stderr output for this test.
        let prev = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let result = lbug_call(|| -> u32 { panic::resume_unwind(Box::new(99u32)) });
        panic::set_hook(prev);

        match result {
            Err(GraphDbError::InternalError(msg)) => {
                assert_eq!(msg, "Rust panic in lbug binding");
            }
            other => panic!("expected InternalError, got {other:?}"),
        }

        assert!(
            logs_contain("ladybug_panic_trapped"),
            "expected 'ladybug_panic_trapped' in logs"
        );
        assert!(
            logs_contain("<non-string panic payload>"),
            "expected generic payload description in logs"
        );
    }
}
