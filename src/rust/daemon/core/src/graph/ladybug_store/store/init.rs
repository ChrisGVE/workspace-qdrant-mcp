//! `ladybug_store/store/init.rs` — `LadybugGraphStore` struct, constructor, FFI
//! choke-point methods, and the two low-level write helpers.
//!
//! Owns the responsibilities that must precede any graph I/O: defining the
//! store struct (holding the `lbug::Database` and the write-serialisation
//! `Mutex`), opening or creating the database, running the idempotent schema
//! DDL on first access, and the two reusable write primitives
//! (`upsert_node_with_conn`, `insert_edge_with_conn`) that serialize individual
//! DML statements through the FFI boundary. Also exposes the three FFI helpers
//! (`connect`, `exec`, `run_prepared`) and the two public Cypher execution APIs
//! (`execute_cypher`, `execute_cypher_with_params`) that all other impl files
//! invoke through `self`. Reads from `helpers::lbug_call` for panic containment.

use std::path::Path;

use lbug::{Connection, Database, SystemConfig, Value};
use tokio::sync::Mutex;

use crate::graph::{
    schema::{GraphDbError, GraphDbResult},
    GraphEdge, GraphNode,
};

use super::super::config::LadybugConfig;
use super::helpers::lbug_call;

// ---- Store struct ------------------------------------------------------------

/// LadybugDB-backed graph store.
///
/// Uses an in-process LadybugDB instance for graph storage and Cypher queries.
/// The `Database` is stored inline; connections are created on demand since
/// `Connection<'a>` borrows `Database` and cannot be stored alongside it.
pub struct LadybugGraphStore {
    /// The LadybugDB database instance.
    pub(super) db: Database,
    /// Serialize write access (LadybugDB supports concurrent reads but
    /// single writer).
    pub(super) write_lock: Mutex<()>,
    pub(super) config: LadybugConfig,
    /// Cross-boundary fan-out caps and fusion settings (shared with SQLite).
    pub(super) graph_rag: crate::config::GraphRagConfig,
}

// `LadybugGraphStore` is automatically Send + Sync: every field is already
// Send + Sync. `lbug 0.14.1` declares `unsafe impl Send/Sync` for `Database`
// (database.rs:13-14) and `Connection` (connection.rs:74-75) — "synchronized on
// the C++ side" — so the inline `Database`, the `Mutex<()>`, `LadybugConfig`,
// and `GraphRagConfig` all carry the marker traits through auto-derivation. The
// previous manual `unsafe impl Send/Sync` here was redundant (CR-005); removing
// it lets the compiler prove thread-safety instead of asserting it by hand.

// ---- Constructor and public accessors ----------------------------------------

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

    /// Get the database path.
    pub fn db_path(&self) -> &Path {
        &self.config.db_path
    }

    /// Execute a raw Cypher query and return results as formatted strings.
    ///
    /// This is LadybugDB-specific (not part of GraphStore trait).
    pub fn execute_cypher(&self, cypher: &str) -> GraphDbResult<Vec<Vec<String>>> {
        let conn = self.connect()?;

        // lbug FFI boundary: both `conn.query` and draining the lazy result
        // iterator cross into C++ kuzu. Wrap both in one [`lbug_call`] so a Rust
        // panic in the binding is contained, never unwinding the tokio runtime
        // (SEC-03 / CR-006).
        let rows = lbug_call(|| -> GraphDbResult<Vec<Vec<Value>>> {
            let result = conn
                .query(cypher)
                .map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {e}")))?;
            Ok(result.collect())
        })
        .and_then(|inner| inner)?;

        Ok(rows
            .iter()
            .map(|row| row.iter().map(|v| format!("{v}")).collect())
            .collect())
    }

    /// Execute a parameterized Cypher query and return results as formatted
    /// strings, with every user value bound as a `$param` (never interpolated).
    ///
    /// LadybugDB-specific (not part of the `GraphStore` trait). Used by the
    /// migrator exporters to bind a tenant filter (`$tid`) safely instead of
    /// `format!`-interpolating it into the query text — closing the Cypher
    /// injection hole an attacker-controlled tenant id would otherwise open
    /// (CR-007). Runs through the [`run_prepared`](Self::run_prepared)
    /// choke-point, so it is also panic-guarded (SEC-03 / CR-006).
    pub fn execute_cypher_with_params(
        &self,
        cypher: &str,
        params: Vec<(&str, Value)>,
    ) -> GraphDbResult<Vec<Vec<String>>> {
        let conn = self.connect()?;
        let rows = self.run_prepared(&conn, cypher, params, "execute_cypher_with_params")?;
        Ok(rows
            .iter()
            .map(|row| row.iter().map(|v| format!("{v}")).collect())
            .collect())
    }
}

// ---- Low-level write primitives ---------------------------------------------
//
// `upsert_node_with_conn` and `insert_edge_with_conn` are thin DML helpers
// shared by the write-path methods in `mutate.rs`. They live here rather than
// in `mutate.rs` so that `mutate.rs` stays under the 500-line size limit while
// keeping the FFI surface (everything that calls `conn.prepare` / `conn.execute`
// directly) collocated with the other FFI choke-point helpers below.

impl LadybugGraphStore {
    /// Upsert a single node using parameterized MERGE+SET.
    ///
    /// MERGE matches on `node_id` (primary key); SET updates all properties.
    /// Per T34 findings, no ON CREATE/ON MATCH needed -- bare SET after MERGE
    /// handles both cases.
    ///
    /// Both `conn.prepare` and `conn.execute` cross into the lbug C++ layer
    /// and are wrapped with [`lbug_call`] for Rust-panic containment (SEC-03).
    pub(super) fn upsert_node_with_conn(
        &self,
        conn: &Connection<'_>,
        node: &GraphNode,
    ) -> GraphDbResult<()> {
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
    pub(super) fn insert_edge_with_conn(
        &self,
        conn: &Connection<'_>,
        edge: &GraphEdge,
    ) -> GraphDbResult<()> {
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
}

// ---- FFI choke-point helpers -------------------------------------------------
//
// These three methods are the ONLY entry-points into the lbug C++ layer.
// All other methods in init/mutate/query call exclusively through these helpers
// so the panic-containment invariant (SEC-03 / CR-006) is never accidentally
// bypassed.

impl LadybugGraphStore {
    /// Create a connection to the database.
    ///
    /// The `Connection::new` call crosses into the lbug C++ layer; it is
    /// wrapped with [`lbug_call`] so a Rust panic in the binding is caught and
    /// converted to [`GraphDbError::InternalError`] rather than unwinding the
    /// tokio runtime (SEC-03).
    pub(super) fn connect(&self) -> GraphDbResult<Connection<'_>> {
        // lbug FFI boundary: Connection::new calls into C++ kuzu internals.
        lbug_call(|| Connection::new(&self.db)).and_then(|result| {
            result.map_err(|e| GraphDbError::Migration(format!("Connection failed: {e}")))
        })
    }

    /// Execute a DDL/mutation query (no parameters), mapping lbug errors.
    ///
    /// The `conn.query` call crosses into the lbug C++ layer; wrapped with
    /// [`lbug_call`] for panic containment (SEC-03).
    pub(super) fn exec(&self, conn: &Connection<'_>, cypher: &str) -> GraphDbResult<()> {
        // lbug FFI boundary: conn.query calls into C++ kuzu query execution.
        // DDL produces no result rows to drain, so guarding the single
        // `conn.query` call fully covers the FFI surface here.
        lbug_call(|| conn.query(cypher)).and_then(|result| {
            result.map_err(|e| GraphDbError::InvalidInput(format!("Cypher failed: {e}")))?;
            Ok(())
        })
    }

    /// Run a parameterized Cypher statement and return its rows as owned values.
    ///
    /// This is the single FFI choke-point for every parameterized read/write in
    /// this backend. It wraps all three lbug operations that cross into the C++
    /// kuzu layer -- `conn.prepare`, `conn.execute`, and draining the lazy
    /// `QueryResult` iterator (whose `next()` itself calls into C++) -- inside
    /// one [`lbug_call`], so a Rust panic anywhere in the binding is contained
    /// and converted to [`GraphDbError`] rather than unwinding the tokio runtime
    /// (SEC-03 / CR-006). Callers receive a fully materialized `Vec<Vec<Value>>`
    /// and iterate it in pure Rust, never touching the FFI boundary themselves.
    ///
    /// # Invariant
    ///
    /// Every FFI call into lbug goes through a `lbug_call`-wrapped helper:
    /// [`connect`](Self::connect), [`exec`](Self::exec) (DDL), or this
    /// `run_prepared` (parameterized queries). No method may call
    /// `conn.prepare` / `conn.execute` / `conn.query` directly -- doing so
    /// reopens the unguarded-panic hole this helper exists to close.
    ///
    /// `op` is a short label (e.g. `"stats nodes"`) used only to build the
    /// prepare/execute error messages, preserving the previous per-site wording.
    pub(super) fn run_prepared(
        &self,
        conn: &Connection<'_>,
        cypher: &str,
        params: Vec<(&str, Value)>,
        op: &str,
    ) -> GraphDbResult<Vec<Vec<Value>>> {
        // lbug FFI boundary: prepare, execute, AND row iteration all cross into
        // C++ kuzu. All three are wrapped in a single panic guard.
        lbug_call(|| -> GraphDbResult<Vec<Vec<Value>>> {
            let mut stmt = conn
                .prepare(cypher)
                .map_err(|e| GraphDbError::InvalidInput(format!("Prepare {op} failed: {e}")))?;
            let result = conn
                .execute(&mut stmt, params)
                .map_err(|e| GraphDbError::InvalidInput(format!("Execute {op} failed: {e}")))?;
            Ok(result.collect())
        })
        // Outer Result: panic containment; inner Result: lbug query error.
        .and_then(|inner| inner)
    }
}

// ---- Schema DDL --------------------------------------------------------------

impl LadybugGraphStore {
    /// Initialize the graph schema (node tables, rel tables).
    ///
    /// Uses `IF NOT EXISTS` so this is idempotent across restarts.
    pub(super) fn init_schema(&self) -> GraphDbResult<()> {
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
}
