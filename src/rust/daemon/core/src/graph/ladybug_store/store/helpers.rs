//! `ladybug_store/helpers.rs` — shared primitives for the LadybugDB backend.
//!
//! Contains everything that the four implementation files (`init`, `mutate`,
//! `query`) all depend on: the lbug FFI panic guard (`lbug_call`), Value
//! extraction helpers (`value_to_string` etc.), Cypher parameterisation
//! utilities (`tenant_param_list`, `distinct_frontier_tails`), and the small
//! private data structures (`PathNode`, `CrossBoundaryPath`,
//! `CrossBoundaryNeighbour`) used during graph traversal. Nothing in this file
//! performs I/O; it is pure helper infrastructure.

use lbug::Value;

use crate::graph::schema::GraphDbError;

use std::panic::AssertUnwindSafe;
use tracing::warn;

// ---- Constants ---------------------------------------------------------------

/// Every relationship type in the schema. Used for operations that must iterate
/// over all rel tables (delete-by-file, delete-tenant, stats, prune) so that
/// narrative and concept edges are not silently skipped.
pub(super) const ALL_REL_TYPES: &[&str] = &[
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
pub(super) const CODE_SYMBOL_TYPES: &[&str] = &[
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
pub(super) const NARRATIVE_FILE_NODE_TYPES: &[&str] = &[
    "document_section",
    "library_section",
    "code_comment",
    "docstring",
];

/// Upper bound on the number of in-progress BFS paths (the frontier) held at
/// once in `find_path` and `query_related`. A dense or highly connected graph
/// can grow the frontier exponentially per hop; without a cap a single query
/// could exhaust process memory (a local denial-of-service). When the frontier
/// would exceed this size we stop expanding and return the best result found so
/// far. This mirrors the intent of `apply_fan_out_caps` in `cross_boundary`,
/// which bounds fan-out for the cross-tenant traversal.
pub(crate) const MAX_FRONTIER_PATHS: usize = 10_000;

// ---- Path traversal data structures ------------------------------------------

/// One node on an in-progress `find_path` BFS path, carrying the identity
/// columns the neighbour query already returned for it. Capturing them here lets
/// [`reconstruct_path`](crate::graph::ladybug_store::query) build the result
/// without a second per-node query (CR-011).
#[derive(Clone)]
pub(super) struct PathNode {
    pub(super) node_id: String,
    pub(super) symbol_name: String,
    pub(super) symbol_type: String,
    pub(super) file_path: String,
}

/// An in-progress cross-boundary BFS path.
///
/// Carries two views of the same path so cycle detection and result formatting
/// stay independent:
/// - `visited` is the ordered set of node-ids already on the path. The acyclic
///   guard tests membership here with exact id comparison.
/// - `display` is the human-readable `"a -> b -> c"` string stored on the
///   resulting [`TraversalNode::path`] and parsed by `apply_fan_out_caps` for
///   concept attribution.
///
/// Splitting the two fixes CR-021: the old code derived the cycle check by
/// splitting `display` on " -> ", so a node_id containing that separator
/// corrupted the parse and could wrongly exclude (or admit) a node.
pub(super) struct CrossBoundaryPath {
    pub(super) visited: Vec<String>,
    pub(super) display: String,
}

impl CrossBoundaryPath {
    /// Start a path at the source node.
    pub(super) fn seed(source_node_id: &str) -> Self {
        Self {
            visited: vec![source_node_id.to_string()],
            display: source_node_id.to_string(),
        }
    }

    /// Whether `node_id` is already on this path (exact match, no string split).
    pub(super) fn visits(&self, node_id: &str) -> bool {
        self.visited.iter().any(|id| id == node_id)
    }

    /// Extend the path by one hop to `node_id`, returning the new path.
    pub(super) fn extend(&self, node_id: &str) -> Self {
        let mut visited = self.visited.clone();
        visited.push(node_id.to_string());
        Self {
            visited,
            display: format!("{} -> {}", self.display, node_id),
        }
    }
}

/// A node reached in one cross-boundary hop, with the reaching edge's type and
/// weight (used to compute per-hop confidence).
pub(super) struct CrossBoundaryNeighbour {
    pub(super) node_id: String,
    pub(super) symbol_name: String,
    pub(super) symbol_type: String,
    pub(super) file_path: String,
    pub(super) tenant_id: String,
    pub(super) edge_type: String,
    pub(super) weight: f64,
}

// ---- Cypher parameterisation utilities ----------------------------------------

/// Build a parameterized Cypher tenant IN-list. Returns the bracketed fragment
/// `[$t0,$t1,...]` and the matching parameter names (`t0`, `t1`, ...), so tenant
/// ids are bound as parameters rather than interpolated into the query text.
/// This avoids Cypher string-literal escaping pitfalls (e.g. a tenant id
/// containing a quote or backslash). Callers bind `Value::String(tenant)` under
/// each returned name in order.
pub(super) fn tenant_param_list(count: usize) -> (String, Vec<String>) {
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

/// Collect the distinct tail (current) node-ids across all frontier paths, so
/// the per-hop neighbour query fetches each source node only once even when
/// several in-progress paths currently end at the same node. Order is
/// deterministic (first-seen) and irrelevant: the result feeds an IN-list and a
/// lookup map.
pub(super) fn distinct_frontier_tails(frontier: &[Vec<PathNode>]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut ids = Vec::new();
    for path in frontier {
        if let Some(tail) = path.last() {
            if seen.insert(tail.node_id.clone()) {
                ids.push(tail.node_id.clone());
            }
        }
    }
    ids
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
pub(crate) fn escape_cypher(s: &str) -> String {
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
pub(super) fn lbug_call<F, R>(f: F) -> Result<R, GraphDbError>
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
