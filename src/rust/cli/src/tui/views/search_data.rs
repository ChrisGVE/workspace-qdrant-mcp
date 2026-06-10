//! Data types and background fetcher for the TUI Search page.
//!
//! Located at: `src/rust/cli/src/tui/views/search_data.rs`
//!
//! # Architecture context
//! The TUI render/event loop is synchronous. Search requires async gRPC calls
//! to the daemon's TextSearchService (grep + exact-text modes) and GraphService
//! (graph related-nodes mode). This module owns the background thread (tokio
//! single-threaded runtime) that makes those calls, and exposes a shared
//! `Arc<Mutex<SearchSnapshot>>` the sync view reads on every draw.
//!
//! Pattern mirrors `graph_data.rs` exactly: FetchRequest channel + background
//! thread + shared snapshot.
//!
//! Neighbors: `search_page.rs` (state + navigation), `search_render.rs`
//! (primary render), `search_render_detail.rs` (preview popup + prompt),
//! `graph_data.rs` (established pattern this mirrors).

use std::sync::{Arc, Mutex};
use std::thread;

use crate::data::db::connect_readonly;
use crate::data::tenants::load_tenants;
use crate::grpc::client::workspace_daemon::{QueryRelatedRequest, TextSearchRequest};

// ─── Per-mode result types ────────────────────────────────────────────────────

/// A single grep/exact-text search match (one line in a file).
#[derive(Debug, Clone)]
pub struct SearchMatch {
    /// File path relative to the project root.
    pub file_path: String,
    /// 1-based line number.
    pub line_number: i32,
    /// Content of the matching line (trimmed).
    pub content: String,
    /// Lines of context before the match (empty when context_lines == 0).
    pub context_before: Vec<String>,
    /// Lines of context after the match (empty when context_lines == 0).
    pub context_after: Vec<String>,
}

/// A single related-node result from the graph mode.
#[derive(Debug, Clone)]
pub struct GraphRelatedNode {
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    /// Edge type (e.g. "CALLS", "IMPORTS").
    pub edge_type: String,
    /// Hops from the source node.
    pub depth: u32,
}

/// Registered tenant entry used by the tenant cycler.
#[derive(Debug, Clone)]
pub struct TenantRef {
    pub tenant_id: String,
    /// Display name (path basename).
    pub name: String,
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────

/// The full data snapshot written by the background thread and read on every draw.
#[derive(Debug, Clone, Default)]
pub struct SearchSnapshot {
    /// Results for Grep or Exact mode.
    pub matches: Vec<SearchMatch>,
    /// Results for Graph mode.
    pub graph_nodes: Vec<GraphRelatedNode>,
    /// Total count reported by the daemon (may exceed `matches.len()` when truncated).
    pub total: i32,
    /// Whether the result set was truncated by the daemon's `max_results` cap.
    pub truncated: bool,
    /// Wall-clock milliseconds the last RPC took.
    pub query_time_ms: i64,
    /// Human-readable error from the last RPC attempt (cleared on next success).
    pub last_error: Option<String>,
    /// Whether a fetch is currently in progress.
    pub loading: bool,
}

// ─── Fetch request ────────────────────────────────────────────────────────────

/// A request sent from the view to the background thread.
#[derive(Debug, Clone)]
pub enum FetchRequest {
    /// Run a grep (literal or regex) or exact-text search via TextSearchService.
    TextSearch {
        tenant_id: String,
        pattern: String,
        /// When true the pattern is treated as a regex; when false, literal.
        regex: bool,
        case_sensitive: bool,
    },
    /// Run a graph related-nodes query via GraphService::QueryRelated.
    GraphQuery {
        tenant_id: String,
        /// Node ID (symbol name) to traverse from.
        node_id: String,
    },
}

// ─── Background fetcher ───────────────────────────────────────────────────────

/// Spawn the background search fetcher. Returns:
/// - A shared snapshot (view reads this every draw).
/// - A channel sender (view sends `FetchRequest`s to trigger fetches).
pub fn spawn_search_fetcher() -> (
    Arc<Mutex<SearchSnapshot>>,
    std::sync::mpsc::Sender<FetchRequest>,
) {
    let shared = Arc::new(Mutex::new(SearchSnapshot::default()));
    let shared_clone = Arc::clone(&shared);
    let (tx, rx) = std::sync::mpsc::channel::<FetchRequest>();

    thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(_) => return,
        };

        rt.block_on(async move {
            while let Ok(req) = rx.recv() {
                if let Ok(mut s) = shared_clone.lock() {
                    s.loading = true;
                    s.last_error = None;
                }

                match req {
                    FetchRequest::TextSearch {
                        tenant_id,
                        pattern,
                        regex,
                        case_sensitive,
                    } => {
                        let result =
                            fetch_text_search(&tenant_id, &pattern, regex, case_sensitive).await;
                        if let Ok(mut s) = shared_clone.lock() {
                            match result {
                                Ok(snap) => {
                                    s.matches = snap.matches;
                                    s.graph_nodes = vec![];
                                    s.total = snap.total;
                                    s.truncated = snap.truncated;
                                    s.query_time_ms = snap.query_time_ms;
                                    s.last_error = None;
                                }
                                Err(e) => {
                                    s.last_error = Some(e.to_string());
                                }
                            }
                            s.loading = false;
                        }
                    }
                    FetchRequest::GraphQuery { tenant_id, node_id } => {
                        let result = fetch_graph_related(&tenant_id, &node_id).await;
                        if let Ok(mut s) = shared_clone.lock() {
                            match result {
                                Ok(snap) => {
                                    s.matches = vec![];
                                    s.graph_nodes = snap.graph_nodes;
                                    s.total = snap.total;
                                    s.truncated = false;
                                    s.query_time_ms = snap.query_time_ms;
                                    s.last_error = None;
                                }
                                Err(e) => {
                                    s.last_error = Some(e.to_string());
                                }
                            }
                            s.loading = false;
                        }
                    }
                }
            }
        });
    });

    (shared, tx)
}

// ─── Async gRPC helpers ───────────────────────────────────────────────────────

/// Run a TextSearchService::Search RPC and return a partial SearchSnapshot.
async fn fetch_text_search(
    tenant_id: &str,
    pattern: &str,
    regex: bool,
    case_sensitive: bool,
) -> anyhow::Result<SearchSnapshot> {
    let mut client = crate::grpc::connect_default().await?;

    let resp = client
        .text_search_client()
        .search(TextSearchRequest {
            pattern: pattern.to_string(),
            regex,
            case_sensitive,
            tenant_id: Some(tenant_id.to_string()),
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 2,
            max_results: 100,
        })
        .await?
        .into_inner();

    let matches = resp
        .matches
        .into_iter()
        .map(|m| SearchMatch {
            file_path: m.file_path,
            line_number: m.line_number,
            content: m.content,
            context_before: m.context_before,
            context_after: m.context_after,
        })
        .collect();

    Ok(SearchSnapshot {
        matches,
        graph_nodes: vec![],
        total: resp.total_matches,
        truncated: resp.truncated,
        query_time_ms: resp.query_time_ms,
        last_error: None,
        loading: false,
    })
}

/// Run a GraphService::QueryRelated RPC and return a partial SearchSnapshot.
async fn fetch_graph_related(tenant_id: &str, node_id: &str) -> anyhow::Result<SearchSnapshot> {
    let mut client = crate::grpc::connect_default().await?;

    let resp = client
        .graph()
        .query_related(QueryRelatedRequest {
            tenant_id: tenant_id.to_string(),
            node_id: node_id.to_string(),
            max_hops: 2,
            edge_types: vec![],
            branch: None,
        })
        .await?
        .into_inner();

    let graph_nodes = resp
        .nodes
        .into_iter()
        .map(|n| GraphRelatedNode {
            symbol_name: n.symbol_name,
            symbol_type: n.symbol_type,
            file_path: n.file_path,
            edge_type: n.edge_type,
            depth: n.depth,
        })
        .collect();

    Ok(SearchSnapshot {
        matches: vec![],
        graph_nodes,
        total: resp.total as i32,
        truncated: false,
        query_time_ms: resp.query_time_ms,
        last_error: None,
        loading: false,
    })
}

// ─── SQLite tenant list ───────────────────────────────────────────────────────

/// Load registered tenants from the local SQLite state database.
///
/// Returns an empty list (rather than an error) when the database is
/// unavailable — the view shows "no tenants" gracefully in that case.
pub fn load_tenant_list() -> Vec<TenantRef> {
    let Ok(conn) = connect_readonly() else {
        return vec![];
    };
    let Ok(entries) = load_tenants(&conn) else {
        return vec![];
    };
    // Deduplicate by tenant_id (load_tenants returns one row per watch folder).
    let mut seen = std::collections::HashSet::new();
    entries
        .into_iter()
        .filter(|e| seen.insert(e.tenant_id.clone()))
        .map(|e| TenantRef {
            name: e.name,
            tenant_id: e.tenant_id,
        })
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_match_fields_accessible() {
        let m = SearchMatch {
            file_path: "src/lib.rs".into(),
            line_number: 42,
            content: "fn main() {}".into(),
            context_before: vec!["// comment".into()],
            context_after: vec!["}".into()],
        };
        assert_eq!(m.file_path, "src/lib.rs");
        assert_eq!(m.line_number, 42);
        assert_eq!(m.context_before.len(), 1);
        assert_eq!(m.context_after.len(), 1);
    }

    #[test]
    fn graph_related_node_fields_accessible() {
        let n = GraphRelatedNode {
            symbol_name: "parse".into(),
            symbol_type: "fn".into(),
            file_path: "src/parser.rs".into(),
            edge_type: "CALLS".into(),
            depth: 1,
        };
        assert_eq!(n.edge_type, "CALLS");
        assert_eq!(n.depth, 1);
    }

    #[test]
    fn snapshot_default_is_empty() {
        let s = SearchSnapshot::default();
        assert!(s.matches.is_empty());
        assert!(s.graph_nodes.is_empty());
        assert_eq!(s.total, 0);
        assert!(!s.truncated);
        assert_eq!(s.query_time_ms, 0);
        assert!(s.last_error.is_none());
        assert!(!s.loading);
    }

    #[test]
    fn tenant_ref_fields_accessible() {
        let t = TenantRef {
            tenant_id: "abc123".into(),
            name: "my-project".into(),
        };
        assert_eq!(t.tenant_id, "abc123");
        assert_eq!(t.name, "my-project");
    }

    #[test]
    fn fetch_request_text_search_variant() {
        let req = FetchRequest::TextSearch {
            tenant_id: "t1".into(),
            pattern: "fn main".into(),
            regex: false,
            case_sensitive: true,
        };
        assert!(matches!(req, FetchRequest::TextSearch { .. }));
    }

    #[test]
    fn fetch_request_graph_query_variant() {
        let req = FetchRequest::GraphQuery {
            tenant_id: "t1".into(),
            node_id: "parse".into(),
        };
        assert!(matches!(req, FetchRequest::GraphQuery { .. }));
    }

    #[test]
    fn load_tenant_list_does_not_panic_without_db() {
        // Without a real database this returns an empty list, not a panic.
        let _tenants = load_tenant_list();
    }
}
