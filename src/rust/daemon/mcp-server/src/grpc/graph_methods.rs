//! GraphService RPC wrappers for [`DaemonClient`].
//!
//! Mirrors `DaemonClientService.queryRelated` from
//! `src/typescript/mcp-server/src/clients/daemon-client/service-methods.ts`
//! lines 94-105.
//!
//! | Rust method      | Proto RPC                    | TS equivalent       |
//! |------------------|------------------------------|---------------------|
//! | `query_related`  | `GraphService::QueryRelated` | `queryRelated()`    |
//!
//! The wire method name is `"queryRelated"` (used in the TS `grpcUnaryWithTimeout`
//! call on service-methods.ts line 100), which does **not** contain `"search"`,
//! so [`super::timeouts::resolve_timeout`] applies the default 5 s budget.
//!
//! ## Internal result type
//!
//! [`TraversalNodeResult`] is a thin Rust struct that mirrors the proto
//! [`TraversalNodeProto`] fields exactly — it provides a typed interface for
//! callers that do not want to work with prost-generated types directly.
//! Field names match the TS `TraversalNodeProto` interface in
//! `grpc-types-search-graph.ts` lines 54-62 exactly:
//! - `node_id`, `symbol_name`, `symbol_type`, `file_path`, `edge_type`, `depth`, `path`
//!
//! ## Response
//!
//! [`QueryRelatedResult`] mirrors the TS `QueryRelatedResponse` interface
//! (grpc-types-search-graph.ts lines 48-52):
//! - `nodes`: related nodes with traversal context
//! - `total`: total related nodes found
//! - `query_time_ms`: query execution time in milliseconds

use tonic::Status;

use crate::proto::{QueryRelatedRequest, QueryRelatedResponse, TraversalNodeProto};

use super::client::DaemonClient;

/// Internal representation of a single graph traversal node.
///
/// Field names match the proto `TraversalNodeProto` message (and the TS
/// `TraversalNodeProto` interface in `grpc-types-search-graph.ts` lines 54-62)
/// exactly:
/// - `node_id`: unique node identifier
/// - `symbol_name`: name of the symbol (function, class, etc.)
/// - `symbol_type`: node type as string (e.g., `"function"`, `"class"`)
/// - `file_path`: relative path from project root
/// - `edge_type`: edge type as string (e.g., `"CALLS"`, `"IMPORTS"`)
/// - `depth`: hops from the source node
/// - `path`: traversal path description (non-filesystem path)
#[derive(Debug, Clone, PartialEq)]
pub struct TraversalNodeResult {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub edge_type: String,
    pub depth: u32,
    pub path: String,
}

impl From<TraversalNodeProto> for TraversalNodeResult {
    fn from(n: TraversalNodeProto) -> Self {
        Self {
            node_id: n.node_id,
            symbol_name: n.symbol_name,
            symbol_type: n.symbol_type,
            file_path: n.file_path,
            edge_type: n.edge_type,
            depth: n.depth,
            path: n.path,
        }
    }
}

/// Internal representation of a [`QueryRelatedResponse`].
///
/// Mirrors the TS `QueryRelatedResponse` interface in
/// `grpc-types-search-graph.ts` lines 48-52.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryRelatedResult {
    pub nodes: Vec<TraversalNodeResult>,
    pub total: u32,
    pub query_time_ms: i64,
}

impl From<QueryRelatedResponse> for QueryRelatedResult {
    fn from(resp: QueryRelatedResponse) -> Self {
        Self {
            nodes: resp
                .nodes
                .into_iter()
                .map(TraversalNodeResult::from)
                .collect(),
            total: resp.total,
            query_time_ms: resp.query_time_ms,
        }
    }
}

impl DaemonClient {
    /// Query graph nodes related to a given node within N hops — mirrors TS `queryRelated()`.
    ///
    /// The wire method name is `"queryRelated"` (see service-methods.ts line 100),
    /// which does **not** contain `"search"`, so the default 5 s budget is applied
    /// by [`super::timeouts::resolve_timeout`].
    ///
    /// # Request fields (mirrors [`QueryRelatedRequest`] proto)
    /// - `tenant_id`: project tenant identifier
    /// - `node_id`: node ID to query from (document ID)
    /// - `max_hops`: max traversal depth (1–5)
    /// - `edge_types`: optional filter list (e.g. `["CALLS", "IMPORTS"]`); empty = all
    /// - `branch`: optional branch scope
    ///
    /// # Errors
    /// Returns `Err(Status)` on transport, timeout, or daemon error.
    pub async fn query_related(
        &mut self,
        request: QueryRelatedRequest,
    ) -> Result<QueryRelatedResponse, Status> {
        // Wire method name "queryRelated" → 5 s default budget (no "search" in name).
        let client = self.graph.clone();
        self.call("queryRelated", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.query_related(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{QueryRelatedResponse, TraversalNodeProto};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_request(tenant_id: &str, node_id: &str, max_hops: u32) -> QueryRelatedRequest {
        QueryRelatedRequest {
            tenant_id: tenant_id.to_string(),
            node_id: node_id.to_string(),
            max_hops,
            edge_types: vec![],
            branch: None,
        }
    }

    fn make_traversal_node(
        node_id: &str,
        symbol_name: &str,
        symbol_type: &str,
        file_path: &str,
        edge_type: &str,
        depth: u32,
        path: &str,
    ) -> TraversalNodeProto {
        TraversalNodeProto {
            node_id: node_id.to_string(),
            symbol_name: symbol_name.to_string(),
            symbol_type: symbol_type.to_string(),
            file_path: file_path.to_string(),
            edge_type: edge_type.to_string(),
            depth,
            path: path.to_string(),
        }
    }

    fn make_response(nodes: Vec<TraversalNodeProto>, total: u32) -> QueryRelatedResponse {
        QueryRelatedResponse {
            nodes,
            total,
            query_time_ms: 7,
        }
    }

    // ── QueryRelatedRequest field mapping ─────────────────────────────────────

    #[test]
    fn request_tenant_id_field() {
        let req = make_request("tenant_abc", "node_42", 2);
        assert_eq!(req.tenant_id, "tenant_abc");
    }

    #[test]
    fn request_node_id_field() {
        let req = make_request("t1", "doc::src/main.rs::my_func", 1);
        assert_eq!(req.node_id, "doc::src/main.rs::my_func");
    }

    #[test]
    fn request_max_hops_field() {
        let req = make_request("t1", "n1", 3);
        assert_eq!(req.max_hops, 3);
    }

    #[test]
    fn request_edge_types_empty_by_default() {
        let req = make_request("t1", "n1", 1);
        assert!(req.edge_types.is_empty());
    }

    #[test]
    fn request_edge_types_filter() {
        let req = QueryRelatedRequest {
            tenant_id: "t1".to_string(),
            node_id: "n1".to_string(),
            max_hops: 2,
            edge_types: vec!["CALLS".to_string(), "IMPORTS".to_string()],
            branch: None,
        };
        assert_eq!(req.edge_types, vec!["CALLS", "IMPORTS"]);
    }

    #[test]
    fn request_branch_none_by_default() {
        let req = make_request("t1", "n1", 1);
        assert!(req.branch.is_none());
    }

    #[test]
    fn request_branch_some() {
        let req = QueryRelatedRequest {
            tenant_id: "t1".to_string(),
            node_id: "n1".to_string(),
            max_hops: 1,
            edge_types: vec![],
            branch: Some("main".to_string()),
        };
        assert_eq!(req.branch.as_deref(), Some("main"));
    }

    // ── TraversalNodeProto → TraversalNodeResult mapping ─────────────────────

    #[test]
    fn traversal_node_result_node_id() {
        let proto = make_traversal_node(
            "n42",
            "my_func",
            "function",
            "src/main.rs",
            "CALLS",
            1,
            "root→my_func",
        );
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.node_id, "n42");
    }

    #[test]
    fn traversal_node_result_symbol_name() {
        let proto = make_traversal_node(
            "n1",
            "process_event",
            "function",
            "src/handler.rs",
            "CALLS",
            2,
            "",
        );
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.symbol_name, "process_event");
    }

    #[test]
    fn traversal_node_result_symbol_type() {
        let proto = make_traversal_node("n1", "Config", "class", "src/config.rs", "IMPORTS", 1, "");
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.symbol_type, "class");
    }

    #[test]
    fn traversal_node_result_file_path() {
        let proto = make_traversal_node("n1", "foo", "function", "src/lib.rs", "CALLS", 1, "");
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.file_path, "src/lib.rs");
    }

    #[test]
    fn traversal_node_result_edge_type() {
        let proto = make_traversal_node("n1", "bar", "function", "src/lib.rs", "IMPORTS", 1, "");
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.edge_type, "IMPORTS");
    }

    #[test]
    fn traversal_node_result_depth() {
        let proto = make_traversal_node("n1", "baz", "function", "src/lib.rs", "CALLS", 3, "");
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.depth, 3);
    }

    #[test]
    fn traversal_node_result_path() {
        let proto = make_traversal_node(
            "n1",
            "baz",
            "function",
            "src/lib.rs",
            "CALLS",
            1,
            "root→a→baz",
        );
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.path, "root→a→baz");
    }

    #[test]
    fn traversal_node_result_depth_zero() {
        let proto =
            make_traversal_node("n1", "root_func", "function", "src/main.rs", "CALLS", 0, "");
        let result = TraversalNodeResult::from(proto);
        assert_eq!(result.depth, 0);
    }

    // ── QueryRelatedResponse → QueryRelatedResult mapping ────────────────────

    #[test]
    fn query_related_result_nodes_empty() {
        let resp = make_response(vec![], 0);
        let result = QueryRelatedResult::from(resp);
        assert!(result.nodes.is_empty());
    }

    #[test]
    fn query_related_result_total() {
        let resp = make_response(vec![], 5);
        let result = QueryRelatedResult::from(resp);
        assert_eq!(result.total, 5);
    }

    #[test]
    fn query_related_result_query_time_ms() {
        let resp = make_response(vec![], 0);
        let result = QueryRelatedResult::from(resp);
        assert_eq!(result.query_time_ms, 7);
    }

    #[test]
    fn query_related_result_nodes_len() {
        let nodes = vec![
            make_traversal_node("n1", "f1", "function", "a.rs", "CALLS", 1, ""),
            make_traversal_node("n2", "f2", "function", "b.rs", "IMPORTS", 2, ""),
        ];
        let resp = make_response(nodes, 2);
        let result = QueryRelatedResult::from(resp);
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.total, 2);
    }

    #[test]
    fn query_related_result_nodes_converted() {
        let nodes = vec![make_traversal_node(
            "n1",
            "my_fn",
            "function",
            "src/lib.rs",
            "CALLS",
            1,
            "root→my_fn",
        )];
        let resp = make_response(nodes, 1);
        let result = QueryRelatedResult::from(resp);
        assert_eq!(result.nodes[0].node_id, "n1");
        assert_eq!(result.nodes[0].symbol_name, "my_fn");
        assert_eq!(result.nodes[0].depth, 1);
    }

    // ── Wire method name: "queryRelated" uses 5 s budget ─────────────────────

    #[tokio::test]
    async fn query_related_wire_name_uses_5s_budget() {
        // "queryRelated" does not contain "search" → 5 s default budget.
        use crate::grpc::timeouts::resolve_timeout;
        use std::time::Duration;
        let budget = resolve_timeout("queryRelated", None);
        assert_eq!(budget, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn query_related_call_times_out_correctly() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("queryRelated", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }

    // ── DaemonClient construction ─────────────────────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_graph_calls() {
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }
}
