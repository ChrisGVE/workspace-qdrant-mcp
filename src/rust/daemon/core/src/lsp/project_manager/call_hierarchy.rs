//! LSP call-hierarchy resolution: precise outgoing calls (callees).
//!
//! `textDocument/references` only finds usage *sites*; the call hierarchy
//! resolves the actual callee *definitions* (name + file + line). That lets the
//! graph build resolved `CALLS` edges pointing at the real callee node instead
//! of a name-only stub target (which carries an empty file_path and never
//! matches the callee's real node_id). This is the community-recommended LSP
//! method for precise caller/callee relations.

use std::path::Path;

use super::{LanguageServerManager, ProjectLspResult};
use crate::graph::{EdgeType, GraphEdge, GraphNode, NodeType};

/// A resolved call target: the callee symbol with its definition site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCall {
    /// Callee symbol name.
    pub name: String,
    /// Filesystem path of the callee's definition (`file://` prefix stripped).
    pub file: String,
    /// 0-indexed line of the callee's definition.
    pub line: u32,
}

impl LanguageServerManager {
    /// Resolve the outgoing calls (callees) of the symbol at `(line, column)`
    /// in `file` via `textDocument/prepareCallHierarchy` +
    /// `callHierarchy/outgoingCalls`.
    ///
    /// Returns an empty vec (never errors) when no server is ready, the symbol
    /// is not callable, or the server lacks call-hierarchy support — callers
    /// then fall back to tree-sitter name-stub edges.
    #[tracing::instrument(
        name = "lsp.outgoing_calls",
        skip_all,
        fields(file = %file.display(), line, column)
    )]
    pub async fn resolved_outgoing_calls(
        &self,
        file: &Path,
        line: u32,
        column: u32,
    ) -> ProjectLspResult<Vec<ResolvedCall>> {
        let (_key, server_instance) = self.find_server_for_file(file).await;
        let Some(instance) = server_instance else {
            return Ok(Vec::new());
        };
        let rpc_client = {
            let inst = instance.lock().await;
            inst.rpc_client()
        };

        // 1. prepareCallHierarchy at the symbol position.
        let prepare_params = serde_json::json!({
            "textDocument": { "uri": Self::file_to_uri(file) },
            "position": { "line": line, "character": column }
        });
        let prepared = match rpc_client
            .send_request("textDocument/prepareCallHierarchy", prepare_params)
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!(file = %file.display(), error = %e, "prepareCallHierarchy failed");
                return Ok(Vec::new());
            }
        };

        let Some(item) = prepared
            .result
            .as_ref()
            .and_then(|r| r.as_array())
            .and_then(|items| items.first())
            .cloned()
        else {
            return Ok(Vec::new());
        };

        // 2. outgoingCalls for the prepared item.
        let outgoing_params = serde_json::json!({ "item": item });
        let response = match rpc_client
            .send_request("callHierarchy/outgoingCalls", outgoing_params)
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!(file = %file.display(), error = %e, "outgoingCalls failed");
                return Ok(Vec::new());
            }
        };

        let calls = response
            .result
            .as_ref()
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(parse_outgoing_call)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        tracing::debug!(file = %file.display(), count = calls.len(), "Resolved outgoing calls");
        Ok(calls)
    }
}

/// Relativize an absolute path returned by the LSP server against the project
/// root, so the result matches the project-relative `file_path` the graph uses
/// to key nodes. Returns `None` for paths outside the project (stdlib/deps),
/// which have no node in this tenant's graph. Separator-agnostic for
/// cross-platform/URI path encodings.
pub(crate) fn relativize_to_project(abs_file: &str, project_root: &str) -> Option<String> {
    let norm_file = abs_file.replace('\\', "/");
    let norm_root = project_root.replace('\\', "/");
    let norm_root = norm_root.trim_end_matches('/');
    let stripped = norm_file.strip_prefix(norm_root)?.trim_start_matches('/');
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// Build resolved `CALLS` graph edges from call-hierarchy results.
///
/// Tree-sitter emits CALLS edges to *stub* nodes (empty file_path → a node_id
/// that never matches the callee's real node). Given LSP-resolved callees, this
/// produces an edge to the callee's REAL node_id — `compute_node_id(tenant,
/// relative_callee_file, name, Function)` — exactly matching the node created
/// when that callee's own file is ingested. Callees outside `project_root`
/// (stdlib/deps) are skipped.
///
/// Pure (no I/O); applied at query-time or by a warm-up backfill pass — not at
/// ingestion time, where the server is usually not yet indexed (the
/// community/Aider pattern: tree-sitter for the baseline graph, LSP precision
/// on demand).
pub(crate) fn resolved_call_edges(
    tenant_id: &str,
    caller_node_id: &str,
    source_file: &str,
    project_root: &str,
    calls: &[ResolvedCall],
) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for call in calls {
        let Some(rel) = relativize_to_project(&call.file, project_root) else {
            continue;
        };
        let target = GraphNode::new(tenant_id, rel, &call.name, NodeType::Function);
        let edge = GraphEdge::new(
            tenant_id,
            caller_node_id,
            &target.node_id,
            EdgeType::Calls,
            source_file,
        );
        nodes.push(target);
        edges.push(edge);
    }
    (nodes, edges)
}

/// Parse one `CallHierarchyOutgoingCall` (its `to` item) into a `ResolvedCall`.
fn parse_outgoing_call(call: &serde_json::Value) -> Option<ResolvedCall> {
    let to = call.get("to")?;
    let name = to.get("name")?.as_str()?.to_string();
    let uri = to.get("uri")?.as_str()?;
    let file = uri.strip_prefix("file://").unwrap_or(uri).to_string();
    let line = to
        .get("selectionRange")
        .or_else(|| to.get("range"))
        .and_then(|r| r.get("start"))
        .and_then(|s| s.get("line"))
        .and_then(|l| l.as_u64())
        .unwrap_or(0) as u32;
    Some(ResolvedCall { name, file, line })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_outgoing_call_extracts_resolved_target() {
        let call = serde_json::json!({
            "to": {
                "name": "add",
                "kind": 12,
                "uri": "file:///home/u/proj/src/lib.rs",
                "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 2, "character": 1 } },
                "selectionRange": { "start": { "line": 0, "character": 7 }, "end": { "line": 0, "character": 10 } }
            },
            "fromRanges": []
        });
        let resolved = parse_outgoing_call(&call).expect("should parse");
        assert_eq!(resolved.name, "add");
        assert_eq!(resolved.file, "/home/u/proj/src/lib.rs");
        // selectionRange wins over range for the definition line.
        assert_eq!(resolved.line, 0);
    }

    #[test]
    fn parse_outgoing_call_handles_missing_fields() {
        assert!(parse_outgoing_call(&serde_json::json!({})).is_none());
        // `to` present but no uri → None (cannot resolve a target file).
        assert!(parse_outgoing_call(&serde_json::json!({ "to": { "name": "x" } })).is_none());
    }

    #[test]
    fn relativize_strips_project_root() {
        assert_eq!(
            relativize_to_project("/home/u/proj/src/lib.rs", "/home/u/proj"),
            Some("src/lib.rs".to_string())
        );
        // Trailing slash on root, Windows-style separators.
        assert_eq!(
            relativize_to_project("C:\\dev\\proj\\src\\a.rs", "C:\\dev\\proj\\"),
            Some("src/a.rs".to_string())
        );
        // Outside the project (dependency/stdlib) → skipped.
        assert_eq!(
            relativize_to_project("/usr/lib/rustlib/std.rs", "/home/u/proj"),
            None
        );
    }

    #[test]
    fn resolved_call_edges_target_real_callee_node() {
        use crate::graph::{compute_node_id, NodeType};

        let tenant = "t1";
        let caller_id = "caller-node-id";
        let calls = vec![ResolvedCall {
            name: "add".to_string(),
            file: "/home/u/proj/src/math.rs".to_string(),
            line: 4,
        }];
        let (nodes, edges) =
            resolved_call_edges(tenant, caller_id, "src/lib.rs", "/home/u/proj", &calls);

        // The resolved target node_id must equal the id of the real callee node
        // (keyed by project-relative path) — NOT a file-less stub.
        let expected_target = compute_node_id(tenant, "src/math.rs", "add", NodeType::Function);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_node_id, caller_id);
        assert_eq!(edges[0].target_node_id, expected_target);
        assert_eq!(edges[0].edge_type, EdgeType::Calls);
        assert_eq!(nodes[0].node_id, expected_target);
        assert_eq!(nodes[0].file_path, "src/math.rs");
    }

    #[test]
    fn resolved_call_edges_skip_out_of_project_callees() {
        let calls = vec![ResolvedCall {
            name: "println".to_string(),
            file: "/usr/lib/rust/std/macros.rs".to_string(),
            line: 1,
        }];
        let (nodes, edges) =
            resolved_call_edges("t1", "caller", "src/lib.rs", "/home/u/proj", &calls);
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }
}
