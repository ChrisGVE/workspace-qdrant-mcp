//! MCP `tools/list` handler and tool module.
//!
//! This module exposes the static tool definitions and wires the rmcp
//! `ServerHandler::list_tools` implementation.  Tool *execution* is handled
//! by tasks 24–30; the handler here returns `unimplemented` for `call_tool`.

pub mod definitions;
pub mod embedding;
pub mod envelope;
pub mod grep;
pub mod retrieve;
#[cfg(test)]
mod tests;

use rmcp::{
    handler::server::ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, ErrorData, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities,
    },
    service::RequestContext,
    RoleServer,
};

pub use definitions::list_tools;

// ---------------------------------------------------------------------------
// ToolsHandler: implements rmcp::ServerHandler for tools/list
// ---------------------------------------------------------------------------

/// Minimal rmcp `ServerHandler` that advertises only the `tools` capability
/// and implements `list_tools`.  Per CC-4, no resources / prompts / logging
/// capabilities are advertised.
///
/// `call_tool` returns an MCP method-not-found error; execution is added in
/// tasks 24–30.
pub struct ToolsHandler;

impl ServerHandler for ToolsHandler {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult::with_all_items(list_tools())))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        let msg = format!("tool '{}' execution not yet implemented", request.name);
        std::future::ready(Err(ErrorData::invalid_params(msg, None)))
    }
}
