//! workspace-qdrant MCP server entry point.
//!
//! Resolves the operating mode (stdio vs HTTP), initialises logging, and will
//! drive the server until shutdown.  Full mode-resolution, transport wiring,
//! and signal handling are added in subsequent tasks (31, 32); this stub
//! initialises logging and emits the version banner via `tracing` (never
//! stdout in stdio mode) so stdout stays reserved for JSON-RPC frames.

use mcp_server::observability::logging::init_logging;
use mcp_server::server_types::{ServerMode, BUILD_NUMBER, SERVER_NAME, SERVER_VERSION_BASE};
use tracing::info;

fn main() {
    // Mode resolution (CLI flags / env) lands in task 31/32; default to stdio,
    // the safest choice for stdout purity, until then.
    let mode = ServerMode::default();
    init_logging(mode);

    info!(
        server = SERVER_NAME,
        version = SERVER_VERSION_BASE,
        build = BUILD_NUMBER,
        "workspace-qdrant MCP server starting"
    );
}
