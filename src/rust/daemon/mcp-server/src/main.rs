//! workspace-qdrant MCP server entry point.
//!
//! Resolves the operating mode (stdio vs HTTP), initialises the server,
//! and drives it until shutdown.  Full mode-resolution and signal handling
//! are added in subsequent tasks; this stub exits 0 after printing the
//! version banner so the scaffold build can be verified.

use mcp_server::server_types::{BUILD_NUMBER, SERVER_NAME, SERVER_VERSION_BASE};

fn main() {
    println!(
        "{} v{} (build {})",
        SERVER_NAME, SERVER_VERSION_BASE, BUILD_NUMBER
    );
}
