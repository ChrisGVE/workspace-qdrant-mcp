//! workspace-qdrant-mcp -- the wqm-0.2 MCP server (surface S2).
//!
//! Phase 0 scaffold: real argument parsing and version reporting, declining to
//! serve because the MCP tool surface (N43 adapter, N49 client seam, N33 rules
//! injection) is added at F-50. Grow-per-phase contract (PRD F-00): minimal but
//! real, never a silent stub.

use clap::Parser;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// The wqm-0.2 MCP server: exposes the 7 JSON-RPC tools over stdio.
#[derive(Debug, Parser)]
#[command(
    name = "workspace-qdrant-mcp",
    version,
    about = "workspace-qdrant MCP server (wqm-0.2)"
)]
struct Cli {}

fn main() -> std::process::ExitCode {
    let _cli = Cli::parse();
    eprintln!(
        "workspace-qdrant-mcp {}: the MCP tool surface is not yet available in \
         this build. It is added at the MCP surface phase (F-50).",
        env!("CARGO_PKG_VERSION")
    );
    std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8)
}
