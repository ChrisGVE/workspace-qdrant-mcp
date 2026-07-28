//! workspace-qdrant-mcp -- the wqm-0.2 MCP server (surface S2).
//!
//! `P04-GT001` scaffold: real argument parsing and version reporting, declining
//! to serve because the MCP tool surface (N43 adapter, N49 client seam, N33 rules
//! injection) arrives with N43's slice, `P04-GT061`. Minimal but real, never a
//! silent stub.

use clap::Parser;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// The wqm-0.2 MCP server: exposes the eight-tool agent surface sealed by
/// P01-GT004 (`query`, `fetch`, `list`, `note`, `ingest`, `project`, `rules`,
/// `status` -- MCP-SURFACE.md §1.2) over stdio.
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
         this build. It is added by the wqm-0.2 P04 build slices.",
        env!("CARGO_PKG_VERSION")
    );
    std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8)
}
