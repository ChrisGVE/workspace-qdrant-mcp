//! workspace-qdrant-mcp -- the wqm-0.2 MCP server (surface S2).
//!
//! `P04-GT001-WO011` makes the first tool real: `status` answers over the whole
//! path the architecture claims -- this bin, the N49 client seam, N24's gRPC, and
//! N48's serve loop in the daemon -- and answers *especially* when the daemon is
//! not there, which is the condition it exists to report (MCP-SURFACE.md §4.3).
//!
//! The other seven tools of the sealed inventory (§1.2), the N43 adapter proper
//! and the N33 rules-injection half arrive with `P04-GT061`. Until then this bin
//! advertises one tool, because advertising eight and serving one is the failure
//! class the surface redesign exists to end.

mod mcp;
mod status;

use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use clap::Parser;
use wqm_client::Client;
use wqm_proto::Address;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;
/// Exit code for a session that failed at the transport.
const EXIT_TRANSPORT_FAILED: i32 = 4;

/// The wqm-0.2 MCP server.
// `CARGO_BIN_NAME` rather than a literal: the binary carries the `-v2` deployment
// suffix, and the MCP registration for the parallel period names that executable.
#[derive(Debug, Parser)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    version,
    about = "workspace-qdrant MCP server (wqm-0.2)"
)]
struct Cli {
    /// The daemon's Unix socket. Required to serve; there is no default address,
    /// for the reason `memexd`'s `--socket` has none -- a defaulted endpoint is
    /// CR-007's first defect, and N7 (`P04-GT013`) owns the derivation that will
    /// supply the real one.
    #[arg(long, value_name = "PATH")]
    daemon_socket: Option<PathBuf>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    let Some(socket) = cli.daemon_socket else {
        eprintln!(
            "{} {}: the MCP tool surface needs a daemon address. Pass \
             --daemon-socket <PATH>; the remaining tools of the sealed inventory \
             are added by the wqm-0.2 P04 build slices.",
            env!("CARGO_BIN_NAME"),
            env!("CARGO_PKG_VERSION")
        );
        return std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8);
    };

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!(
                "{}: could not start the async runtime: {e}",
                env!("CARGO_BIN_NAME")
            );
            return std::process::ExitCode::from(EXIT_TRANSPORT_FAILED as u8);
        }
    };

    let client = Client::new(Address::Uds(socket));
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    match mcp::serve(
        BufReader::new(stdin.lock()),
        BufWriter::new(stdout.lock()),
        &client,
        &runtime,
    ) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: the stdio session failed: {e}", env!("CARGO_BIN_NAME"));
            std::process::ExitCode::from(EXIT_TRANSPORT_FAILED as u8)
        }
    }
}
