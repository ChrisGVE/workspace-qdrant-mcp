//! memexd -- the wqm-0.2 daemon (surface S1).
//!
//! Phase 0 scaffold: real argument parsing and version reporting, but the daemon
//! declines to run because its engine (host runtime, gRPC serve loop, queue
//! processor, product sync) is added in later phases. This is the grow-per-phase
//! contract from PRD F-00 -- minimal but real, never a silent stub.

use clap::Parser;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// The wqm-0.2 daemon: owns all persistent state and serves the read/write API.
#[derive(Debug, Parser)]
#[command(name = "memexd", version, about = "workspace-qdrant daemon (wqm-0.2)")]
struct Cli {
    /// Path to the configuration file (honored once the host runtime lands).
    #[arg(long, value_name = "FILE")]
    config: Option<String>,
}

fn main() -> std::process::ExitCode {
    let _cli = Cli::parse();
    eprintln!(
        "memexd {}: the daemon runtime is not yet available in this build. \
         It is assembled across the wqm-0.2 rebuild phases (F-52).",
        env!("CARGO_PKG_VERSION")
    );
    std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8)
}
