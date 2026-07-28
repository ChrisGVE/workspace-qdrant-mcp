//! memexd -- the wqm-0.2 daemon (surface S1).
//!
//! `P04-GT001` scaffold: real argument parsing and version reporting, but the
//! daemon declines to run because its engine (host runtime N46 at `P04-GT056`,
//! gRPC serve loop N48 at `P04-GT062`, queue processor, product sync) arrives
//! with those slices. Minimal but real, never a silent stub (DELIVERABLES: "No
//! stubs, no silent loss").

use clap::Parser;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// The wqm-0.2 daemon: owns all persistent state and serves the read/write API.
#[derive(Debug, Parser)]
// `CARGO_BIN_NAME` rather than a literal: the binary carries the `-v2` deployment
// suffix (bins/memexd/Cargo.toml), and `--help` must name what the user actually
// invoked. One spelling, in the manifest.
#[command(
    name = env!("CARGO_BIN_NAME"),
    version,
    about = "workspace-qdrant daemon (wqm-0.2)"
)]
struct Cli {
    /// Path to the configuration file (honored once the host runtime lands).
    #[arg(long, value_name = "FILE")]
    config: Option<String>,
}

fn main() -> std::process::ExitCode {
    let _cli = Cli::parse();
    eprintln!(
        "{} {}: the daemon runtime is not yet available in this build. \
         It is assembled across the wqm-0.2 P04 build slices.",
        env!("CARGO_BIN_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8)
}
