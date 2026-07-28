//! memexd -- the wqm-0.2 daemon (surface S1).
//!
//! `P04-GT001-WO011` gives the daemon its first real behaviour: it serves N24's
//! `SystemService` over N48's serve loop, so the walking skeleton has something to
//! talk to. Everything else it will eventually own -- the host runtime (N46,
//! `P04-GT056`), the queue processor, the write path, product sync -- still
//! arrives with those slices, and asking for them still declines rather than
//! pretending.
//!
//! # Why `--socket` has no default
//!
//! CR-007's first defect is a default live endpoint: v0.1's tests defaulted to
//! `localhost:6333` and therefore wrote to production. The same reasoning applies
//! to the product. A daemon that defaults its listening address is a daemon that
//! can be started against the wrong one by omission, and the deployment-directory
//! derivation that will supply the real default belongs to N7 (`P04-GT013`) --
//! which does not exist yet. So the address is required, and nothing is guessed.

use std::path::PathBuf;

use clap::Parser;
use wqm_proto::Address;
use wqm_serve::DaemonFacts;

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// Exit code for a serve loop that could not start or failed while running.
const EXIT_SERVE_FAILED: i32 = 4;

/// The wqm-0.2 daemon: owns all persistent state and serves the read/write API.
// `CARGO_BIN_NAME` rather than a literal: the binary carries the `-v2` deployment
// suffix (bins/memexd/Cargo.toml), and `--help` must name what the user actually
// invoked. One spelling, in the manifest.
#[derive(Debug, Parser)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    version,
    about = "workspace-qdrant daemon (wqm-0.2)"
)]
struct Cli {
    /// Path to the configuration file (honored once the host runtime lands).
    #[arg(long, value_name = "FILE")]
    config: Option<String>,

    /// Serve N24 on this Unix socket. Required to serve; there is no default
    /// address, by design.
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    let Some(socket) = cli.socket else {
        eprintln!(
            "{} {}: the daemon runtime is not yet available in this build. \
             It is assembled across the wqm-0.2 P04 build slices. Pass --socket \
             <PATH> to serve the subset that is: the N24 system service.",
            env!("CARGO_BIN_NAME"),
            env!("CARGO_PKG_VERSION")
        );
        return std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8);
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!(
                "{}: could not start the async runtime: {e}",
                env!("CARGO_BIN_NAME")
            );
            return std::process::ExitCode::from(EXIT_SERVE_FAILED as u8);
        }
    };

    let address = Address::Uds(socket);
    let facts = DaemonFacts::starting_now(env!("CARGO_PKG_VERSION"));

    // Announced on stderr before the loop starts, so a supervisor (or a test)
    // knows the socket is the one it asked for and not a defaulted guess.
    eprintln!("{}: serving {address}", env!("CARGO_BIN_NAME"));

    match runtime.block_on(wqm_serve::serve(&address, facts, shutdown_signal())) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: {e}", env!("CARGO_BIN_NAME"));
            std::process::ExitCode::from(EXIT_SERVE_FAILED as u8)
        }
    }
}

/// Stop serving on SIGINT or SIGTERM, so the socket is unlinked on the way out
/// rather than left for the next start to refuse.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            // A daemon that cannot install a handler still serves; it just cannot
            // be asked to stop gracefully, and saying so is better than exiting.
            Err(e) => {
                eprintln!(
                    "{}: no SIGTERM handler ({e}); shutdown will be abrupt",
                    env!("CARGO_BIN_NAME")
                );
                return std::future::pending().await;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
