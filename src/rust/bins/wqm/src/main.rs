//! wqm -- the wqm-0.2 command-line surface (surface S3).
//!
//! `P04-GT001` scaffold: real subcommand parsing for the known command groups,
//! each returning a typed not-yet-available decline. The command groups are
//! declared now so `wqm --help` is honest about the eventual surface; their
//! handlers are filled in as the client library and services land (N32's slice
//! `P04-GT059`, with the TUI N42 at `P04-GT060` behind the `tui` feature).
//! Minimal but real, never a silent stub.
//!
//! **The verb tree here predates P01 and is NOT the designed CLI.** P01-GT001/2/3
//! own the human surfaces and are PARKED (Chris, 20260727), so these groups are
//! the v0.1-shaped placeholder they were seeded as. `P04-GT059` replaces them
//! with the P01 design; nothing should be built against this shape meanwhile.

use clap::{Parser, Subcommand};

/// Exit code for a surface that parsed a valid request it cannot yet serve.
const EXIT_NOT_YET_AVAILABLE: i32 = 3;

/// The wqm-0.2 CLI: manage the daemon, collections, queue, projects, and graph.
#[derive(Debug, Parser)]
#[command(name = "wqm", version, about = "workspace-qdrant CLI (wqm-0.2)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Top-level command groups. Each maps to a subsystem that is wired in a later
/// phase; until then every group declines with a stable, typed message.
#[derive(Debug, Subcommand)]
enum Command {
    /// Daemon service control (status, install, start/stop).
    Service,
    /// Health and status diagnostics.
    Status,
    /// Ingestion queue monitoring.
    Queue,
    /// Project registration and watch control.
    Project,
    /// Administrative operations (collections, rebuild, performance).
    Admin,
    /// Code relationship graph queries.
    Graph,
    /// First-run setup (completions, man pages, hooks).
    Init,
}

impl Command {
    /// The subsystem name a decline message names, so the not-yet-available text
    /// is specific rather than generic.
    fn subsystem(&self) -> &'static str {
        match self {
            Command::Service => "service control",
            Command::Status => "status diagnostics",
            Command::Queue => "queue monitoring",
            Command::Project => "project management",
            Command::Admin => "administration",
            Command::Graph => "graph queries",
            Command::Init => "first-run setup",
        }
    }
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    eprintln!(
        "wqm {}: {} is not yet available in this build. The CLI surface is wired \
         to the daemon across the wqm-0.2 P04 build slices.",
        env!("CARGO_PKG_VERSION"),
        cli.command.subsystem()
    );
    std::process::ExitCode::from(EXIT_NOT_YET_AVAILABLE as u8)
}
