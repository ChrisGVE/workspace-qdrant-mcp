//! `wqm docker` — Docker deployment helpers.
//!
//! Currently provides `wqm docker generate-compose` for emitting a
//! `docker-compose.override.yaml` derived from the live `config.yaml`
//! mount map. See `docs/specs/16-path-abstraction.md` §9 for the design.

use anyhow::Result;
use clap::{Args, Subcommand};

pub mod generate_compose;

/// Arguments for the `docker` subcommand group.
#[derive(Args)]
#[command(
    about = "Docker deployment helpers (compose generation)",
    long_about = "Helpers for managing wqm under Docker. Currently provides \
                  `generate-compose`, which derives a docker-compose override \
                  from the live config.yaml mount map (spec 16 §9).",
    after_long_help = "Examples:\n  \
        wqm docker generate-compose                    Generate docker-compose.override.yaml\n  \
        wqm docker generate-compose --check            Detect drift between config and override\n  \
        wqm docker generate-compose --clean            Delete the override file"
)]
pub struct DockerArgs {
    #[command(subcommand)]
    command: DockerCommand,
}

/// `docker` subcommands.
#[derive(Subcommand)]
pub enum DockerCommand {
    /// Generate or validate `docker-compose.override.yaml` from config.yaml.
    GenerateCompose(generate_compose::GenerateComposeArgs),
}

/// Execute a `wqm docker …` invocation.
pub async fn execute(args: DockerArgs) -> Result<()> {
    match args.command {
        DockerCommand::GenerateCompose(a) => generate_compose::execute(a).await,
    }
}
