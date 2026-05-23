//! Groups command -- view project group memberships.
//!
//! Subcommands: list

use anyhow::Result;
use clap::{Args, Subcommand};

mod list;

/// Groups command arguments
#[derive(Args)]
pub struct GroupsArgs {
    #[command(subcommand)]
    command: GroupsCommand,
}

/// Groups subcommands
#[derive(Subcommand)]
enum GroupsCommand {
    /// List all project group memberships
    List {
        /// Filter by tenant ID
        #[arg(long)]
        tenant: Option<String>,

        /// Filter by group type (dependency, workspace, git_org, affinity, tag_affinity)
        #[arg(long, alias = "type")]
        group_type: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Output as tab-separated script-friendly format
        #[arg(long)]
        script: bool,

        /// Omit headers in script output
        #[arg(long)]
        no_headers: bool,
    },
}

pub async fn execute(args: GroupsArgs) -> Result<()> {
    match args.command {
        GroupsCommand::List {
            tenant,
            group_type,
            json,
            script,
            no_headers,
        } => list::list_groups(
            tenant.as_deref(),
            group_type.as_deref(),
            json,
            script,
            no_headers,
        ),
    }
}
