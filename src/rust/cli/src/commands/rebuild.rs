//! Rebuild command - trigger index rebuilds via the daemon.
//!
//! Sends RebuildIndex gRPC requests to the daemon for computed indexes
//! (tag hierarchy, FTS, sparse vectors, etc.).

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::workspace_daemon::RebuildIndexRequest;
use crate::grpc::client::DaemonClient;
use crate::output;

/// Rebuild command arguments
#[derive(Args)]
pub struct RebuildArgs {
    #[command(subcommand)]
    command: RebuildCommand,
}

/// Rebuild subcommands
#[derive(Subcommand)]
enum RebuildCommand {
    /// Rebuild canonical tag hierarchy
    Tags {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Rebuild all computed indexes
    All {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },
}

/// Execute rebuild command
pub async fn execute(args: RebuildArgs) -> Result<()> {
    let (target, tenant, collection) = match args.command {
        RebuildCommand::Tags { tenant, collection } => ("tags", tenant, collection),
        RebuildCommand::All { tenant, collection } => ("all", tenant, collection),
    };

    let mut client = DaemonClient::connect_default()
        .await
        .context("Failed to connect to daemon. Is memexd running?")?;

    output::info(format!(
        "Rebuilding '{}'{}...",
        target,
        tenant
            .as_ref()
            .map(|t| format!(" for tenant {}", t))
            .unwrap_or_default()
    ));

    let mut request = tonic::Request::new(RebuildIndexRequest {
        target: target.into(),
        tenant_id: tenant,
        collection: Some(collection),
    });
    request.set_timeout(std::time::Duration::from_secs(300));

    let response = client
        .system()
        .rebuild_index(request)
        .await
        .context("RebuildIndex RPC failed")?;

    let resp = response.into_inner();
    if resp.success {
        output::success(resp.message);
        output::info("Check daemon logs for rebuild progress and results.");
    } else {
        anyhow::bail!("Rebuild failed: {}", resp.message);
    }

    Ok(())
}
