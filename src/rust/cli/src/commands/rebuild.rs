//! Rebuild command - trigger index rebuilds via the daemon.
//!
//! Sends RebuildIndex gRPC requests to the daemon for computed indexes
//! (tag hierarchy, FTS5, sparse vectors, etc.).

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

    /// Rebuild FTS5 code search index
    Search,

    /// Rebuild BM25 sparse vocabulary (cleanup + reset)
    Vocabulary {
        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Re-extract keywords/tags for all documents
    Keywords {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Sync rules between Qdrant and SQLite
    Rules {
        /// Sync direction (default: qdrant-to-db)
        #[arg(long, value_parser = ["qdrant-to-db", "db-to-qdrant"], default_value = "qdrant-to-db")]
        direction: String,
    },

    /// Rescan all project watch folders
    Projects {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,
    },

    /// Rescan all library watch folders
    Libraries {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,
    },

    /// Rebuild all computed indexes in sequence
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
        RebuildCommand::Tags { tenant, collection } => ("tags".to_string(), tenant, Some(collection)),
        RebuildCommand::Search => ("search".to_string(), None, None),
        RebuildCommand::Vocabulary { collection } => ("vocabulary".to_string(), None, Some(collection)),
        RebuildCommand::Keywords { tenant, collection } => ("keywords".to_string(), tenant, Some(collection)),
        RebuildCommand::Rules { direction } => (format!("rules:{}", direction), None, None),
        RebuildCommand::Projects { tenant } => ("projects".to_string(), tenant, None),
        RebuildCommand::Libraries { tenant } => ("libraries".to_string(), tenant, None),
        RebuildCommand::All { tenant, collection } => ("all".to_string(), tenant, Some(collection)),
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
        target,
        tenant_id: tenant,
        collection,
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
