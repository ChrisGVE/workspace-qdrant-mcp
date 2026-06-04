//! Admin command - system administration operations
//!
//! Provides administrative operations that affect the daemon's persistent state.
//! Subcommands: rename-tenant, idle-history, prune-logs, cleanup-orphans, recover-state,
//! collections, rebuild, backup, restore, stats

use anyhow::Result;
use clap::{Args, Subcommand};

use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

mod cleanup_orphans;
mod idle_history;
mod metrics;
mod metrics_setup;
mod perf;
mod perf_data;
mod perf_queries;
mod prune_logs;
mod rebalance_idf;
mod reembed;
mod rename_tenant;

/// Canonical collection names (validated against wqm-common constants)
pub(super) const VALID_COLLECTIONS: &[&str] =
    &[COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES];

/// All 4 canonical collections for orphan scanning
pub(super) const ALL_COLLECTIONS: &[&str] = &[
    COLLECTION_PROJECTS,
    COLLECTION_LIBRARIES,
    COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
];

/// Admin command arguments
#[derive(Args)]
pub struct AdminArgs {
    #[command(subcommand)]
    command: AdminCommand,
}

/// Admin subcommands
#[derive(Subcommand)]
enum AdminCommand {
    /// Rename a tenant_id across SQLite tables (watch_folders, unified_queue, tracked_files)
    RenameTenant {
        /// Current tenant_id to rename
        old_id: String,

        /// New tenant_id value
        new_id: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },
    /// Show idle/active state transition history and flip-flop analysis
    IdleHistory {
        /// Hours of history to analyze (default: 24)
        #[arg(short = 'H', long, default_value = "24")]
        hours: f64,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long)]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },
    /// Prune old log files from the canonical log directory
    PruneLogs {
        /// Only list files that would be deleted, without deleting
        #[arg(long)]
        dry_run: bool,

        /// Retention period in hours (default: 36)
        #[arg(long, default_value = "36")]
        retention_hours: u64,
    },
    /// Detect and optionally delete orphaned tenants across all collections
    ///
    /// Orphans are tenant_ids (or library_names) that exist in Qdrant but have
    /// no corresponding watch_folder or tracked_files entry in SQLite.
    CleanupOrphans {
        /// Actually delete orphaned points (default: dry-run report only)
        #[arg(long)]
        delete: bool,

        /// Limit to a specific collection (default: all 4 canonical collections)
        #[arg(long)]
        collection: Option<String>,
    },
    /// Rebuild state.db from Qdrant collections
    ///
    /// Scrolls all Qdrant collections and reconstructs watch_folders, tracked_files,
    /// qdrant_chunks, and rules_mirror. Existing state.db is backed up first.
    RecoverState {
        /// Actually perform recovery (default: dry-run description)
        #[arg(long)]
        confirm: bool,
    },
    /// Correct IDF drift in stored sparse vectors
    ///
    /// Sparse vectors are computed with IDF statistics from the corpus size N
    /// at ingest time. As N grows, stored weights become stale. This command
    /// applies correction factors to bring all points to the current N.
    RebalanceIdf {
        /// Target collection (default: all collections with corpus statistics)
        #[arg(long)]
        collection: Option<String>,

        /// Report drift without updating any vectors
        #[arg(long)]
        dry_run: bool,

        /// Minimum corpus growth (%) required before correction runs (default: 10.0)
        #[arg(long, default_value = "10.0")]
        min_growth_pct: f64,
    },

    /// Trigger a full re-embed: drop and recreate the four canonical Qdrant
    /// collections at the configured embedding output_dim and re-enqueue all
    /// indexed sources. Used after switching embedding provider/model to a
    /// model with a different dimensionality.
    Reembed {
        /// Required safety flag — without it the command refuses to run.
        #[arg(long)]
        confirm: bool,
    },

    /// Display pipeline performance statistics (per-phase timing breakdown)
    #[command(
        after_long_help = "See also:\n  wqm admin stats processing  operation-level breakdown with Q1/Q3 quartiles\n  wqm status --performance    system resource metrics (CPU, memory, disk)"
    )]
    Perf {
        /// Time window in hours (default: 24)
        #[arg(short = 'w', long, default_value = "24")]
        window: f64,

        /// Output in JSON format
        #[arg(long)]
        json: bool,

        /// Break down by 1-2 dimensions (comma-separated): project, phase,
        /// language, operation (alias: op), collection, file_type,
        /// embedding_engine. `--by` is an alias of `--group-by`.
        #[arg(short = 'g', long, visible_alias = "by")]
        group_by: Option<String>,

        /// Sort by column:direction (e.g. avg_ms:desc, count:asc)
        #[arg(short = 's', long)]
        sort: Option<String>,

        /// Filter by collection (projects, libraries, rules, scratchpad)
        #[arg(short = 'c', long)]
        collection: Option<String>,
    },
    /// Manage and fetch Prometheus metrics from the daemon
    Metrics {
        #[command(subcommand)]
        command: MetricsCommand,
    },

    /// Rebuild indexes and sync state (tags, search, vocabulary, keywords, rules, projects, libraries, all)
    Rebuild(super::rebuild::RebuildArgs),

    /// Backup Qdrant collections (create, list, delete snapshots)
    Backup(super::backup::BackupArgs),

    /// Restore Qdrant collections from snapshots (snapshot, from-backup, list, verify)
    Restore(super::restore::RestoreArgs),

    /// Search instrumentation analytics (overview, processing, log-search)
    Stats(super::stats::StatsArgs),
}

/// Metrics subcommands
#[derive(Subcommand)]
enum MetricsCommand {
    /// Fetch live Prometheus metrics from the daemon
    Show {
        /// Metrics server port (default: 9090)
        #[arg(short = 'p', long, default_value = "9090")]
        port: u16,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
    /// Enable the metrics endpoint (adds --metrics-port to daemon launch args)
    Enable {
        /// Metrics server port (default: 9090)
        #[arg(short = 'p', long)]
        port: Option<u16>,
    },
    /// Disable the metrics endpoint (removes --metrics-port from daemon launch args)
    Disable,
    /// Check if metrics endpoint is configured and responding
    Status {
        /// Metrics server port to check (default: 9090)
        #[arg(short = 'p', long, default_value = "9090")]
        port: u16,
    },
}

/// Execute admin command
pub async fn execute(args: AdminArgs) -> Result<()> {
    match args.command {
        AdminCommand::RenameTenant {
            old_id,
            new_id,
            yes,
        } => rename_tenant::execute(old_id, new_id, yes).await,
        AdminCommand::IdleHistory {
            hours,
            script,
            no_headers,
        } => idle_history::execute(hours, script, no_headers),
        AdminCommand::PruneLogs {
            dry_run,
            retention_hours,
        } => prune_logs::execute(dry_run, retention_hours).await,
        AdminCommand::CleanupOrphans { delete, collection } => {
            cleanup_orphans::execute(delete, collection).await
        }
        AdminCommand::RecoverState { confirm } => super::recover_state::execute(confirm).await,
        AdminCommand::RebalanceIdf {
            collection,
            dry_run,
            min_growth_pct,
        } => rebalance_idf::execute(collection, dry_run, min_growth_pct).await,
        AdminCommand::Reembed { confirm } => reembed::execute(confirm).await,
        AdminCommand::Perf {
            window,
            json,
            group_by,
            sort,
            collection,
        } => perf::execute(window, json, group_by, sort, collection).await,
        AdminCommand::Metrics { command } => match command {
            MetricsCommand::Show { port, json } => metrics::execute(port, json).await,
            MetricsCommand::Enable { port } => metrics_setup::enable(port).await,
            MetricsCommand::Disable => metrics_setup::disable().await,
            MetricsCommand::Status { port } => metrics_setup::status(port).await,
        },
        AdminCommand::Rebuild(args) => super::rebuild::execute(args).await,
        AdminCommand::Backup(args) => super::backup::execute(args).await,
        AdminCommand::Restore(args) => super::restore::execute(args).await,
        AdminCommand::Stats(args) => super::stats::execute(args).await,
    }
}
