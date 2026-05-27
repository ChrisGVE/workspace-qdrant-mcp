//! Admin command - system administration operations
//!
//! Provides administrative operations that affect the daemon's persistent state.
//! Subcommands: rename-tenant, idle-history, prune-logs, cleanup-orphans, recover-state,
//! collections, rebuild, backup, restore, stats

use anyhow::Result;
use clap::{Args, Subcommand};

use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES};

mod clean_orphan_queue_items;
mod cleanup_orphans;
mod idle_history;
mod ignore_candidates;
mod metrics;
mod metrics_setup;
mod perf;
mod perf_data;
mod perf_queries;
mod prune_logs;
mod rebalance_idf;
mod reembed;
mod rename_tenant;
mod requeue_failed;
mod token_savings;

/// Subset of canonical collections that participate in tenant renames
/// (scratchpad is global-only, so it is excluded).
pub(super) const VALID_COLLECTIONS: &[&str] =
    &[COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES];

/// All 4 canonical collections for orphan scanning. Aliased to
/// `wqm_common::constants::CANONICAL_COLLECTIONS` for drift safety.
pub(super) use wqm_common::constants::CANONICAL_COLLECTIONS as ALL_COLLECTIONS;

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
    /// Delete `unified_queue` rows whose tenant_id no longer matches any watch_folder
    ///
    /// Use after disabling/removing watch folders to drop queue items the
    /// daemon will never be able to process. Dry-run by default; pass `--apply`
    /// to actually delete. Optional `--limit N` caps the batch size.
    CleanOrphanQueueItems {
        /// Actually delete matching rows (default: dry-run report only)
        #[arg(long)]
        apply: bool,

        /// Cap the delete batch size (default: unlimited)
        #[arg(long)]
        limit: Option<u64>,
    },
    /// Reset retry-exhausted `failed` queue rows back to `pending`
    ///
    /// Filters by `error_message LIKE '%<reason>%'`. Use after fixing a bug
    /// that caused a batch of items to retry-exhaust. Dry-run by default;
    /// pass `--apply` to commit. Default `--max-rows=100` for safety.
    RequeueFailed {
        /// Substring matched against `error_message` (case-sensitive)
        #[arg(long)]
        reason_substring: String,

        /// Cap the number of rows updated (default: 100)
        #[arg(long, default_value = "100")]
        max_rows: u64,

        /// Actually reset matching rows (default: dry-run report only)
        #[arg(long)]
        apply: bool,
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

    /// Rank directories by how strongly they look like ignore candidates
    ///
    /// Aggregates `tracked_files` by parent directory (up to `--depth` segments)
    /// and scores each group by `file_count × (1 + 2·failure_rate + ext_homogeneity)`.
    /// Use the output to decide what to add to `.wqmignore` / `global.wqmignore`
    /// — this command never edits ignore files.
    #[command(
        after_long_help = "See also:\n  docs/specs/14-future-development.md  planned Phase 2 (cost-vs-usage scoring)"
    )]
    IgnoreCandidates {
        /// Show the top N scoring directories (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        top: usize,

        /// Parent-directory segment depth for aggregation (default: 3)
        #[arg(short = 'd', long, default_value = "3")]
        depth: usize,

        /// Skip directories with fewer than this many files (default: 10)
        #[arg(long, default_value = "10")]
        min_files: u64,

        /// Output in JSON format
        #[arg(long)]
        json: bool,

        /// Override the SQLite DB path (defaults to WQM_DATABASE_PATH or XDG state dir)
        #[arg(long)]
        db: Option<std::path::PathBuf>,
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

        /// Group by 1-2 dimensions: project, phase, language, op (comma-separated)
        #[arg(short = 'g', long)]
        group_by: Option<String>,

        /// Sort by column:direction (e.g. avg_ms:desc, count:asc)
        #[arg(short = 's', long)]
        sort: Option<String>,

        /// Filter by collection (projects, libraries, rules, scratchpad)
        #[arg(short = 'c', long)]
        collection: Option<String>,
    },
    /// Display token-economy savings (how much context the MCP server
    /// saved the agent vs. an unshaped Read of referenced files).
    ///
    /// Reads the `token_savings` view (added in schema v38). Spec:
    /// docs/specs/20-token-economy-instrumentation.md
    #[command(
        after_long_help = "See also:\n  wqm admin perf        per-phase pipeline timing\n  wqm admin stats       search instrumentation analytics"
    )]
    TokenSavings {
        /// Time window, e.g. `7d`, `24h`, `30m`, or a bare number (hours).
        #[arg(short = 'w', long, default_value = "7d")]
        window: String,

        /// Output in JSON format.
        #[arg(long)]
        json: bool,

        /// Filter by project_id.
        #[arg(long)]
        project: Option<String>,

        /// Filter by tool name (e.g. `mcp_qdrant`).
        #[arg(long)]
        tool: Option<String>,
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
        } => prune_logs::execute(dry_run, retention_hours),
        AdminCommand::CleanupOrphans { delete, collection } => {
            cleanup_orphans::execute(delete, collection).await
        }
        AdminCommand::CleanOrphanQueueItems { apply, limit } => {
            clean_orphan_queue_items::execute(apply, limit)
        }
        AdminCommand::RequeueFailed {
            reason_substring,
            max_rows,
            apply,
        } => requeue_failed::execute(reason_substring, max_rows, apply),
        AdminCommand::RecoverState { confirm } => super::recover_state::execute(confirm).await,
        AdminCommand::RebalanceIdf {
            collection,
            dry_run,
            min_growth_pct,
        } => rebalance_idf::execute(collection, dry_run, min_growth_pct).await,
        AdminCommand::Reembed { confirm } => reembed::execute(confirm).await,
        AdminCommand::IgnoreCandidates {
            top,
            depth,
            min_files,
            json,
            db,
        } => ignore_candidates::execute(top, depth, min_files, json, db),
        AdminCommand::Perf {
            window,
            json,
            group_by,
            sort,
            collection,
        } => perf::execute(window, json, group_by, sort, collection).await,
        AdminCommand::TokenSavings {
            window,
            json,
            project,
            tool,
        } => token_savings::execute(window, json, project, tool).await,
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
