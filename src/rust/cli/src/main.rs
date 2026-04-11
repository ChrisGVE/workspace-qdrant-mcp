//! WQM - Workspace Qdrant MCP Companion
//!
//! A high-performance companion for the workspace-qdrant-mcp daemon.
//! Designed for <100ms startup time using minimal tokio runtime.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

mod commands;
mod config;
mod error;
mod grpc;
mod output;
pub(crate) mod path_arg;
#[cfg(feature = "tui")]
mod tui;

/// Custom help template with grouped subcommands
const HELP_TEMPLATE: &str = "\
{before-help}{name} {version}
{about-with-newline}
USAGE: {usage}

OPTIONS:
{options}
INTERACTIVE:
  tui          Interactive terminal UI

CONTENT:
  project      Project management (list, info, register, delete, search, check)
  library      Library management (list, info, register, remove, search)
  rules        Behavioral rules (list, add, remove, search)
  scratchpad   Scratchpad entries (list, search)

QUEUE & MONITORING:
  queue        Queue inspector (list, show, stats, cancel)
  status       System status and monitoring

SERVICE & ADMIN:
  service      Daemon management (start, stop, restart, status)
  admin        Administration (collections, backup, restore, rebuild, stats, perf, metrics)
  config       Configuration management

CODE ANALYSIS:
  graph        Code relationship graph
  language     Language tools (LSP, Tree-sitter)
  tags         Keyword/tag hierarchy

SETUP & DIAGNOSTICS:
  init         Setup (shell completions, hooks)
  update       Update from GitHub releases
  debug        Diagnostic tools (logs, errors)
  benchmark    Benchmarking tools
{after-help}";

/// Workspace Qdrant MCP Companion
#[derive(Parser)]
#[command(name = "wqm")]
#[command(author, version, about = "Workspace Qdrant MCP Companion", long_about = None)]
#[command(long_version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("BUILD_NUMBER"), ")"))]
#[command(arg_required_else_help = true)]
#[command(help_template = HELP_TEMPLATE)]
#[command(disable_help_subcommand = true)]
struct Cli {
    /// Output format (table, json, plain)
    #[arg(long, global = true, default_value = "table")]
    format: String,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Daemon address (default: http://127.0.0.1:50051)
    #[arg(long, global = true, env = "WQM_DAEMON_ADDR")]
    daemon_addr: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

/// CLI commands — hidden from default help (rendered by custom template)
#[derive(Subcommand)]
enum Commands {
    // --- Search & Content ---
    /// [Deprecated] Use: project search, rules search, library search, scratchpad search
    #[command(display_order = 10, hide = true)]
    Search(commands::search::SearchArgs),

    /// [Deprecated] Use: library register, library ingest
    #[command(display_order = 11, hide = true)]
    Ingest(commands::ingest::IngestArgs),

    /// Behavioral rules management (list, add, remove, search, scope)
    #[command(
        display_order = 12,
        long_about = "Manage behavioral rules that guide AI assistant behavior. Rules are stored \
            in the rules collection and can be scoped globally or per-project. Use rules to \
            encode preferences, constraints, and patterns that persist across sessions.",
        after_help = "Examples:\n  \
            wqm rules list                              List all rules\n  \
            wqm rules list --global                     List global rules only\n  \
            wqm rules add -l no-emoji -c 'Never use emojis' --global   Add a global rule\n  \
            wqm rules remove --label no-emoji --global  Remove a global rule\n  \
            wqm rules search 'testing'                  Search rules by content"
    )]
    Rules(commands::rules::RulesArgs),

    /// Scratchpad entries (list, search)
    #[command(
        display_order = 13,
        alias = "scratch",
        long_about = "Browse and search scratchpad entries. The scratchpad is a persistent \
            knowledge store for analytical findings, design rationale, and research notes. \
            Entries are stored per-project or globally and support semantic search.",
        after_help = "Examples:\n  \
            wqm scratchpad list                         List all entries\n  \
            wqm scratchpad list --project .             List entries for current project\n  \
            wqm scratchpad search 'architecture'        Semantic search across entries"
    )]
    Scratchpad(commands::scratchpad::ScratchArgs),

    // --- Project & Library ---
    /// Project lifecycle (list, info, register, watch, branch)
    #[command(
        display_order = 20,
        long_about = "Manage projects tracked by the workspace-qdrant daemon. Projects are \
            automatically detected via Git repositories and indexed for semantic search. \
            Use subcommands to register, inspect, search content, manage watch folders, \
            and view branch information.",
        after_help = "Examples:\n  \
            wqm project list                            List all registered projects\n  \
            wqm project register .                      Register current directory\n  \
            wqm project info                            Show info for current project\n  \
            wqm project search 'TODO'                   Full-text search in project\n  \
            wqm project check                           Compare tracked vs filesystem\n  \
            wqm project watch list                      List watch folders\n  \
            wqm project branch list                     List indexed branches"
    )]
    Project(commands::project::ProjectArgs),

    /// Library management (list, add, ingest, watch, remove)
    #[command(
        display_order = 21,
        long_about = "Manage reference libraries for documentation, specs, and external content. \
            Libraries are indexed into the libraries collection for semantic search. \
            Supports file watching for automatic re-ingestion on changes.",
        after_help = "Examples:\n  \
            wqm library list                            List all libraries\n  \
            wqm library add docs ./docs                 Add a library (no watching)\n  \
            wqm library watch api-spec ./spec -p '*.yaml'  Watch with file patterns\n  \
            wqm library ingest ./README.md -l docs      Ingest a single document\n  \
            wqm library search 'authentication'         Semantic search across libraries\n  \
            wqm library info docs                       Show library details\n  \
            wqm library remove docs                     Remove a library and its data"
    )]
    Library(commands::library::LibraryArgs),

    /// Watch folder management (hidden alias for `project watch`)
    #[command(display_order = 22, hide = true)]
    Watch(commands::watch::WatchArgs),

    // --- Code Analysis ---
    /// Keyword/tag hierarchy (list, keywords, tree, stats, search, baskets)
    #[command(
        display_order = 23,
        long_about = "Browse and search the keyword/tag hierarchy extracted from indexed code. \
            Tags represent semantic concepts (functions, classes, modules) organized in a \
            tree structure. Use to explore code organization and find related symbols.",
        after_help = "Examples:\n  \
            wqm tags list                               List top-level tags\n  \
            wqm tags tree                               Show full tag hierarchy\n  \
            wqm tags search 'parse'                     Search tags by name\n  \
            wqm tags stats                              Show tag statistics\n  \
            wqm tags keywords                           List extracted keywords"
    )]
    Tags(commands::tags::TagsArgs),

    /// Code relationship graph (query, impact, stats, pagerank, communities, betweenness, migrate)
    #[command(
        display_order = 24,
        long_about = "Analyze code relationships using the dependency graph built from indexed \
            projects. Supports PageRank for importance ranking, community detection for \
            module clustering, betweenness centrality for bridge identification, and \
            impact analysis for change propagation.",
        after_help = "Examples:\n  \
            wqm graph stats                             Show graph statistics\n  \
            wqm graph query 'src/main.rs'               Query relationships for a file\n  \
            wqm graph impact 'src/lib.rs'               Analyze change impact\n  \
            wqm graph pagerank                          Rank files by importance\n  \
            wqm graph communities                       Detect module clusters\n  \
            wqm graph betweenness                       Find bridge files"
    )]
    Graph(commands::graph::GraphArgs),

    /// Language tools (LSP, Tree-sitter)
    #[command(
        display_order = 25,
        long_about = "Inspect and manage language support tooling. Shows available LSP servers, \
            Tree-sitter grammars, and their installation status. Useful for diagnosing \
            language-specific indexing issues.",
        after_help = "Examples:\n  \
            wqm language list                           List supported languages\n  \
            wqm language lsp status                     Show LSP server status\n  \
            wqm language treesitter status              Show grammar status"
    )]
    Language(commands::language::LanguageArgs),

    // --- Queue & Monitoring ---
    /// Queue inspector (list, show, stats, cancel)
    #[command(
        display_order = 30,
        long_about = "Inspect the daemon's unified processing queue. View pending, in-progress, \
            and completed items. Use to monitor ingestion progress, diagnose stuck items, \
            or cancel queued operations.",
        after_help = "Examples:\n  \
            wqm queue list                              List queued items\n  \
            wqm queue stats                             Show queue statistics\n  \
            wqm queue show <id>                         Show details for a queue item\n  \
            wqm queue cancel <id>                       Cancel a pending item"
    )]
    Queue(commands::queue::QueueArgs),

    /// System status and monitoring
    #[command(
        display_order = 31,
        long_about = "Show system health and status information. Displays daemon connectivity, \
            Qdrant availability, collection sizes, active projects, and resource usage. \
            Use for quick health checks and troubleshooting.",
        after_help = "Examples:\n  \
            wqm status                                  Show overall system status\n  \
            wqm status health                           Detailed health diagnostics"
    )]
    Status(commands::status::StatusArgs),

    /// Interactive terminal UI for browsing and monitoring
    #[command(
        display_order = 32,
        long_about = "Launch an interactive terminal UI for browsing projects, libraries, \
            queue status, and system health. Provides a real-time dashboard view of the \
            workspace-qdrant system.",
        after_help = "Examples:\n  \
            wqm tui                                     Launch the interactive UI"
    )]
    #[cfg(feature = "tui")]
    Tui,

    // --- Service & Admin ---
    /// Daemon management (start, stop, restart, status)
    #[command(
        display_order = 40,
        long_about = "Control the memexd daemon lifecycle. Start, stop, or restart the daemon \
            process. The daemon handles file watching, indexing, embedding generation, \
            and serves the gRPC API that the CLI and MCP server connect to.",
        after_help = "Examples:\n  \
            wqm service status                          Check if daemon is running\n  \
            wqm service start                           Start the daemon\n  \
            wqm service stop                            Stop the daemon\n  \
            wqm service restart                         Restart the daemon\n  \
            wqm service install                         Install as system service\n  \
            wqm service logs                            View daemon logs"
    )]
    Service(commands::service::ServiceArgs),

    /// Administration (collections, backup, restore, rebuild, stats, perf, metrics)
    #[command(
        display_order = 41,
        long_about = "Administrative operations for managing Qdrant collections, rebuilding \
            indexes, viewing performance metrics, and running backups. Most operations \
            require a running daemon.",
        after_help = "Examples:\n  \
            wqm admin collections list                  List Qdrant collections\n  \
            wqm admin rebuild all                       Rebuild all indexes\n  \
            wqm admin perf                              Pipeline performance stats\n  \
            wqm admin backup                            Backup collections\n  \
            wqm admin restore                           Restore from backup"
    )]
    Admin(commands::admin::AdminArgs),

    /// Configuration management
    #[command(
        display_order = 42,
        long_about = "View and modify wqm configuration. Settings include daemon address, \
            output format, and environment-specific overrides. Configuration is read from \
            environment variables and can be overridden per-invocation via CLI flags.",
        after_help = "Examples:\n  \
            wqm config show                             Show current configuration\n  \
            wqm config path                             Show config file path"
    )]
    Config(commands::config_cmd::ConfigCmdArgs),

    // --- Setup & Diagnostics ---
    /// Setup (shell completions, hooks)
    #[command(
        display_order = 60,
        long_about = "Initial setup utilities: generate shell completions for zsh/bash/fish, \
            install man pages, and configure Claude Code hooks for automatic rule injection \
            at session start.",
        after_help = "Examples:\n  \
            wqm init completions zsh                    Generate zsh completions\n  \
            wqm init completions bash                   Generate bash completions\n  \
            wqm init man install                        Install man pages\n  \
            wqm init hooks install                      Install Claude Code hooks"
    )]
    Init(commands::init::InitArgs),

    /// Update from GitHub releases
    #[command(
        display_order = 61,
        long_about = "Check for and install updates from GitHub releases. Downloads the latest \
            release binaries for both the daemon (memexd) and CLI (wqm), replacing the \
            currently installed versions.",
        after_help = "Examples:\n  \
            wqm update                                  Check and install updates\n  \
            wqm update --check                          Check for updates only"
    )]
    Update(commands::update::UpdateArgs),

    /// Diagnostic tools (logs, errors, queue-errors, language)
    #[command(
        display_order = 62,
        long_about = "Diagnostic tools for troubleshooting. View daemon logs, recent errors, \
            queue processing failures, and language-specific indexing issues. Useful when \
            files are not being indexed or search results are unexpected.",
        after_help = "Examples:\n  \
            wqm debug logs                              View recent daemon logs\n  \
            wqm debug errors                            Show recent errors\n  \
            wqm debug queue-errors                      Show queue processing failures\n  \
            wqm debug language                          Diagnose language support issues"
    )]
    Debug(commands::debug::DebugArgs),

    /// Benchmarking tools (sparse vectors, search engines)
    #[command(
        display_order = 63,
        long_about = "Run performance benchmarks for sparse vector generation and search \
            engine components. Results help tune chunking parameters and identify \
            performance bottlenecks.",
        after_help = "Examples:\n  \
            wqm benchmark sparse                        Benchmark sparse vector generation\n  \
            wqm benchmark search                        Benchmark search performance"
    )]
    Benchmark(commands::benchmark::BenchmarkArgs),

    // --- Hidden backward-compat aliases ---
    /// Man page generation and installation (alias for `init man`)
    #[command(display_order = 900, hide = true)]
    Man(commands::man::ManArgs),

    /// Claude Code hooks management (alias for `init hooks`)
    #[command(display_order = 901, hide = true)]
    Hooks(commands::hooks::HooksArgs),

    /// Rebuild indexes and sync state (hidden alias for `admin rebuild`)
    #[command(display_order = 902, hide = true)]
    Rebuild(commands::rebuild::RebuildArgs),

    /// Collection management (hidden alias for `admin collections`)
    #[command(display_order = 903, hide = true)]
    Collections(commands::collections::CollectionsArgs),

    /// Backup Qdrant collections (hidden alias for `admin backup`)
    #[command(display_order = 904, hide = true)]
    Backup(commands::backup::BackupArgs),

    /// Restore Qdrant collections from snapshots (hidden alias for `admin restore`)
    #[command(display_order = 905, hide = true)]
    Restore(commands::restore::RestoreArgs),

    /// Search instrumentation analytics (hidden alias for `admin stats`)
    #[command(display_order = 906, hide = true)]
    Stats(commands::stats::StatsArgs),
}

/// Apply CLI argument overrides to configuration and validate it.
///
/// Exits the process with code 1 if validation fails.
fn apply_cli_overrides(
    mut cfg: config::Config,
    daemon_addr: Option<String>,
    format: &str,
    verbose: bool,
) -> config::Config {
    if let Some(addr) = daemon_addr {
        cfg = cfg.with_daemon_address(addr);
    }
    if let Some(fmt) = config::OutputFormat::from_str(format) {
        cfg = cfg.with_output_format(fmt);
    }
    cfg = cfg.with_verbose(verbose);

    if let Err(e) = cfg.validate() {
        output::error(e);
        std::process::exit(1);
    }
    cfg
}

/// Main entry point with minimal tokio runtime for fast startup
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let cfg = apply_cli_overrides(
        config::Config::from_env(),
        cli.daemon_addr,
        &cli.format,
        cli.verbose,
    );

    // Set up environment PATH (expand, merge, deduplicate, save)
    // Non-fatal: warn on failure but continue CLI execution
    if let Err(e) = config::setup_environment_path() {
        if cfg.verbose {
            eprintln!("Warning: PATH setup failed: {}", e);
        }
    }

    // Execute the command
    let result = match cli.command {
        // Search & Content
        Commands::Search(args) => commands::search::execute(args).await,
        Commands::Ingest(args) => commands::ingest::execute(args).await,
        Commands::Rules(args) => commands::rules::execute(args).await,
        Commands::Scratchpad(args) => commands::scratchpad::execute(args).await,

        // Project & Library
        Commands::Project(args) => commands::project::execute(args).await,
        Commands::Library(args) => commands::library::execute(args).await,
        // Hidden alias: `wqm watch` delegates to the same handler as `wqm project watch`
        Commands::Watch(args) => commands::watch::execute(args).await,

        // Code Analysis
        Commands::Tags(args) => commands::tags::execute(args).await,
        Commands::Graph(args) => commands::graph::execute(args).await,
        Commands::Language(args) => commands::language::execute(args).await,

        // Queue & Monitoring
        Commands::Queue(args) => commands::queue::execute(args).await,
        Commands::Status(args) => commands::status::execute(args).await,
        #[cfg(feature = "tui")]
        Commands::Tui => tui::run_tui(cfg.daemon_address.clone()),

        // Service & Admin
        Commands::Service(args) => {
            let mut cmd = Cli::command();
            commands::service::execute(args, Some(&mut cmd)).await
        }
        Commands::Admin(args) => commands::admin::execute(args).await,
        Commands::Config(args) => commands::config_cmd::execute(args).await,

        // Setup & Diagnostics
        Commands::Init(args) => {
            let mut cmd = Cli::command();
            commands::init::execute(args, &mut cmd).await
        }
        Commands::Update(args) => commands::update::execute(args).await,
        Commands::Debug(args) => commands::debug::execute(args).await,
        Commands::Benchmark(args) => commands::benchmark::execute(args).await,

        // Hidden backward-compat aliases (delegate to same handlers)
        Commands::Man(args) => {
            let mut cmd = Cli::command();
            commands::man::execute(args, &mut cmd).await
        }
        Commands::Hooks(args) => commands::hooks::execute(args).await,
        Commands::Rebuild(args) => commands::rebuild::execute(args).await,
        Commands::Collections(args) => commands::collections::execute(args).await,
        Commands::Backup(args) => commands::backup::execute(args).await,
        Commands::Restore(args) => commands::restore::execute(args).await,
        Commands::Stats(args) => commands::stats::execute(args).await,
    };

    // Handle result
    if let Err(e) = result {
        output::error(format!("{:#}", e));
        std::process::exit(1);
    }

    Ok(())
}
