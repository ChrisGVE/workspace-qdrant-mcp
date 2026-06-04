//! WQM - Workspace Qdrant MCP Companion
//!
//! A high-performance companion for the workspace-qdrant-mcp daemon.
//! Designed for <100ms startup time using minimal tokio runtime.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

mod commands;
mod config;
pub(crate) mod data;
mod error;
mod grpc;
mod output;
pub(crate) mod path_arg;
#[cfg(feature = "tui")]
mod tui;

/// Custom help template with grouped subcommands.
///
/// Per PRD: title first, then version; blank line between options and commands;
/// "Commands by domain:" label; bold command names via ANSI codes.
const HELP_TEMPLATE: &str = "\
{before-help}{about}
version {version}

USAGE: {usage}

OPTIONS:
{options}
Commands by domain:

  INTERACTIVE:
    \x1b[1mtui\x1b[0m          Interactive terminal UI

  CONTENT:
    \x1b[1mproject\x1b[0m      Project management (list, status, register, delete, search, groups)
    \x1b[1mlibrary\x1b[0m      Library management (list, register, delete, add, search)
    \x1b[1mrules\x1b[0m        Behavioral rules (list, add, remove, search)
    \x1b[1mscratchpad\x1b[0m   Scratchpad entries (list, search)

  QUEUE & MONITORING:
    \x1b[1mqueue\x1b[0m        Queue inspector (list, show, stats, cancel)
    \x1b[1mstatus\x1b[0m       System status and monitoring

  SERVICE & ADMIN:
    \x1b[1mservice\x1b[0m      Daemon management (start, stop, restart, status)
    \x1b[1madmin\x1b[0m        Administration (collections, backup, restore, rebuild)
    \x1b[1mconfig\x1b[0m       Configuration management
    \x1b[1mdocker\x1b[0m       Docker deployment helpers (compose generation)

  CODE ANALYSIS:
    \x1b[1mgraph\x1b[0m        Code relationship graph
    \x1b[1mgroups\x1b[0m       Project group memberships
    \x1b[1mlanguage\x1b[0m     Language tools (LSP, Tree-sitter)
    \x1b[1mtags\x1b[0m         Keyword/tag hierarchy

  SETUP & DIAGNOSTICS:
    \x1b[1minit\x1b[0m         Setup (shell completions, hooks)
    \x1b[1mupdate\x1b[0m       Update from GitHub releases
    \x1b[1mdebug\x1b[0m        Diagnostic tools (logs, errors)
    \x1b[1mbenchmark\x1b[0m    Benchmarking tools

Use --help for detailed information on any command.
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

    /// Behavioral rules management (list, add, remove, search)
    #[command(
        display_order = 12,
        long_about = "Manage behavioral rules that guide AI assistant behavior. Rules are stored \
            in the rules collection and can be scoped globally or per-project. Use rules to \
            encode preferences, constraints, and patterns that persist across sessions.",
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
            wqm scratchpad list                         List all entries\n  \
            wqm scratchpad list --project .             List entries for current project\n  \
            wqm scratchpad search 'architecture'        Semantic search across entries"
    )]
    Scratchpad(commands::scratchpad::ScratchArgs),

    // --- Project & Library ---
    /// Project management (list, status, register, delete, search, groups)
    #[command(
        display_order = 20,
        long_about = "Manage projects tracked by the workspace-qdrant daemon. Projects are \
            automatically detected via Git repositories and indexed for semantic search. \
            Use subcommands to list, inspect status, register new projects, search content, and view group memberships.",
        after_long_help = "Examples:\n  \
            wqm project list                            List all registered projects\n  \
            wqm project status                          Status for current project\n  \
            wqm project register .                      Register current directory\n  \
            wqm project delete                          Delete current project\n  \
            wqm project search 'TODO'                   Full-text search in project\n  \
            wqm project groups                          Show group memberships"
    )]
    Project(commands::project::ProjectArgs),

    /// Library management (list, register, delete, add, search)
    #[command(
        display_order = 21,
        long_about = "Manage reference libraries for documentation, specs, and external content. \
            Libraries are indexed into the libraries collection for semantic search. \
            Supports file watching for automatic re-ingestion on changes.",
        after_long_help = "Examples:\n  \
            wqm library list                            List all libraries\n  \
            wqm library register docs ./docs            Register a library\n  \
            wqm library add ./README.md -l docs         Add a document to a library\n  \
            wqm library search 'authentication'         Semantic search across libraries\n  \
            wqm library delete docs                     Delete a library and its data"
    )]
    Library(commands::library::LibraryArgs),

    /// Watch folder management (hidden alias for `project watch`)
    #[command(display_order = 22, hide = true)]
    Watch(commands::watch::WatchArgs),

    // --- Code Analysis ---
    /// Keyword/tag hierarchy (list, keywords, tree, stats, search, baskets, summary)
    #[command(
        display_order = 23,
        long_about = "Browse and search the keyword/tag hierarchy extracted from indexed code. \
            Tags represent semantic concepts (functions, classes, modules) organized in a \
            tree structure. Use to explore code organization and find related symbols.",
        after_long_help = "Examples:\n  \
            wqm tags list                               List top-level tags\n  \
            wqm tags tree                               Show full tag hierarchy\n  \
            wqm tags search 'parse'                     Search tags by name\n  \
            wqm tags stats                              Show tag statistics\n  \
            wqm tags keywords                           List extracted keywords\n  \
            wqm tags summary --tenant <id>              Tag frequency summary"
    )]
    Tags(commands::tags::TagsArgs),

    /// Project group memberships (list)
    #[command(
        display_order = 24,
        long_about = "View project group memberships discovered by the grouping scheduler. \
            Groups represent relationships between projects such as shared dependencies, \
            workspace co-location, Git organization, or tag affinity.",
        after_long_help = "Examples:\n  \
            wqm groups list                             List all group memberships\n  \
            wqm groups list --tenant <id>               Filter by tenant ID\n  \
            wqm groups list --type affinity             Filter by group type\n  \
            wqm groups list --json                      Output as JSON"
    )]
    Groups(commands::groups::GroupsArgs),

    /// Code relationship graph (query, impact, stats, pagerank, communities, betweenness, migrate)
    #[command(
        display_order = 25,
        long_about = "Analyze code relationships using the dependency graph built from indexed \
            projects. Supports PageRank for importance ranking, community detection for \
            module clustering, betweenness centrality for bridge identification, and \
            impact analysis for change propagation.",
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
            wqm status                                  Show overall system status\n  \
            wqm status health                           Detailed health diagnostics\n  \
            wqm status watch                            Show active file watchers\n  \
            wqm status history -r 1h                    Show metrics history\n  \
            wqm status messages                         Show system messages"
    )]
    Status(commands::status::StatusArgs),

    /// Interactive terminal UI for browsing and monitoring
    #[command(
        display_order = 32,
        long_about = "Launch an interactive terminal UI for browsing projects, libraries, \
            queue status, and system health. Provides a real-time dashboard view of the \
            workspace-qdrant system.",
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
            wqm config show                             Show current configuration\n  \
            wqm config path                             Show config file path"
    )]
    Config(commands::config_cmd::ConfigCmdArgs),

    /// Docker deployment helpers (compose generation)
    #[command(
        display_order = 43,
        long_about = "Helpers for managing wqm under Docker. Currently provides \
            `generate-compose`, which derives a docker-compose override from the \
            live config.yaml mount map (spec 16 §9). The override embeds a \
            content-hash header used by the daemon entrypoint and `--check` \
            mode to detect drift.",
        after_long_help = "Examples:\n  \
            wqm docker generate-compose                 Generate the override\n  \
            wqm docker generate-compose --check         Detect drift (exit 1 if stale)\n  \
            wqm docker generate-compose --clean         Delete the override"
    )]
    Docker(commands::docker::DockerArgs),

    // --- Setup & Diagnostics ---
    /// Setup (shell completions, hooks)
    #[command(
        display_order = 60,
        long_about = "Initial setup utilities: generate shell completions for zsh/bash/fish, \
            install man pages, and configure Claude Code hooks for automatic rule injection \
            at session start.",
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
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
        after_long_help = "Examples:\n  \
            wqm debug logs                              View recent daemon logs\n  \
            wqm debug errors                            Show recent errors\n  \
            wqm debug queue-errors                      Show queue processing failures\n  \
            wqm debug language                          Diagnose language support issues"
    )]
    Debug(commands::debug::DebugArgs),

    /// Benchmarking tools (sparse vectors, search engines) — `bench` feature only
    #[cfg(feature = "bench")]
    #[command(
        display_order = 63,
        long_about = "Run performance benchmarks for sparse vector generation and search \
            engine components. Results help tune chunking parameters and identify \
            performance bottlenecks.",
        after_long_help = "Examples:\n  \
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
        Commands::Groups(args) => commands::groups::execute(args).await,
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
        Commands::Docker(args) => commands::docker::execute(args).await,

        // Setup & Diagnostics
        Commands::Init(args) => {
            let mut cmd = Cli::command();
            commands::init::execute(args, &mut cmd).await
        }
        Commands::Update(args) => commands::update::execute(args).await,
        Commands::Debug(args) => commands::debug::execute(args).await,
        #[cfg(feature = "bench")]
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
