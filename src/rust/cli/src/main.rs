//! WQM - Workspace Qdrant MCP CLI
//!
//! A high-performance CLI for managing workspace-qdrant-mcp daemon.
//! Designed for <100ms startup time using minimal tokio runtime.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

mod commands;
mod config;
mod error;
mod grpc;
mod output;
mod queue;

/// Custom help template with grouped subcommands
const HELP_TEMPLATE: &str = "\
{before-help}{name} {version}
{about-with-newline}
{usage-heading} {usage}

Options:
{options}
Search & Content:
  search       Search collections (project, library, rules, global)
  ingest       Ingest documents (file, folder, web)
  rules        Behavioral rules management
  scratch      Scratchpad entries

Project & Library:
  project      Project lifecycle (list, info, remove)
  library      Library management (list, add, ingest, watch, remove, config)
  watch        Watch folder management (list, enable, disable, show)
  tags         Keyword/tag management and hierarchy

Queue & Analytics:
  queue        Unified queue inspector (list, show, stats)
  stats        Search instrumentation analytics

Service & Admin:
  service      Daemon service management (start, stop, restart, status)
  status       System status monitoring (queue, watch, health)
  admin        Administrative operations
  config       Configuration management (generate, default, xdg, show, path)
  collections  Collection management (list, reset)
  language     Language tools (LSP, Tree-sitter)
  update       Update system from GitHub releases

Maintenance & Recovery:
  rebuild      Rebuild indexes and sync state (tags, search, vocabulary, keywords, rules, projects, libraries, all)

Data Management:
  backup       Backup Qdrant collections (create snapshots)
  restore      Restore Qdrant collections from snapshots

Code Graph:
  graph        Code relationship graph (query, impact, stats, pagerank, communities, betweenness, migrate)

Setup & Diagnostics:
  init         Shell completion setup (bash, zsh, fish)
  man          Man page generation and installation
  hooks        Claude Code hooks management
  debug        Diagnostic tools (logs, errors)

Benchmarking:
  benchmark    Benchmarking tools (sparse vectors, search engines)
{after-help}";

/// Workspace Qdrant MCP CLI
#[derive(Parser)]
#[command(name = "wqm")]
#[command(author, version, about = "Workspace Qdrant MCP CLI", long_about = None)]
#[command(long_version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("BUILD_NUMBER"), ")"))]
#[command(propagate_version = true)]
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
    /// Search collections (project, library, rules, global)
    #[command(display_order = 10)]
    Search(commands::search::SearchArgs),

    /// Ingest documents (file, folder, web)
    #[command(display_order = 11)]
    Ingest(commands::ingest::IngestArgs),

    /// Behavioral rules management (list, add, remove, search, scope)
    #[command(display_order = 12)]
    Rules(commands::rules::RulesArgs),

    /// Scratchpad entries (add, list)
    #[command(display_order = 13)]
    Scratch(commands::scratch::ScratchArgs),

    // --- Project & Library ---
    /// Project lifecycle (list, info, remove)
    #[command(display_order = 20)]
    Project(commands::project::ProjectArgs),

    /// Library management with tags (list, add, ingest, watch, unwatch, remove, config)
    #[command(display_order = 21)]
    Library(commands::library::LibraryArgs),

    /// Watch folder management (list, enable, disable, show)
    #[command(display_order = 22)]
    Watch(commands::watch::WatchArgs),

    /// Keyword/tag management and hierarchy inspection (list, keywords, tree, stats, search, baskets)
    #[command(display_order = 23)]
    Tags(commands::tags::TagsArgs),

    /// Code graph queries and algorithms (query, impact, stats, pagerank, communities, betweenness, migrate)
    #[command(display_order = 24)]
    Graph(commands::graph::GraphArgs),

    // --- Queue & Analytics ---
    /// Unified queue inspector (list, show, stats)
    #[command(display_order = 30)]
    Queue(commands::queue::QueueArgs),

    /// Search instrumentation analytics (overview, log-search)
    #[command(display_order = 31)]
    Stats(commands::stats::StatsArgs),

    // --- Maintenance ---
    /// Rebuild indexes and sync state (tags, search, vocabulary, keywords, rules, projects, libraries, all)
    #[command(display_order = 35)]
    Rebuild(commands::rebuild::RebuildArgs),

    // --- Service & Admin ---
    /// Daemon service management (start, stop, restart, status)
    #[command(display_order = 40)]
    Service(commands::service::ServiceArgs),

    /// Consolidated status monitoring (queue, watch, performance, health)
    #[command(display_order = 41)]
    Status(commands::status::StatusArgs),

    /// Administrative operations (rename-tenant, idle-history, prune-logs)
    #[command(display_order = 42)]
    Admin(commands::admin::AdminArgs),

    /// Configuration management (generate, show, path)
    #[command(display_order = 43)]
    Config(commands::config_cmd::ConfigCmdArgs),

    /// Collection management (list, reset)
    #[command(display_order = 44)]
    Collections(commands::collections::CollectionsArgs),

    /// Language tools - LSP and Tree-sitter (list, ts-install, ts-remove, lsp-install, lsp-remove, status)
    #[command(display_order = 45)]
    Language(commands::language::LanguageArgs),

    /// Update system from GitHub releases
    #[command(display_order = 46)]
    Update(commands::update::UpdateArgs),

    // --- Data Management ---
    /// Backup Qdrant collections (create snapshots)
    #[command(display_order = 50)]
    Backup(commands::backup::BackupArgs),

    /// Restore Qdrant collections from snapshots
    #[command(display_order = 51)]
    Restore(commands::restore::RestoreArgs),

    // --- Setup & Diagnostics ---
    /// Shell completion setup (bash, zsh, fish)
    #[command(display_order = 60)]
    Init(commands::init::InitArgs),

    /// Man page generation and installation
    #[command(display_order = 61)]
    Man(commands::man::ManArgs),

    /// Claude Code hooks management (install, uninstall, status)
    #[command(display_order = 62)]
    Hooks(commands::hooks::HooksArgs),

    /// Diagnostic tools (logs, errors, queue-errors, language)
    #[command(display_order = 63)]
    Debug(commands::debug::DebugArgs),

    /// Benchmarking tools (sparse vectors, search engines)
    #[command(display_order = 70)]
    Benchmark(commands::benchmark::BenchmarkArgs),
}

/// Main entry point with minimal tokio runtime for fast startup
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Build configuration from CLI args and environment
    let mut cfg = config::Config::from_env();

    // Override with CLI arguments
    if let Some(addr) = cli.daemon_addr {
        cfg = cfg.with_daemon_address(addr);
    }
    if let Some(fmt) = config::OutputFormat::from_str(&cli.format) {
        cfg = cfg.with_output_format(fmt);
    }
    cfg = cfg.with_verbose(cli.verbose);

    // Validate configuration
    if let Err(e) = cfg.validate() {
        output::error(e);
        std::process::exit(1);
    }

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
        Commands::Scratch(args) => commands::scratch::execute(args).await,

        // Project & Library
        Commands::Project(args) => commands::project::execute(args).await,
        Commands::Library(args) => commands::library::execute(args).await,
        Commands::Watch(args) => commands::watch::execute(args).await,
        Commands::Tags(args) => commands::tags::execute(args).await,
        Commands::Graph(args) => commands::graph::execute(args).await,

        // Queue & Analytics
        Commands::Queue(args) => commands::queue::execute(args).await,
        Commands::Stats(args) => commands::stats::execute(args).await,

        // Maintenance
        Commands::Rebuild(args) => commands::rebuild::execute(args).await,

        // Service & Admin
        Commands::Service(args) => {
            let mut cmd = Cli::command();
            commands::service::execute(args, Some(&mut cmd)).await
        },
        Commands::Status(args) => commands::status::execute(args).await,
        Commands::Admin(args) => commands::admin::execute(args).await,
        Commands::Config(args) => commands::config_cmd::execute(args).await,
        Commands::Collections(args) => commands::collections::execute(args).await,
        Commands::Language(args) => commands::language::execute(args).await,
        Commands::Update(args) => commands::update::execute(args).await,

        // Data Management
        Commands::Backup(args) => commands::backup::execute(args).await,
        Commands::Restore(args) => commands::restore::execute(args).await,

        // Setup & Diagnostics
        Commands::Init(args) => {
            let mut cmd = Cli::command();
            commands::init::execute(args, &mut cmd).await
        },
        Commands::Man(args) => {
            let mut cmd = Cli::command();
            commands::man::execute(args, &mut cmd).await
        },
        Commands::Hooks(args) => commands::hooks::execute(args).await,
        Commands::Debug(args) => commands::debug::execute(args).await,

        // Benchmarking
        Commands::Benchmark(args) => commands::benchmark::execute(args).await,
    };

    // Handle result
    if let Err(e) = result {
        output::error(format!("{:#}", e));
        std::process::exit(1);
    }

    Ok(())
}
