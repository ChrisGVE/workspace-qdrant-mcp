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
pub(crate) mod path_arg;
#[cfg(feature = "tui")]
mod tui;

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
  project      Project lifecycle (list, info, register, watch, branch)
  library      Library management (list, add, ingest, watch, remove)

Queue & Monitoring:
  queue        Queue inspector (list, show, stats, cancel)
  status       System status and monitoring
  tui          Interactive terminal UI

Service & Admin:
  service      Daemon management (start, stop, restart, status)
  admin        Administration (collections, backup, restore, rebuild, stats, perf, metrics)
  config       Configuration management

Code Analysis:
  graph        Code relationship graph
  language     Language tools (LSP, Tree-sitter)
  tags         Keyword/tag hierarchy

Setup & Diagnostics:
  init         Setup tools (completions, man pages, hooks)
  update       Update from GitHub releases
  debug        Diagnostic tools (logs, errors)
  benchmark    Benchmarking tools
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
    /// Project lifecycle (list, info, register, watch, branch)
    #[command(display_order = 20)]
    Project(commands::project::ProjectArgs),

    /// Library management (list, add, ingest, watch, remove)
    #[command(display_order = 21)]
    Library(commands::library::LibraryArgs),

    /// Watch folder management (hidden alias for `project watch`)
    #[command(display_order = 22, hide = true)]
    Watch(commands::watch::WatchArgs),

    // --- Code Analysis ---
    /// Keyword/tag hierarchy (list, keywords, tree, stats, search, baskets)
    #[command(display_order = 23)]
    Tags(commands::tags::TagsArgs),

    /// Code relationship graph (query, impact, stats, pagerank, communities, betweenness, migrate)
    #[command(display_order = 24)]
    Graph(commands::graph::GraphArgs),

    /// Language tools (LSP, Tree-sitter)
    #[command(display_order = 25)]
    Language(commands::language::LanguageArgs),

    // --- Queue & Monitoring ---
    /// Queue inspector (list, show, stats, cancel)
    #[command(display_order = 30)]
    Queue(commands::queue::QueueArgs),

    /// System status and monitoring
    #[command(display_order = 31)]
    Status(commands::status::StatusArgs),

    /// Interactive terminal UI for browsing and monitoring
    #[command(display_order = 32)]
    #[cfg(feature = "tui")]
    Tui,

    // --- Service & Admin ---
    /// Daemon management (start, stop, restart, status)
    #[command(display_order = 40)]
    Service(commands::service::ServiceArgs),

    /// Administration (collections, backup, restore, rebuild, stats, perf, metrics)
    #[command(display_order = 41)]
    Admin(commands::admin::AdminArgs),

    /// Configuration management
    #[command(display_order = 42)]
    Config(commands::config_cmd::ConfigCmdArgs),

    // --- Setup & Diagnostics ---
    /// Setup tools (completions, man pages, hooks)
    #[command(display_order = 60)]
    Init(commands::init::InitArgs),

    /// Update from GitHub releases
    #[command(display_order = 61)]
    Update(commands::update::UpdateArgs),

    /// Diagnostic tools (logs, errors, queue-errors, language)
    #[command(display_order = 62)]
    Debug(commands::debug::DebugArgs),

    /// Benchmarking tools (sparse vectors, search engines)
    #[command(display_order = 63)]
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
        Commands::Scratch(args) => commands::scratch::execute(args).await,

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
