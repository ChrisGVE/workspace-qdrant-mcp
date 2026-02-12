//! WQM - Workspace Qdrant MCP CLI
//!
//! A high-performance CLI for managing workspace-qdrant-mcp daemon.
//! Designed for <100ms startup time using minimal tokio runtime.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;
mod config;
mod error;
mod grpc;
mod output;
mod queue;

/// Workspace Qdrant MCP CLI
#[derive(Parser)]
#[command(name = "wqm")]
#[command(author, version, about = "Workspace Qdrant MCP CLI", long_about = None)]
#[command(long_version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("BUILD_NUMBER"), ")"))]
#[command(propagate_version = true)]
#[command(arg_required_else_help = true)]
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

/// CLI commands
#[derive(Subcommand)]
enum Commands {
    // =========================================================================
    // Service & Status
    // =========================================================================
    /// Daemon service management (start, stop, restart, status)
    Service(commands::service::ServiceArgs),

    /// Consolidated status monitoring (queue, watch, performance, health)
    Status(commands::status::StatusArgs),

    // =========================================================================
    // Content Management
    // =========================================================================
    /// Library management with tags (list, add, ingest, watch, unwatch, remove, config)
    Library(commands::library::LibraryArgs),

    /// Project lifecycle (list, info, remove)
    Project(commands::project::ProjectArgs),

    /// Memory rules management (list, add, remove, update, search, scope)
    Memory(commands::memory::MemoryArgs),

    // =========================================================================
    // Search & Queue
    // =========================================================================
    /// Search collections (project, library, memory, global)
    Search(commands::search::SearchArgs),

    /// Unified queue inspector (list, show, stats)
    Queue(commands::queue::QueueArgs),

    // =========================================================================
    // Language Support
    // =========================================================================
    /// Language tools - LSP and Tree-sitter (list, ts-install, ts-remove, lsp-install, lsp-remove, status)
    Language(commands::language::LanguageArgs),

    // =========================================================================
    // System Administration
    // =========================================================================
    /// Update system from GitHub releases
    Update(commands::update::UpdateArgs),

    /// Backup Qdrant collections (create snapshots)
    Backup(commands::backup::BackupArgs),

    /// Restore Qdrant collections from snapshots
    Restore(commands::restore::RestoreArgs),

    /// Ingest documents (file, folder, web)
    Ingest(commands::ingest::IngestArgs),

    /// Watch folder management (list, enable, disable, show)
    Watch(commands::watch::WatchArgs),

    // =========================================================================
    // Diagnostics & Setup
    // =========================================================================
    /// Diagnostic tools (logs, errors, queue-errors, language)
    Debug(commands::debug::DebugArgs),

    /// Claude Code hooks management (install, uninstall, status)
    Hooks(commands::hooks::HooksArgs),

    /// Shell completion setup (bash, zsh, fish)
    Init(commands::init::InitArgs),
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
        // Service & Status
        Commands::Service(args) => commands::service::execute(args).await,
        Commands::Status(args) => commands::status::execute(args).await,

        // Content Management
        Commands::Library(args) => commands::library::execute(args).await,
        Commands::Project(args) => commands::project::execute(args).await,
        Commands::Memory(args) => commands::memory::execute(args).await,

        // Search & Queue
        Commands::Search(args) => commands::search::execute(args).await,
        Commands::Queue(args) => commands::queue::execute(args).await,

        // Language Support
        Commands::Language(args) => commands::language::execute(args).await,

        // System Administration
        Commands::Update(args) => commands::update::execute(args).await,
        Commands::Backup(args) => commands::backup::execute(args).await,
        Commands::Restore(args) => commands::restore::execute(args).await,
        Commands::Ingest(args) => commands::ingest::execute(args).await,
        Commands::Watch(args) => commands::watch::execute(args).await,

        // Diagnostics & Setup
        Commands::Hooks(args) => commands::hooks::execute(args).await,
        Commands::Debug(args) => commands::debug::execute(args).await,
        Commands::Init(args) => commands::init::execute(args).await,
    };

    // Handle result
    if let Err(e) = result {
        output::error(format!("{:#}", e));
        std::process::exit(1);
    }

    Ok(())
}
