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
#[command(propagate_version = true)]
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

/// CLI commands organized by priority phase
#[derive(Subcommand)]
enum Commands {
    // =========================================================================
    // Phase 1 - HIGH priority (always available)
    // =========================================================================
    /// Daemon service management (install, start, stop, status, logs)
    Service(commands::service::ServiceArgs),

    /// System administration (status, collections, health, projects)
    Admin(commands::admin::AdminArgs),

    /// Consolidated status monitoring (queue, watch, performance, messages)
    Status(commands::status::StatusArgs),

    /// Library management with tags (list, add, watch, unwatch)
    Library(commands::library::LibraryArgs),

    /// Unified queue inspector for debugging (list, show, stats, clean)
    Queue(commands::queue::QueueArgs),

    // =========================================================================
    // Phase 2 - MEDIUM priority (behind feature flag)
    // =========================================================================
    /// Search collections (project, collection, global, memory)
    #[cfg(feature = "phase2")]
    Search(commands::search::SearchArgs),

    /// Ingest documents (file, folder, web)
    #[cfg(feature = "phase2")]
    Ingest(commands::ingest::IngestArgs),

    /// Backup management - Qdrant snapshot wrapper
    #[cfg(feature = "phase2")]
    Backup(commands::backup::BackupArgs),

    /// Memory rules management (list, add, edit, remove)
    #[cfg(feature = "phase2")]
    Memory(commands::memory::MemoryArgs),

    /// Language tools - LSP and grammar (merged)
    #[cfg(feature = "phase2")]
    Language(commands::language::LanguageArgs),

    /// Project management - watch and branch (merged)
    #[cfg(feature = "phase2")]
    Project(commands::project::ProjectArgs),

    // =========================================================================
    // Phase 3 - LOW priority (behind feature flag)
    // =========================================================================
    /// Shell completion setup (bash, zsh, fish)
    #[cfg(feature = "phase3")]
    Init(commands::init::InitArgs),

    /// Extended help system
    #[cfg(feature = "phase3")]
    Help(commands::help::HelpArgs),

    /// Setup wizards for guided configuration
    #[cfg(feature = "phase3")]
    Wizard(commands::wizard::WizardArgs),
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

    // Execute the command
    let result = match cli.command {
        // Phase 1 commands
        Commands::Service(args) => commands::service::execute(args).await,
        Commands::Admin(args) => commands::admin::execute(args).await,
        Commands::Status(args) => commands::status::execute(args).await,
        Commands::Library(args) => commands::library::execute(args).await,
        Commands::Queue(args) => commands::queue::execute(args).await,

        // Phase 2 commands
        #[cfg(feature = "phase2")]
        Commands::Search(args) => commands::search::execute(args).await,
        #[cfg(feature = "phase2")]
        Commands::Ingest(args) => commands::ingest::execute(args).await,
        #[cfg(feature = "phase2")]
        Commands::Backup(args) => commands::backup::execute(args).await,
        #[cfg(feature = "phase2")]
        Commands::Memory(args) => commands::memory::execute(args).await,
        #[cfg(feature = "phase2")]
        Commands::Language(args) => commands::language::execute(args).await,
        #[cfg(feature = "phase2")]
        Commands::Project(args) => commands::project::execute(args).await,

        // Phase 3 commands
        #[cfg(feature = "phase3")]
        Commands::Init(args) => commands::init::execute(args).await,
        #[cfg(feature = "phase3")]
        Commands::Help(args) => commands::help::execute(args).await,
        #[cfg(feature = "phase3")]
        Commands::Wizard(args) => commands::wizard::execute(args).await,
    };

    // Handle result
    if let Err(e) = result {
        output::error(format!("{:#}", e));
        std::process::exit(1);
    }

    Ok(())
}
