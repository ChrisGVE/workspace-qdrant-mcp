//! Help command - extended help system
//!
//! Phase 3 LOW priority command for extended help.
//! Provides topic-based help, examples, and quick reference.

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::output;

/// Help command arguments
#[derive(Args)]
pub struct HelpArgs {
    #[command(subcommand)]
    command: Option<HelpCommand>,

    /// Topic to get help for
    topic: Option<String>,
}

/// Help subcommands
#[derive(Subcommand)]
enum HelpCommand {
    /// Show quick reference card
    Quick,

    /// Show common examples
    Examples,

    /// Show topic-specific help
    Topic {
        /// Help topic
        name: String,
    },
}

/// Execute help command
pub async fn execute(args: HelpArgs) -> Result<()> {
    match args.command {
        Some(HelpCommand::Quick) => show_quick_reference().await,
        Some(HelpCommand::Examples) => show_examples().await,
        Some(HelpCommand::Topic { name }) => show_topic_help(&name).await,
        None => {
            if let Some(topic) = args.topic {
                show_topic_help(&topic).await
            } else {
                show_overview().await
            }
        }
    }
}

async fn show_overview() -> Result<()> {
    output::section("WQM CLI Help");

    output::info("Workspace Qdrant MCP CLI - High-performance command-line interface");
    output::separator();

    output::info("Command Groups:");
    output::kv("  service", "Daemon management (start, stop, status, logs)");
    output::kv("  admin", "System administration (status, collections, health)");
    output::kv("  status", "Monitoring (queue, watch, performance, live)");
    output::kv("  library", "Library management with tags");
    output::kv("  search", "Semantic search operations");
    output::kv("  ingest", "Document ingestion");
    output::kv("  backup", "Qdrant snapshot management");
    output::kv("  memory", "LLM rules management");
    output::kv("  language", "LSP and grammar tools");
    output::kv("  project", "Project and branch management");

    output::separator();
    output::info("Quick Start:");
    output::info("  wqm service start      # Start the daemon");
    output::info("  wqm status             # Check system status");
    output::info("  wqm search project \"query\"  # Search project");

    output::separator();
    output::info("More help:");
    output::info("  wqm help quick         # Quick reference");
    output::info("  wqm help examples      # Common examples");
    output::info("  wqm <command> --help   # Command help");

    Ok(())
}

async fn show_quick_reference() -> Result<()> {
    output::section("Quick Reference");

    output::info("=== Daemon Management ===");
    output::info("  wqm service start      Start daemon");
    output::info("  wqm service stop       Stop daemon");
    output::info("  wqm service status     Check daemon status");
    output::info("  wqm service logs -f    Follow daemon logs");

    output::separator();
    output::info("=== System Status ===");
    output::info("  wqm status             Overall status");
    output::info("  wqm status live        Live dashboard");
    output::info("  wqm admin health       Health check");

    output::separator();
    output::info("=== Search ===");
    output::info("  wqm search project \"query\"   Search current project");
    output::info("  wqm search global \"query\"    Search all projects");
    output::info("  wqm search memory \"query\"    Search memory rules");

    output::separator();
    output::info("=== Content Ingestion ===");
    output::info("  wqm ingest file <path>       Ingest single file");
    output::info("  wqm ingest text \"content\"    Ingest text directly");
    output::info("  wqm library watch <tag> <path>  Watch library folder");

    output::separator();
    output::info("=== Project Management ===");
    output::info("  wqm project list             List registered projects");
    output::info("  wqm project register         Register current project");
    output::info("  wqm project branch info      Current branch info");

    Ok(())
}

async fn show_examples() -> Result<()> {
    output::section("Common Examples");

    output::info("=== Getting Started ===");
    output::info("");
    output::info("# 1. Start the daemon");
    output::info("wqm service start");
    output::info("");
    output::info("# 2. Register your project");
    output::info("cd /path/to/project");
    output::info("wqm project register --name \"My Project\"");
    output::info("");
    output::info("# 3. Check status");
    output::info("wqm status");

    output::separator();
    output::info("=== Searching Content ===");
    output::info("");
    output::info("# Search current project");
    output::info("wqm search project \"authentication flow\"");
    output::info("");
    output::info("# Search with filters");
    output::info("wqm search project \"config\" --file-type config");
    output::info("");
    output::info("# Search across all projects");
    output::info("wqm search global \"error handling\" -n 20");

    output::separator();
    output::info("=== Library Management ===");
    output::info("");
    output::info("# Add a documentation library");
    output::info("wqm library watch rust-docs /path/to/rust/docs -p \"*.md\"");
    output::info("");
    output::info("# List libraries");
    output::info("wqm library list -v");
    output::info("");
    output::info("# Rescan a library");
    output::info("wqm library rescan rust-docs --force");

    output::separator();
    output::info("=== Memory Rules ===");
    output::info("");
    output::info("# Add a coding preference");
    output::info("wqm memory add \"Always use async/await instead of promises\" -t preference");
    output::info("");
    output::info("# Search rules");
    output::info("wqm memory search \"async\"");

    Ok(())
}

async fn show_topic_help(topic: &str) -> Result<()> {
    output::section(format!("Help: {}", topic));

    match topic.to_lowercase().as_str() {
        "daemon" | "service" => {
            output::info("The daemon (memexd) handles background processing:");
            output::info("  - File watching and automatic ingestion");
            output::info("  - Embedding generation");
            output::info("  - Qdrant operations");
            output::info("  - gRPC service for CLI communication");
            output::separator();
            output::info("Start: wqm service start");
            output::info("Stop:  wqm service stop");
            output::info("Logs:  wqm service logs -f");
        }
        "search" => {
            output::info("Search uses hybrid retrieval (semantic + keyword):");
            output::separator();
            output::info("Scopes:");
            output::info("  project    - Current project only");
            output::info("  global     - All registered projects");
            output::info("  collection - Specific collection");
            output::info("  memory     - LLM memory rules");
            output::separator();
            output::info("Note: Search requires the MCP server for embedding generation.");
        }
        "collections" => {
            output::info("Collection naming conventions:");
            output::separator();
            output::info("  {project_id}       - Project content (auto-created)");
            output::info("  _{library_name}    - Library content (underscore prefix)");
            output::info("  _memory            - LLM memory rules");
            output::separator();
            output::info("List collections: wqm admin collections");
        }
        "memory" | "rules" => {
            output::info("Memory rules guide LLM behavior:");
            output::separator();
            output::info("Types:");
            output::info("  preference  - Coding style preferences");
            output::info("  behavior    - Expected behaviors");
            output::info("  constraint  - Hard constraints");
            output::info("  pattern     - Code patterns to use");
            output::separator();
            output::info("Scopes:");
            output::info("  global      - Apply everywhere");
            output::info("  project     - Specific project");
            output::info("  language    - Specific programming language");
        }
        _ => {
            output::warning(format!("Unknown topic: {}", topic));
            output::info("Available topics: daemon, search, collections, memory");
        }
    }

    Ok(())
}
