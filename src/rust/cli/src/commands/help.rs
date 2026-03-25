//! Help command - extended help system
//!
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
    output::kv(
        "  search",
        "Search collections (project, library, rules, global)",
    );
    output::kv("  ingest", "Ingest documents (file, folder, web)");
    output::kv("  rules", "Behavioral rules management");
    output::kv("  scratch", "Scratchpad entries");
    output::kv(
        "  project",
        "Project lifecycle (list, info, register, watch, branch)",
    );
    output::kv(
        "  library",
        "Library management (list, add, ingest, watch, remove)",
    );
    output::kv("  queue", "Queue inspector (list, show, stats, cancel)");
    output::kv("  status", "System status and monitoring");
    output::kv(
        "  service",
        "Daemon management (start, stop, restart, status)",
    );
    output::kv(
        "  admin",
        "Administration (collections, backup, restore, rebuild, stats, perf, metrics)",
    );
    output::kv("  config", "Configuration management");
    output::kv("  graph", "Code relationship graph");
    output::kv("  language", "Language tools (LSP, Tree-sitter)");
    output::kv("  tags", "Keyword/tag hierarchy");
    output::kv("  init", "Setup tools (completions, man pages, hooks)");
    output::kv("  update", "Update from GitHub releases");
    output::kv("  debug", "Diagnostic tools (logs, errors)");
    output::kv("  benchmark", "Benchmarking tools");

    output::separator();
    output::info("Quick Start:");
    output::info("  wqm service start           # Start the daemon");
    output::info("  wqm status                  # Check system status");
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

    output::separator();
    output::info("=== System Status ===");
    output::info("  wqm status             Overall status");
    output::info("  wqm status live        Live dashboard");
    output::info("  wqm admin health       Health check");
    output::info("  wqm admin perf         Pipeline performance stats");

    output::separator();
    output::info("=== Search ===");
    output::info("  wqm search project \"query\"   Search current project");
    output::info("  wqm search global \"query\"    Search all projects");
    output::info("  wqm search rules \"query\"     Search behavioral rules");

    output::separator();
    output::info("=== Content Ingestion ===");
    output::info("  wqm ingest file <path>       Ingest single file");
    output::info("  wqm ingest text \"content\"    Ingest text directly");
    output::info("  wqm library watch <tag> <path>  Watch library folder");

    output::separator();
    output::info("=== Project Management ===");
    output::info("  wqm project list             List registered projects");
    output::info("  wqm project register         Register current project");
    output::info("  wqm project watch list       List watch folders");
    output::info("  wqm project branch info      Current branch info");

    output::separator();
    output::info("=== Administration ===");
    output::info("  wqm admin collections        Manage collections");
    output::info("  wqm admin backup             Backup Qdrant snapshots");
    output::info("  wqm admin restore            Restore from snapshots");
    output::info("  wqm admin rebuild            Rebuild indexes");
    output::info("  wqm admin stats              Search analytics");

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

    output::separator();
    output::info("=== Behavioral Rules ===");
    output::info("");
    output::info("# Add a coding preference");
    output::info("wqm rules add \"Always use async/await\" -t preference");
    output::info("");
    output::info("# Search rules");
    output::info("wqm rules search \"async\"");

    Ok(())
}

async fn show_topic_help(topic: &str) -> Result<()> {
    output::section(format!("Help: {topic}"));

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
        }
        "search" => {
            output::info("Search uses hybrid retrieval (semantic + keyword):");
            output::separator();
            output::info("Scopes:");
            output::info("  project    - Current project only");
            output::info("  global     - All registered projects");
            output::info("  collection - Specific collection");
            output::info("  rules      - Behavioral rules");
            output::separator();
            output::info("Note: Search requires the MCP server for embedding generation.");
        }
        "collections" => {
            output::info("Collection naming conventions:");
            output::separator();
            output::info("  {project_id}       - Project content (auto-created)");
            output::info("  _{library_name}    - Library content (underscore prefix)");
            output::info("  _rules             - Behavioral rules");
            output::info("  _scratchpad        - Scratchpad entries");
            output::separator();
            output::info("Manage collections: wqm admin collections");
        }
        "rules" => {
            output::info("Behavioral rules guide LLM behavior:");
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
        "admin" => {
            output::info("Administrative commands under `wqm admin`:");
            output::separator();
            output::info("  collections    - Collection management (list, reset)");
            output::info("  backup         - Backup Qdrant snapshots");
            output::info("  restore        - Restore from snapshots");
            output::info("  rebuild        - Rebuild indexes and sync state");
            output::info("  stats          - Search instrumentation analytics");
            output::info("  perf           - Pipeline performance statistics");
            output::info("  metrics        - Prometheus metrics management");
            output::info("  rename-tenant  - Rename a tenant across tables");
            output::info("  cleanup-orphans - Detect/delete orphaned tenants");
        }
        _ => {
            output::warning(format!("Unknown topic: {topic}"));
            output::info("Available topics: daemon, search, collections, rules, admin");
        }
    }

    Ok(())
}
