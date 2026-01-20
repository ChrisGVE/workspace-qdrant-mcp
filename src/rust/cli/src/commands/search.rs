//! Search command - semantic and hybrid search
//!
//! Phase 2 MEDIUM priority command for search operations.
//! Subcommands: project, collection, global, memory
//!
//! Note: Search operations primarily go through the MCP server which handles
//! embedding generation and Qdrant queries. The CLI provides guidance.

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::output;

/// Search command arguments
#[derive(Args)]
pub struct SearchArgs {
    #[command(subcommand)]
    command: SearchCommand,

    /// Maximum number of results
    #[arg(short = 'n', long, default_value = "10", global = true)]
    limit: usize,
}

/// Search subcommands
#[derive(Subcommand)]
enum SearchCommand {
    /// Search within current project
    Project {
        /// Search query
        query: String,

        /// Include library content
        #[arg(long)]
        include_libs: bool,

        /// Filter by file type (code, doc, test, config)
        #[arg(short, long)]
        file_type: Option<String>,

        /// Filter by branch
        #[arg(short, long)]
        branch: Option<String>,
    },

    /// Search a specific collection
    Collection {
        /// Collection name
        name: String,

        /// Search query
        query: String,

        /// Filter by metadata (JSON format)
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// Search globally across all projects
    Global {
        /// Search query
        query: String,

        /// Exclude specific projects
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Search memory rules
    Memory {
        /// Search query
        query: String,

        /// Filter by scope (global, project, language)
        #[arg(short, long)]
        scope: Option<String>,
    },
}

/// Execute search command
pub async fn execute(args: SearchArgs) -> Result<()> {
    let limit = args.limit;

    match args.command {
        SearchCommand::Project {
            query,
            include_libs,
            file_type,
            branch,
        } => search_project(&query, limit, include_libs, file_type, branch).await,
        SearchCommand::Collection { name, query, filter } => {
            search_collection(&name, &query, limit, filter).await
        }
        SearchCommand::Global { query, exclude } => search_global(&query, limit, &exclude).await,
        SearchCommand::Memory { query, scope } => search_memory(&query, limit, scope).await,
    }
}

async fn search_project(
    query: &str,
    limit: usize,
    include_libs: bool,
    file_type: Option<String>,
    branch: Option<String>,
) -> Result<()> {
    output::section("Project Search");

    output::kv("Query", query);
    output::kv("Limit", &limit.to_string());
    output::kv("Include Libraries", &include_libs.to_string());
    if let Some(ft) = &file_type {
        output::kv("File Type", ft);
    }
    if let Some(b) = &branch {
        output::kv("Branch", b);
    }
    output::separator();

    // Check daemon connection
    match DaemonClient::connect_default().await {
        Ok(_) => {
            output::info("Daemon connected.");
            output::separator();
            output::info("Project search requires embedding generation via MCP server.");
            output::info("Use the MCP search tool:");
            output::info("  mcp__workspace_qdrant__search(");
            output::info(&format!("    query=\"{}\",", query));
            output::info("    scope=\"project\",");
            output::info(&format!("    limit={}", limit));
            output::info("  )");
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

async fn search_collection(
    name: &str,
    query: &str,
    limit: usize,
    filter: Option<String>,
) -> Result<()> {
    output::section(format!("Collection Search: {}", name));

    output::kv("Query", query);
    output::kv("Limit", &limit.to_string());
    if let Some(f) = &filter {
        output::kv("Filter", f);
    }
    output::separator();

    output::info("Collection search requires embedding generation.");
    output::info("Options:");
    output::info("  1. MCP server: mcp__workspace_qdrant__search(scope=\"collection\", collection=\"...\"");
    output::info(&format!(
        "  2. Direct Qdrant with pre-computed vector:"
    ));
    output::info(&format!(
        "     curl -X POST 'http://localhost:6333/collections/{}/points/search'",
        name
    ));

    Ok(())
}

async fn search_global(query: &str, limit: usize, exclude: &[String]) -> Result<()> {
    output::section("Global Search");

    output::kv("Query", query);
    output::kv("Limit", &limit.to_string());
    if !exclude.is_empty() {
        output::kv("Excluding", &exclude.join(", "));
    }
    output::separator();

    output::info("Global search queries all project collections.");
    output::info("Use the MCP search tool:");
    output::info("  mcp__workspace_qdrant__search(");
    output::info(&format!("    query=\"{}\",", query));
    output::info("    scope=\"global\",");
    output::info(&format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}

async fn search_memory(query: &str, limit: usize, scope: Option<String>) -> Result<()> {
    output::section("Memory Search");

    output::kv("Query", query);
    output::kv("Limit", &limit.to_string());
    if let Some(s) = &scope {
        output::kv("Scope", s);
    }
    output::separator();

    output::info("Memory search queries the _memory collection for LLM rules.");
    output::info("Use the MCP search tool:");
    output::info("  mcp__workspace_qdrant__search(");
    output::info(&format!("    query=\"{}\",", query));
    output::info("    scope=\"memory\",");
    output::info(&format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}
