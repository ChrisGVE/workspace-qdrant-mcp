//! Search command - semantic and hybrid search
//!
//! Phase 2 MEDIUM priority command for search operations.
//! Subcommands: project, collection, global, rules
//!
//! Note: Search operations primarily go through the MCP server which handles
//! embedding generation and Qdrant queries. The CLI provides guidance.

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::output;

/// Sentinel value meaning "search across all branches".
const BRANCH_WILDCARD: &str = "*";

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

        /// Branch to search (auto-detected from CWD if omitted; use "*" for all branches)
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

    /// Search behavioral rules
    Rules {
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
        SearchCommand::Collection {
            name,
            query,
            filter,
        } => search_collection(&name, &query, limit, filter).await,
        SearchCommand::Global { query, exclude } => search_global(&query, limit, &exclude).await,
        SearchCommand::Rules { query, scope } => search_rules(&query, limit, scope).await,
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

    // Resolve branch: explicit arg > auto-detect from CWD > cross-branch fallback
    let resolved_branch = resolve_branch(branch);

    output::kv("Query", query);
    output::kv("Limit", limit.to_string());
    output::kv("Include Libraries", include_libs.to_string());
    if let Some(ft) = &file_type {
        output::kv("File Type", ft);
    }
    match &resolved_branch {
        BranchFilter::Specific(b) => output::kv("Branch", b),
        BranchFilter::All => output::kv("Branch", "*  (all branches)"),
    }
    output::separator();

    // Check daemon connection
    match crate::grpc::connect_default().await {
        Ok(_) => {
            output::info("Daemon connected.");
            output::separator();
            output::info("Project search requires embedding generation via MCP server.");
            output::info("Use the MCP search tool:");
            output::info("  mcp__workspace_qdrant__search(");
            output::info(format!("    query=\"{}\",", query));
            output::info("    scope=\"project\",");
            if let BranchFilter::Specific(b) = &resolved_branch {
                output::info(format!("    branch=\"{}\",", b));
            }
            output::info(format!("    limit={}", limit));
            output::info("  )");
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

/// The resolved branch filter for a search operation.
enum BranchFilter {
    /// Search a specific named branch.
    Specific(String),
    /// Cross-branch search (no branch filter applied).
    All,
}

/// Resolve the branch to use for a project search.
///
/// Resolution order:
/// 1. Explicit `"*"` → cross-branch (no filter)
/// 2. Any other explicit value → use as-is
/// 3. Omitted → auto-detect from the current working directory via git
/// 4. Git unavailable or HEAD detached → cross-branch fallback
fn resolve_branch(arg: Option<String>) -> BranchFilter {
    match arg {
        Some(b) if b == BRANCH_WILDCARD => BranchFilter::All,
        Some(b) => BranchFilter::Specific(b),
        None => match detect_branch_from_cwd() {
            Some(branch) => BranchFilter::Specific(branch),
            None => BranchFilter::All,
        },
    }
}

/// Ask git for the current branch in the process's working directory.
/// Returns `None` when git is unavailable or HEAD is detached.
fn detect_branch_from_cwd() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let output = std::process::Command::new("git")
        .args(["-C", cwd.to_str()?, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch.is_empty() || branch == "HEAD" {
            None
        } else {
            Some(branch)
        }
    } else {
        None
    }
}

async fn search_collection(
    name: &str,
    query: &str,
    limit: usize,
    filter: Option<String>,
) -> Result<()> {
    output::section(format!("Collection Search: {}", name));

    output::kv("Query", query);
    output::kv("Limit", limit.to_string());
    if let Some(f) = &filter {
        output::kv("Filter", f);
    }
    output::separator();

    output::info("Collection search requires embedding generation.");
    output::info("Options:");
    output::info(
        "  1. MCP server: mcp__workspace_qdrant__search(scope=\"collection\", collection=\"...\"",
    );
    output::info("  2. Direct Qdrant with pre-computed vector:".to_string());
    output::info(format!(
        "     curl -X POST 'http://localhost:6333/collections/{}/points/search'",
        name
    ));

    Ok(())
}

async fn search_global(query: &str, limit: usize, exclude: &[String]) -> Result<()> {
    output::section("Global Search");

    output::kv("Query", query);
    output::kv("Limit", limit.to_string());
    if !exclude.is_empty() {
        output::kv("Excluding", exclude.join(", "));
    }
    output::separator();

    output::info("Global search queries all project collections.");
    output::info("Use the MCP search tool:");
    output::info("  mcp__workspace_qdrant__search(");
    output::info(format!("    query=\"{}\",", query));
    output::info("    scope=\"global\",");
    output::info(format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}

async fn search_rules(query: &str, limit: usize, scope: Option<String>) -> Result<()> {
    output::section("Rules Search");

    output::kv("Query", query);
    output::kv("Limit", limit.to_string());
    if let Some(s) = &scope {
        output::kv("Scope", s);
    }
    output::separator();

    output::info("Rules search queries the rules collection for behavioral rules.");
    output::info("Use the MCP search tool:");
    output::info("  mcp__workspace_qdrant__search(");
    output::info(format!("    query=\"{}\",", query));
    output::info("    scope=\"rules\",");
    output::info(format!("    limit={}", limit));
    output::info("  )");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_specific(f: BranchFilter, expected: &str) -> bool {
        matches!(f, BranchFilter::Specific(b) if b == expected)
    }

    fn is_all(f: BranchFilter) -> bool {
        matches!(f, BranchFilter::All)
    }

    #[test]
    fn test_resolve_branch_wildcard() {
        let f = resolve_branch(Some("*".to_string()));
        assert!(is_all(f), "\"*\" should resolve to BranchFilter::All");
    }

    #[test]
    fn test_resolve_branch_explicit_name() {
        let f = resolve_branch(Some("main".to_string()));
        assert!(
            is_specific(f, "main"),
            "explicit branch name should pass through"
        );
    }

    #[test]
    fn test_resolve_branch_explicit_slash() {
        let f = resolve_branch(Some("feature/my-work".to_string()));
        assert!(
            is_specific(f, "feature/my-work"),
            "branch with slash should pass through unchanged"
        );
    }

    #[test]
    fn test_resolve_branch_none_produces_valid_filter() {
        // When branch is None, auto-detect runs. In a git repo it returns the
        // current branch; outside git it falls back to All. Either is valid.
        let _f = resolve_branch(None);
    }
}
