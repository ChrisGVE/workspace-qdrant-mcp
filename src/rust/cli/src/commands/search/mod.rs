//! Search command -- real hybrid search from the CLI (#125).
//!
//! Located at: `src/rust/cli/src/commands/search/mod.rs`
//!
//! Subcommands: project, collection, global, rules. All execute the shared
//! `wqm_client::search` pipeline (embed via daemon -> dense+sparse Qdrant ->
//! RRF fusion) -- the same code path the MCP server's `search` tool runs.
//!
//! **Search scope (F17):**
//!
//! | Subcommand   | Effective scope | Notes                                        |
//! |--------------|-----------------|----------------------------------------------|
//! | `project`    | `scope=project` | Current project only.                        |
//! | `global`     | `scope=all`     | All registered projects. Blocked above the   |
//! |              |                 | 50-project cliff with a ScopeTooBroad error. |
//! |              |                 | Banner: "Scope too broad: N projects > cliff |
//! |              |                 | K -- retry with --scope group". Exit non-0.  |
//! | `collection` | n/a             | Named collection, no project scope.          |
//! | `rules`      | n/a             | Rules collection only.                       |
//!
//! The `scope=group` value is available via the MCP `search` tool (searches all
//! projects sharing a `project_groups` group with the current project). The CLI
//! `global` subcommand maps to `scope=all`; a dedicated `group` subcommand is
//! a planned addition.
//!
//! The cliff (default 50) is configurable; see `FanoutConfig` in
//! `storage/src/facade/read/fanout.rs` and arch §8.
//!
//! Neighbors: `hybrid.rs` (pipeline glue + project/scope resolution),
//! `render.rs` (terminal output).

pub mod hybrid;
mod render;

use anyhow::Result;
use clap::{Args, Subcommand};

use wqm_client::models::SearchScope;
use wqm_client::search::options::{SearchInput, SearchOptions};

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
        /// Collection name (projects, libraries, rules, scratchpad)
        name: String,

        /// Search query
        query: String,
    },

    /// Search globally across all projects
    Global {
        /// Search query
        query: String,

        /// Exclude specific projects by tenant id
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Search behavioral rules
    Rules {
        /// Search query
        query: String,
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
        SearchCommand::Collection { name, query } => search_collection(&name, &query, limit).await,
        SearchCommand::Global { query, exclude } => search_global(&query, limit, &exclude).await,
        SearchCommand::Rules { query } => search_collection("rules", &query, limit).await,
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

    let Some(project_id) = hybrid::resolve_project_id_from_cwd() else {
        output::error("Current directory is not inside a registered project.");
        output::info("Register it first:  wqm project register");
        output::info("Or search everything:  wqm search global \"<query>\"");
        return Ok(());
    };

    // Resolve branch: explicit arg > auto-detect from CWD > cross-branch fallback
    let resolved_branch = resolve_branch(branch);

    output::kv("Query", query);
    match &resolved_branch {
        BranchFilter::Specific(b) => output::kv("Branch", b),
        BranchFilter::All => output::kv("Branch", "*  (all branches)"),
    }
    output::separator();

    let input = SearchInput {
        query: query.to_string(),
        limit: Some(limit),
        scope: Some(SearchScope::Project),
        branch: match resolved_branch {
            BranchFilter::Specific(b) => Some(b),
            BranchFilter::All => None,
        },
        file_type,
        include_libraries: Some(include_libs),
        ..Default::default()
    };
    let opts = SearchOptions::from_input(input, None);

    let resp = hybrid::run_hybrid_search(&opts, Some(&project_id)).await?;
    render::print_response(&resp);
    Ok(())
}

async fn search_collection(name: &str, query: &str, limit: usize) -> Result<()> {
    output::section(format!("Collection Search: {}", name));
    output::kv("Query", query);
    output::separator();

    let input = SearchInput {
        query: query.to_string(),
        limit: Some(limit),
        collection: Some(name.to_string()),
        scope: Some(SearchScope::All),
        ..Default::default()
    };
    let opts = SearchOptions::from_input(input, None);

    let resp = hybrid::run_hybrid_search(&opts, None).await?;
    render::print_response(&resp);
    Ok(())
}

async fn search_global(query: &str, limit: usize, exclude: &[String]) -> Result<()> {
    output::section("Global Search");
    output::kv("Query", query);
    if !exclude.is_empty() {
        output::kv("Excluding", exclude.join(", "));
    }
    output::separator();

    let input = SearchInput {
        query: query.to_string(),
        limit: Some(limit),
        scope: Some(SearchScope::All),
        ..Default::default()
    };
    let opts = SearchOptions::from_input(input, None);

    // Relevance decay for scope=all needs a reference project — use the cwd
    // project when available; global search works without one.
    let project_id = hybrid::resolve_project_id_from_cwd();

    let mut resp = hybrid::run_hybrid_search(&opts, project_id.as_deref()).await?;

    // Tenant exclusion is a client-side post-filter: Qdrant filters support
    // inclusion only at the shared FilterParams level.
    if !exclude.is_empty() {
        resp.results.retain(|r| {
            r.metadata
                .get("tenant_id")
                .and_then(|v| v.as_str())
                .is_none_or(|t| !exclude.iter().any(|e| e == t))
        });
        resp.total = resp.results.len();
    }

    render::print_response(&resp);
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
