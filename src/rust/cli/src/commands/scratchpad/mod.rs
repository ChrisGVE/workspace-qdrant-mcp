//! Scratchpad command - persistent LLM scratch space
//!
//! Manages scratchpad entries stored in the `scratchpad` collection.
//! Subcommands: add, list, search, info

mod add;
mod client;
mod info;
mod list;
mod manage;
mod search;
mod types;

use anyhow::Result;
use clap::{Args, Subcommand};

/// Scratchpad command arguments
#[derive(Args)]
pub struct ScratchArgs {
    #[command(subcommand)]
    command: ScratchCommand,
}

/// Scratchpad subcommands
#[derive(Subcommand)]
enum ScratchCommand {
    /// Search scratchpad entries (semantic)
    #[command(
        long_about = "Perform semantic search across scratchpad entries using vector embeddings. \
            Returns entries ranked by relevance to the query. Optionally filter to a \
            specific project.",
        after_long_help = "Examples:\n  \
            wqm scratchpad search 'architecture decisions'   Semantic search\n  \
            wqm scratchpad search 'auth flow' --project .    Filter to current project\n  \
            wqm scratchpad search 'design' -n 5              Limit to 5 results"
    )]
    Search {
        /// Search query
        query: String,

        /// Filter to a specific project
        #[arg(short, long)]
        project: Option<String>,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Show detailed information about a scratchpad entry
    #[command(
        long_about = "Display full details for a scratchpad entry, including title, content, \
            tags, tenant, and creation time. Searches by title substring.",
        after_long_help = "Examples:\n  \
            wqm scratchpad info 'auth design'           Look up by title\n  \
            wqm scratchpad info 'auth design' --json    Output as JSON"
    )]
    Info {
        /// Entry title or substring to search for
        identifier: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Add a scratchpad entry (use MCP for programmatic access)
    #[command(hide = true)]
    Add {
        /// Content text
        content: String,

        /// Optional title
        #[arg(short, long)]
        title: Option<String>,

        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,

        /// Project ID or path (defaults to _global_)
        #[arg(short, long)]
        project: Option<String>,
    },

    /// Delete a scratchpad entry
    #[command(
        long_about = "Permanently delete one scratchpad entry, identified by title. \
            Requires typing the exact \"Delete <title>\" confirmation unless --yes is given.",
        after_long_help = "Examples:\n  \
            wqm scratchpad delete 'auth design'         Delete (typed confirmation)\n  \
            wqm scratchpad delete 'auth design' --yes   Delete without prompting"
    )]
    Delete {
        /// Entry title to delete
        identifier: String,

        /// Skip the typed "Delete <title>" confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Update (amend) a scratchpad entry
    #[command(
        long_about = "Amend one scratchpad entry, identified by title. Any combination of \
            --content, --title, and --tags can be changed; omitted fields keep their \
            current value.",
        after_long_help = "Examples:\n  \
            wqm scratchpad update 'auth design' --content 'new text'   Replace content\n  \
            wqm scratchpad update 'auth design' --tags arch,auth       Replace tags\n  \
            wqm scratchpad update 'auth design' --title 'auth notes'   Rename"
    )]
    Update {
        /// Entry title to update
        identifier: String,

        /// New content (replaces the stored content)
        #[arg(long)]
        content: Option<String>,

        /// New title
        #[arg(long)]
        title: Option<String>,

        /// New tags (comma-separated, replaces existing tags)
        #[arg(long)]
        tags: Option<String>,
    },

    /// Reassign a scratchpad entry to another project (or global)
    #[command(
        long_about = "Move one scratchpad entry, identified by title, to a different scope: \
            a specific project (path or tenant id) or the global scope when --to is omitted.",
        after_long_help = "Examples:\n  \
            wqm scratchpad reassign 'auth design' --to .          Move to current project\n  \
            wqm scratchpad reassign 'auth design'                 Move to global scope"
    )]
    Reassign {
        /// Entry title to reassign
        identifier: String,

        /// Target project (path or tenant id; omit for global)
        #[arg(long)]
        to: Option<String>,
    },

    /// List scratchpad entries
    #[command(
        long_about = "Display all scratchpad entries, optionally filtered by project. Shows \
            title, tenant, tags, and creation time. Use --verbose for full content.",
        after_long_help = "Examples:\n  \
            wqm scratchpad list                         List all entries\n  \
            wqm scratchpad list --project .             Filter to current project\n  \
            wqm scratchpad list --verbose               Show full content\n  \
            wqm scratchpad list --format json           Output as JSON\n  \
            wqm scratchpad list --script --no-headers   Machine-readable output"
    )]
    List {
        /// Project ID or path (defaults to showing all)
        #[arg(short, long)]
        project: Option<String>,

        /// Maximum entries to show
        #[arg(short = 'n', long, default_value = "50")]
        limit: usize,

        /// Show detailed info including full content
        #[arg(short, long)]
        verbose: bool,

        /// Output format: table (default) or json
        #[arg(short, long, default_value = "table")]
        format: String,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long)]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },
}

/// Execute scratchpad command
pub async fn execute(args: ScratchArgs) -> Result<()> {
    match args.command {
        ScratchCommand::Search {
            query,
            project,
            limit,
        } => search::search_entries(&query, project, limit).await,
        ScratchCommand::Info { identifier, json } => info::scratchpad_info(&identifier, json).await,
        ScratchCommand::Add {
            content,
            title,
            tags,
            project,
        } => add::add_entry(content, title, tags, project).await,
        ScratchCommand::Delete { identifier, yes } => manage::delete_entry(&identifier, yes).await,
        ScratchCommand::Update {
            identifier,
            content,
            title,
            tags,
        } => manage::update_entry(&identifier, content, title, tags).await,
        ScratchCommand::Reassign { identifier, to } => {
            manage::reassign_entry(&identifier, to).await
        }
        ScratchCommand::List {
            project,
            limit,
            verbose,
            format,
            script,
            no_headers,
        } => list::list_entries(project, limit, verbose, &format, script, no_headers).await,
    }
}
