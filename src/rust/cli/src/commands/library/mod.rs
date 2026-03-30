//! Library command - tag-based library management
//!
//! Subcommands: list, add, watch, unwatch, remove, rescan, info, status,
//!              ingest, config, set-incremental
//!
//! Note: Library management uses SQLite for watch configuration.
//! The daemon reads from SQLite and manages the actual watching.
//! Orphan cleanup has moved to: `wqm admin cleanup-orphans`

mod add;
mod config;
mod helpers;
mod info;
mod ingest;
mod list;
mod remove;
mod rescan;
mod search;
mod set_incremental;
mod status;
mod unwatch;
mod watch_cmd;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::output;
pub use helpers::LibraryMode;

/// Library command arguments
#[derive(Args)]
pub struct LibraryArgs {
    #[command(subcommand)]
    command: LibraryCommand,
}

/// Library subcommands
#[derive(Subcommand)]
enum LibraryCommand {
    /// List all libraries
    #[command(
        long_about = "Show all registered libraries with their tags, paths, watch status, \
            and document counts. Use --verbose for additional details including file \
            patterns and sync mode.",
        after_help = "Examples:\n  \
            wqm library list                            List all libraries\n  \
            wqm library list --verbose                  Show detailed information"
    )]
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Add a library (unwatched - metadata only)
    #[command(
        long_about = "Register a directory as a library without enabling file watching. The \
            library is tracked by its tag (a short identifier). Files must be ingested \
            manually with 'library ingest' or watching enabled later with 'library watch'.",
        after_help = "Examples:\n  \
            wqm library add docs ./docs                 Add with default incremental mode\n  \
            wqm library add api-spec ./spec -m sync     Add with sync mode (deletes on remove)"
    )]
    Add {
        /// Library tag (identifier)
        tag: String,

        /// Path to library content
        #[arg(value_parser = crate::path_arg::parse_path)]
        path: PathBuf,

        /// Sync mode: 'sync' (delete vectors for removed files) or 'incremental' (append-only, default)
        #[arg(short, long, value_enum, default_value_t = LibraryMode::Incremental)]
        mode: LibraryMode,
    },

    /// Watch a library path for changes
    #[command(
        long_about = "Register a directory as a watched library. The daemon will automatically \
            detect file changes and re-ingest updated content. Use --patterns to filter \
            which files are indexed.",
        after_help = "Examples:\n  \
            wqm library watch docs ./docs               Watch all files in ./docs\n  \
            wqm library watch api ./spec -p '*.yaml' -p '*.json'  Watch specific patterns\n  \
            wqm library watch notes ./notes -m sync     Watch with sync mode"
    )]
    Watch {
        /// Library tag (identifier)
        tag: String,

        /// Path to library content
        #[arg(value_parser = crate::path_arg::parse_path)]
        path: PathBuf,

        /// File patterns to include (e.g., "*.pdf", "*.md")
        #[arg(short, long)]
        patterns: Vec<String>,

        /// Sync mode: 'sync' (delete vectors for removed files) or 'incremental' (append-only, default)
        #[arg(short, long, value_enum, default_value_t = LibraryMode::Incremental)]
        mode: LibraryMode,
    },

    /// Search library content (semantic)
    #[command(
        long_about = "Perform semantic search across library content using vector embeddings. \
            Returns the most relevant passages ranked by similarity. Optionally filter to \
            a specific library by tag.",
        after_help = "Examples:\n  \
            wqm library search 'authentication flow'    Search all libraries\n  \
            wqm library search 'error codes' -l api     Search a specific library\n  \
            wqm library search 'setup guide' -n 5       Limit to 5 results"
    )]
    Search {
        /// Search query
        query: String,

        /// Filter to a specific library tag
        #[arg(short = 'l', long)]
        library: Option<String>,

        /// Maximum results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Stop watching a library
    Unwatch {
        /// Library tag to unwatch
        tag: String,
    },

    /// Remove a library (deletes watch config AND all vectors from Qdrant)
    #[command(
        long_about = "Permanently remove a library, deleting both its watch configuration in \
            SQLite and all associated vector data from Qdrant. This action cannot be undone.",
        after_help = "Examples:\n  \
            wqm library remove docs                     Remove with confirmation prompt\n  \
            wqm library remove docs --yes               Remove without confirmation"
    )]
    Remove {
        /// Library tag to remove
        tag: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// Rescan and re-ingest a library
    Rescan {
        /// Library tag to rescan
        tag: String,

        /// Force re-ingestion of all files
        #[arg(short, long)]
        force: bool,
    },

    /// Show library information
    #[command(
        long_about = "Display detailed information about a library, including its path, watch \
            status, file patterns, sync mode, document count, and last scan time. Shows \
            all libraries if no tag is specified.",
        after_help = "Examples:\n  \
            wqm library info docs                       Show info for 'docs' library\n  \
            wqm library info                            Show info for all libraries"
    )]
    Info {
        /// Library tag (optional - shows all if omitted)
        tag: Option<String>,
    },

    /// Show watch status for all libraries
    Status,

    /// Ingest a single document into a library
    #[command(
        long_about = "Manually ingest a single file into a library. The file is chunked, \
            embedded, and stored in Qdrant. Use --chunk-tokens and --overlap-tokens to \
            control chunking granularity.",
        after_help = "Examples:\n  \
            wqm library ingest ./README.md -l docs      Ingest with default chunking\n  \
            wqm library ingest ./spec.pdf -l api --chunk-tokens 200  Larger chunks"
    )]
    Ingest {
        /// Path to the document file
        #[arg(value_parser = crate::path_arg::parse_path)]
        file: PathBuf,

        /// Library tag to ingest into
        #[arg(short, long)]
        library: String,

        /// Target tokens per chunk (default: 105)
        #[arg(long, default_value_t = 105)]
        chunk_tokens: usize,

        /// Overlap tokens between chunks (default: 12)
        #[arg(long, default_value_t = 12)]
        overlap_tokens: usize,
    },

    /// Configure library settings
    Config {
        /// Library tag to configure
        tag: String,

        /// Set sync mode: 'sync' or 'incremental'
        #[arg(long)]
        mode: Option<LibraryMode>,

        /// Set file patterns (comma-separated, e.g., "*.pdf,*.md")
        #[arg(long)]
        patterns: Option<String>,

        /// Enable watching
        #[arg(long, conflicts_with = "disable")]
        enable: bool,

        /// Disable watching
        #[arg(long, conflicts_with = "enable")]
        disable: bool,

        /// Show current configuration
        #[arg(long)]
        show: bool,
    },

    /// Set or clear the incremental (do-not-delete) flag on tracked files
    SetIncremental {
        /// File path(s) to set the flag on (absolute paths)
        #[arg(required = true, value_parser = crate::path_arg::parse_path)]
        files: Vec<PathBuf>,

        /// Clear the incremental flag (allow deletions)
        #[arg(long)]
        clear: bool,
    },

    /// [Deprecated] Use `wqm admin cleanup-orphans` instead (works across all collections)
    #[command(hide = true)]
    CleanupOrphans {
        #[arg(long)]
        delete: bool,
    },
}

/// Execute library command
pub async fn execute(args: LibraryArgs) -> Result<()> {
    match args.command {
        LibraryCommand::List { verbose } => list::execute(verbose).await,
        LibraryCommand::Add { tag, path, mode } => add::execute(&tag, &path, mode).await,
        LibraryCommand::Watch {
            tag,
            path,
            patterns,
            mode,
        } => watch_cmd::execute(&tag, &path, &patterns, mode).await,
        LibraryCommand::Search {
            query,
            library,
            limit,
        } => search::search_library(&query, library, limit).await,
        LibraryCommand::Unwatch { tag } => unwatch::execute(&tag).await,
        LibraryCommand::Remove { tag, yes } => remove::execute(&tag, yes).await,
        LibraryCommand::Rescan { tag, force } => rescan::execute(&tag, force).await,
        LibraryCommand::Info { tag } => info::execute(tag.as_deref()).await,
        LibraryCommand::Status => status::execute().await,
        LibraryCommand::Ingest {
            file,
            library,
            chunk_tokens,
            overlap_tokens,
        } => ingest::execute(&file, &library, chunk_tokens, overlap_tokens).await,
        LibraryCommand::Config {
            tag,
            mode,
            patterns,
            enable,
            disable,
            show,
        } => config::execute(&tag, mode, patterns, enable, disable, show).await,
        LibraryCommand::SetIncremental { files, clear } => {
            set_incremental::execute(&files, clear).await
        }
        LibraryCommand::CleanupOrphans { delete: _ } => {
            output::warning("This command has moved. Use: wqm admin cleanup-orphans");
            output::info(
                "The new command works across all 4 collections \
                 (projects, libraries, memory, scratchpad).",
            );
            Ok(())
        }
    }
}
