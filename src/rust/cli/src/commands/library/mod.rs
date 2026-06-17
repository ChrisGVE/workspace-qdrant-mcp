//! Library command - tag-based library management
//!
//! Subcommands: list, register, delete, add, search, info, status, config, rescan

mod add;
mod config;
mod helpers;
mod info;
mod ingest;
mod list;
mod recover;
mod remove;
mod rescan;
mod search;
mod set_incremental;
mod status;
mod watch_cmd;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

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
            and document counts.",
        after_long_help = "Examples:\n  \
            wqm library list                            List all libraries"
    )]
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Register a new library
    #[command(
        long_about = "Register a directory as a library for indexing. By default, file watching \
            is enabled so the daemon automatically detects changes. Use --no-watch to register \
            without watching (files must be added manually with 'library add').\n\n\
            The library is identified by its tag (a short identifier like 'docs' or 'api').",
        after_long_help = "Examples:\n  \
            wqm library register docs ./docs            Register with watching\n  \
            wqm library register api ./spec -p '*.yaml' Watch specific patterns\n  \
            wqm library register ref ./ref --no-watch   Register without watching\n  \
            wqm library register notes ./notes -m sync  Use sync mode (deletes on remove)"
    )]
    Register {
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

        /// Register without enabling file watching
        #[arg(long)]
        no_watch: bool,
    },

    /// Delete a library and all its data
    #[command(
        long_about = "Permanently remove a library, deleting both its watch configuration and \
            all associated vector data from Qdrant. This action cannot be undone.",
        after_long_help = "Examples:\n  \
            wqm library delete docs                     Delete with confirmation prompt\n  \
            wqm library delete docs --yes               Delete without confirmation"
    )]
    Delete {
        /// Library tag to delete
        tag: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// Re-point a library to a new source path
    #[command(
        long_about = "Re-point a library whose source folder moved on disk. Updates the stored \
            source path and rewrites every stored file path old->new in both SQLite and Qdrant. \
            A library's identity is its tag, so re-pointing never changes tenancy.\n\n\
            Re-running with the library already at the given path is a no-op.",
        after_long_help = "Examples:\n  \
            wqm library recover docs --new-path /new/docs --dry-run   Preview the move\n  \
            wqm library recover docs --new-path /new/docs             Apply the re-point"
    )]
    Recover {
        /// Library tag to recover
        tag: String,

        /// New source path the library moved to
        #[arg(long)]
        new_path: Option<PathBuf>,

        /// Report old->new and counts without writing
        #[arg(long)]
        dry_run: bool,
    },

    /// Add a single file to a library
    #[command(
        long_about = "Manually add a single file to a library. The file is chunked, \
            embedded, and stored in Qdrant.",
        after_long_help = "Examples:\n  \
            wqm library add ./README.md -l docs         Add file to 'docs' library\n  \
            wqm library add ./spec.pdf -l api           Add PDF to 'api' library"
    )]
    Add {
        /// Path to the document file
        #[arg(value_parser = crate::path_arg::parse_path)]
        file: PathBuf,

        /// Library tag to add the file to
        #[arg(short, long)]
        library: String,

        /// Target tokens per chunk (default: 105)
        #[arg(long, default_value_t = 105)]
        chunk_tokens: usize,

        /// Overlap tokens between chunks (default: 12)
        #[arg(long, default_value_t = 12)]
        overlap_tokens: usize,
    },

    /// Search library content (semantic)
    #[command(
        long_about = "Perform semantic search across library content using vector embeddings. \
            Returns the most relevant passages ranked by similarity.",
        after_long_help = "Examples:\n  \
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

    /// Show library information
    #[command(
        long_about = "Display detailed information about a library including path, watch \
            status, file patterns, sync mode, document count, and last scan time.",
        after_long_help = "Examples:\n  \
            wqm library info docs                       Show info for 'docs' library\n  \
            wqm library info                            Show info for all libraries"
    )]
    Info {
        /// Library tag (shows all if omitted)
        tag: Option<String>,
    },

    /// Show watch status for all libraries
    #[command(hide = true)]
    Status,

    /// Rescan and re-ingest a library
    #[command(hide = true)]
    Rescan {
        /// Library tag to rescan
        tag: String,

        /// Force re-ingestion of all files
        #[arg(short, long)]
        force: bool,
    },

    /// Configure library settings
    #[command(hide = true)]
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

    // ── Hidden commands (internal) ────────────────────────────────────────
    /// Set or clear the incremental flag on tracked files
    #[command(hide = true)]
    SetIncremental {
        #[arg(required = true, value_parser = crate::path_arg::parse_path)]
        files: Vec<PathBuf>,
        #[arg(long)]
        clear: bool,
        #[arg(long, required = true, help = "Library tag to scope the update")]
        tag: String,
    },
}

/// Execute library command
pub async fn execute(args: LibraryArgs) -> Result<()> {
    match args.command {
        LibraryCommand::List { verbose } => list::execute(verbose).await,
        LibraryCommand::Register {
            tag,
            path,
            patterns,
            mode,
            no_watch,
        } => {
            if no_watch {
                add::execute(&tag, &path, mode).await
            } else {
                watch_cmd::execute(&tag, &path, &patterns, mode).await
            }
        }
        LibraryCommand::Delete { tag, yes } => remove::execute(&tag, yes).await,
        LibraryCommand::Recover {
            tag,
            new_path,
            dry_run,
        } => recover::execute(&tag, new_path, dry_run).await,
        LibraryCommand::Add {
            file,
            library,
            chunk_tokens,
            overlap_tokens,
        } => ingest::execute(&file, &library, chunk_tokens, overlap_tokens).await,
        LibraryCommand::Search {
            query,
            library,
            limit,
        } => search::search_library(&query, library, limit).await,
        LibraryCommand::Info { tag } => info::execute(tag.as_deref()).await,
        LibraryCommand::Status => status::execute().await,
        LibraryCommand::Rescan { tag, force } => rescan::execute(&tag, force).await,
        LibraryCommand::Config {
            tag,
            mode,
            patterns,
            enable,
            disable,
            show,
        } => config::execute(&tag, mode, patterns, enable, disable, show).await,
        LibraryCommand::SetIncremental { files, clear, tag } => {
            set_incremental::execute(&files, clear, &tag).await
        }
    }
}
