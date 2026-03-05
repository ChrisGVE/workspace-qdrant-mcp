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
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Add a library (unwatched - metadata only)
    Add {
        /// Library tag (identifier)
        tag: String,

        /// Path to library content
        path: PathBuf,

        /// Sync mode: 'sync' (delete vectors for removed files) or 'incremental' (append-only, default)
        #[arg(short, long, value_enum, default_value_t = LibraryMode::Incremental)]
        mode: LibraryMode,
    },

    /// Watch a library path for changes
    Watch {
        /// Library tag (identifier)
        tag: String,

        /// Path to library content
        path: PathBuf,

        /// File patterns to include (e.g., "*.pdf", "*.md")
        #[arg(short, long)]
        patterns: Vec<String>,

        /// Sync mode: 'sync' (delete vectors for removed files) or 'incremental' (append-only, default)
        #[arg(short, long, value_enum, default_value_t = LibraryMode::Incremental)]
        mode: LibraryMode,
    },

    /// Stop watching a library
    Unwatch {
        /// Library tag to unwatch
        tag: String,
    },

    /// Remove a library (deletes watch config AND all vectors from Qdrant)
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
    Info {
        /// Library tag (optional - shows all if omitted)
        tag: Option<String>,
    },

    /// Show watch status for all libraries
    Status,

    /// Ingest a single document into a library
    Ingest {
        /// Path to the document file
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
        #[arg(required = true)]
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
