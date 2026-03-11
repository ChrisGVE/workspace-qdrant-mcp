//! Ingest command - document ingestion
//!
//! Phase 2 MEDIUM priority command for document processing.
//! Subcommands: file, folder, text, url, status
//!
//! Ingestion routes through the daemon which handles embedding and Qdrant writes.

mod detect;
mod file_folder;
mod status;
mod text;
mod url;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

use file_folder::{ingest_file, ingest_folder};
use status::ingest_status;
use text::ingest_text;
use url::ingest_url;

/// Ingest command arguments
#[derive(Args)]
pub struct IngestArgs {
    #[command(subcommand)]
    command: IngestCommand,
}

/// Ingest subcommands
#[derive(Subcommand)]
enum IngestCommand {
    /// Ingest a single file
    File {
        /// Path to file
        #[arg(value_parser = crate::path_arg::parse_path)]
        path: PathBuf,

        /// Target collection (optional, auto-detected)
        #[arg(short, long)]
        collection: Option<String>,

        /// Library tag (for library ingestion)
        #[arg(short, long)]
        tag: Option<String>,
    },

    /// Ingest a folder recursively
    Folder {
        /// Path to folder
        #[arg(value_parser = crate::path_arg::parse_path)]
        path: PathBuf,

        /// Target collection (optional, auto-detected)
        #[arg(short, long)]
        collection: Option<String>,

        /// Library tag (for library ingestion)
        #[arg(short, long)]
        tag: Option<String>,

        /// File patterns to include
        #[arg(short = 'p', long)]
        patterns: Vec<String>,

        /// Maximum files to process
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Ingest raw text content
    Text {
        /// Text content to ingest
        content: String,

        /// Target collection
        #[arg(short, long)]
        collection: String,

        /// Document title/identifier
        #[arg(short, long)]
        title: Option<String>,
    },

    /// Fetch and ingest a URL (web page)
    Url {
        /// URL to fetch and ingest
        url: String,

        /// Target collection (default: projects or libraries)
        #[arg(short, long)]
        collection: Option<String>,

        /// Library name (stores in libraries collection)
        #[arg(short, long)]
        library: Option<String>,

        /// Document title (auto-extracted from HTML if omitted)
        #[arg(short, long)]
        title: Option<String>,
    },

    /// Show ingestion queue status
    Status {
        /// Show detailed queue items
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute ingest command
pub async fn execute(args: IngestArgs) -> Result<()> {
    match args.command {
        IngestCommand::File {
            path,
            collection,
            tag,
        } => ingest_file(&path, collection, tag).await,
        IngestCommand::Folder {
            path,
            collection,
            tag,
            patterns,
            limit,
        } => ingest_folder(&path, collection, tag, &patterns, limit).await,
        IngestCommand::Text {
            content,
            collection,
            title,
        } => ingest_text(&content, &collection, title).await,
        IngestCommand::Url {
            url,
            collection,
            library,
            title,
        } => ingest_url(&url, collection, library, title).await,
        IngestCommand::Status { verbose } => ingest_status(verbose).await,
    }
}
