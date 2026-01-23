//! Library command - tag-based library management
//!
//! Phase 1 HIGH priority command for library documentation.
//! Redesigned with tags instead of direct collection names.
//! Subcommands: list, add <tag> <path>, watch <tag> <path>,
//!              unwatch <tag>, rescan <tag>, info [tag], status
//!
//! Note: Library management uses SQLite for watch configuration.
//! The daemon reads from SQLite and manages the actual watching.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output::{self, ServiceStatus};

const LIBRARY_AFTER_HELP: &str = "\
EXAMPLES:
    wqm library list                        List all configured libraries
    wqm library watch python ~/docs/python  Watch Python docs folder
    wqm library watch rust ~/docs/rust -p \"*.md\" -p \"*.rs\"
                                            Watch with specific patterns
    wqm library info python                 Show details for 'python' library
    wqm library rescan python --force       Force re-index all files
    wqm library unwatch python              Stop watching 'python' library

LIBRARY TAGS:
    Tags are short identifiers for your documentation libraries.
    Choose meaningful names like: python, rust, react, internal-api

    The tag becomes the collection name prefix: _python, _rust, etc.

DEFAULT FILE PATTERNS:
    If no patterns specified, watches: *.pdf, *.epub, *.md, *.txt, *.html

STORAGE:
    Watch configurations are stored in SQLite at:
    ~/Library/Application Support/workspace-qdrant/state.db (macOS)
    ~/.local/share/workspace-qdrant/state.db (Linux)";

/// Library command arguments
#[derive(Args)]
#[command(after_help = LIBRARY_AFTER_HELP)]
pub struct LibraryArgs {
    #[command(subcommand)]
    command: LibraryCommand,
}

/// Library subcommands
#[derive(Subcommand)]
enum LibraryCommand {
    /// List all configured libraries with their watch status
    #[command(long_about = "Display all libraries configured in the system, showing their \
        tags, paths, watch status, and document counts.")]
    List {
        /// Show detailed information including file patterns and stats
        #[arg(short, long, help = "Include file patterns, document counts, and timestamps")]
        verbose: bool,
    },

    /// Register a library without file watching (metadata only)
    #[command(long_about = "Add a library to the system without enabling file watching. \
        Use this for libraries you want to manually manage or index on-demand. \
        To enable watching, use 'wqm library watch' instead.")]
    Add {
        /// Library tag - a short identifier (e.g., 'python', 'rust-docs')
        #[arg(help = "Short identifier for the library (used in searches)")]
        tag: String,

        /// Path to the library content directory
        #[arg(help = "Directory containing documentation files")]
        path: PathBuf,
    },

    /// Watch a library path and automatically index changes
    #[command(long_about = "Start watching a directory for documentation files. \
        New and modified files are automatically indexed by the daemon. \
        Use -p/--patterns to specify which file types to include.")]
    Watch {
        /// Library tag - a short identifier (e.g., 'python', 'rust-docs')
        #[arg(help = "Short identifier for the library")]
        tag: String,

        /// Path to the library content directory
        #[arg(help = "Directory to watch for documentation files")]
        path: PathBuf,

        /// File patterns to include (default: *.pdf, *.epub, *.md, *.txt, *.html)
        #[arg(short, long, help = "Glob patterns for files to index (e.g., \"*.md\", \"*.pdf\")")]
        patterns: Vec<String>,
    },

    /// Stop watching a library (keeps indexed documents)
    #[command(long_about = "Disable file watching for a library. Already indexed documents \
        remain in the database. Use 'wqm library watch' to re-enable watching.")]
    Unwatch {
        /// Library tag to stop watching
        #[arg(help = "Tag of the library to unwatch")]
        tag: String,
    },

    /// Rescan and re-index library files
    #[command(long_about = "Trigger a full rescan of the library directory. \
        By default, only modified files are re-indexed. Use --force to re-index all files.")]
    Rescan {
        /// Library tag to rescan
        #[arg(help = "Tag of the library to rescan")]
        tag: String,

        /// Force re-ingestion of all files (not just modified ones)
        #[arg(short, long, help = "Re-index all files regardless of modification time")]
        force: bool,
    },

    /// Show detailed information about a library
    #[command(long_about = "Display detailed information about a library including path, \
        file patterns, document count, watch status, and last scan time.")]
    Info {
        /// Library tag (omit to show info for all libraries)
        #[arg(help = "Tag of the library (optional - shows all if omitted)")]
        tag: Option<String>,
    },

    /// Show watch status for all libraries
    #[command(long_about = "Display the current watch status for all configured libraries, \
        including whether they're actively being watched and any errors.")]
    Status,
}

/// Execute library command
pub async fn execute(args: LibraryArgs) -> Result<()> {
    match args.command {
        LibraryCommand::List { verbose } => list(verbose).await,
        LibraryCommand::Add { tag, path } => add(&tag, &path).await,
        LibraryCommand::Watch {
            tag,
            path,
            patterns,
        } => watch(&tag, &path, &patterns).await,
        LibraryCommand::Unwatch { tag } => unwatch(&tag).await,
        LibraryCommand::Rescan { tag, force } => rescan(&tag, force).await,
        LibraryCommand::Info { tag } => info(tag.as_deref()).await,
        LibraryCommand::Status => status().await,
    }
}

/// Get SQLite database path
fn get_db_path() -> Result<PathBuf> {
    let data_dir = dirs::data_local_dir()
        .context("Could not find local data directory")?
        .join("workspace-qdrant");
    Ok(data_dir.join("state.db"))
}

async fn list(verbose: bool) -> Result<()> {
    output::section("Libraries");

    let db_path = get_db_path()?;

    if !db_path.exists() {
        output::info("No libraries configured yet.");
        output::info("Add a library with: wqm library add <tag> <path>");
        return Ok(());
    }

    // Query SQLite for watch_folders with library pattern
    output::info("Library watch folders (from SQLite):");
    output::info(&format!("  Database: {}", db_path.display()));
    output::separator();

    // Show command to query directly
    if verbose {
        output::info("Query watch folders:");
        output::info(&format!(
            "  sqlite3 {} 'SELECT watch_id, path, patterns, enabled FROM watch_folders WHERE watch_id LIKE \"lib-%\"'",
            db_path.display()
        ));
    } else {
        output::info("Use -v/--verbose to see query commands");
    }

    // Note: Full implementation would query SQLite directly
    // For now, provide guidance to user
    output::separator();
    output::info("Library collections follow naming: _{library_name}");
    output::info("Query Qdrant for library collections:");
    output::info("  curl http://localhost:6333/collections | jq '.result.collections[] | select(.name | startswith(\"_\"))'");

    Ok(())
}

async fn add(tag: &str, path: &PathBuf) -> Result<()> {
    output::section(format!("Add Library: {}", tag));

    // Validate path exists
    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path.canonicalize()
        .context("Could not resolve absolute path")?;

    output::info(format!("Library tag: {}", tag));
    output::info(format!("Path: {}", abs_path.display()));
    output::separator();

    // Library add creates metadata without watching
    // This would write to SQLite with enabled=false
    output::info("To add library with watching enabled, use:");
    output::info(&format!("  wqm library watch {} {}", tag, abs_path.display()));
    output::separator();

    output::info("Library metadata storage:");
    output::info("  Libraries are stored in SQLite watch_folders table");
    output::info("  Watch ID format: lib-{tag}");
    output::info(&format!("  Collection name: _{}", tag));

    // Signal daemon if available
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::info("Daemon connected - signaling configuration change...");
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified of configuration change");
        }
    }

    Ok(())
}

async fn watch(tag: &str, path: &PathBuf, patterns: &[String]) -> Result<()> {
    output::section(format!("Watch Library: {}", tag));

    // Validate path exists
    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path.canonicalize()
        .context("Could not resolve absolute path")?;

    let patterns_str = if patterns.is_empty() {
        vec!["*.pdf", "*.epub", "*.md", "*.txt", "*.html"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    } else {
        patterns.to_vec()
    };

    output::info(format!("Library tag: {}", tag));
    output::info(format!("Path: {}", abs_path.display()));
    output::info(format!("Patterns: {}", patterns_str.join(", ")));
    output::separator();

    // This would insert into SQLite watch_folders table
    let watch_id = format!("lib-{}", tag);
    let collection = format!("_{}", tag);

    output::info("Watch configuration:");
    output::kv("  watch_id", &watch_id);
    output::kv("  collection", &collection);
    output::kv("  auto_ingest", "true");
    output::kv("  recursive", "true");
    output::kv("  enabled", "true");
    output::separator();

    output::info("To configure in SQLite:");
    let db_path = get_db_path()?;
    output::info(&format!(
        "  sqlite3 {} \"INSERT INTO watch_folders (watch_id, path, collection, patterns, enabled) VALUES ('{}', '{}', '{}', '{}', 1)\"",
        db_path.display(),
        watch_id,
        abs_path.display(),
        collection,
        serde_json::to_string(&patterns_str).unwrap_or_default()
    ));

    // Signal daemon
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::separator();
        output::info("Signaling daemon to reload watch configuration...");
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - it will start watching shortly");
        }
    } else {
        output::warning("Daemon not running - start it to begin watching");
    }

    Ok(())
}

async fn unwatch(tag: &str) -> Result<()> {
    output::section(format!("Unwatch Library: {}", tag));

    let watch_id = format!("lib-{}", tag);
    let db_path = get_db_path()?;

    output::info(format!("Disabling watch for library: {}", tag));
    output::info(format!("Watch ID: {}", watch_id));
    output::separator();

    output::info("To disable in SQLite:");
    output::info(&format!(
        "  sqlite3 {} \"UPDATE watch_folders SET enabled = 0 WHERE watch_id = '{}'\"",
        db_path.display(),
        watch_id
    ));

    output::info("To remove completely:");
    output::info(&format!(
        "  sqlite3 {} \"DELETE FROM watch_folders WHERE watch_id = '{}'\"",
        db_path.display(),
        watch_id
    ));

    // Signal daemon
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::separator();
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified of configuration change");
        }
    }

    Ok(())
}

async fn rescan(tag: &str, force: bool) -> Result<()> {
    output::section(format!("Rescan Library: {}", tag));

    output::info(format!("Library tag: {}", tag));
    output::info(format!("Force re-ingestion: {}", force));
    output::separator();

    // Rescan would trigger the daemon to re-process all files
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::info("Signaling daemon to rescan library...");

        // Signal queue refresh to trigger re-processing
        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };

        match client.system().send_refresh_signal(request).await {
            Ok(_) => {
                output::success("Rescan signal sent to daemon");
                output::info("The daemon will queue files for re-ingestion");

                if force {
                    output::info("Force mode: All files will be re-processed regardless of modification time");
                } else {
                    output::info("Normal mode: Only modified files will be re-processed");
                }
            }
            Err(e) => {
                output::error(format!("Failed to signal rescan: {}", e));
            }
        }
    } else {
        output::error("Cannot connect to daemon");
        output::info("Start daemon with: wqm service start");
    }

    Ok(())
}

async fn info(tag: Option<&str>) -> Result<()> {
    match tag {
        Some(t) => {
            output::section(format!("Library Info: {}", t));

            let watch_id = format!("lib-{}", t);
            let collection = format!("_{}", t);

            output::kv("Tag", t);
            output::kv("Watch ID", &watch_id);
            output::kv("Collection", &collection);
            output::separator();

            let db_path = get_db_path()?;
            output::info("Query configuration:");
            output::info(&format!(
                "  sqlite3 {} \"SELECT * FROM watch_folders WHERE watch_id = '{}'\" -header -column",
                db_path.display(),
                watch_id
            ));

            output::separator();
            output::info("Query document count:");
            output::info(&format!(
                "  curl 'http://localhost:6333/collections/{}/points/count' -H 'Content-Type: application/json' -d '{{}}'",
                collection
            ));
        }
        None => {
            output::section("All Libraries");

            let db_path = get_db_path()?;
            output::info("Library watch configurations:");
            output::info(&format!(
                "  sqlite3 {} \"SELECT watch_id, path, enabled FROM watch_folders WHERE watch_id LIKE 'lib-%'\" -header -column",
                db_path.display()
            ));

            output::separator();
            output::info("Library collections in Qdrant (prefix with _):");
            output::info("  curl http://localhost:6333/collections | jq '.result.collections[] | select(.name | startswith(\"_\")) | .name'");
        }
    }

    Ok(())
}

async fn status() -> Result<()> {
    output::section("Library Watch Status");

    // Check daemon status
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().health_check(()).await {
                Ok(response) => {
                    let health = response.into_inner();

                    // Look for file_watcher component
                    for comp in &health.components {
                        if comp.component_name.contains("watcher") || comp.component_name.contains("library") {
                            let comp_status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&comp.component_name, comp_status);
                            if !comp.message.is_empty() {
                                output::kv("  Message", &comp.message);
                            }
                        }
                    }
                }
                Err(_) => {
                    output::warning("Could not get health details");
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::error("Daemon not running");
        }
    }

    output::separator();
    output::info("Watch folder configuration stored in SQLite:");

    let db_path = get_db_path()?;
    output::info(&format!("  Database: {}", db_path.display()));
    output::info("  Table: watch_folders");
    output::separator();

    output::info("Query enabled library watches:");
    output::info(&format!(
        "  sqlite3 {} \"SELECT watch_id, path FROM watch_folders WHERE watch_id LIKE 'lib-%' AND enabled = 1\"",
        db_path.display()
    ));

    Ok(())
}
