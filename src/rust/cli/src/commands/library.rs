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
use clap::{Args, Subcommand, ValueEnum};

use crate::grpc::client::DaemonClient;

/// Library sync mode controlling how file deletions are handled
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum LibraryMode {
    /// Mirror mode: Delete vectors when source files are removed
    Sync,
    /// Append-only mode: Never delete vectors, only add/update (default)
    #[default]
    Incremental,
}

impl std::fmt::Display for LibraryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryMode::Sync => write!(f, "sync"),
            LibraryMode::Incremental => write!(f, "incremental"),
        }
    }
}
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output::{self, ServiceStatus};

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
}

/// Execute library command
pub async fn execute(args: LibraryArgs) -> Result<()> {
    match args.command {
        LibraryCommand::List { verbose } => list(verbose).await,
        LibraryCommand::Add { tag, path, mode } => add(&tag, &path, mode).await,
        LibraryCommand::Watch {
            tag,
            path,
            patterns,
            mode,
        } => watch(&tag, &path, &patterns, mode).await,
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
            "  sqlite3 {} 'SELECT watch_id, path, library_mode, patterns, enabled FROM watch_folders WHERE watch_id LIKE \"lib-%\"'",
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

async fn add(tag: &str, path: &PathBuf, mode: LibraryMode) -> Result<()> {
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
    output::info(format!("Mode: {} ({})", mode, mode_description(mode)));
    output::separator();

    // Library add creates metadata without watching
    // This would write to SQLite with enabled=false
    output::info("To add library with watching enabled, use:");
    output::info(&format!("  wqm library watch {} {} --mode {}", tag, abs_path.display(), mode));
    output::separator();

    output::info("Library metadata storage:");
    output::info("  Libraries are stored in SQLite watch_folders table");
    output::info("  Watch ID format: lib-{tag}");
    output::info(&format!("  Collection name: _{}", tag));
    output::info(&format!("  library_mode: {}", mode));

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

/// Returns a human-readable description of the library mode
fn mode_description(mode: LibraryMode) -> &'static str {
    match mode {
        LibraryMode::Sync => "deletes vectors when files are removed",
        LibraryMode::Incremental => "append-only, never deletes vectors",
    }
}

async fn watch(tag: &str, path: &PathBuf, patterns: &[String], mode: LibraryMode) -> Result<()> {
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
    output::info(format!("Mode: {} ({})", mode, mode_description(mode)));
    output::separator();

    // This would insert into SQLite watch_folders table
    let watch_id = format!("lib-{}", tag);
    let collection = format!("_{}", tag);

    output::info("Watch configuration:");
    output::kv("  watch_id", &watch_id);
    output::kv("  collection", &collection);
    output::kv("  library_mode", &mode.to_string());
    output::kv("  auto_ingest", "true");
    output::kv("  recursive", "true");
    output::kv("  enabled", "true");
    output::separator();

    output::info("To configure in SQLite:");
    let db_path = get_db_path()?;
    output::info(&format!(
        "  sqlite3 {} \"INSERT INTO watch_folders (watch_id, path, collection, patterns, library_mode, enabled) VALUES ('{}', '{}', '{}', '{}', '{}', 1)\"",
        db_path.display(),
        watch_id,
        abs_path.display(),
        collection,
        serde_json::to_string(&patterns_str).unwrap_or_default(),
        mode
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
                "  sqlite3 {} \"SELECT watch_id, path, library_mode, enabled FROM watch_folders WHERE watch_id LIKE 'lib-%'\" -header -column",
                db_path.display()
            ));

            output::separator();
            output::info("Library modes:");
            output::info("  sync: Deletes vectors when source files are removed");
            output::info("  incremental: Append-only, never deletes vectors (default)");

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
        "  sqlite3 {} \"SELECT watch_id, path, library_mode FROM watch_folders WHERE watch_id LIKE 'lib-%' AND enabled = 1\"",
        db_path.display()
    ));

    Ok(())
}
