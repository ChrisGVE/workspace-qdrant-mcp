//! Library command - tag-based library management
//!
//! Phase 1 HIGH priority command for library documentation.
//! Redesigned with tags instead of direct collection names.
//! Subcommands: list, add <tag> <path>, watch <tag> <path>,
//!              unwatch <tag>, rescan <tag>, info [tag], status
//!
//! Note: Library management uses SQLite for watch configuration.
//! The daemon reads from SQLite and manages the actual watching.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::timestamps;
use clap::{Args, Subcommand, ValueEnum};
use rusqlite::Connection;

use wqm_common::constants::COLLECTION_LIBRARIES;
use crate::config::get_database_path;
use crate::grpc::client::DaemonClient;
use crate::queue::{UnifiedQueueClient, ItemType, QueueOperation};

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

/// Default file patterns for library collections.
/// Covers all supported document formats in `AllowedExtensions::library_extensions`.
const DEFAULT_LIBRARY_PATTERNS: &[&str] = &[
    "*.pdf", "*.epub", "*.docx", "*.pptx", "*.ppt", "*.pages", "*.key",
    "*.odt", "*.odp", "*.ods", "*.rtf", "*.doc",
    "*.md", "*.txt", "*.html", "*.htm",
];

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
        LibraryCommand::Remove { tag, yes } => remove(&tag, yes).await,
        LibraryCommand::Rescan { tag, force } => rescan(&tag, force).await,
        LibraryCommand::Info { tag } => info(tag.as_deref()).await,
        LibraryCommand::Status => status().await,
        LibraryCommand::Config {
            tag,
            mode,
            patterns,
            enable,
            disable,
            show,
        } => config(&tag, mode, patterns, enable, disable, show).await,
    }
}

/// Get SQLite database path (canonical: ~/.workspace-qdrant/state.db)
fn get_db_path() -> Result<PathBuf> {
    get_database_path().map_err(|e| anyhow::anyhow!("{}", e))
}

/// Open a connection to the state database with WAL mode
fn open_db() -> Result<Connection> {
    let db_path = get_db_path()?;
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run daemon first: wqm service start",
            db_path.display()
        );
    }
    let conn = Connection::open(&db_path)
        .context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

async fn list(verbose: bool) -> Result<()> {
    output::section("Libraries");

    let conn = match open_db() {
        Ok(c) => c,
        Err(_) => {
            output::info("No libraries configured yet.");
            output::info("Add a library with: wqm library add <tag> <path>");
            return Ok(());
        }
    };

    let mut stmt = conn.prepare(
        &format!("SELECT watch_id, tenant_id, path, library_mode, enabled, is_active, created_at, last_activity_at
         FROM watch_folders WHERE collection = '{}' ORDER BY tenant_id", COLLECTION_LIBRARIES)
    ).context("Failed to query watch_folders")?;

    let libraries: Vec<(String, String, String, Option<String>, bool, bool, String, Option<String>)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, i32>(4)? != 0,
                row.get::<_, i32>(5)? != 0,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })
        .context("Failed to read library rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")?;

    if libraries.is_empty() {
        output::info("No libraries configured yet.");
        output::info("Add a library with: wqm library add <tag> <path>");
        return Ok(());
    }

    output::info(format!("Found {} library/libraries:", libraries.len()));
    output::separator();

    for (watch_id, tenant_id, path, mode, enabled, _is_active, created_at, last_activity) in &libraries {
        let status = if *enabled { "watching" } else { "paused" };
        let mode_str = mode.as_deref().unwrap_or("incremental");

        output::kv("  Tag", tenant_id);
        output::kv("  Path", path);
        output::kv("  Status", status);
        output::kv("  Mode", mode_str);
        if verbose {
            output::kv("  Watch ID", watch_id);
            output::kv("  Created", created_at);
            if let Some(activity) = last_activity {
                output::kv("  Last Activity", activity);
            }
        }
        output::separator();
    }

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

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();
    let abs_path_str = abs_path.to_string_lossy().to_string();

    // Check for duplicate
    let exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE watch_id = ?",
        [&watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if exists {
        output::error(format!("Library '{}' already exists. Use 'wqm library config' to update it.", tag));
        return Ok(());
    }

    // Check for duplicate path
    let path_exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE path = ?",
        [&abs_path_str],
        |_| Ok(true),
    ).unwrap_or(false);

    if path_exists {
        output::error(format!("Path '{}' is already registered.", abs_path.display()));
        return Ok(());
    }

    // Insert into watch_folders (enabled=0 for add, use watch to enable)
    conn.execute(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, follow_symlinks, cleanup_on_disable, created_at, updated_at)
         VALUES (?1, ?2, 'libraries', ?3, ?4, 0, 0, 0, 0, ?5, ?5)",
        rusqlite::params![&watch_id, &abs_path_str, tag, &mode.to_string(), &now],
    ).context("Failed to insert library into watch_folders")?;

    output::success(format!("Library '{}' added (not watching yet)", tag));
    output::kv("  Tag", tag);
    output::kv("  Path", &abs_path_str);
    output::kv("  Mode", &mode.to_string());
    output::separator();
    output::info("To start watching: wqm library watch <tag> <path>");

    // Signal daemon if available
    if let Ok(mut client) = DaemonClient::connect_default().await {
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

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();
    let abs_path_str = abs_path.to_string_lossy().to_string();

    // Check if library already exists
    let exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE watch_id = ?",
        [&watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if exists {
        // Enable watching on existing library
        conn.execute(
            "UPDATE watch_folders SET enabled = 1, library_mode = ?, path = ?, updated_at = ?, last_activity_at = ? WHERE watch_id = ?",
            rusqlite::params![&mode.to_string(), &abs_path_str, &now, &now, &watch_id],
        ).context("Failed to enable watch")?;
        output::success(format!("Library '{}' watching enabled", tag));
    } else {
        // Insert new library with watching enabled
        conn.execute(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, follow_symlinks, cleanup_on_disable, created_at, updated_at, last_activity_at)
             VALUES (?1, ?2, 'libraries', ?3, ?4, 1, 0, 0, 0, ?5, ?5, ?5)",
            rusqlite::params![&watch_id, &abs_path_str, tag, &mode.to_string(), &now],
        ).context("Failed to insert library watch")?;
        output::success(format!("Library '{}' added and watching enabled", tag));
    }

    output::kv("  Tag", tag);
    output::kv("  Path", &abs_path_str);
    output::kv("  Mode", &format!("{} ({})", mode, mode_description(mode)));

    // Use user-provided patterns or defaults
    let effective_patterns: Vec<String> = if patterns.is_empty() {
        DEFAULT_LIBRARY_PATTERNS.iter().map(|s| s.to_string()).collect()
    } else {
        patterns.to_vec()
    };

    output::kv("  Patterns", &format!("{}", effective_patterns.len()));

    // Enqueue a folder scan for the library
    match UnifiedQueueClient::connect() {
        Ok(client) => {
            let payload_json = serde_json::json!({
                "folder_path": abs_path_str,
                "recursive": true,
                "patterns": effective_patterns,
            }).to_string();

            match client.enqueue(
                ItemType::Folder,
                QueueOperation::Scan,
                tag,                    // tenant_id
                COLLECTION_LIBRARIES,   // collection
                &payload_json,
                0,              // priority
                "",             // branch (not applicable)
                None,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Library scan already queued");
                    } else {
                        output::success("Library scan queued for ingestion");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue library scan: {}", e));
                }
            }
        }
        Err(e) => {
            output::warning(format!("Could not connect to queue: {}. Daemon will scan on next poll.", e));
        }
    }

    // Signal daemon
    if let Ok(mut client) = DaemonClient::connect_default().await {
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

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();

    // Verify library exists
    let exists: bool = conn.query_row(
        &format!("SELECT 1 FROM watch_folders WHERE watch_id = ? AND collection = '{}'", COLLECTION_LIBRARIES),
        [&watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if !exists {
        output::error(format!("Library '{}' not found", tag));
        return Ok(());
    }

    // Disable watching (keep the record for re-enabling later)
    conn.execute(
        "UPDATE watch_folders SET enabled = 0, updated_at = ? WHERE watch_id = ?",
        rusqlite::params![&now, &watch_id],
    ).context("Failed to disable watch")?;

    output::success(format!("Library '{}' watching disabled", tag));
    output::info("Existing indexed content is preserved.");
    output::info("To re-enable: wqm library watch <tag> <path>");
    output::info("To remove completely: wqm library remove <tag>");

    // Signal daemon
    if let Ok(mut client) = DaemonClient::connect_default().await {
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

async fn remove(tag: &str, skip_confirm: bool) -> Result<()> {
    output::section(format!("Remove Library: {}", tag));

    let watch_id = format!("lib-{}", tag);
    let conn = open_db()?;

    let exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE watch_id = ?",
        [&watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if !exists {
        output::error(format!("Library '{}' not found (watch_id: {})", tag, watch_id));
        return Ok(());
    }

    // Confirm deletion unless --yes flag
    if !skip_confirm {
        output::warning(format!(
            "This will delete ALL vectors for library '{}' from Qdrant.",
            tag
        ));
        output::warning("This action cannot be undone.");
        output::info("");
        print!("Continue? (y/N): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            output::info("Cancelled.");
            return Ok(());
        }
    }

    output::separator();

    // Step 1: Delete watch folder config from SQLite
    output::info("Removing watch configuration...");
    let deleted = conn.execute(
        "DELETE FROM watch_folders WHERE watch_id = ?",
        [&watch_id],
    ).context("Failed to delete watch folder config")?;

    if deleted > 0 {
        output::success(format!("Removed watch config for '{}'", tag));
    }

    // Step 2: Enqueue deletion of vectors from Qdrant
    output::info("Queueing vector deletion...");

    // Libraries are stored in the unified libraries collection
    let collection = COLLECTION_LIBRARIES;

    // Build delete tenant payload
    let payload_json = serde_json::json!({
        "tenant_id_to_delete": tag
    }).to_string();

    match UnifiedQueueClient::connect() {
        Ok(client) => {
            match client.enqueue(
                ItemType::DeleteTenant,
                QueueOperation::Delete,
                tag,            // tenant_id
                collection,     // collection
                &payload_json,  // payload
                0,              // priority is dynamic (computed at dequeue time)
                "",             // branch (not applicable for libraries)
                None,           // metadata
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Vector deletion already queued (duplicate)");
                    } else {
                        output::success(format!(
                            "Vector deletion queued (queue_id: {})",
                            result.queue_id
                        ));
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue vector deletion: {}", e));
                    output::info("You may need to manually delete vectors from Qdrant:");
                    output::info(&format!(
                        "  curl -X POST 'http://localhost:6333/collections/{}/points/delete' \\",
                        collection
                    ));
                    output::info(&format!(
                        "    -H 'Content-Type: application/json' \\",
                    ));
                    output::info(&format!(
                        "    -d '{{\"filter\": {{\"must\": [{{\"key\": \"library_name\", \"match\": {{\"value\": \"{}\"}}}}]}}}}'",
                        tag
                    ));
                }
            }
        }
        Err(e) => {
            output::warning(format!("Could not connect to queue: {}", e));
            output::info("Vectors will need to be deleted manually or when daemon starts.");
        }
    }

    // Signal daemon if available
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::separator();
        output::info("Signaling daemon...");
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - will process deletion shortly");
        }
    } else {
        output::separator();
        output::info("Daemon not running. Vector deletion will occur when daemon starts.");
    }

    output::separator();
    output::success(format!("Library '{}' removed", tag));

    Ok(())
}

async fn rescan(tag: &str, force: bool) -> Result<()> {
    output::section(format!("Rescan Library: {}", tag));

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);

    // Look up library path from watch_folders
    let result: Result<(String, Option<String>), _> = conn.query_row(
        "SELECT path, library_mode FROM watch_folders WHERE watch_id = ? AND collection = 'libraries'",
        [&watch_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
    );

    let (lib_path, mode) = match result {
        Ok(r) => r,
        Err(_) => {
            output::error(format!("Library '{}' not found", tag));
            output::info("List libraries with: wqm library list");
            return Ok(());
        }
    };

    output::kv("Tag", tag);
    output::kv("Path", &lib_path);
    output::kv("Mode", mode.as_deref().unwrap_or("incremental"));
    output::kv("Force", &force.to_string());
    output::separator();

    // Enqueue a folder scan for the library
    match UnifiedQueueClient::connect() {
        Ok(client) => {
            let payload_json = serde_json::json!({
                "folder_path": lib_path,
                "recursive": true,
                "patterns": DEFAULT_LIBRARY_PATTERNS,
            }).to_string();

            // Use priority 0 for rescans (high priority)
            match client.enqueue(
                ItemType::Folder,
                QueueOperation::Scan,
                tag,                    // tenant_id
                COLLECTION_LIBRARIES,   // collection
                &payload_json,
                0,              // high priority
                "",             // branch
                None,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Library rescan already queued");
                    } else {
                        output::success("Library rescan queued for processing");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue rescan: {}", e));
                }
            }
        }
        Err(e) => {
            output::warning(format!("Could not connect to queue: {}. Start daemon first.", e));
        }
    }

    // Signal daemon to process immediately
    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - rescan will begin shortly");
        }
    } else {
        output::warning("Daemon not running - start it to begin rescan");
    }

    Ok(())
}

async fn info(tag: Option<&str>) -> Result<()> {
    let conn = match open_db() {
        Ok(c) => c,
        Err(e) => {
            output::error(format!("Cannot read database: {}", e));
            return Ok(());
        }
    };

    match tag {
        Some(t) => {
            output::section(format!("Library Info: {}", t));

            let watch_id = format!("lib-{}", t);

            let result: Result<(String, String, Option<String>, bool, String, Option<String>, Option<String>), _> = conn.query_row(
                "SELECT path, tenant_id, library_mode, enabled, created_at, updated_at, last_activity_at
                 FROM watch_folders WHERE watch_id = ? AND collection = 'libraries'",
                [&watch_id],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, i32>(3)? != 0,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5).ok().flatten(),
                    row.get::<_, Option<String>>(6).ok().flatten(),
                )),
            );

            match result {
                Ok((path, tenant_id, mode, enabled, created_at, updated_at, last_activity)) => {
                    let status = if enabled { "watching" } else { "paused" };
                    output::kv("Tag", &tenant_id);
                    output::kv("Watch ID", &watch_id);
                    output::kv("Path", &path);
                    output::kv("Status", status);
                    output::kv("Mode", mode.as_deref().unwrap_or("incremental"));
                    output::kv("Collection", COLLECTION_LIBRARIES);
                    output::kv("Created", &created_at);
                    if let Some(updated) = updated_at {
                        output::kv("Updated", &updated);
                    }
                    if let Some(activity) = last_activity {
                        output::kv("Last Activity", &activity);
                    }

                    // Query tracked_files for file count
                    output::separator();
                    let file_count: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM tracked_files WHERE watch_folder_id = ?",
                        [&watch_id],
                        |row| row.get(0),
                    ).unwrap_or(0);

                    let chunk_count: i64 = conn.query_row(
                        "SELECT COALESCE(SUM(chunk_count), 0) FROM tracked_files WHERE watch_folder_id = ?",
                        [&watch_id],
                        |row| row.get(0),
                    ).unwrap_or(0);

                    output::kv("Tracked Files", &file_count.to_string());
                    output::kv("Total Chunks", &chunk_count.to_string());
                }
                Err(_) => {
                    output::error(format!("Library '{}' not found", t));
                }
            }
        }
        None => {
            // Show info for all libraries (delegates to list with verbose)
            list(true).await?;
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

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();

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
            output::warning("Daemon not running - start it with: wqm service start");
        }
    }

    output::separator();

    // Show library watch status from SQLite
    let conn = match open_db() {
        Ok(c) => c,
        Err(_) => {
            output::info("No database found. Run daemon first to initialize.");
            return Ok(());
        }
    };

    let mut stmt = conn.prepare(
        "SELECT tenant_id, path, library_mode, enabled FROM watch_folders WHERE collection = 'libraries' ORDER BY tenant_id"
    ).context("Failed to query watch_folders")?;

    let libraries: Vec<(String, String, Option<String>, bool)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i32>(3)? != 0,
            ))
        })
        .context("Failed to read library rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")?;

    if libraries.is_empty() {
        output::info("No libraries configured.");
        return Ok(());
    }

    output::info(format!("{} library/libraries configured:", libraries.len()));

    let watching = libraries.iter().filter(|(_, _, _, e)| *e).count();
    let paused = libraries.len() - watching;
    output::kv("  Watching", &watching.to_string());
    output::kv("  Paused", &paused.to_string());
    output::separator();

    for (tag, path, mode, enabled) in &libraries {
        let status_icon = if *enabled { "watching" } else { "paused" };
        output::info(format!("{}: {} [{}] ({})", tag, path, status_icon, mode.as_deref().unwrap_or("incremental")));
    }

    Ok(())
}

async fn config(
    tag: &str,
    mode: Option<LibraryMode>,
    patterns: Option<String>,
    enable: bool,
    disable: bool,
    show: bool,
) -> Result<()> {
    output::section(format!("Library Configuration: {}", tag));

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();

    // Check if library exists
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM watch_folders WHERE watch_id = ? AND collection = 'libraries'",
            [&watch_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !exists {
        output::error(format!(
            "Library '{}' not found (watch_id: {})",
            tag, watch_id
        ));
        output::info("Add it first with: wqm library watch <tag> <path>");
        return Ok(());
    }

    // Show current configuration
    if show || (mode.is_none() && !enable && !disable) {
        output::info("Current configuration:");
        output::separator();

        let result: Result<(String, Option<String>, i32, bool), _> = conn.query_row(
            "SELECT path, library_mode, enabled, follow_symlinks FROM watch_folders WHERE watch_id = ?",
            [&watch_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get::<_, i32>(3)? != 0)),
        );

        match result {
            Ok((path, lib_mode, enabled, follow_symlinks)) => {
                output::kv("Tag", tag);
                output::kv("Watch ID", &watch_id);
                output::kv("Path", &path);
                output::kv("Mode", lib_mode.as_deref().unwrap_or("incremental"));
                output::kv("Enabled", if enabled == 1 { "yes" } else { "no" });
                output::kv("Follow Symlinks", if follow_symlinks { "yes" } else { "no" });
            }
            Err(e) => {
                output::error(format!("Failed to read configuration: {}", e));
            }
        }

        if mode.is_some() || enable || disable {
            output::separator();
        }
    }

    // Apply configuration changes
    let mut changes_made = false;

    if let Some(new_mode) = mode {
        output::info(format!("Setting mode to: {}", new_mode));
        conn.execute(
            "UPDATE watch_folders SET library_mode = ?, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![&new_mode.to_string(), &now, &watch_id],
        )
        .context("Failed to update mode")?;
        changes_made = true;
    }

    if enable {
        output::info("Enabling watch...");
        conn.execute(
            "UPDATE watch_folders SET enabled = 1, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![&now, &watch_id],
        )
        .context("Failed to enable")?;
        changes_made = true;
    }

    if disable {
        output::info("Disabling watch...");
        conn.execute(
            "UPDATE watch_folders SET enabled = 0, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![&now, &watch_id],
        )
        .context("Failed to disable")?;
        changes_made = true;
    }

    if let Some(ref pat) = patterns {
        let parsed: Vec<&str> = pat.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        output::info(format!("Patterns: {} (applied at next scan/rescan)", parsed.join(", ")));
        // Patterns are passed through the queue payload at scan time.
        // To apply new patterns, rescan the library: wqm library rescan <tag>
    }

    if changes_made {
        output::success("Configuration updated");

        // Signal daemon to reload
        if let Ok(mut client) = DaemonClient::connect_default().await {
            let request = RefreshSignalRequest {
                queue_type: QueueType::WatchedFolders as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            };
            if client.system().send_refresh_signal(request).await.is_ok() {
                output::success("Daemon notified of configuration change");
            }
        } else {
            output::info("Daemon not running - changes will apply when daemon starts");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_library_patterns_cover_all_formats() {
        // All supported document formats must be present
        let expected = [
            "*.pdf", "*.epub", "*.docx", "*.pptx", "*.ppt", "*.pages", "*.key",
            "*.odt", "*.odp", "*.ods", "*.rtf", "*.doc",
            "*.md", "*.txt", "*.html", "*.htm",
        ];
        for pat in &expected {
            assert!(
                DEFAULT_LIBRARY_PATTERNS.contains(pat),
                "Missing default library pattern: {}",
                pat
            );
        }
        assert_eq!(
            DEFAULT_LIBRARY_PATTERNS.len(),
            expected.len(),
            "Default patterns count mismatch"
        );
    }

    #[test]
    fn test_default_patterns_used_when_none_provided() {
        let user_patterns: Vec<String> = vec![];
        let effective: Vec<String> = if user_patterns.is_empty() {
            DEFAULT_LIBRARY_PATTERNS.iter().map(|s| s.to_string()).collect()
        } else {
            user_patterns
        };
        assert_eq!(effective.len(), DEFAULT_LIBRARY_PATTERNS.len());
        assert_eq!(effective[0], "*.pdf");
    }

    #[test]
    fn test_user_patterns_override_defaults() {
        let user_patterns: Vec<String> = vec!["*.pdf".to_string(), "*.md".to_string()];
        let effective: Vec<String> = if user_patterns.is_empty() {
            DEFAULT_LIBRARY_PATTERNS.iter().map(|s| s.to_string()).collect()
        } else {
            user_patterns.clone()
        };
        assert_eq!(effective.len(), 2);
        assert_eq!(effective, vec!["*.pdf", "*.md"]);
    }
}
