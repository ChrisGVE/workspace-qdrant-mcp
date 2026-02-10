//! Watch command - watch folder management
//!
//! Phase 1 HIGH priority command for managing file watch configurations.
//! Subcommands: list, enable, disable
//!
//! Task 25: Top-level wqm watch command per spec.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Args, Subcommand};
use colored::Colorize;
use rusqlite::{Connection, params};
use serde::Serialize;
use tabled::{settings::Style, Table, Tabled};

use crate::config::get_database_path_checked;
use crate::output;

/// Watch command arguments
#[derive(Args)]
pub struct WatchArgs {
    #[command(subcommand)]
    command: WatchCommand,
}

/// Watch subcommands
#[derive(Subcommand)]
enum WatchCommand {
    /// List all watch configurations
    List {
        /// Show only enabled watches
        #[arg(long)]
        enabled: bool,

        /// Show only disabled watches
        #[arg(long, conflicts_with = "enabled")]
        disabled: bool,

        /// Filter by collection name
        #[arg(short, long)]
        collection: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show more columns
        #[arg(short, long)]
        verbose: bool,
    },

    /// Enable a watch configuration
    Enable {
        /// Watch ID to enable
        watch_id: String,
    },

    /// Disable a watch configuration
    Disable {
        /// Watch ID to disable
        watch_id: String,
    },

    /// Show detailed information for a specific watch
    Show {
        /// Watch ID or path prefix
        watch_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Pause all enabled watchers (stops file event processing)
    Pause,

    /// Resume all paused watchers (restarts file event processing)
    Resume,
}

/// Watch item for list display
#[derive(Debug, Tabled, Serialize)]
pub struct WatchListItem {
    #[tabled(rename = "Watch ID")]
    pub watch_id: String,
    #[tabled(rename = "Path")]
    pub path: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Enabled")]
    pub enabled: String,
    #[tabled(rename = "Active")]
    pub is_active: String,
    #[tabled(rename = "Paused")]
    pub is_paused: String,
    #[tabled(rename = "Last Scan")]
    pub last_scan: String,
}

/// Watch item for verbose list display
#[derive(Debug, Tabled, Serialize)]
pub struct WatchListItemVerbose {
    #[tabled(rename = "Watch ID")]
    pub watch_id: String,
    #[tabled(rename = "Path")]
    pub path: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Tenant ID")]
    pub tenant_id: String,
    #[tabled(rename = "Enabled")]
    pub enabled: String,
    #[tabled(rename = "Active")]
    pub is_active: String,
    #[tabled(rename = "Paused")]
    pub is_paused: String,
    #[tabled(rename = "Recursive")]
    pub recursive: String,
    #[tabled(rename = "Patterns")]
    pub patterns: String,
    #[tabled(rename = "Last Scan")]
    pub last_scan: String,
}

/// Watch item detail view
#[derive(Debug, Serialize)]
pub struct WatchDetailItem {
    pub watch_id: String,
    pub path: String,
    pub collection: String,
    pub tenant_id: String,
    pub patterns: String,
    pub ignore_patterns: String,
    pub auto_ingest: bool,
    pub recursive: bool,
    pub recursive_depth: i32,
    pub debounce_seconds: f64,
    pub enabled: bool,
    pub is_active: bool,
    pub is_paused: bool,
    pub created_at: String,
    pub updated_at: String,
    pub last_scan: Option<String>,
    pub last_activity_at: Option<String>,
    pub parent_watch_id: Option<String>,
    pub metadata: Option<String>,
}

/// Execute watch command
pub async fn execute(args: WatchArgs) -> Result<()> {
    match args.command {
        WatchCommand::List {
            enabled,
            disabled,
            collection,
            json,
            verbose,
        } => list(enabled, disabled, collection, json, verbose).await,
        WatchCommand::Enable { watch_id } => enable(&watch_id).await,
        WatchCommand::Disable { watch_id } => disable(&watch_id).await,
        WatchCommand::Show { watch_id, json } => show(&watch_id, json).await,
        WatchCommand::Pause => pause().await,
        WatchCommand::Resume => resume().await,
    }
}

/// Connect to the state database (read-only for list/show)
fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    Ok(conn)
}

/// Connect to the state database (read-write for enable/disable)
fn connect_readwrite() -> Result<Connection> {
    let db_path = get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open(&db_path)
        .context(format!("Failed to open state database at {:?}", db_path))?;

    // Enable WAL mode for better concurrency
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;

    Ok(conn)
}

/// Format relative time from ISO timestamp
fn format_relative_time(timestamp_str: &str) -> String {
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        let now = Utc::now();
        let duration = now.signed_duration_since(dt.with_timezone(&Utc));

        let secs = duration.num_seconds();
        if secs < 0 {
            return "future".to_string();
        }

        if secs < 60 {
            format!("{}s ago", secs)
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else if secs < 86400 {
            format!("{}h ago", secs / 3600)
        } else {
            format!("{}d ago", secs / 86400)
        }
    } else {
        "never".to_string()
    }
}

/// Truncate string for display
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Format enabled/active status with color
fn format_bool(value: bool) -> String {
    if value {
        "yes".green().to_string()
    } else {
        "no".red().to_string()
    }
}

/// Format paused status with color (paused = yellow, not paused = green)
fn format_bool_paused(value: bool) -> String {
    if value {
        "yes".yellow().to_string()
    } else {
        "no".green().to_string()
    }
}

/// Resolve watch_id by exact match or prefix match
/// Returns Some(resolved_id) if found, None if not found (error already printed)
fn resolve_watch_id(conn: &Connection, watch_id: &str) -> Result<Option<String>> {
    // First check if exact match exists
    let exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE watch_id = ?1",
        params![watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if exists {
        return Ok(Some(watch_id.to_string()));
    }

    // Try prefix match
    let prefix = format!("{}%", watch_id);
    let matches: Vec<String> = {
        let mut stmt = conn.prepare("SELECT watch_id FROM watch_folders WHERE watch_id LIKE ?1 LIMIT 5")?;
        let rows = stmt.query_map(params![&prefix], |row| row.get(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };

    if matches.is_empty() {
        output::error(format!("Watch not found: {}", watch_id));
        output::info("Use 'wqm watch list' to see available watches");
        return Ok(None);
    } else if matches.len() == 1 {
        // Single match, use it
        return Ok(Some(matches[0].clone()));
    } else {
        output::error(format!("Ambiguous watch ID '{}', multiple matches:", watch_id));
        for m in &matches {
            println!("  - {}", m);
        }
        return Ok(None);
    }
}

async fn list(
    enabled_only: bool,
    disabled_only: bool,
    collection: Option<String>,
    json: bool,
    verbose: bool,
) -> Result<()> {
    let conn = connect_readonly()?;

    // Build WHERE clause
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if enabled_only {
        conditions.push("enabled = 1".to_string());
    } else if disabled_only {
        conditions.push("enabled = 0".to_string());
    }

    if let Some(ref c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c.clone()));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!(
        r#"
        SELECT watch_id, path, collection, tenant_id, patterns, ignore_patterns,
               enabled, is_active, recursive, last_scan_at,
               COALESCE(is_paused, 0) as is_paused
        FROM watch_folders
        {}
        ORDER BY path ASC
        "#,
        where_clause
    );

    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,  // watch_id
            row.get::<_, String>(1)?,  // path
            row.get::<_, String>(2)?,  // collection
            row.get::<_, String>(3)?,  // tenant_id
            row.get::<_, String>(4)?,  // patterns
            row.get::<_, String>(5)?,  // ignore_patterns
            row.get::<_, i32>(6)? != 0,  // enabled
            row.get::<_, i32>(7)? != 0,  // is_active
            row.get::<_, i32>(8)? != 0,  // recursive
            row.get::<_, Option<String>>(9)?,  // last_scan_at
            row.get::<_, i32>(10)? != 0,  // is_paused
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json {
            println!("[]");
        } else {
            output::info("No watch configurations found");
            output::info("Add watches via project registration or wqm project add");
        }
        return Ok(());
    }

    if verbose {
        let display_items: Vec<WatchListItemVerbose> = items
            .iter()
            .map(|(watch_id, path, collection, tenant_id, patterns, _ignore_patterns, enabled, is_active, recursive, last_scan, is_paused)| {
                WatchListItemVerbose {
                    watch_id: truncate(watch_id, 20),
                    path: truncate(path, 40),
                    collection: truncate(collection, 20),
                    tenant_id: truncate(tenant_id, 20),
                    enabled: format_bool(*enabled),
                    is_active: format_bool(*is_active),
                    is_paused: format_bool_paused(*is_paused),
                    recursive: if *recursive { "yes" } else { "no" }.to_string(),
                    patterns: truncate(patterns, 20),
                    last_scan: last_scan.as_ref().map(|s| format_relative_time(s)).unwrap_or_else(|| "never".to_string()),
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            let table = Table::new(&display_items).with(Style::rounded()).to_string();
            println!("{}", table);
            output::info(format!("Showing {} watch configurations", display_items.len()));
        }
    } else {
        let display_items: Vec<WatchListItem> = items
            .iter()
            .map(|(watch_id, path, collection, _tenant_id, _patterns, _ignore_patterns, enabled, is_active, _recursive, last_scan, is_paused)| {
                WatchListItem {
                    watch_id: truncate(watch_id, 20),
                    path: truncate(path, 50),
                    collection: truncate(collection, 20),
                    enabled: format_bool(*enabled),
                    is_active: format_bool(*is_active),
                    is_paused: format_bool_paused(*is_paused),
                    last_scan: last_scan.as_ref().map(|s| format_relative_time(s)).unwrap_or_else(|| "never".to_string()),
                }
            })
            .collect();

        if json {
            output::print_json(&display_items);
        } else {
            let table = Table::new(&display_items).with(Style::rounded()).to_string();
            println!("{}", table);
            output::info(format!("Showing {} watch configurations", display_items.len()));
        }
    }

    Ok(())
}

async fn enable(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    // Resolve watch_id (exact match or prefix match)
    let resolved_id = resolve_watch_id(&conn, watch_id)?;

    let resolved_id = match resolved_id {
        Some(id) => id,
        None => return Ok(()),  // Error already printed
    };

    // Update enabled flag
    let updated = conn.execute(
        "UPDATE watch_folders SET enabled = 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE watch_id = ?1",
        params![resolved_id],
    )?;

    if updated > 0 {
        output::success(format!("Watch '{}' enabled", resolved_id));
        output::info("Daemon will pick up this watch on next poll cycle");
    } else {
        output::warning(format!("Watch '{}' not found or already enabled", resolved_id));
    }

    Ok(())
}

async fn disable(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    // Resolve watch_id (exact match or prefix match)
    let resolved_id = resolve_watch_id(&conn, watch_id)?;

    let resolved_id = match resolved_id {
        Some(id) => id,
        None => return Ok(()),  // Error already printed
    };

    // Update enabled flag
    let updated = conn.execute(
        "UPDATE watch_folders SET enabled = 0, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE watch_id = ?1",
        params![resolved_id],
    )?;

    if updated > 0 {
        output::success(format!("Watch '{}' disabled", resolved_id));
        output::info("Daemon will stop watching this folder on next poll cycle");
    } else {
        output::warning(format!("Watch '{}' not found or already disabled", resolved_id));
    }

    Ok(())
}

async fn show(watch_id: &str, json: bool) -> Result<()> {
    let conn = connect_readonly()?;

    // Try exact match first, then prefix match
    let query = r#"
        SELECT watch_id, path, collection, tenant_id, patterns, ignore_patterns,
               auto_ingest, recursive, recursive_depth, debounce_seconds,
               enabled, is_active, created_at, updated_at, last_scan_at,
               last_activity_at, parent_watch_id, metadata,
               COALESCE(is_paused, 0) as is_paused
        FROM watch_folders
        WHERE watch_id = ? OR watch_id LIKE ?
        LIMIT 1
    "#;

    let prefix = format!("{}%", watch_id);
    let mut stmt = conn.prepare(query)?;
    let result = stmt.query_row(params![watch_id, &prefix], |row| {
        Ok(WatchDetailItem {
            watch_id: row.get(0)?,
            path: row.get(1)?,
            collection: row.get(2)?,
            tenant_id: row.get(3)?,
            patterns: row.get(4)?,
            ignore_patterns: row.get(5)?,
            auto_ingest: row.get::<_, i32>(6)? != 0,
            recursive: row.get::<_, i32>(7)? != 0,
            recursive_depth: row.get(8)?,
            debounce_seconds: row.get(9)?,
            enabled: row.get::<_, i32>(10)? != 0,
            is_active: row.get::<_, i32>(11)? != 0,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
            last_scan: row.get(14)?,
            last_activity_at: row.get(15)?,
            parent_watch_id: row.get(16)?,
            metadata: row.get(17)?,
            is_paused: row.get::<_, i32>(18)? != 0,
        })
    });

    match result {
        Ok(item) => {
            if json {
                output::print_json(&item);
            } else {
                output::section("Watch Configuration Details");
                output::kv("Watch ID", &item.watch_id);
                output::kv("Path", &item.path);
                output::separator();
                output::kv("Collection", &item.collection);
                output::kv("Tenant ID", &item.tenant_id);
                output::separator();
                output::kv("Enabled", &format_bool(item.enabled));
                output::kv("Active", &format_bool(item.is_active));
                output::kv("Paused", &format_bool_paused(item.is_paused));
                output::kv("Auto-ingest", &format_bool(item.auto_ingest));
                output::kv("Recursive", &format_bool(item.recursive));
                if item.recursive {
                    output::kv("Recursive Depth", &item.recursive_depth.to_string());
                }
                output::kv("Debounce", &format!("{:.1}s", item.debounce_seconds));
                output::separator();
                output::kv("Patterns", &item.patterns);
                output::kv("Ignore Patterns", &item.ignore_patterns);
                output::separator();
                output::kv("Created At", &item.created_at);
                output::kv("Updated At", &item.updated_at);
                if let Some(ref scan) = item.last_scan {
                    output::kv("Last Scan", &format!("{} ({})", scan, format_relative_time(scan)));
                } else {
                    output::kv("Last Scan", "never");
                }
                if let Some(ref activity) = item.last_activity_at {
                    output::kv("Last Activity", &format!("{} ({})", activity, format_relative_time(activity)));
                }
                if let Some(ref parent) = item.parent_watch_id {
                    output::separator();
                    output::kv("Parent Watch", parent);
                    output::info("This is a submodule watch");
                }

                if let Some(ref meta) = item.metadata {
                    if meta != "{}" {
                        output::separator();
                        println!("{}", "Metadata:".bold());
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
                            if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                                for line in pretty.lines() {
                                    println!("  {}", line);
                                }
                            } else {
                                println!("  {}", meta);
                            }
                        } else {
                            println!("  {}", meta);
                        }
                    }
                }
            }
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Watch not found: {}", watch_id));
            output::info("Use 'wqm watch list' to see available watches");
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    Ok(())
}

async fn pause() -> Result<()> {
    let conn = connect_readwrite()?;

    let updated = conn.execute(
        "UPDATE watch_folders SET is_paused = 1, \
         pause_start_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE enabled = 1 AND is_paused = 0",
        [],
    )?;

    if updated > 0 {
        output::success(format!("Paused {} watch folder(s)", updated));
        output::info("File events will be buffered until watchers are resumed");
    } else {
        output::info("No active watchers to pause (all already paused or none enabled)");
    }

    Ok(())
}

async fn resume() -> Result<()> {
    let conn = connect_readwrite()?;

    let updated = conn.execute(
        "UPDATE watch_folders SET is_paused = 0, \
         pause_start_time = NULL, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE enabled = 1 AND is_paused = 1",
        [],
    )?;

    if updated > 0 {
        output::success(format!("Resumed {} watch folder(s)", updated));
        output::info("Buffered file events will be processed");
    } else {
        output::info("No paused watchers to resume");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_relative_time() {
        let now = Utc::now();
        let timestamp = (now - chrono::Duration::seconds(30)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.contains("s ago") || result.contains("0m ago"));

        let result = format_relative_time("invalid");
        assert_eq!(result, "never");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long string", 10), "this is...");
    }

    #[test]
    fn test_format_bool() {
        // Just verify it doesn't panic
        let _ = format_bool(true);
        let _ = format_bool(false);
    }
}
