//! Admin command - system administration operations
//!
//! Provides administrative operations that affect the daemon's persistent state.
//! Subcommands: rename-tenant

use std::io::Write as _;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::output;

/// Canonical collection names (validated against wqm-common constants)
const VALID_COLLECTIONS: &[&str] = &[
    wqm_common::constants::COLLECTION_PROJECTS,
    wqm_common::constants::COLLECTION_LIBRARIES,
    wqm_common::constants::COLLECTION_MEMORY,
];

/// Admin command arguments
#[derive(Args)]
pub struct AdminArgs {
    #[command(subcommand)]
    command: AdminCommand,
}

/// Admin subcommands
#[derive(Subcommand)]
enum AdminCommand {
    /// Rename a tenant_id across SQLite tables (watch_folders, unified_queue, tracked_files)
    RenameTenant {
        /// Current tenant_id to rename
        old_id: String,

        /// New tenant_id value
        new_id: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },
    /// Show idle/active state transition history and flip-flop analysis
    IdleHistory {
        /// Hours of history to analyze (default: 24)
        #[arg(short = 'H', long, default_value = "24")]
        hours: f64,
    },
    /// Prune old log files from the canonical log directory
    PruneLogs {
        /// Only list files that would be deleted, without deleting
        #[arg(long)]
        dry_run: bool,

        /// Retention period in hours (default: 36)
        #[arg(long, default_value = "36")]
        retention_hours: u64,
    },
}

/// Execute admin command
pub async fn execute(args: AdminArgs) -> Result<()> {
    match args.command {
        AdminCommand::RenameTenant { old_id, new_id, yes } => {
            rename_tenant(old_id, new_id, yes).await
        }
        AdminCommand::IdleHistory { hours } => {
            show_idle_history(hours)
        }
        AdminCommand::PruneLogs { dry_run, retention_hours } => {
            prune_logs(dry_run, retention_hours)
        }
    }
}

/// Rename a tenant_id across SQLite (via daemon gRPC) and verify
async fn rename_tenant(old_id: String, new_id: String, yes: bool) -> Result<()> {
    output::section("Tenant Rename");

    output::kv("From", &old_id);
    output::kv("To", &new_id);
    output::separator();

    output::warning("This will rename the tenant_id in SQLite tables (watch_folders, unified_queue, tracked_files).");
    output::info("Qdrant payloads (project_id field) will NOT be updated automatically.");
    output::info("To update Qdrant, reset the affected collections after rename: wqm collections reset projects");
    println!();

    // Confirmation
    if !yes {
        print!(
            "  To confirm, type the old tenant_id '{}': ",
            old_id
        );
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input != old_id {
            output::error(format!("Expected '{}', got '{}'. Aborting.", old_id, input));
            return Ok(());
        }
        println!();
    }

    // Call daemon gRPC
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = crate::grpc::proto::RenameTenantRequest {
                old_tenant_id: old_id.clone(),
                new_tenant_id: new_id.clone(),
                collections: VALID_COLLECTIONS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            };

            match client.project().rename_tenant(request).await {
                Ok(response) => {
                    let resp = response.into_inner();
                    if resp.success {
                        output::success(format!(
                            "Renamed '{}' -> '{}': {} SQLite rows updated",
                            old_id, new_id, resp.sqlite_rows_updated
                        ));
                    } else {
                        output::error(format!("Rename failed: {}", resp.message));
                    }
                }
                Err(e) => {
                    output::error(format!("gRPC call failed: {}", e));
                    output::info("Falling back to direct SQLite rename...");

                    // Fallback to direct SQLite
                    match rename_tenant_direct(&old_id, &new_id).await {
                        Ok(count) => {
                            output::success(format!(
                                "Direct rename '{}' -> '{}': {} SQLite rows updated",
                                old_id, new_id, count
                            ));
                        }
                        Err(e) => {
                            output::error(format!("Direct rename failed: {}", e));
                        }
                    }
                }
            }
        }
        Err(_) => {
            output::warning("Daemon not running, using direct SQLite access");

            match rename_tenant_direct(&old_id, &new_id).await {
                Ok(count) => {
                    output::success(format!(
                        "Direct rename '{}' -> '{}': {} SQLite rows updated",
                        old_id, new_id, count
                    ));
                }
                Err(e) => {
                    output::error(format!("Direct rename failed: {}", e));
                }
            }
        }
    }

    Ok(())
}

/// Show idle state transition history and flip-flop analysis
fn show_idle_history(hours: f64) -> Result<()> {
    use serde::Deserialize;
    use std::io::BufRead;
    use tabled::Tabled;

    use crate::output::ColumnHints;

    #[derive(Deserialize)]
    struct Entry {
        timestamp: String,
        from_mode: String,
        to_mode: String,
        idle_seconds: f64,
        duration_in_previous_secs: f64,
    }

    #[derive(Tabled)]
    struct IdleHistoryRow {
        #[tabled(rename = "Timestamp")]
        timestamp: String,
        #[tabled(rename = "From")]
        from_mode: String,
        #[tabled(rename = "To")]
        to_mode: String,
        #[tabled(rename = "Idle")]
        idle: String,
        #[tabled(rename = "Duration")]
        duration: String,
    }

    impl ColumnHints for IdleHistoryRow {
        fn content_columns() -> &'static [usize] { &[] }
    }

    let config_dir = wqm_common::paths::get_config_dir()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let history_path = config_dir.join("idle_history.jsonl");

    if !history_path.exists() {
        output::info("No idle history file found. The daemon records transitions automatically.");
        output::kv("Expected path", &history_path.display().to_string());
        return Ok(());
    }

    let file = std::fs::File::open(&history_path)
        .context("Failed to open idle history file")?;
    let reader = std::io::BufReader::new(file);

    // Compute cutoff timestamp using wqm_common (avoids chrono dependency)
    let cutoff_str = wqm_common::timestamps::hours_ago(hours);

    let entries: Vec<Entry> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str::<Entry>(&line).ok())
        .filter(|e| e.timestamp >= cutoff_str)
        .collect();

    output::section(&format!("Idle History (last {:.0}h)", hours));
    output::kv("File", &history_path.display().to_string());
    output::kv("Transitions", &entries.len().to_string());

    if entries.is_empty() {
        output::info("No transitions in the specified window.");
        return Ok(());
    }

    // Analysis
    let transitions_per_hour = entries.len() as f64 / hours;
    let avg_duration = entries.iter().map(|e| e.duration_in_previous_secs).sum::<f64>()
        / entries.len() as f64;
    let short_count = entries.iter().filter(|e| e.duration_in_previous_secs < 30.0).count();
    let is_flip_flopping = transitions_per_hour > 10.0;

    output::separator();
    output::kv("Rate", &format!("{:.1} transitions/hr", transitions_per_hour));
    output::kv("Avg mode duration", &wqm_common::duration_fmt::format_duration(avg_duration, 0));
    output::kv("Short (<30s)", &short_count.to_string());

    if is_flip_flopping {
        output::separator();
        output::warning("Flip-flop detected! Consider increasing idle_cooloff_polls in config.");
        let recommended = ((transitions_per_hour / 10.0).ceil() as u32).saturating_sub(1);
        output::kv("Recommended +cooloff", &format!("+{} polls", recommended));
    }

    // Show last 20 transitions in a table
    let tail: Vec<&Entry> = entries.iter().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect();
    let idle_secs: Vec<f64> = tail.iter().map(|e| e.idle_seconds).collect();
    let dur_secs: Vec<f64> = tail.iter().map(|e| e.duration_in_previous_secs).collect();
    let idle_fmt = wqm_common::duration_fmt::format_duration_column(&idle_secs);
    let dur_fmt = wqm_common::duration_fmt::format_duration_column(&dur_secs);

    let rows: Vec<IdleHistoryRow> = tail.iter().enumerate().map(|(i, entry)| {
        IdleHistoryRow {
            timestamp: wqm_common::timestamp_fmt::format_local(&entry.timestamp),
            from_mode: entry.from_mode.clone(),
            to_mode: entry.to_mode.clone(),
            idle: idle_fmt[i].clone(),
            duration: dur_fmt[i].clone(),
        }
    }).collect();

    output::separator();
    output::print_table_auto(&rows);

    Ok(())
}

/// Prune old log files from the canonical log directory
fn prune_logs(dry_run: bool, retention_hours: u64) -> Result<()> {
    use tabled::Tabled;
    use crate::output::ColumnHints;

    let log_dir = wqm_common::paths::get_canonical_log_dir();

    output::section("Log Pruning");
    output::kv("Log directory", &log_dir.display().to_string());
    output::kv("Retention", &format!("{}h", retention_hours));
    if dry_run {
        output::info("Dry run — no files will be deleted");
    }
    output::separator();

    let result = workspace_qdrant_core::log_pruner::prune_now(&log_dir, retention_hours, dry_run)
        .context("Failed to prune logs")?;

    if result.candidates.is_empty() {
        output::success("No log files older than the retention period.");
        return Ok(());
    }

    #[derive(Tabled)]
    struct PruneRow {
        #[tabled(rename = "File")]
        file: String,
        #[tabled(rename = "Size")]
        size: String,
        #[tabled(rename = "Age")]
        age: String,
    }

    impl ColumnHints for PruneRow {
        fn content_columns() -> &'static [usize] { &[] }
    }

    let rows: Vec<PruneRow> = result.candidates.iter().map(|c| {
        PruneRow {
            file: c.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string(),
            size: format_bytes(c.size),
            age: wqm_common::duration_fmt::format_duration(c.age_hours * 3600.0, 0),
        }
    }).collect();

    output::print_table_auto(&rows);
    output::separator();

    if dry_run {
        output::info(format!(
            "Would delete {} file(s), freeing {}",
            result.candidates.len(),
            format_bytes(result.candidates.iter().map(|c| c.size).sum()),
        ));
    } else {
        output::success(format!(
            "Deleted {} file(s), freed {}",
            result.files_deleted,
            format_bytes(result.bytes_freed),
        ));
    }

    Ok(())
}

/// Format bytes into a human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Direct SQLite rename (fallback when daemon is not running)
async fn rename_tenant_direct(old_id: &str, new_id: &str) -> Result<usize> {
    let db_path = crate::config::get_database_path()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn = rusqlite::Connection::open(&db_path)
        .context("Failed to open state database")?;

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;

    let tx = conn.unchecked_transaction()
        .context("Failed to begin transaction")?;

    let mut total = 0usize;

    // Update watch_folders
    let count = tx.execute(
        "UPDATE watch_folders SET tenant_id = ?1 WHERE tenant_id = ?2",
        rusqlite::params![new_id, old_id],
    ).context("Failed to update watch_folders")?;
    total += count;

    // Update unified_queue
    let count = tx.execute(
        "UPDATE unified_queue SET tenant_id = ?1 WHERE tenant_id = ?2",
        rusqlite::params![new_id, old_id],
    ).context("Failed to update unified_queue")?;
    total += count;

    // Update tracked_files (may not exist)
    match tx.execute(
        "UPDATE tracked_files SET tenant_id = ?1 WHERE tenant_id = ?2",
        rusqlite::params![new_id, old_id],
    ) {
        Ok(count) => total += count,
        Err(_) => {} // Table may not exist, ignore
    }

    tx.commit().context("Failed to commit rename")?;

    Ok(total)
}
