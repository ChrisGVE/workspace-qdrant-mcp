//! Configuration migration helpers.
//!
//! Provides safety checks and the shared migration logic used by the
//! `default` and `xdg` subcommands when moving configuration (and
//! optionally the state database) to a new location.

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::daemon::{is_daemon_running, start_daemon, stop_daemon};
use crate::output;

// =========================================================================
// Safety checks
// =========================================================================

/// Check for active MCP server sessions via operational_state table.
/// Returns true if an active session was detected.
pub(super) fn check_active_mcp_sessions() -> Result<bool> {
    let db_path = wqm_common::paths::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        // No database means no sessions
        return Ok(false);
    }

    let conn = rusqlite::Connection::open(&db_path).context("Failed to open state database")?;

    // Check if operational_state table exists (schema v17+)
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='operational_state'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        return Ok(false);
    }

    // Look for any server component entries with recent updated_at (within last 10 minutes)
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM operational_state
             WHERE component = 'server'
             AND updated_at > datetime('now', '-10 minutes')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    Ok(count > 0)
}

/// Perform safety checks before config migration.
/// Returns an error if migration should not proceed.
pub(super) async fn pre_migration_checks() -> Result<()> {
    // Check for active MCP sessions
    if check_active_mcp_sessions()? {
        bail!(
            "Active MCP server session detected.\n\
             Close all MCP server connections before moving configuration.\n\
             Then retry this command."
        );
    }
    Ok(())
}

// =========================================================================
// File operations
// =========================================================================

/// Move a config file from source to target, creating parent dirs.
/// Uses copy+delete for cross-device safety.
pub(super) fn move_file(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    // Copy then delete for cross-device safety (atomic rename only works same-device)
    std::fs::copy(source, target)
        .with_context(|| format!("Failed to copy {} → {}", source.display(), target.display()))?;

    std::fs::remove_file(source).with_context(|| {
        format!(
            "Failed to remove source file after copy: {}",
            source.display()
        )
    })?;

    Ok(())
}

/// Copy a file from source to target, creating parent dirs.
pub(super) fn copy_file(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    std::fs::copy(source, target)
        .with_context(|| format!("Failed to copy {} → {}", source.display(), target.display()))?;

    Ok(())
}

// =========================================================================
// Shared migration logic
// =========================================================================

/// Shared migration logic: move config (and optionally database) to target location.
/// Returns Ok(()) on success or user abort.
pub(super) async fn migrate_config(
    target_config: &Path,
    target_data_dir: Option<&Path>,
) -> Result<()> {
    pre_migration_checks().await?;

    let current_config = wqm_common::paths::find_config_file();
    let current_db = wqm_common::paths::get_database_path().ok();

    // Determine what will happen
    output::section("Configuration Migration Plan");

    let already_at_target = current_config.as_ref().is_some_and(|p| p == target_config);

    if already_at_target {
        output::success(format!(
            "Configuration is already at {}",
            target_config.display()
        ));
        return Ok(());
    }

    // Show dry-run summary
    match &current_config {
        Some(source) => {
            output::kv("Current config", source.display().to_string());
            output::kv("Target config", target_config.display().to_string());
            output::info("Action: move config file to target location");
        }
        None => {
            output::info("No existing config file found");
            output::kv("Target config", target_config.display().to_string());
            output::info("Action: write default config to target location");
        }
    }

    // Show database migration plan if applicable
    if let (Some(data_dir), Some(db_path)) = (target_data_dir, &current_db) {
        if db_path.exists() {
            let target_db = data_dir.join("state.db");
            let db_already_there = db_path == &target_db;
            if !db_already_there {
                output::kv("Current database", db_path.display().to_string());
                output::kv("Target database", target_db.display().to_string());
                output::info("Action: copy database to target location");
            }
        }
    }

    output::separator();

    // Ask for confirmation
    if !output::confirm("Proceed with migration?") {
        output::info("Aborted.");
        return Ok(());
    }

    // Stop daemon
    let daemon_was_running = is_daemon_running().await;
    if daemon_was_running {
        stop_daemon().await?;
    }

    // Execute migration
    match &current_config {
        Some(source) => {
            move_file(source, target_config)?;
            output::success(format!(
                "Moved config: {} → {}",
                source.display(),
                target_config.display()
            ));
        }
        None => {
            if let Some(parent) = target_config.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
            }
            std::fs::write(target_config, wqm_common::yaml_defaults::DEFAULT_YAML).with_context(
                || {
                    format!(
                        "Failed to write default config to {}",
                        target_config.display()
                    )
                },
            )?;
            output::success(format!(
                "Wrote default config to {}",
                target_config.display()
            ));
        }
    }

    // Copy database if target data dir is different
    if let (Some(data_dir), Some(db_path)) = (target_data_dir, &current_db) {
        if db_path.exists() {
            let target_db = data_dir.join("state.db");
            if db_path != &target_db && !target_db.exists() {
                copy_file(db_path, &target_db)?;
                output::success(format!(
                    "Copied database: {} → {}",
                    db_path.display(),
                    target_db.display()
                ));
            }
        }
    }

    // Restart daemon if it was running
    if daemon_was_running {
        start_daemon().await?;
    }

    output::separator();
    output::success("Configuration migration complete");

    Ok(())
}
