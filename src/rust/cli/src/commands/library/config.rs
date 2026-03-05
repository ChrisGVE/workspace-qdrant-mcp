//! Library config subcommand

use anyhow::{Context, Result};
use wqm_common::timestamps;

use super::helpers::{open_db, signal_daemon_watch_folders, LibraryMode};
use crate::output;

/// Configure library settings
pub async fn execute(
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
        show_current_config(&conn, tag, &watch_id)?;

        if mode.is_some() || enable || disable {
            output::separator();
        }
    }

    // Apply configuration changes
    let changes_made = apply_changes(&conn, &watch_id, &now, mode, enable, disable)?;

    if let Some(ref pat) = patterns {
        let parsed: Vec<&str> = pat
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        output::info(format!(
            "Patterns: {} (applied at next scan/rescan)",
            parsed.join(", ")
        ));
    }

    if changes_made {
        output::success("Configuration updated");
        signal_daemon_watch_folders().await;
    }

    Ok(())
}

/// Display the current library configuration
fn show_current_config(conn: &rusqlite::Connection, tag: &str, watch_id: &str) -> Result<()> {
    output::info("Current configuration:");
    output::separator();

    let result: Result<(String, Option<String>, i32, bool), _> = conn.query_row(
        "SELECT path, library_mode, enabled, follow_symlinks \
         FROM watch_folders WHERE watch_id = ?",
        [watch_id],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get::<_, i32>(3)? != 0,
            ))
        },
    );

    match result {
        Ok((path, lib_mode, enabled, follow_symlinks)) => {
            output::kv("Tag", tag);
            output::kv("Watch ID", watch_id);
            output::kv("Path", &path);
            output::kv("Mode", lib_mode.as_deref().unwrap_or("incremental"));
            output::kv("Enabled", if enabled == 1 { "yes" } else { "no" });
            output::kv(
                "Follow Symlinks",
                if follow_symlinks { "yes" } else { "no" },
            );
        }
        Err(e) => {
            output::error(format!("Failed to read configuration: {}", e));
        }
    }

    Ok(())
}

/// Apply mode, enable, and disable changes. Returns whether any changes were made.
fn apply_changes(
    conn: &rusqlite::Connection,
    watch_id: &str,
    now: &str,
    mode: Option<LibraryMode>,
    enable: bool,
    disable: bool,
) -> Result<bool> {
    let mut changes_made = false;

    if let Some(new_mode) = mode {
        output::info(format!("Setting mode to: {}", new_mode));
        conn.execute(
            "UPDATE watch_folders SET library_mode = ?, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![&new_mode.to_string(), now, watch_id],
        )
        .context("Failed to update mode")?;
        changes_made = true;
    }

    if enable {
        output::info("Enabling watch...");
        conn.execute(
            "UPDATE watch_folders SET enabled = 1, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![now, watch_id],
        )
        .context("Failed to enable")?;
        changes_made = true;
    }

    if disable {
        output::info("Disabling watch...");
        conn.execute(
            "UPDATE watch_folders SET enabled = 0, updated_at = ? WHERE watch_id = ?",
            rusqlite::params![now, watch_id],
        )
        .context("Failed to disable")?;
        changes_made = true;
    }

    Ok(changes_made)
}
