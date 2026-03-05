//! Library add subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::timestamps;

use super::helpers::{open_db, signal_daemon_watch_folders, LibraryMode};
use crate::output;

/// Add a library (unwatched - metadata only)
pub async fn execute(tag: &str, path: &PathBuf, mode: LibraryMode) -> Result<()> {
    output::section(format!("Add Library: {}", tag));

    // Validate path exists
    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path
        .canonicalize()
        .context("Could not resolve absolute path")?;

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();
    let abs_path_str = abs_path.to_string_lossy().to_string();

    // Check for duplicate
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM watch_folders WHERE watch_id = ?",
            [&watch_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if exists {
        output::error(format!(
            "Library '{}' already exists. Use 'wqm library config' to update it.",
            tag
        ));
        return Ok(());
    }

    // Check for duplicate path
    let path_exists: bool = conn
        .query_row(
            "SELECT 1 FROM watch_folders WHERE path = ?",
            [&abs_path_str],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if path_exists {
        output::error(format!(
            "Path '{}' is already registered.",
            abs_path.display()
        ));
        return Ok(());
    }

    // Insert into watch_folders (enabled=0 for add, use watch to enable)
    conn.execute(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
          follow_symlinks, cleanup_on_disable, created_at, updated_at) \
         VALUES (?1, ?2, 'libraries', ?3, ?4, 0, 0, 0, 0, ?5, ?5)",
        rusqlite::params![&watch_id, &abs_path_str, tag, &mode.to_string(), &now],
    )
    .context("Failed to insert library into watch_folders")?;

    output::success(format!("Library '{}' added (not watching yet)", tag));
    output::kv("  Tag", tag);
    output::kv("  Path", &abs_path_str);
    output::kv("  Mode", &mode.to_string());
    output::separator();
    output::info("To start watching: wqm library watch <tag> <path>");

    // Signal daemon if available
    signal_daemon_watch_folders().await;

    Ok(())
}
