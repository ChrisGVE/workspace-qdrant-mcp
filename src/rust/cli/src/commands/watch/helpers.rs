//! Shared helpers for watch subcommands

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use colored::Colorize;
use rusqlite::Connection;

use crate::config::get_database_path_checked;

/// Connect to the state database (read-only for list/show)
pub fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    Ok(conn)
}

/// Connect to the state database (read-write for enable/disable)
pub fn connect_readwrite() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open(&db_path)
        .context(format!("Failed to open state database at {:?}", db_path))?;

    // Enable WAL mode for better concurrency
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
    )
    .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

/// Format relative time from ISO timestamp
pub fn format_relative_time(timestamp_str: &str) -> String {
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

/// Format enabled/active status with color
pub fn format_bool(value: bool) -> String {
    if value {
        "yes".green().to_string()
    } else {
        "no".red().to_string()
    }
}

/// Format paused status with color (paused = yellow, not paused = green)
pub fn format_bool_paused(value: bool) -> String {
    if value {
        "yes".yellow().to_string()
    } else {
        "no".green().to_string()
    }
}

/// Format archived status with color (archived = yellow, not archived = green)
pub fn format_bool_archived(value: bool) -> String {
    if value {
        "yes".yellow().to_string()
    } else {
        "no".green().to_string()
    }
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
    fn test_format_bool() {
        // Just verify it doesn't panic
        let _ = format_bool(true);
        let _ = format_bool(false);
    }
}
