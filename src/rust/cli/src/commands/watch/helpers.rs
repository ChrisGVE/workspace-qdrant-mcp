//! Shared helpers for watch subcommands

use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use colored::Colorize;
use rusqlite::Connection;

use crate::config::get_database_path_checked;
use crate::output::short_id;

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

/// Build a `tenant_id` to project name mapping from `watch_folders`.
///
/// When multiple projects share the same directory name, the tenant_id is
/// appended in parentheses to disambiguate.  Falls back gracefully if the
/// `watch_folders` table does not exist.
pub fn build_tenant_name_map(conn: &Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();

    let mut entries: Vec<(String, String)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name =
                    r.1.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&r.0)
                        .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

    for (tenant_id, name) in entries {
        let display = if name_count.get(&name).copied().unwrap_or(0) > 1 {
            format!("{} ({})", name, short_id(&tenant_id))
        } else {
            name
        };
        map.insert(tenant_id, display);
    }

    map
}

/// Resolve a tenant_id to a human-readable project name, falling back to
/// a shortened tenant_id when no mapping exists.
pub fn resolve_project_name(tenant_id: &str, tenant_names: &HashMap<String, String>) -> String {
    tenant_names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| short_id(tenant_id))
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
