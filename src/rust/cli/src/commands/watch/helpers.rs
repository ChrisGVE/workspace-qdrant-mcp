//! Shared helpers for watch subcommands

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use colored::Colorize;
use rusqlite::Connection;

use crate::output::short_id;

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

/// Build a combined tenant_id → display name mapping that covers all
/// collection types: projects (from `watch_folders`), libraries (from
/// `watch_folders`), and uses the tenant_id itself for rules/scratchpad.
///
/// The mapping does NOT include collection prefixes — call
/// [`prefixed_display_name`] to add them when mixing collection types.
pub fn build_full_tenant_name_map(conn: &Connection) -> HashMap<String, String> {
    let mut map = build_tenant_name_map(conn);

    // Add library entries
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'libraries'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let (tenant_id, path) = r;
                map.entry(tenant_id.clone()).or_insert_with(|| {
                    path.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&tenant_id)
                        .to_string()
                });
            }
        }
    }

    map
}

/// Return the collection prefix for display: `prj:`, `lib:`, `rls:`, `scp:`.
pub fn collection_prefix(collection: &str) -> &'static str {
    match collection {
        "projects" => "prj:",
        "libraries" => "lib:",
        "rules" => "rls:",
        "scratchpad" => "scp:",
        _ => "",
    }
}

/// Build a prefixed display name from collection + tenant_id.
///
/// Uses `tenant_names` for project/library name lookup, falls back to
/// the tenant_id (or shortened version) for unknown entries.
pub fn prefixed_display_name(
    collection: &str,
    tenant_id: &str,
    tenant_names: &HashMap<String, String>,
) -> String {
    let prefix = collection_prefix(collection);
    let base = tenant_names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| tenant_id.to_string());
    format!("{prefix}{base}")
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

    #[test]
    fn test_collection_prefix() {
        assert_eq!(collection_prefix("projects"), "prj:");
        assert_eq!(collection_prefix("libraries"), "lib:");
        assert_eq!(collection_prefix("rules"), "rls:");
        assert_eq!(collection_prefix("scratchpad"), "scp:");
        assert_eq!(collection_prefix("unknown"), "");
    }

    #[test]
    fn test_prefixed_display_name_with_known_tenant() {
        let mut names = HashMap::new();
        names.insert("abc123".to_string(), "my-project".to_string());
        assert_eq!(
            prefixed_display_name("projects", "abc123", &names),
            "prj:my-project"
        );
        assert_eq!(
            prefixed_display_name("libraries", "abc123", &names),
            "lib:my-project"
        );
    }

    #[test]
    fn test_prefixed_display_name_with_unknown_tenant() {
        let names = HashMap::new();
        assert_eq!(
            prefixed_display_name("projects", "deadbeef", &names),
            "prj:deadbeef"
        );
        assert_eq!(
            prefixed_display_name("scratchpad", "global", &names),
            "scp:global"
        );
    }
}
