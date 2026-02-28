//! Watch ID resolution logic

use anyhow::Result;
use rusqlite::{params, Connection};

use crate::output;

/// Resolve watch_id by exact match or prefix match.
///
/// Returns `Some(resolved_id)` if found, `None` if not found (error already
/// printed).
pub fn resolve_watch_id(conn: &Connection, watch_id: &str) -> Result<Option<String>> {
    // First check if exact match exists
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM watch_folders WHERE watch_id = ?1",
            params![watch_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if exists {
        return Ok(Some(watch_id.to_string()));
    }

    // Try prefix match
    let prefix = format!("{}%", watch_id);
    let matches: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT watch_id FROM watch_folders \
             WHERE watch_id LIKE ?1 LIMIT 5",
        )?;
        let rows = stmt.query_map(params![&prefix], |row| row.get(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };

    if matches.is_empty() {
        output::error(format!("Watch not found: {}", watch_id));
        output::info("Use 'wqm watch list' to see available watches");
        Ok(None)
    } else if matches.len() == 1 {
        // Single match, use it
        Ok(Some(matches[0].clone()))
    } else {
        output::error(format!(
            "Ambiguous watch ID '{}', multiple matches:",
            watch_id
        ));
        for m in &matches {
            println!("  - {}", m);
        }
        Ok(None)
    }
}
