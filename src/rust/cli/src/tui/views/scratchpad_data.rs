//! Data types and SQLite fetching logic for the TUI scratchpad browser.

use crate::data::db::connect_readonly;

/// A scratchpad entry for display in the TUI.
#[derive(Debug, Clone)]
pub struct ScratchpadRow {
    pub scratchpad_id: String,
    pub title: String,
    pub content: String,
    pub tags: String,
    pub tenant_id: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Fetch all scratchpad entries from the scratchpad_mirror table.
pub fn fetch_scratchpad_rows() -> Vec<ScratchpadRow> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut stmt = match conn.prepare(
        "SELECT scratchpad_id, COALESCE(title, '(untitled)'), content, \
         COALESCE(tags, '[]'), tenant_id, created_at, updated_at \
         FROM scratchpad_mirror ORDER BY updated_at DESC",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let rows = stmt
        .query_map([], |row| {
            Ok(ScratchpadRow {
                scratchpad_id: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
                tags: row.get(3)?,
                tenant_id: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })
        .ok();

    match rows {
        Some(iter) => iter.filter_map(|r| r.ok()).collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_scratchpad_rows_does_not_panic() {
        let _ = fetch_scratchpad_rows();
    }
}
