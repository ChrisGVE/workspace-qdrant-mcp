//! Data types and SQLite fetching logic for the TUI rules browser.

use crate::data::db::connect_readonly;

/// A rule for display in the TUI rules list.
#[derive(Debug, Clone)]
pub struct RuleRow {
    pub rule_id: String,
    pub rule_text: String,
    pub scope: String,
    pub tenant_id: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Fetch all rules from the rules_mirror table.
pub fn fetch_rule_rows() -> Vec<RuleRow> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut stmt = match conn.prepare(
        "SELECT rule_id, rule_text, COALESCE(scope, 'global'), \
         COALESCE(tenant_id, ''), created_at, updated_at \
         FROM rules_mirror ORDER BY scope, created_at",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let rows = stmt
        .query_map([], |row| {
            Ok(RuleRow {
                rule_id: row.get(0)?,
                rule_text: row.get(1)?,
                scope: row.get(2)?,
                tenant_id: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
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
    fn fetch_rule_rows_does_not_panic() {
        // May return empty if no DB, but should not panic
        let _ = fetch_rule_rows();
    }
}
