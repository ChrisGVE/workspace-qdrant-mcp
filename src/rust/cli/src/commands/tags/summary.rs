//! Summary subcommand handler: `tags summary` — tag frequency aggregation per tenant.

use anyhow::Result;
use tabled::Tabled;

use crate::data::db::{connect_readonly, table_exists};
use crate::output;

#[derive(Tabled, serde::Serialize)]
pub(super) struct SummaryRow {
    #[tabled(rename = "Tag")]
    pub tag: String,
    #[tabled(rename = "Documents")]
    pub doc_count: i64,
    #[tabled(rename = "Avg Score")]
    pub avg_score: String,
}

/// Show tag frequency summary for a tenant, ordered by document count descending.
pub(super) fn show_summary(
    tenant_id: &str,
    collection: &str,
    top: usize,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let conn = connect_readonly()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let mut stmt = conn.prepare(
        "SELECT tag, COUNT(DISTINCT doc_id) AS doc_count, AVG(score) AS avg_score \
         FROM tags \
         WHERE tenant_id = ? AND collection = ? AND tag_type = 'concept' \
         GROUP BY tag \
         ORDER BY doc_count DESC \
         LIMIT ?",
    )?;

    let limit = top as i64;
    let rows: Vec<SummaryRow> = stmt
        .query_map(rusqlite::params![tenant_id, collection, limit], |row| {
            let avg: f64 = row.get(2)?;
            Ok(SummaryRow {
                tag: row.get(0)?,
                doc_count: row.get(1)?,
                avg_score: format!("{:.3}", avg),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.is_empty() {
        output::info("No tags found for project. Tags populate after the tagging pipeline runs.");
        return Ok(());
    }

    if json {
        output::print_json(&rows);
    } else if script {
        output::print_script(&rows, !no_headers);
    } else {
        let count = rows.len();
        output::print_table(&rows);
        output::summary(output::summary_line(count, count, "tags"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_row_fields() {
        let row = SummaryRow {
            tag: "async runtime".to_string(),
            doc_count: 12,
            avg_score: "0.854".to_string(),
        };
        assert_eq!(row.tag, "async runtime");
        assert_eq!(row.doc_count, 12);
        assert_eq!(row.avg_score, "0.854");
    }

    #[test]
    fn summary_row_serializes_to_json() {
        let row = SummaryRow {
            tag: "error handling".to_string(),
            doc_count: 7,
            avg_score: "0.721".to_string(),
        };
        let json = serde_json::to_string(&row).unwrap();
        assert!(json.contains("\"tag\":\"error handling\""));
        assert!(json.contains("\"doc_count\":7"));
        assert!(json.contains("\"avg_score\":\"0.721\""));
    }
}
