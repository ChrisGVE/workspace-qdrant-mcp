//! Queue clean subcommand

use anyhow::Result;

use crate::output;

use super::db::connect_readwrite;

pub async fn execute(days: i64, status_filter: Option<String>, yes: bool) -> Result<()> {
    // Validate status filter
    let valid_statuses = match status_filter.as_deref() {
        Some("done") => vec!["done"],
        Some("failed") => vec!["failed"],
        Some(other) => anyhow::bail!("Invalid status '{}'. Use 'done' or 'failed'.", other),
        None => vec!["done", "failed"],
    };

    let conn = connect_readwrite()?;

    // Count items to be cleaned
    let placeholders = valid_statuses
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(",");
    let count_query = format!(
        "SELECT COUNT(*) FROM unified_queue \
         WHERE status IN ({}) AND updated_at < datetime('now', '-{} days')",
        placeholders, days
    );

    let mut stmt = conn.prepare(&count_query)?;
    let params_slice: Vec<&dyn rusqlite::ToSql> = valid_statuses
        .iter()
        .map(|s| s as &dyn rusqlite::ToSql)
        .collect();
    let count: i64 = stmt.query_row(params_slice.as_slice(), |row| row.get(0))?;

    if count == 0 {
        output::success(format!(
            "No {} items older than {} days to clean",
            valid_statuses.join("/"),
            days
        ));
        return Ok(());
    }

    if !yes {
        output::warning(format!(
            "Will remove {} {} items older than {} days. Use -y to skip this confirmation.",
            count,
            valid_statuses.join("/"),
            days
        ));
        return Ok(());
    }

    let delete_query = format!(
        "DELETE FROM unified_queue \
         WHERE status IN ({}) AND updated_at < datetime('now', '-{} days')",
        placeholders, days
    );

    let mut stmt = conn.prepare(&delete_query)?;
    let deleted = stmt.execute(params_slice.as_slice())?;

    output::success(format!("Removed {} old queue items", deleted));

    Ok(())
}
