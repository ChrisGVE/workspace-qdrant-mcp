//! Queue remove subcommand

use anyhow::Result;
use rusqlite::params;

use crate::output;

use super::db::connect_readwrite;
use super::formatters::format_status;

pub async fn execute(queue_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    // Look up item using exact match first, then prefix match (same as show)
    let lookup_query = r#"
        SELECT queue_id, item_type, op, collection, status
        FROM unified_queue
        WHERE queue_id = ? OR queue_id LIKE ? OR idempotency_key LIKE ?
        LIMIT 1
    "#;

    let prefix = format!("{}%", queue_id);
    let mut stmt = conn.prepare(lookup_query)?;
    let result = stmt.query_row(params![queue_id, &prefix, &prefix], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    });

    match result {
        Ok((resolved_id, item_type, op, collection, status)) => {
            // Warn if item is currently being processed
            if status == "in_progress" {
                output::warning(format!(
                    "Item {} is currently in_progress -- removing may cause the processor to error",
                    &resolved_id[..8.min(resolved_id.len())]
                ));
            }

            // Delete using the resolved exact ID
            conn.execute(
                "DELETE FROM unified_queue WHERE queue_id = ?",
                params![&resolved_id],
            )?;

            output::success(format!(
                "Removed queue item {} ({} {} in {}), was {}",
                &resolved_id[..8.min(resolved_id.len())],
                item_type,
                op,
                collection,
                format_status(&status),
            ));
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Queue item not found: {}", queue_id));
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    Ok(())
}
