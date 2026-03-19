//! Queue drop subcommand — remove failed items from the queue.

use anyhow::Result;
use rusqlite::params;

use crate::output;

use super::db::connect_readwrite;

pub async fn execute(
    queue_id: Option<String>,
    all_permanent: bool,
    all_stale: bool,
    yes: bool,
) -> Result<()> {
    if !all_permanent && !all_stale && queue_id.is_none() {
        anyhow::bail!("Specify a queue_id, --all-permanent, or --all-stale");
    }

    let conn = connect_readwrite()?;

    if all_permanent {
        drop_all_permanent(&conn, yes)
    } else if all_stale {
        drop_all_stale(&conn, yes)
    } else {
        drop_one(&conn, &queue_id.unwrap())
    }
}

fn drop_all_permanent(conn: &rusqlite::Connection, yes: bool) -> Result<()> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM unified_queue WHERE status = 'failed' \
         AND (error_message LIKE '[permanent_%' OR error_message LIKE '[permanent_exhausted]%')",
        [],
        |row| row.get(0),
    )?;

    if count == 0 {
        output::success("No permanently failed items to drop");
        return Ok(());
    }

    if !yes {
        output::warning(format!(
            "This will remove {} permanently failed item(s). Use -y to confirm.",
            count
        ));
        return Ok(());
    }

    let deleted = conn.execute(
        "DELETE FROM unified_queue WHERE status = 'failed' \
         AND (error_message LIKE '[permanent_%' OR error_message LIKE '[permanent_exhausted]%')",
        [],
    )?;

    output::success(format!("Dropped {} permanently failed item(s)", deleted));
    Ok(())
}

fn drop_all_stale(conn: &rusqlite::Connection, yes: bool) -> Result<()> {
    // Find failed file items where the file no longer exists on disk
    let mut stmt = conn.prepare(
        "SELECT queue_id, file_path FROM unified_queue \
         WHERE status = 'failed' AND item_type = 'file' AND file_path IS NOT NULL",
    )?;

    let stale_ids: Vec<String> = stmt
        .query_map([], |row| {
            let queue_id: String = row.get(0)?;
            let file_path: String = row.get(1)?;
            Ok((queue_id, file_path))
        })?
        .filter_map(|r| r.ok())
        .filter(|(_, path)| !std::path::Path::new(path).exists())
        .map(|(id, _)| id)
        .collect();

    if stale_ids.is_empty() {
        output::success("No stale failed items to drop");
        return Ok(());
    }

    if !yes {
        output::warning(format!(
            "This will remove {} stale failed item(s) (files no longer on disk). Use -y to confirm.",
            stale_ids.len()
        ));
        return Ok(());
    }

    let mut deleted = 0;
    for id in &stale_ids {
        deleted += conn.execute("DELETE FROM unified_queue WHERE queue_id = ?1", params![id])?;
    }

    output::success(format!("Dropped {} stale failed item(s)", deleted));
    Ok(())
}

fn drop_one(conn: &rusqlite::Connection, id: &str) -> Result<()> {
    let prefix = format!("{}%", id);

    let result: std::result::Result<(String, String), _> = conn.query_row(
        "SELECT queue_id, status FROM unified_queue \
         WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
        params![id, &prefix],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    match result {
        Ok((found_id, status)) => {
            if status != "failed" {
                output::warning(format!(
                    "Item {} has status '{}', not 'failed'. Only failed items can be dropped.",
                    found_id, status
                ));
                return Ok(());
            }

            conn.execute(
                "DELETE FROM unified_queue WHERE queue_id = ?1",
                params![&found_id],
            )?;

            output::success(format!("Dropped failed item {}", found_id));
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Queue item not found: {}", id));
        }
        Err(e) => return Err(e.into()),
    }

    Ok(())
}
