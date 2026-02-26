//! Queue retry subcommand

use anyhow::Result;
use rusqlite::params;

use crate::output;

use super::db::connect_readwrite;

pub async fn execute(queue_id: Option<String>, all: bool) -> Result<()> {
    if !all && queue_id.is_none() {
        anyhow::bail!("Specify a queue_id or use --all to retry all failed items");
    }

    let conn = connect_readwrite()?;

    if all {
        retry_all(&conn)
    } else {
        retry_one(&conn, &queue_id.unwrap())
    }
}

fn retry_all(conn: &rusqlite::Connection) -> Result<()> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM unified_queue WHERE status = 'failed'",
        [],
        |row| row.get(0),
    )?;

    if count == 0 {
        output::success("No failed items to retry");
        return Ok(());
    }

    let updated = conn.execute(
        r#"
        UPDATE unified_queue
        SET status = 'pending',
            retry_count = 0,
            error_message = NULL,
            last_error_at = NULL,
            lease_until = NULL,
            worker_id = NULL,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE status = 'failed'
        "#,
        [],
    )?;

    output::success(format!("Reset {} failed items to pending", updated));
    Ok(())
}

fn retry_one(conn: &rusqlite::Connection, id: &str) -> Result<()> {
    let prefix = format!("{}%", id);

    let result: Result<(String, String, i32), _> = conn.query_row(
        "SELECT queue_id, status, retry_count FROM unified_queue \
         WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
        params![id, &prefix],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    );

    match result {
        Ok((found_id, status, retry_count)) => {
            if status != "failed" {
                output::warning(format!(
                    "Item {} has status '{}', not 'failed'. \
                     Use --status filter with list to find failed items.",
                    found_id, status
                ));
                return Ok(());
            }

            conn.execute(
                r#"
                UPDATE unified_queue
                SET status = 'pending',
                    retry_count = 0,
                    error_message = NULL,
                    last_error_at = NULL,
                    lease_until = NULL,
                    worker_id = NULL,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE queue_id = ?1
                "#,
                params![&found_id],
            )?;

            output::success(format!(
                "Reset item {} to pending (was retry {})",
                found_id, retry_count
            ));
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Queue item not found: {}", id));
        }
        Err(e) => return Err(e.into()),
    }

    Ok(())
}
