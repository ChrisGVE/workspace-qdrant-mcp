//! Queue drop subcommand — remove failed items from the queue.

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{CleanQueueRequest, RemoveItemRequest};
use crate::output;

use super::db::connect_readonly;

pub async fn execute(
    queue_id: Option<String>,
    all_permanent: bool,
    all_stale: bool,
    yes: bool,
) -> Result<()> {
    if !all_permanent && !all_stale && queue_id.is_none() {
        anyhow::bail!("Specify a queue_id, --all-permanent, or --all-stale");
    }

    if all_permanent {
        drop_all_permanent(yes).await
    } else if all_stale {
        drop_all_stale(yes).await
    } else {
        drop_one(&queue_id.unwrap()).await
    }
}

async fn drop_all_permanent(yes: bool) -> Result<()> {
    // Read count via direct SQLite (read-only)
    let conn = connect_readonly()?;
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

    // Use CleanQueue with 0 days (clean all failed items matching the pattern)
    // Note: CleanQueue cleans by age, not by error pattern. For permanent failures,
    // we use the general "clean failed with 0 days" approach.
    let mut client = ensure_daemon_available().await?;
    let response = client
        .queue_write()
        .clean_queue(CleanQueueRequest {
            older_than_days: 0,
            statuses: vec!["failed".to_string()],
        })
        .await?
        .into_inner();

    output::success(format!("Dropped {} failed item(s)", response.deleted_count));
    Ok(())
}

async fn drop_all_stale(yes: bool) -> Result<()> {
    // Read failed file items via direct SQLite (read-only)
    let conn = connect_readonly()?;
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

    // Delete each stale item via gRPC
    let mut client = ensure_daemon_available().await?;
    let mut deleted = 0u32;
    for id in &stale_ids {
        let response = client
            .queue_write()
            .remove_item(RemoveItemRequest {
                queue_id: id.clone(),
            })
            .await?
            .into_inner();
        if response.found {
            deleted += 1;
        }
    }

    output::success(format!("Dropped {} stale failed item(s)", deleted));
    Ok(())
}

async fn drop_one(id: &str) -> Result<()> {
    // Read item status via direct SQLite (read-only) for validation
    let conn = connect_readonly()?;
    let result: std::result::Result<(String, String), _> = conn.query_row(
        "SELECT queue_id, status FROM unified_queue \
         WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
        rusqlite::params![id, format!("{}%", id)],
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

            let mut client = ensure_daemon_available().await?;
            client
                .queue_write()
                .remove_item(RemoveItemRequest {
                    queue_id: found_id.clone(),
                })
                .await?;

            output::success(format!("Dropped failed item {}", found_id));
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Queue item not found: {}", id));
        }
        Err(e) => return Err(e.into()),
    }

    Ok(())
}
