//! Library unwatch subcommand

use anyhow::{Context, Result};
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::timestamps;

use super::helpers::{open_db, signal_daemon_watch_folders};
use crate::output;

/// Stop watching a library (preserves indexed content)
pub async fn execute(tag: &str) -> Result<()> {
    output::section(format!("Unwatch Library: {}", tag));

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();

    // Verify library exists
    let exists: bool = conn
        .query_row(
            &format!(
                "SELECT 1 FROM watch_folders WHERE watch_id = ? AND collection = '{}'",
                COLLECTION_LIBRARIES
            ),
            [&watch_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !exists {
        output::error(format!("Library '{}' not found", tag));
        return Ok(());
    }

    // Disable watching (keep the record for re-enabling later)
    conn.execute(
        "UPDATE watch_folders SET enabled = 0, updated_at = ? WHERE watch_id = ?",
        rusqlite::params![&now, &watch_id],
    )
    .context("Failed to disable watch")?;

    output::success(format!("Library '{}' watching disabled", tag));
    output::info("Existing indexed content is preserved.");
    output::info("To re-enable: wqm library watch <tag> <path>");
    output::info("To remove completely: wqm library remove <tag>");

    // Signal daemon
    signal_daemon_watch_folders().await;

    Ok(())
}
