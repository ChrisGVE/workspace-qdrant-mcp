//! Watch pause and resume subcommands

use anyhow::Result;

use crate::output;

use super::helpers::connect_readwrite;

pub async fn pause() -> Result<()> {
    let conn = connect_readwrite()?;

    let updated = conn.execute(
        "UPDATE watch_folders SET is_paused = 1, \
         pause_start_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE enabled = 1 AND is_paused = 0",
        [],
    )?;

    if updated > 0 {
        output::success(format!("Paused {} watch folder(s)", updated));
        output::info("File events will be buffered until watchers are resumed");
    } else {
        output::info(
            "No active watchers to pause \
             (all already paused or none enabled)",
        );
    }

    Ok(())
}

pub async fn resume() -> Result<()> {
    let conn = connect_readwrite()?;

    let updated = conn.execute(
        "UPDATE watch_folders SET is_paused = 0, \
         pause_start_time = NULL, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE enabled = 1 AND is_paused = 1",
        [],
    )?;

    if updated > 0 {
        output::success(format!("Resumed {} watch folder(s)", updated));
        output::info("Buffered file events will be processed");
    } else {
        output::info("No paused watchers to resume");
    }

    Ok(())
}
