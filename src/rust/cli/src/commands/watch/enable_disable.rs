//! Watch enable and disable subcommands

use anyhow::Result;
use rusqlite::params;

use crate::output;

use super::helpers::connect_readwrite;
use super::resolver::resolve_watch_id;

pub async fn enable(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    let resolved_id = match resolve_watch_id(&conn, watch_id)? {
        Some(id) => id,
        None => return Ok(()), // Error already printed
    };

    let updated = conn.execute(
        "UPDATE watch_folders SET enabled = 1, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = ?1",
        params![resolved_id],
    )?;

    if updated > 0 {
        output::success(format!("Watch '{}' enabled", resolved_id));
        output::info("Daemon will pick up this watch on next poll cycle");
    } else {
        output::warning(format!(
            "Watch '{}' not found or already enabled",
            resolved_id
        ));
    }

    Ok(())
}

pub async fn disable(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    let resolved_id = match resolve_watch_id(&conn, watch_id)? {
        Some(id) => id,
        None => return Ok(()), // Error already printed
    };

    let updated = conn.execute(
        "UPDATE watch_folders SET enabled = 0, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = ?1",
        params![resolved_id],
    )?;

    if updated > 0 {
        output::success(format!("Watch '{}' disabled", resolved_id));
        output::info("Daemon will stop watching this folder on next poll cycle");
    } else {
        output::warning(format!(
            "Watch '{}' not found or already disabled",
            resolved_id
        ));
    }

    Ok(())
}
