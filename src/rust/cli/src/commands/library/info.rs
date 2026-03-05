//! Library info subcommand

use anyhow::Result;
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::open_db;
use crate::output;

/// Show library information for a specific tag or all libraries
pub async fn execute(tag: Option<&str>) -> Result<()> {
    let conn = match open_db() {
        Ok(c) => c,
        Err(e) => {
            output::error(format!("Cannot read database: {}", e));
            return Ok(());
        }
    };

    match tag {
        Some(t) => show_single(&conn, t),
        None => {
            // Show info for all libraries (delegates to list with verbose)
            super::list::execute(true).await
        }
    }
}

/// Show detailed info for a single library tag
fn show_single(conn: &rusqlite::Connection, tag: &str) -> Result<()> {
    output::section(format!("Library Info: {}", tag));

    let watch_id = format!("lib-{}", tag);

    let result: Result<
        (
            String,
            String,
            Option<String>,
            bool,
            String,
            Option<String>,
            Option<String>,
        ),
        _,
    > = conn.query_row(
        "SELECT path, tenant_id, library_mode, enabled, created_at, updated_at, \
         last_activity_at \
         FROM watch_folders WHERE watch_id = ? AND collection = 'libraries'",
        [&watch_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i32>(3)? != 0,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5).ok().flatten(),
                row.get::<_, Option<String>>(6).ok().flatten(),
            ))
        },
    );

    match result {
        Ok((path, tenant_id, mode, enabled, created_at, updated_at, last_activity)) => {
            let status = if enabled { "watching" } else { "paused" };
            output::kv("Tag", &tenant_id);
            output::kv("Watch ID", &watch_id);
            output::kv("Path", &path);
            output::kv("Status", status);
            output::kv("Mode", mode.as_deref().unwrap_or("incremental"));
            output::kv("Collection", COLLECTION_LIBRARIES);
            output::kv("Created", &created_at);
            if let Some(updated) = updated_at {
                output::kv("Updated", &updated);
            }
            if let Some(activity) = last_activity {
                output::kv("Last Activity", &activity);
            }

            // Query tracked_files for file count
            output::separator();
            let file_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM tracked_files WHERE watch_folder_id = ?",
                    [&watch_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            let chunk_count: i64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(chunk_count), 0) FROM tracked_files \
                 WHERE watch_folder_id = ?",
                    [&watch_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            output::kv("Tracked Files", &file_count.to_string());
            output::kv("Total Chunks", &chunk_count.to_string());
        }
        Err(_) => {
            output::error(format!("Library '{}' not found", tag));
        }
    }

    Ok(())
}
