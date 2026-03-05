//! Library set-incremental subcommand

use std::path::PathBuf;

use anyhow::Result;
use wqm_common::timestamps;

use super::helpers::open_db;
use crate::output;

/// Set or clear the incremental (do-not-delete) flag on tracked files.
///
/// When the incremental flag is set on a tracked file, the file watcher will
/// not enqueue a deletion when the source file is removed from disk. This is
/// useful for library-routed files (PDFs, etc.) that should persist in the
/// vector database even after the source is moved or deleted.
///
/// Note: Full project deletion (wqm project remove) always cascades
/// regardless of the incremental flag.
pub async fn execute(files: &[PathBuf], clear: bool) -> Result<()> {
    let conn = open_db()?;
    let value = if clear { 0i32 } else { 1i32 };
    let action = if clear { "cleared" } else { "set" };
    let now = timestamps::now_utc();

    let mut updated = 0u32;
    let mut not_found = 0u32;

    for file in files {
        let abs_path = std::fs::canonicalize(file)
            .unwrap_or_else(|_| file.to_path_buf())
            .to_string_lossy()
            .to_string();

        let rows = conn.execute(
            "UPDATE tracked_files SET incremental = ?1, updated_at = ?2 WHERE file_path = ?3",
            rusqlite::params![value, now, abs_path],
        )?;

        if rows > 0 {
            updated += 1;
            output::success(&format!("{} incremental flag: {}", action, abs_path));
        } else {
            not_found += 1;
            output::warning(&format!("Not found in tracked_files: {}", abs_path));
        }
    }

    if updated > 0 || not_found > 0 {
        output::info(&format!(
            "Updated: {}, not found: {} (total: {})",
            updated,
            not_found,
            files.len()
        ));
    }

    Ok(())
}
