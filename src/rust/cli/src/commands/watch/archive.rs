//! Watch archive and unarchive subcommands

use anyhow::Result;
use rusqlite::{params, Connection};

use crate::output;

use super::helpers::connect_readwrite;
use super::resolver::resolve_watch_id;

/// Resolve a watch_id input that may be either an ID or a filesystem path.
fn resolve_watch_id_or_path(conn: &Connection, watch_id: &str) -> Result<Option<String>> {
    if let Some(id) = resolve_watch_id(conn, watch_id)? {
        return Ok(Some(id));
    }

    // Try to resolve as a path: canonicalize and look up by path column
    let path = std::path::Path::new(watch_id);
    if let Ok(canonical) = path.canonicalize() {
        let path_str = canonical.to_string_lossy().to_string();
        let found: Option<String> = conn
            .query_row(
                "SELECT watch_id FROM watch_folders WHERE path = ?1",
                params![&path_str],
                |row| row.get(0),
            )
            .ok();
        Ok(found)
    } else {
        // resolve_watch_id already printed the error
        Ok(None)
    }
}

pub async fn archive(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    let resolved_id = match resolve_watch_id_or_path(&conn, watch_id)? {
        Some(id) => id,
        None => {
            // Try canonicalize for a better error message
            let path = std::path::Path::new(watch_id);
            if let Ok(canonical) = path.canonicalize() {
                let path_str = canonical.to_string_lossy().to_string();
                output::error(format!("Watch folder not found for path: {}", path_str));
                output::info("Use 'wqm watch list' to see available watches");
            }
            return Ok(());
        }
    };

    // Archive the parent watch folder
    let updated = conn.execute(
        "UPDATE watch_folders SET is_archived = 1, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = ?1 AND COALESCE(is_archived, 0) = 0",
        params![resolved_id],
    )?;

    if updated == 0 {
        output::warning(format!(
            "Watch '{}' not found or already archived",
            resolved_id
        ));
        return Ok(());
    }

    output::success(format!("Archived watch '{}'", resolved_id));

    // Check and archive submodules with cross-reference safety
    let (archived_count, skipped_count) = archive_submodules_safely(&conn, &resolved_id)?;

    if archived_count > 0 || skipped_count > 0 {
        output::info(format!(
            "{} submodule(s) archived, {} shared submodule(s) kept active",
            archived_count, skipped_count
        ));
    }

    output::info("Watching and ingesting stopped; data remains fully searchable");

    Ok(())
}

pub async fn unarchive(watch_id: &str) -> Result<()> {
    let conn = connect_readwrite()?;

    let resolved_id = match resolve_watch_id_or_path(&conn, watch_id)? {
        Some(id) => id,
        None => {
            let path = std::path::Path::new(watch_id);
            if let Ok(canonical) = path.canonicalize() {
                let path_str = canonical.to_string_lossy().to_string();
                output::error(format!("Watch folder not found for path: {}", path_str));
                output::info(
                    "Use 'wqm watch list --show-archived' \
                     to see archived watches",
                );
            }
            return Ok(());
        }
    };

    let updated = conn.execute(
        "UPDATE watch_folders SET is_archived = 0, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = ?1 AND COALESCE(is_archived, 0) = 1",
        params![resolved_id],
    )?;

    if updated > 0 {
        output::success(format!("Unarchived watch '{}'", resolved_id));
        output::info("Watching and ingesting will resume on next poll cycle");
    } else {
        output::warning(format!("Watch '{}' not found or not archived", resolved_id));
    }

    Ok(())
}

/// Archive submodules of a parent project with cross-reference safety
/// checks.
///
/// For each submodule: if other active projects reference the same
/// remote, the submodule is skipped (stays active). Otherwise it is
/// archived with the parent. Returns `(archived_count, skipped_count)`.
fn archive_submodules_safely(conn: &Connection, parent_watch_id: &str) -> Result<(usize, usize)> {
    // Get submodules of this parent via junction table (Task 14)
    let mut stmt = conn.prepare(
        "SELECT wf.watch_id, wf.remote_hash, wf.git_remote_url \
         FROM watch_folders wf \
         INNER JOIN watch_folder_submodules j \
           ON wf.watch_id = j.child_watch_id \
         WHERE j.parent_watch_id = ?1",
    )?;
    let submodules: Vec<(String, Option<String>, Option<String>)> = stmt
        .query_map(params![parent_watch_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut archived_count = 0;
    let mut skipped_count = 0;

    for (sub_watch_id, remote_hash, git_remote_url) in &submodules {
        let rh = remote_hash.as_deref().unwrap_or("");
        let url = git_remote_url.as_deref().unwrap_or("");

        if rh.is_empty() && url.is_empty() {
            // No remote info, archive with parent
            let updated = conn.execute(
                "UPDATE watch_folders SET is_archived = 1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                 WHERE watch_id = ?1 \
                 AND COALESCE(is_archived, 0) = 0",
                params![sub_watch_id],
            )?;
            if updated > 0 {
                archived_count += 1;
            }
            continue;
        }

        // Count other active references to this submodule
        let other_refs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM watch_folders sub \
             WHERE sub.remote_hash = ?1 \
             AND sub.git_remote_url = ?2 \
             AND sub.parent_watch_id != ?3 \
             AND COALESCE(sub.is_archived, 0) = 0 \
             AND EXISTS ( \
               SELECT 1 FROM watch_folders parent \
               WHERE parent.watch_id = sub.parent_watch_id \
               AND COALESCE(parent.is_archived, 0) = 0 \
             )",
            params![rh, url, parent_watch_id],
            |row| row.get(0),
        )?;

        if other_refs > 0 {
            skipped_count += 1;
        } else {
            let updated = conn.execute(
                "UPDATE watch_folders SET is_archived = 1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                 WHERE watch_id = ?1 \
                 AND COALESCE(is_archived, 0) = 0",
                params![sub_watch_id],
            )?;
            if updated > 0 {
                archived_count += 1;
            }
        }
    }

    Ok((archived_count, skipped_count))
}
