//! Library list subcommand

use std::collections::HashSet;

use anyhow::{Context, Result};

use super::super::qdrant_helpers;
use super::helpers::open_db;
use crate::output;
use wqm_common::constants::COLLECTION_LIBRARIES;

/// List all libraries, including watched, format-routed, and orphaned
pub async fn execute(verbose: bool) -> Result<()> {
    output::section("Libraries");

    let conn = match open_db() {
        Ok(c) => c,
        Err(_) => {
            output::info("No libraries configured yet.");
            output::info("Add a library with: wqm library add <tag> <path>");
            return Ok(());
        }
    };

    // Collect known library tags from SQLite
    let mut known_tags = HashSet::new();
    let mut found_any = false;

    // Try to get Qdrant point counts (non-fatal if Qdrant is down)
    let qdrant_counts = match qdrant_helpers::build_qdrant_http_client() {
        Ok(client) => {
            let base_url = qdrant_helpers::qdrant_base_url();
            qdrant_helpers::scroll_tenant_point_counts(
                &client,
                &base_url,
                COLLECTION_LIBRARIES,
                "library_name",
            )
            .await
            .unwrap_or_default()
        }
        Err(_) => std::collections::HashMap::new(),
    };

    list_watch_folders(
        &conn,
        verbose,
        &qdrant_counts,
        &mut known_tags,
        &mut found_any,
    )?;
    list_format_routed(&conn, &qdrant_counts, &mut known_tags, &mut found_any)?;
    list_orphans(&qdrant_counts, &known_tags, &mut found_any);

    if !found_any {
        output::info("No libraries configured and no format-routed library files found.");
        output::info("Add a library with: wqm library add <tag> <path>");
        output::info("Or add PDFs/documents to a watched project folder.");
    }

    Ok(())
}

/// Display explicit library watch folders from SQLite
fn list_watch_folders(
    conn: &rusqlite::Connection,
    verbose: bool,
    qdrant_counts: &std::collections::HashMap<String, usize>,
    known_tags: &mut HashSet<String>,
    found_any: &mut bool,
) -> Result<()> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT watch_id, tenant_id, path, library_mode, enabled, is_active, \
             created_at, last_activity_at \
             FROM watch_folders WHERE collection = '{}' ORDER BY tenant_id",
            COLLECTION_LIBRARIES
        ))
        .context("Failed to query watch_folders")?;

    let libraries: Vec<(
        String,
        String,
        String,
        Option<String>,
        bool,
        bool,
        String,
        Option<String>,
    )> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, i32>(4)? != 0,
                row.get::<_, i32>(5)? != 0,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })
        .context("Failed to read library rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")?;

    if libraries.is_empty() {
        return Ok(());
    }

    *found_any = true;
    output::info(format!("Library watch folders ({}):", libraries.len()));
    output::separator();

    for (watch_id, tenant_id, path, mode, enabled, _is_active, created_at, last_activity) in
        &libraries
    {
        known_tags.insert(tenant_id.clone());
        let status = if *enabled { "watching" } else { "paused" };
        let mode_str = mode.as_deref().unwrap_or("incremental");

        output::kv("  Tag", tenant_id);
        output::kv("  Path", path);
        output::kv("  Status", status);
        output::kv("  Mode", mode_str);
        if let Some(count) = qdrant_counts.get(tenant_id) {
            output::kv("  Points", count.to_string());
        }
        if verbose {
            output::kv("  Watch ID", watch_id);
            output::kv("  Created", created_at);
            if let Some(activity) = last_activity {
                output::kv("  Last Activity", activity);
            }
        }
        output::separator();
    }

    Ok(())
}

/// Display format-routed files (PDFs etc. auto-routed from project folders)
fn list_format_routed(
    conn: &rusqlite::Connection,
    qdrant_counts: &std::collections::HashMap<String, usize>,
    known_tags: &mut HashSet<String>,
    found_any: &mut bool,
) -> Result<()> {
    let mut routed_stmt = conn
        .prepare(
            "SELECT wf.tenant_id, wf.path, COUNT(tf.file_id) as file_count
         FROM tracked_files tf
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
         WHERE tf.collection = 'libraries' AND wf.collection = 'projects'
         GROUP BY wf.tenant_id
         ORDER BY wf.tenant_id",
        )
        .context("Failed to query format-routed library files")?;

    let routed: Vec<(String, String, i64)> = routed_stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .context("Failed to read format-routed rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse format-routed rows")?;

    if routed.is_empty() {
        return Ok(());
    }

    *found_any = true;
    output::info(format!(
        "Format-routed from projects ({} project(s)):",
        routed.len()
    ));
    output::separator();

    for (tenant_id, project_path, file_count) in &routed {
        known_tags.insert(tenant_id.clone());
        output::kv("  Project", tenant_id);
        output::kv("  Path", project_path);
        output::kv("  Library Files", file_count.to_string());
        output::kv("  Source", "auto-routed (PDF, DOCX, etc.)");
        if let Some(count) = qdrant_counts.get(tenant_id) {
            output::kv("  Points", count.to_string());
        }
        output::separator();
    }

    Ok(())
}

/// Display orphaned libraries that exist in Qdrant but not in SQLite
fn list_orphans(
    qdrant_counts: &std::collections::HashMap<String, usize>,
    known_tags: &HashSet<String>,
    found_any: &mut bool,
) {
    let mut orphan_tags: Vec<(&String, &usize)> = qdrant_counts
        .iter()
        .filter(|(tag, _)| !known_tags.contains(*tag))
        .collect();
    orphan_tags.sort_by_key(|(tag, _)| (*tag).clone());

    if orphan_tags.is_empty() {
        return;
    }

    *found_any = true;
    output::warning(format!("Orphaned libraries ({}):", orphan_tags.len()));
    output::separator();

    for (tag, count) in &orphan_tags {
        output::kv("  Tag", format!("{} (ORPHAN)", tag));
        output::kv("  Points", count.to_string());
        output::kv(
            "  Status",
            "no watch folder — run: wqm admin cleanup-orphans",
        );
        output::separator();
    }
}
