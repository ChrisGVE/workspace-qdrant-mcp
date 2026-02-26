//! Watch show subcommand

use anyhow::Result;
use rusqlite::params;

use crate::output;

use super::helpers::{
    connect_readonly, format_bool, format_bool_archived, format_bool_paused, format_relative_time,
};
use super::types::WatchDetailItem;

pub async fn show(watch_id: &str, json: bool) -> Result<()> {
    let conn = connect_readonly()?;

    // Try exact match first, then prefix match
    let query = r#"
        SELECT watch_id, path, collection, tenant_id,
               git_remote_url, remote_hash, disambiguation_path,
               enabled, is_active, library_mode, follow_symlinks,
               created_at, updated_at, last_scan, last_activity_at,
               parent_watch_id, submodule_path,
               COALESCE(is_paused, 0) as is_paused,
               COALESCE(is_archived, 0) as is_archived
        FROM watch_folders
        WHERE watch_id = ? OR watch_id LIKE ?
        LIMIT 1
    "#;

    let prefix = format!("{}%", watch_id);
    let mut stmt = conn.prepare(query)?;
    let result = stmt.query_row(params![watch_id, &prefix], |row| {
        Ok(WatchDetailItem {
            watch_id: row.get(0)?,
            path: row.get(1)?,
            collection: row.get(2)?,
            tenant_id: row.get(3)?,
            git_remote_url: row.get(4)?,
            remote_hash: row.get(5)?,
            disambiguation_path: row.get(6)?,
            enabled: row.get::<_, i32>(7)? != 0,
            is_active: row.get::<_, i32>(8)? != 0,
            library_mode: row.get(9)?,
            follow_symlinks: row.get::<_, i32>(10)? != 0,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
            last_scan: row.get(13)?,
            last_activity_at: row.get(14)?,
            parent_watch_id: row.get(15)?,
            submodule_path: row.get(16)?,
            is_paused: row.get::<_, i32>(17)? != 0,
            is_archived: row.get::<_, i32>(18)? != 0,
        })
    });

    match result {
        Ok(item) => {
            if json {
                output::print_json(&item);
            } else {
                print_detail(&item);
            }
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            output::error(format!("Watch not found: {}", watch_id));
            output::info("Use 'wqm watch list' to see available watches");
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    Ok(())
}

fn print_detail(item: &WatchDetailItem) {
    output::section("Watch Configuration Details");
    output::kv("Watch ID", &item.watch_id);
    output::kv("Path", &item.path);
    output::separator();
    output::kv("Collection", &item.collection);
    output::kv("Tenant ID", &item.tenant_id);
    if let Some(ref url) = item.git_remote_url {
        output::kv("Git Remote", url);
    }
    if let Some(ref hash) = item.remote_hash {
        output::kv("Remote Hash", hash);
    }
    if let Some(ref dp) = item.disambiguation_path {
        output::kv("Disambiguation", dp);
    }
    output::separator();
    output::kv("Enabled", &format_bool(item.enabled));
    output::kv("Active", &format_bool(item.is_active));
    output::kv("Paused", &format_bool_paused(item.is_paused));
    output::kv("Archived", &format_bool_archived(item.is_archived));
    output::kv("Follow Symlinks", &format_bool(item.follow_symlinks));
    if let Some(ref mode) = item.library_mode {
        output::kv("Library Mode", mode);
    }
    output::separator();
    output::kv(
        "Created At",
        &wqm_common::timestamp_fmt::format_local(&item.created_at),
    );
    output::kv(
        "Updated At",
        &wqm_common::timestamp_fmt::format_local(&item.updated_at),
    );
    if let Some(ref scan) = item.last_scan {
        output::kv(
            "Last Scan",
            &format!(
                "{} ({})",
                wqm_common::timestamp_fmt::format_local(scan),
                format_relative_time(scan)
            ),
        );
    } else {
        output::kv("Last Scan", "never");
    }
    if let Some(ref activity) = item.last_activity_at {
        output::kv(
            "Last Activity",
            &format!(
                "{} ({})",
                wqm_common::timestamp_fmt::format_local(activity),
                format_relative_time(activity)
            ),
        );
    }
    if let Some(ref parent) = item.parent_watch_id {
        output::separator();
        output::kv("Parent Watch", parent);
        if let Some(ref sp) = item.submodule_path {
            output::kv("Submodule Path", sp);
        }
        output::info("This is a submodule watch");
    }
}
