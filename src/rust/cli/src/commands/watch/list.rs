//! Watch list subcommand

use anyhow::Result;

use crate::output;

use super::helpers::{
    connect_readonly, format_bool, format_bool_archived, format_bool_paused, format_relative_time,
};
use super::types::{WatchListItem, WatchListItemVerbose};

pub async fn list(
    enabled_only: bool,
    disabled_only: bool,
    collection: Option<String>,
    json: bool,
    verbose: bool,
    show_archived: bool,
) -> Result<()> {
    let conn = connect_readonly()?;

    // Build WHERE clause
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    // By default, hide archived folders unless --show-archived is set
    if !show_archived {
        conditions.push("COALESCE(is_archived, 0) = 0".to_string());
    }

    if enabled_only {
        conditions.push("enabled = 1".to_string());
    } else if disabled_only {
        conditions.push("enabled = 0".to_string());
    }

    if let Some(ref c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c.clone()));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!(
        r#"
        SELECT watch_id, path, collection, tenant_id,
               enabled, is_active, last_scan, last_activity_at,
               COALESCE(is_paused, 0) as is_paused,
               COALESCE(is_archived, 0) as is_archived,
               git_remote_url, library_mode
        FROM watch_folders
        {}
        ORDER BY path ASC
        "#,
        where_clause
    );

    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,          // watch_id
            row.get::<_, String>(1)?,          // path
            row.get::<_, String>(2)?,          // collection
            row.get::<_, String>(3)?,          // tenant_id
            row.get::<_, i32>(4)? != 0,        // enabled
            row.get::<_, i32>(5)? != 0,        // is_active
            row.get::<_, Option<String>>(6)?,  // last_scan
            row.get::<_, Option<String>>(7)?,  // last_activity_at
            row.get::<_, i32>(8)? != 0,        // is_paused
            row.get::<_, i32>(9)? != 0,        // is_archived
            row.get::<_, Option<String>>(10)?, // git_remote_url
            row.get::<_, Option<String>>(11)?, // library_mode
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json {
            println!("[]");
        } else {
            output::info("No watch configurations found");
            output::info("Add watches via project registration or wqm project add");
        }
        return Ok(());
    }

    if verbose {
        print_verbose(&items, json);
    } else {
        print_compact(&items, json);
    }

    Ok(())
}

type WatchRow = (
    String,         // watch_id
    String,         // path
    String,         // collection
    String,         // tenant_id
    bool,           // enabled
    bool,           // is_active
    Option<String>, // last_scan
    Option<String>, // last_activity_at
    bool,           // is_paused
    bool,           // is_archived
    Option<String>, // git_remote_url
    Option<String>, // library_mode
);

fn print_verbose(items: &[WatchRow], json: bool) {
    let display_items: Vec<WatchListItemVerbose> = items
        .iter()
        .map(
            |(
                watch_id,
                path,
                collection,
                tenant_id,
                enabled,
                is_active,
                last_scan,
                _last_activity_at,
                is_paused,
                is_archived,
                git_remote_url,
                library_mode,
            )| {
                WatchListItemVerbose {
                    watch_id: watch_id.clone(),
                    path: path.clone(),
                    collection: collection.clone(),
                    tenant_id: tenant_id.clone(),
                    enabled: format_bool(*enabled),
                    is_active: format_bool(*is_active),
                    is_paused: format_bool_paused(*is_paused),
                    archived: format_bool_archived(*is_archived),
                    git_remote_url: git_remote_url.as_deref().unwrap_or("-").to_string(),
                    library_mode: library_mode.as_deref().unwrap_or("-").to_string(),
                    last_scan: last_scan
                        .as_ref()
                        .map(|s| format_relative_time(s))
                        .unwrap_or_else(|| "never".to_string()),
                }
            },
        )
        .collect();

    if json {
        output::print_json(&display_items);
    } else {
        output::print_table_auto(&display_items);
        output::info(format!(
            "Showing {} watch configurations",
            display_items.len()
        ));
    }
}

fn print_compact(items: &[WatchRow], json: bool) {
    let display_items: Vec<WatchListItem> = items
        .iter()
        .map(
            |(
                watch_id,
                path,
                collection,
                _tenant_id,
                enabled,
                is_active,
                last_scan,
                _last_activity_at,
                is_paused,
                is_archived,
                _git_remote_url,
                _library_mode,
            )| {
                WatchListItem {
                    watch_id: watch_id.clone(),
                    path: path.clone(),
                    collection: collection.clone(),
                    enabled: format_bool(*enabled),
                    is_active: format_bool(*is_active),
                    is_paused: format_bool_paused(*is_paused),
                    archived: format_bool_archived(*is_archived),
                    last_scan: last_scan
                        .as_ref()
                        .map(|s| format_relative_time(s))
                        .unwrap_or_else(|| "never".to_string()),
                }
            },
        )
        .collect();

    if json {
        output::print_json(&display_items);
    } else {
        output::print_table_auto(&display_items);
        output::info(format!(
            "Showing {} watch configurations",
            display_items.len()
        ));
    }
}
