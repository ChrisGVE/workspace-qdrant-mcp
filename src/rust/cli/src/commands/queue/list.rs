//! Queue list subcommand

use std::collections::HashMap;

use anyhow::Result;

use crate::output;
use crate::output::style::short_id;

use super::db::connect_readonly;
use super::formatters::{
    extract_subject, format_relative_time, format_status, truncate_str, QueueListItem,
    QueueListItemVerbose, QueueListItemWithError,
};

/// Maximum character width for error messages in the table view.
const ERROR_TRUNCATE_LEN: usize = 60;

pub async fn execute(
    status: Option<String>,
    collection: Option<String>,
    item_type: Option<String>,
    limit: i64,
    offset: i64,
    order_by: &str,
    desc: bool,
    json: bool,
    script: bool,
    no_headers: bool,
    verbose: bool,
) -> Result<()> {
    let conn = connect_readonly()?;
    let (query, params_vec) = build_list_query(
        status.clone(),
        collection.clone(),
        item_type.clone(),
        order_by,
        desc,
        limit,
        offset,
    );

    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,          // queue_id
            row.get::<_, String>(1)?,          // idempotency_key
            row.get::<_, String>(2)?,          // item_type
            row.get::<_, String>(3)?,          // op
            row.get::<_, String>(4)?,          // collection
            row.get::<_, String>(5)?,          // status
            row.get::<_, String>(6)?,          // created_at
            row.get::<_, i32>(7)?,             // retry_count
            row.get::<_, Option<String>>(8)?,  // worker_id
            row.get::<_, String>(9)?,          // tenant_id
            row.get::<_, String>(10)?,         // payload_json
            row.get::<_, Option<String>>(11)?, // error_message
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json {
            println!("[]");
        } else {
            output::info(
                "No queue items found. Projects are processed automatically when registered.",
            );
        }
        return Ok(());
    }

    // Get total count for summary line
    let total = count_total(&conn, status, collection, item_type)?;

    // Build tenant_id -> project_name mapping
    let tenant_names = build_tenant_name_map(&conn);

    if verbose {
        print_verbose(&items, &tenant_names, total, json, script, no_headers);
    } else {
        print_compact(&items, &tenant_names, total, json, script, no_headers);
    }

    Ok(())
}

/// Count total matching items (ignoring LIMIT/OFFSET) for the summary line.
fn count_total(
    conn: &rusqlite::Connection,
    status: Option<String>,
    collection: Option<String>,
    item_type: Option<String>,
) -> Result<usize> {
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(s) = status {
        conditions.push("status = ?".to_string());
        params_vec.push(Box::new(s));
    }
    if let Some(c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c));
    }
    if let Some(t) = item_type {
        conditions.push("item_type = ?".to_string());
        params_vec.push(Box::new(t));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!("SELECT COUNT(*) FROM unified_queue {}", where_clause);
    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
    let count: i64 = conn.query_row(&query, params_slice.as_slice(), |row| row.get(0))?;
    Ok(count as usize)
}

/// Build the SELECT query and parameter list for the queue list command.
fn build_list_query(
    status: Option<String>,
    collection: Option<String>,
    item_type: Option<String>,
    order_by: &str,
    desc: bool,
    limit: i64,
    offset: i64,
) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(s) = status {
        conditions.push("status = ?".to_string());
        params_vec.push(Box::new(s));
    }
    if let Some(c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c));
    }
    if let Some(t) = item_type {
        conditions.push("item_type = ?".to_string());
        params_vec.push(Box::new(t));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let order_column = match order_by {
        "created_at" | "status" | "item_type" | "collection" => order_by,
        _ => "created_at",
    };
    let order_direction = if desc { "DESC" } else { "ASC" };

    let query = format!(
        "SELECT queue_id, idempotency_key, item_type, op, collection, status, \
         created_at, retry_count, worker_id, tenant_id, \
         COALESCE(payload_json, '{{}}'), error_message \
         FROM unified_queue {} ORDER BY {} {} \
         LIMIT ? OFFSET ?",
        where_clause, order_column, order_direction
    );

    params_vec.push(Box::new(limit));
    params_vec.push(Box::new(offset));

    (query, params_vec)
}

/// Build a `tenant_id` to project name mapping from `watch_folders`.
///
/// When multiple projects share the same directory name, the tenant_id is
/// appended in parentheses to disambiguate.  Falls back gracefully if the
/// `watch_folders` table does not exist.
fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();

    let mut entries: Vec<(String, String)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects'",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name =
                    r.1.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&r.0)
                        .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

    for (tenant_id, name) in entries {
        let display = if name_count.get(&name).copied().unwrap_or(0) > 1 {
            format!("{} ({})", name, short_id(&tenant_id))
        } else {
            name
        };
        map.insert(tenant_id, display);
    }

    map
}

/// Resolve a tenant_id to a human-readable project name, falling back to
/// a shortened tenant_id when no mapping exists.
fn resolve_project_name(tenant_id: &str, tenant_names: &HashMap<String, String>) -> String {
    tenant_names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| short_id(tenant_id))
}

type RowTuple = (
    String,
    String,
    String,
    String,
    String,
    String,
    String,
    i32,
    Option<String>,
    String,
    String,
    Option<String>,
);

fn print_verbose(
    items: &[RowTuple],
    tenant_names: &HashMap<String, String>,
    total: usize,
    json: bool,
    script: bool,
    no_headers: bool,
) {
    let display_items: Vec<QueueListItemVerbose> = items
        .iter()
        .map(
            |(
                queue_id,
                idempotency_key,
                item_type,
                op,
                collection,
                status,
                created_at,
                retry_count,
                worker_id,
                tenant_id,
                payload_json,
                _error_message,
            )| {
                QueueListItemVerbose {
                    queue_id: queue_id.clone(),
                    idempotency_key: idempotency_key.clone(),
                    project: resolve_project_name(tenant_id, tenant_names),
                    subject: extract_subject(item_type, payload_json),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: collection.clone(),
                    status: status.clone(),
                    created_at: wqm_common::timestamp_fmt::format_local(created_at),
                    retry_count: *retry_count,
                    worker_id: worker_id.clone().unwrap_or_default(),
                }
            },
        )
        .collect();

    if json {
        output::print_json(&display_items);
    } else if script {
        output::print_script(&display_items, !no_headers);
    } else {
        output::print_table_auto(&display_items);
        output::summary(output::summary_line(
            display_items.len(),
            total,
            "queue items",
        ));
    }
}

fn print_compact(
    items: &[RowTuple],
    tenant_names: &HashMap<String, String>,
    total: usize,
    json: bool,
    script: bool,
    no_headers: bool,
) {
    // Show the Error column when any item in the result set has a non-empty error_message
    let has_errors = items
        .iter()
        .any(|(_, _, _, _, _, _, _, _, _, _, _, err)| err.is_some());

    if has_errors {
        print_compact_with_error(items, tenant_names, total, json, script, no_headers);
    } else {
        print_compact_plain(items, tenant_names, total, json, script, no_headers);
    }
}

fn print_compact_plain(
    items: &[RowTuple],
    tenant_names: &HashMap<String, String>,
    total: usize,
    json: bool,
    script: bool,
    no_headers: bool,
) {
    let display_items: Vec<QueueListItem> = items
        .iter()
        .map(
            |(
                queue_id,
                _idempotency_key,
                item_type,
                op,
                _collection,
                status,
                created_at,
                retry_count,
                _worker_id,
                tenant_id,
                payload_json,
                _error_message,
            )| {
                QueueListItem {
                    queue_id: short_id(queue_id),
                    project: resolve_project_name(tenant_id, tenant_names),
                    subject: extract_subject(item_type, payload_json),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    status: format_status(status),
                    age: format_relative_time(created_at),
                    retry_count: *retry_count,
                }
            },
        )
        .collect();

    if json {
        output::print_json(&display_items);
    } else if script {
        output::print_script(&display_items, !no_headers);
    } else {
        output::print_table_auto(&display_items);
        output::summary(output::summary_line(
            display_items.len(),
            total,
            "queue items",
        ));
    }
}

fn print_compact_with_error(
    items: &[RowTuple],
    tenant_names: &HashMap<String, String>,
    total: usize,
    json: bool,
    script: bool,
    no_headers: bool,
) {
    let display_items: Vec<QueueListItemWithError> = items
        .iter()
        .map(
            |(
                queue_id,
                _idempotency_key,
                item_type,
                op,
                _collection,
                status,
                created_at,
                retry_count,
                _worker_id,
                tenant_id,
                payload_json,
                error_message,
            )| {
                QueueListItemWithError {
                    queue_id: short_id(queue_id),
                    project: resolve_project_name(tenant_id, tenant_names),
                    subject: extract_subject(item_type, payload_json),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    status: format_status(status),
                    age: format_relative_time(created_at),
                    retry_count: *retry_count,
                    error_message: error_message
                        .as_deref()
                        .map(|e| truncate_str(e, ERROR_TRUNCATE_LEN))
                        .unwrap_or_default(),
                }
            },
        )
        .collect();

    if json {
        output::print_json(&display_items);
    } else if script {
        output::print_script(&display_items, !no_headers);
    } else {
        output::print_table_auto(&display_items);
        output::summary(output::summary_line(
            display_items.len(),
            total,
            "queue items",
        ));
    }
}
