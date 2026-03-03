//! Queue list subcommand

use anyhow::Result;

use crate::output;

use super::db::connect_readonly;
use super::formatters::{
    QueueListItem, QueueListItemVerbose, format_relative_time, format_status,
};

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
    let (query, params_vec) =
        build_list_query(status, collection, item_type, order_by, desc, limit, offset);

    let params_slice: Vec<&dyn rusqlite::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,         // queue_id
            row.get::<_, String>(1)?,         // idempotency_key
            row.get::<_, String>(2)?,         // item_type
            row.get::<_, String>(3)?,         // op
            row.get::<_, String>(4)?,         // collection
            row.get::<_, String>(5)?,         // status
            row.get::<_, String>(6)?,         // created_at
            row.get::<_, i32>(7)?,            // retry_count
            row.get::<_, Option<String>>(8)?, // worker_id
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json { println!("[]"); } else { output::info("No queue items found"); }
        return Ok(());
    }

    if verbose {
        print_verbose(&items, json, script, no_headers);
    } else {
        print_compact(&items, json, script, no_headers);
    }

    Ok(())
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
         created_at, retry_count, worker_id FROM unified_queue {} ORDER BY {} {} \
         LIMIT ? OFFSET ?",
        where_clause, order_column, order_direction
    );

    params_vec.push(Box::new(limit));
    params_vec.push(Box::new(offset));

    (query, params_vec)
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
);

fn print_verbose(items: &[RowTuple], json: bool, script: bool, no_headers: bool) {
    let display_items: Vec<QueueListItemVerbose> = items
        .iter()
        .map(
            |(queue_id, idempotency_key, item_type, op, collection, status, created_at, retry_count, worker_id)| {
                QueueListItemVerbose {
                    queue_id: queue_id.clone(),
                    idempotency_key: idempotency_key.clone(),
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
        output::info(format!("Showing {} items", display_items.len()));
    }
}

fn print_compact(items: &[RowTuple], json: bool, script: bool, no_headers: bool) {
    let display_items: Vec<QueueListItem> = items
        .iter()
        .map(
            |(queue_id, _idempotency_key, item_type, op, collection, status, created_at, retry_count, _worker_id)| {
                QueueListItem {
                    queue_id: queue_id.clone(),
                    item_type: item_type.clone(),
                    op: op.clone(),
                    collection: collection.clone(),
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
        output::info(format!("Showing {} items", display_items.len()));
    }
}
