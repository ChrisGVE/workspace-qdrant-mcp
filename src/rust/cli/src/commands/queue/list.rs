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
    verbose: bool,
) -> Result<()> {
    let conn = connect_readonly()?;

    // Build WHERE clause
    let mut conditions: Vec<String> = Vec::new();
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if let Some(ref s) = status {
        conditions.push("status = ?".to_string());
        params_vec.push(Box::new(s.clone()));
    }
    if let Some(ref c) = collection {
        conditions.push("collection = ?".to_string());
        params_vec.push(Box::new(c.clone()));
    }
    if let Some(ref t) = item_type {
        conditions.push("item_type = ?".to_string());
        params_vec.push(Box::new(t.clone()));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    // Validate order_by
    let order_column = match order_by {
        "created_at" | "status" | "item_type" | "collection" => order_by,
        _ => "created_at",
    };
    let order_direction = if desc { "DESC" } else { "ASC" };

    let query = format!(
        r#"
        SELECT queue_id, idempotency_key, item_type, op, collection, status,
               created_at, retry_count, worker_id
        FROM unified_queue
        {}
        ORDER BY {} {}
        LIMIT ? OFFSET ?
        "#,
        where_clause, order_column, order_direction
    );

    params_vec.push(Box::new(limit));
    params_vec.push(Box::new(offset));

    let params_slice: Vec<&dyn rusqlite::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?, // queue_id
            row.get::<_, String>(1)?, // idempotency_key
            row.get::<_, String>(2)?, // item_type
            row.get::<_, String>(3)?, // op
            row.get::<_, String>(4)?, // collection
            row.get::<_, String>(5)?, // status
            row.get::<_, String>(6)?, // created_at
            row.get::<_, i32>(7)?,    // retry_count
            row.get::<_, Option<String>>(8)?, // worker_id
        ))
    })?;

    let items: Vec<_> = rows.filter_map(|r| r.ok()).collect();

    if items.is_empty() {
        if json {
            println!("[]");
        } else {
            output::info("No queue items found");
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

fn print_verbose(items: &[RowTuple], json: bool) {
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
    } else {
        output::print_table_auto(&display_items);
        output::info(format!("Showing {} items", display_items.len()));
    }
}

fn print_compact(items: &[RowTuple], json: bool) {
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
    } else {
        output::print_table_auto(&display_items);
        output::info(format!("Showing {} items", display_items.len()));
    }
}
