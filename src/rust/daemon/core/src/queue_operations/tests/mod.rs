//! Tests for queue operations.

use super::*;
use crate::queue_config::QueueConnectionConfig;
use crate::unified_queue_schema::{
    DestinationStatus, ItemType, QueueOperation as UnifiedOp, QueueStatus,
};
use sqlx::{Row, SqlitePool};
use std::sync::Arc;
use tempfile::tempdir;

mod cascade_priority_tests;
mod concurrency_tests;
mod destination_tests;
mod enqueue_dequeue_tests;
mod failure_retry_tests;
mod stats_cleanup_tests;
mod validation_tests;

async fn apply_sql_script(pool: &SqlitePool, script: &str) -> Result<(), sqlx::Error> {
    let mut conn = pool.acquire().await?;
    let mut statement = String::new();
    let mut in_trigger = false;

    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") {
            continue;
        }

        if trimmed.to_uppercase().starts_with("CREATE TRIGGER") {
            in_trigger = true;
        }

        statement.push_str(line);
        statement.push('\n');

        if in_trigger {
            if trimmed.eq_ignore_ascii_case("END;") || trimmed.eq_ignore_ascii_case("END") {
                in_trigger = false;
                let stmt = statement.trim();
                if !stmt.is_empty() {
                    sqlx::query(stmt).execute(&mut *conn).await?;
                }
                statement.clear();
            }
            continue;
        }

        if trimmed.ends_with(';') {
            let stmt = statement.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&mut *conn).await?;
            }
            statement.clear();
        }
    }

    let remainder = statement.trim();
    if !remainder.is_empty() {
        sqlx::query(remainder).execute(&mut *conn).await?;
    }

    Ok(())
}
