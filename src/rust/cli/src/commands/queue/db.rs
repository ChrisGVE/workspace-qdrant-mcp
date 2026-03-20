//! Database connection helpers for queue commands

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::config::get_database_path_checked;

/// Connect to the state database (read-only).
/// All write operations now go through gRPC to the daemon.
pub fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    Ok(conn)
}
