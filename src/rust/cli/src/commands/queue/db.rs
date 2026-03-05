//! Database connection helpers for queue commands

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::config::get_database_path_checked;

/// Connect to the state database (read-only for safety)
pub fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    Ok(conn)
}

/// Connect to the state database (read-write for retry/clean/remove)
pub fn connect_readwrite() -> Result<Connection> {
    let db_path = get_database_path_checked().map_err(|e| anyhow::anyhow!("{}", e))?;

    let conn = Connection::open(&db_path)
        .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;

    Ok(conn)
}
