//! Database helpers shared across tags subcommand handlers.

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::config::get_database_path;

pub(super) fn open_db() -> Result<Connection> {
    let db_path = get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run daemon first: wqm service start",
            db_path.display()
        );
    }
    let conn = Connection::open(&db_path).context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

pub(super) fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        [name],
        |_| Ok(true),
    )
    .unwrap_or(false)
}
