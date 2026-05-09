//! Unified SQLite database connection helpers.
//!
//! Single source of truth for read-only database access.
//! All CLI commands must use these functions instead of
//! creating their own connection helpers.

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::config::get_database_path;

/// Open a read-only connection to the state database.
///
/// Validates that the database file exists before opening.
/// All write operations go through gRPC to the daemon.
pub fn connect_readonly() -> Result<Connection> {
    let db_path = get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Is the daemon running? Start it with: wqm service start",
            db_path.display()
        );
    }

    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context(format!("Failed to open state database at {:?}", db_path))?;

    conn.execute_batch("PRAGMA busy_timeout=5000;")
        .context("Failed to set busy_timeout")?;

    Ok(conn)
}

/// Check if a table or view exists in the database.
pub fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        [name],
        |_| Ok(true),
    )
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_readonly_fails_gracefully_when_no_db() {
        // With no database file, should return an error, not panic
        std::env::set_var("WQM_DATABASE_PATH", "/tmp/nonexistent_wqm_test.db");
        let result = connect_readonly();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("Failed"),
            "Unexpected error: {}",
            err
        );
        std::env::remove_var("WQM_DATABASE_PATH");
    }

    #[test]
    fn table_exists_returns_false_for_missing_table() {
        // Create an in-memory database to test table_exists
        let conn = Connection::open_in_memory().unwrap();
        assert!(!table_exists(&conn, "nonexistent_table"));

        conn.execute_batch("CREATE TABLE test_table (id INTEGER)")
            .unwrap();
        assert!(table_exists(&conn, "test_table"));
    }
}
