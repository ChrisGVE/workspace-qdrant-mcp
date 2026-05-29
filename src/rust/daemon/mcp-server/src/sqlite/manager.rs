//! SQLite `StateManager` — read-only facade over the daemon's `state.db`.
//!
//! Mirrors the behaviour of `SqliteStateManager` in `sqlite-state-manager.ts`:
//! - Opens the database READ-ONLY (`SQLITE_OPEN_READ_ONLY`).
//! - Sets `busy_timeout = 5000 ms` as a defensive measure (RISK-15).
//! - Returns a *degraded* instance (status `database_not_found`) when the
//!   file does not exist yet — the daemon has not been initialised.
//! - Never creates the file, never runs migrations, never changes journal_mode
//!   (ADR-003: the Rust daemon owns all schema changes).

use rusqlite::{Connection, OpenFlags};
use std::path::{Path, PathBuf};
use tracing::warn;

use wqm_common::paths::get_database_path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Why a query returned degraded (empty/null) data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DegradedReason {
    DatabaseNotFound,
    TableNotFound,
    DatabaseError,
}

/// The result of any read operation: either good data or a degraded fallback.
#[derive(Debug, Clone)]
pub struct QueryResult<T> {
    pub data: T,
    pub status: QueryStatus,
    pub reason: Option<DegradedReason>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    Ok,
    Degraded,
}

impl<T> QueryResult<T> {
    pub fn ok(data: T) -> Self {
        Self {
            data,
            status: QueryStatus::Ok,
            reason: None,
            message: None,
        }
    }

    pub fn degraded(data: T, reason: DegradedReason, message: impl Into<String>) -> Self {
        Self {
            data,
            status: QueryStatus::Degraded,
            reason: Some(reason),
            message: Some(message.into()),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.status == QueryStatus::Ok
    }
}

// ---------------------------------------------------------------------------
// StateManager
// ---------------------------------------------------------------------------

/// Read-only wrapper around the daemon's SQLite state database.
///
/// Construct via [`StateManager::open`] or [`StateManager::open_at`].
/// When the database file does not exist the manager is in *degraded* mode:
/// [`StateManager::connection`] returns `None` and every query module treats
/// `None` as "return empty results" (matching the TS behaviour).
pub struct StateManager {
    conn: Option<Connection>,
    db_path: PathBuf,
    degraded_reason: Option<DegradedReason>,
    degraded_message: Option<String>,
}

impl StateManager {
    // ── Constructors ────────────────────────────────────────────────────────

    /// Open the database at the default XDG path (`WQM_DATABASE_PATH` or
    /// `~/.local/share/workspace-qdrant/state.db`).
    ///
    /// Returns a degraded manager if the file does not exist.
    pub fn open() -> Self {
        let path = match get_database_path() {
            Ok(p) => p,
            Err(e) => {
                let msg = format!("Could not resolve database path: {e}");
                warn!("{msg}");
                return Self::degraded_at(PathBuf::from("<unknown>"), msg);
            }
        };
        Self::open_at(path)
    }

    /// Open the database at an explicit path.
    ///
    /// Returns a degraded manager if the file does not exist.
    pub fn open_at(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            let msg = format!(
                "Database not found at {}. Daemon has not initialized yet.",
                path.display()
            );
            return Self::degraded_at(path, msg);
        }

        match Self::connect(&path) {
            Ok(conn) => Self {
                conn: Some(conn),
                db_path: path,
                degraded_reason: None,
                degraded_message: None,
            },
            Err(e) => {
                let msg = format!("Failed to open database: {e}");
                warn!("{msg}");
                Self {
                    conn: None,
                    db_path: path,
                    degraded_reason: Some(DegradedReason::DatabaseError),
                    degraded_message: Some(msg),
                }
            }
        }
    }

    fn degraded_at(path: PathBuf, msg: String) -> Self {
        Self {
            conn: None,
            db_path: path,
            degraded_reason: Some(DegradedReason::DatabaseNotFound),
            degraded_message: Some(msg),
        }
    }

    fn connect(path: &Path) -> rusqlite::Result<Connection> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        // Defensive busy_timeout (RISK-15): avoids SQLITE_BUSY when the
        // daemon holds a write lock briefly during checkpointing.
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        Ok(conn)
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    /// Returns the underlying connection, or `None` when degraded.
    ///
    /// Query modules accept `Option<&Connection>` and return empty results
    /// when it is `None`, matching the TS `if (!db) return []` pattern.
    pub fn connection(&self) -> Option<&Connection> {
        self.conn.as_ref()
    }

    /// Filesystem path of the database.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// `true` when the database is open and usable.
    pub fn is_connected(&self) -> bool {
        self.conn.is_some()
    }

    /// Initialization result suitable for callers that need to surface health
    /// information (mirrors `initialize()` in `sqlite-state-manager.ts`).
    pub fn init_result(&self) -> QueryResult<bool> {
        if self.is_connected() {
            QueryResult::ok(true)
        } else {
            QueryResult::degraded(
                false,
                self.degraded_reason
                    .clone()
                    .unwrap_or(DegradedReason::DatabaseError),
                self.degraded_message.clone().unwrap_or_default(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Create an ephemeral WAL-mode SQLite database with the minimal schema
    /// for testing, then close and reopen READ-ONLY (mirrors the real flow).
    pub(crate) fn make_wal_db(dir: &TempDir) -> (PathBuf, Connection) {
        let path = dir.path().join("state.db");

        // Write-mode to create + set WAL
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .unwrap();
        setup
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS unified_queue (
                queue_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                lease_until TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
            )
            .unwrap();
        drop(setup);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        (path, conn)
    }

    #[test]
    fn open_at_missing_file_is_degraded() {
        let mgr = StateManager::open_at("/nonexistent/path/state.db");
        assert!(!mgr.is_connected());
        assert!(mgr.connection().is_none());
        let result = mgr.init_result();
        assert_eq!(result.status, QueryStatus::Degraded);
        assert_eq!(result.reason, Some(DegradedReason::DatabaseNotFound));
        assert!(result
            .message
            .unwrap()
            .contains("Daemon has not initialized yet"));
    }

    #[test]
    fn open_at_existing_wal_db_succeeds() {
        let dir = TempDir::new().unwrap();
        let (path, _conn) = make_wal_db(&dir);
        drop(_conn);

        let mgr = StateManager::open_at(&path);
        assert!(mgr.is_connected());
        assert!(mgr.connection().is_some());
        let result = mgr.init_result();
        assert_eq!(result.status, QueryStatus::Ok);
        assert!(result.data);
    }

    #[test]
    fn db_path_is_returned() {
        let dir = TempDir::new().unwrap();
        let (path, _) = make_wal_db(&dir);
        let mgr = StateManager::open_at(&path);
        assert_eq!(mgr.db_path(), &path);
    }

    #[test]
    fn connection_is_read_only() {
        let dir = TempDir::new().unwrap();
        let (path, _) = make_wal_db(&dir);
        let mgr = StateManager::open_at(&path);
        let conn = mgr.connection().unwrap();
        // Attempting a write on a read-only connection must fail.
        let result = conn.execute("INSERT INTO unified_queue VALUES ('x','t','o','tenant','coll','pending',NULL,datetime('now'),datetime('now'))", []);
        assert!(
            result.is_err(),
            "expected write on read-only connection to fail"
        );
    }

    #[test]
    fn busy_timeout_set() {
        // Verify the manager opens without panicking (timeout is set internally).
        let dir = TempDir::new().unwrap();
        let (path, _) = make_wal_db(&dir);
        let mgr = StateManager::open_at(&path);
        assert!(mgr.is_connected());
    }
}
